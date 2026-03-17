import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict

from llm_eval.core.config import import_from_path, load_agent_configuration
from llm_eval.core.model import AIModel
from llm_eval.core.tracing import configure_logging, configure_tracing
from llm_eval.evaluation.runner import multi_variant_evaluation, run_and_eval_flow
from llm_eval.variants.generator import generate_variants
from llm_eval.variants.prompt_generator import PromptGenerator

logger = configure_logging()
tracer = configure_tracing("llm-eval")

PROMPT_GENERATOR_FOLDER = "agents/prompt_generator"
PROMPT_GENERATOR_CONFIG_FILE = "prompt_generator_config.yaml"
CONFIG_SCHEMA = os.path.join("agents", "schemas", "agent_config.yaml")
NUMBER_OF_VARIANTS_GENERATED = 50


class OrchestratorSingleOutput(TypedDict):
    best_agent_variant: str
    reasoning: str


class OrchestratorOutput(TypedDict):
    prompts: List[OrchestratorSingleOutput]


class Orchestrator:
    def __init__(self, agent_config: Optional[Dict] = None) -> None:
        logger.info("Initializing Orchestrator")
        with tracer.start_as_current_span("Orchestrator.__init__") as span:
            try:
                if agent_config is None:
                    agent_config = load_agent_configuration(
                        "agents/orchestrator", "evaluation_orchestrator_agent_config.yaml"
                    )

                api_key = os.getenv("AZURE_OPENAI_KEY")
                if agent_config is None or api_key is None:
                    raise ValueError("Agent config and api_key are required")

                self.agent_config = agent_config
                span.set_attribute("orchestrator_agent_config", str(agent_config))

                self.aimodel = AIModel(
                    azure_deployment=agent_config["AgentConfiguration"]["deployment"]["name"],
                    openai_api_version=agent_config["AgentConfiguration"]["deployment"]["openai_api_version"],
                    azure_endpoint=agent_config["AgentConfiguration"]["deployment"]["endpoint"],
                    api_key=api_key,
                    model_parameters={"temperature": agent_config["AgentConfiguration"]["model_parameters"]["temperature"]},
                )
            except Exception as e:
                logger.exception("Error initializing Orchestrator: %s", e)
                raise

    def find_optimal_agent_configuration(
        self,
        agent: Type,
        eval_fn: Callable[[pd.DataFrame, bool], Tuple[pd.DataFrame, pd.DataFrame]],
        agent_config_file_dir: str,
        agent_config_file_name: str,
        evaluation_dataset: str,
        base_variant: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Runs: baseline eval -> generate prompts -> generate variants -> multi-eval -> return results JSON."""
        # Step 1: Baseline evaluation with current config
        eval_res = run_and_eval_flow(
            agent, eval_fn, agent_config_file_dir, agent_config_file_name, evaluation_dataset, dump_output=False
        )

        # Step 2: Use baseline results to generate improved prompt variants via LLM
        pgen = PromptGenerator(load_agent_configuration(PROMPT_GENERATOR_FOLDER, PROMPT_GENERATOR_CONFIG_FILE))
        evaluated_agent_config = load_agent_configuration(agent_config_file_dir, agent_config_file_name)
        generated_prompts = pgen.generate_prompts(
            evaluated_agent_config["AgentConfiguration"]["chat_system_prompt"], eval_res
        )

        # Step 3: Generate YAML config variants (cartesian product of models, params, prompts)
        try:
            with open(base_variant, "r", encoding="utf-8") as f:
                variants = json.load(f)
            generate_variants(
                CONFIG_SCHEMA, agent_config_file_dir, agent_config_file_name,
                NUMBER_OF_VARIANTS_GENERATED, generated_prompts, variants, output_dir,
            )
        except Exception as e:
            logger.exception("Error generating variants: %s", e)
            raise

        # Step 4: Evaluate all generated variants in parallel
        all_results = multi_variant_evaluation(agent, eval_fn, output_dir, evaluation_dataset)
        evaluation_results = json.dumps(all_results, default=self._serializer)
        return evaluation_results

    def analyze(self, evaluation_results: str) -> Any:
        """Uses LLM to review all variant results and pick the best one with reasoning."""
        logger.info("Analyzing evaluation results")
        with tracer.start_as_current_span("Orchestrator.analyze") as span:
            prompt_template = PromptTemplate.from_template(
                self.agent_config["AgentConfiguration"]["system_prompt"]
            )
            span.set_attribute("evaluation_results", evaluation_results)
            chain = prompt_template | self.aimodel.llm().with_structured_output(OrchestratorOutput)
            return chain.invoke({"evaluation_results": evaluation_results})

    def _serializer(self, obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        raise TypeError("Type not serializable")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation orchestrator.")
    parser.add_argument("--agent_class", type=str, required=True, help="Full module path to agent class")
    parser.add_argument("--eval_fn", type=str, required=True, help="Full module path to evaluation function")
    parser.add_argument("--agent_config_file_dir", type=str, required=True, help="Agent config directory")
    parser.add_argument("--agent_config_file_name", type=str, required=True, help="Agent config file name")
    parser.add_argument("--evaluation_dataset", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--base_variant", type=str, required=True, help="Path to variant definitions JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated configs")
    args = parser.parse_args()

    agent_class = import_from_path(args.agent_class)
    eval_fn = import_from_path(args.eval_fn)
    orchestrator = Orchestrator()

    evaluation_results = orchestrator.find_optimal_agent_configuration(
        agent=agent_class,
        eval_fn=eval_fn,
        agent_config_file_dir=args.agent_config_file_dir,
        agent_config_file_name=args.agent_config_file_name,
        evaluation_dataset=args.evaluation_dataset,
        base_variant=args.base_variant,
        output_dir=args.output_dir,
    )
    answer = orchestrator.analyze(evaluation_results)
    logger.info("Best configuration: %s", answer)


if __name__ == "__main__":
    main()
