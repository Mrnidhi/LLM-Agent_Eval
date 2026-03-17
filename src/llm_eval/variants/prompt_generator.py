import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict

from llm_eval.core.config import load_agent_configuration
from llm_eval.core.model import AIModel
from llm_eval.core.tracing import configure_logging, configure_tracing

load_dotenv()

logger = configure_logging()
tracer = configure_tracing("llm-eval")


class Prompt(TypedDict):
    prompt: str
    description: str


class PromptGeneratorOutput(TypedDict):
    prompts: List[Prompt]


class PromptGenerator:
    def __init__(self, agent_config: dict = None) -> None:
        logger.info("Initializing PromptGenerator")
        with tracer.start_as_current_span("PromptGenerator.__init__") as span:
            try:
                if agent_config is None:
                    agent_config = load_agent_configuration("agents/prompt_generator", "prompt_generator_config.yaml")

                span.set_attribute("agent_config", str(agent_config))

                self.api_key = os.getenv("AZURE_OPENAI_KEY")
                if agent_config is None or self.api_key is None:
                    raise ValueError("agent config and api_key are required")

                self.agent_config = agent_config

                self.aimodel = AIModel(
                    azure_deployment=agent_config["AgentConfiguration"]["deployment"]["name"],
                    openai_api_version=agent_config["AgentConfiguration"]["deployment"]["openai_api_version"],
                    azure_endpoint=agent_config["AgentConfiguration"]["deployment"]["endpoint"],
                    api_key=self.api_key,
                    model_parameters={"temperature": agent_config["AgentConfiguration"]["model_parameters"]["temperature"]},
                )
            except Exception as e:
                logger.error("PromptGenerator init failed: %s", e)
                raise

    def __call__(self, prompt: str, evaluation_dataset: str) -> dict:
        return self.generate_prompts(prompt, evaluation_dataset)

    def load_prompt(self, config_path: str, config_file_name: str, prompt_attribute_name: str) -> str:
        agent_config = load_agent_configuration(config_path, config_file_name)
        return agent_config["AgentConfiguration"][prompt_attribute_name]

    def generate_prompts(self, prompt: str, evaluation_dataset: pd.DataFrame) -> dict:
        """Uses LLM to create improved prompt variants based on the current prompt and eval results."""
        logger.info("Generating prompt variants")
        with tracer.start_as_current_span("PromptGenerator.generate_prompts") as span:
            # system_prompt in the config contains instructions for the LLM on how to generate better prompts
            prompt_template = PromptTemplate.from_template(self.agent_config["AgentConfiguration"]["system_prompt"])
            span.set_attribute("prompt", prompt)
            chain = prompt_template | self.aimodel.llm().with_structured_output(PromptGeneratorOutput)
            return chain.invoke({"prompt": prompt, "evaluation_dataset": evaluation_dataset.to_json(orient="records")})


if __name__ == "__main__":
    pg = PromptGenerator()
    prompt = pg.load_prompt("agents/rag", "rag_agent_config.yaml", "chat_system_prompt")
    print(f"Prompt = {prompt}")
