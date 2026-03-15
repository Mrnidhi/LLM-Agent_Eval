import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Tuple, Type

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_eval.core.config import import_from_path, load_agent_configuration
from llm_eval.core.tracing import configure_logging, configure_tracing

TRACING_COLLECTION_NAME = "agent_evaluation"

logger = configure_logging()
tracer = configure_tracing(collection_name=TRACING_COLLECTION_NAME)


def run_evaluation_ds(agent_class: Type, agent_config: dict, evaluation_dataset_path: str, dump_output: bool = False) -> pd.DataFrame:
    logger.info("Running batch evaluation on dataset.")

    if not Path(evaluation_dataset_path).exists():
        raise ValueError(f"Dataset path does not exist: {evaluation_dataset_path}")

    df = pd.read_json(evaluation_dataset_path, lines=True)
    outputs = []
    for _, row in df.iterrows():
        agent_instance = agent_class(agent_config)
        output = agent_instance(row["session_id"], row["question"])
        outputs.append(output)
    df["outputs.output"] = outputs

    if dump_output:
        df.to_json("batch_flow_output.json", index=False)
    return df


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2), retry=retry_if_exception_type(Exception))
def run_evaluation_ds_with_retry(agent_class, agent_config, evaluation_dataset, dump_output=False):
    return run_evaluation_ds(agent_class, agent_config, evaluation_dataset, dump_output=dump_output)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2), retry=retry_if_exception_type(Exception))
def evaluate_output_with_retry(eval_fn, batch_output, dump_output=False):
    return eval_fn(batch_output, dump_output=dump_output)


def run_and_eval_flow(agent_class: Type, eval_fn: Callable, agent_config_file_dir: str,
                      agent_config_file_name: str, agent_evaluation_dataset: str,
                      dump_output: bool = False):
    with tracer.start_as_current_span("batch::evaluation::run_and_eval_flow"):
        agent_config = load_agent_configuration(agent_config_file_dir, agent_config_file_name)

        try:
            batch_output = run_evaluation_ds_with_retry(agent_class, agent_config, agent_evaluation_dataset, dump_output=dump_output)
            eval_res, eval_metrics = evaluate_output_with_retry(eval_fn, batch_output, dump_output=dump_output)
        except Exception as e:
            logger.error("Error processing agent %s: %s", agent_config_file_name, e)
            raise

        logger.info(json.dumps({
            "name": "batch-evaluation-flow-raw",
            "metadata": agent_config,
            "result": eval_res.to_dict(orient="records"),
        }))
        logger.info(json.dumps({
            "name": "batch-evaluation-flow-metrics",
            "metadata": agent_config,
            "result": eval_metrics.to_dict(orient="records"),
        }))

        return eval_res


def multi_variant_evaluation(agent_class: Type, eval_fn: Callable, variants_path: str, evaluation_dataset: str):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    multi_config_path = os.path.join(project_root, variants_path)

    files = [f for f in os.listdir(multi_config_path) if f.endswith(".yaml")]
    logger.info("Evaluating %d variant files", len(files))

    all_eval_results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(run_and_eval_flow, agent_class, eval_fn, variants_path, f, evaluation_dataset, False): f
            for f in files
        }
        for future in as_completed(futures):
            file = futures[future]
            try:
                all_eval_results[file] = future.result()
            except Exception as e:
                logger.error("Error processing %s: %s", file, e)
                raise

    return all_eval_results


def main():
    parser = argparse.ArgumentParser(description="Run agent evaluation.")
    parser.add_argument("--agent_class", type=str, required=True, help="Full module path to agent class")
    parser.add_argument("--eval_fn", type=str, required=True, help="Full module path to evaluation function")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory with agent config YAML")
    parser.add_argument("--config_file", type=str, required=False, help="Agent config file name (single mode)")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--dump_output", action="store_true", default=False, help="Save results to JSON files")
    parser.add_argument("--mode", choices=["single", "multiple"], default="single", help="Evaluation mode")
    args = parser.parse_args()

    agent_class = import_from_path(args.agent_class)
    eval_fn = import_from_path(args.eval_fn)

    if args.mode == "single":
        results = run_and_eval_flow(agent_class, eval_fn, args.config_dir, args.config_file, args.eval_dataset, dump_output=args.dump_output)
    else:
        results = multi_variant_evaluation(agent_class, eval_fn, args.config_dir, args.eval_dataset)

    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
