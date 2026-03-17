import os

import pandas as pd
from azure.ai.evaluation import (
    CoherenceEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    SimilarityEvaluator,
)
from dotenv import load_dotenv

from llm_eval.core.tracing import configure_logging

load_dotenv()

logger = configure_logging()

# Uses a separate deployment for evaluation (LLM-as-judge) to avoid rate-limit conflicts
# with the agent's own model. Set AZURE_OPENAI_KEY_EVALUATION and related vars in .env
model_config = {
    "azure_endpoint": os.getenv("AZURE_OPENAI_EVALUATION_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_KEY_EVALUATION"),
    "azure_deployment": os.getenv("AZURE_OPENAI_EVALUATION_DEPLOYMENT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
}

evaluators = {
    "relevance": RelevanceEvaluator(model_config),
    "groundedness": GroundednessEvaluator(model_config),
    "similarity": SimilarityEvaluator(model_config),
    "coherence": CoherenceEvaluator(model_config),
}


# Each evaluator returns a dict; the score key is prefixed with "gpt_" by the Azure SDK
def relevance(row):
    return evaluators["relevance"](
        response=row["outputs.output"], context=row["context"], query=row["question"]
    )["gpt_relevance"]


def groundedness(row):
    return evaluators["groundedness"](
        response=row["outputs.output"], context=row["context"]
    )["gpt_groundedness"]


def similarity(row):
    return evaluators["similarity"](
        response=row["outputs.output"], ground_truth=row["answer"], query=row["question"]
    )["gpt_similarity"]


def coherence(row):
    return evaluators["coherence"](
        response=row["outputs.output"], query=row["question"]
    )["gpt_coherence"]


evaluator_funcs = {
    "relevance": relevance,
    "groundedness": groundedness,
    "similarity": similarity,
    "coherence": coherence,
}


def calc_score(scores):
    return sum(scores) / len(scores) if scores else 0


def calculate_overall_score(scores):
    calc_scores = {"metric": [], "score": []}
    for key, values in scores.items():
        calc_scores["metric"].append(key)
        calc_scores["score"].append(calc_score(values))
    return pd.DataFrame(calc_scores)


def eval_batch(batch_output: pd.DataFrame, dump_output: bool = False):
    """Expects a DataFrame with columns: question, answer, context, outputs.output.
    Returns (detailed_results_df, aggregated_metrics_df)."""
    results = []
    scores = {key: [] for key in evaluator_funcs}

    try:
        for _, row in batch_output.iterrows():
            new_row = row.to_dict()
            for eval_name, func in evaluator_funcs.items():
                score = func(row)
                new_row[eval_name] = score
                scores[eval_name].append(score)
            results.append(new_row)

        eval_res = pd.DataFrame(results)
        eval_metrics = calculate_overall_score(scores)
    except Exception as e:
        logger.exception("Error during batch evaluation: %s", e)
        raise

    if dump_output:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        eval_res.to_json(f"eval_results_{timestamp}.json", orient="records", lines=True)
        eval_metrics.to_json(f"batch_eval_output_{timestamp}.json", orient="records", lines=True)

    return eval_res, eval_metrics
