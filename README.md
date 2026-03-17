# LLM Agents Evaluation

A framework for evaluating LLM-based agents. It lets you run agents against test datasets, score results with metrics, compare different configurations, and pick the best one.

The project includes a RAG (Retrieval Augmented Generation) agent as a working example, but the evaluation tools work with any agent that follows the same interface.

## What it does

- Runs an agent on a test dataset and collects outputs
- Scores outputs using four metrics: relevance, groundedness, similarity, coherence
- Generates multiple agent configurations (different prompts, models, parameters)
- Evaluates all configurations and picks the best performer
- Sends traces and logs to Azure Application Insights for monitoring

## Project structure

```
├── src/llm_eval/            # main Python package
│   ├── core/                # config loading, tracing, model/search/session wrappers
│   ├── agents/rag/          # RAG agent implementation + Prompt Flow config
│   ├── evaluation/          # evaluation runner and metrics
│   ├── orchestrator/        # runs multi-variant evaluation end-to-end
│   ├── variants/            # generates config variants and prompt variants
│   └── ingestion/           # ingests documents into Azure AI Search
├── docs/                    # project documentation and screenshots
├── examples/                # sample configs, evaluation outputs
├── data/sample_docs/        # sample PDF documents for ingestion
├── infrastructure/
│   ├── kusto/               # KQL scripts for Azure Data Explorer / Fabric
│   └── fabric/              # Microsoft Fabric dashboard and pipeline assets
├── pyproject.toml           # Python dependencies
└── .env.example             # environment variable template
```

## Prerequisites

- Python 3.11+
- Azure AI Foundry project
- Azure AI Search instance
- Azure Document Intelligence instance
- Azure OpenAI deployment

## Setup

```sh
git clone <repo-url>
cd llm-agents-evaluation
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# fill in your Azure credentials in .env
```

## Ingest documents

Upload PDF files to an Azure AI Search index:

```sh
python -m llm_eval.ingestion.chunking \
  --index_name <index-name> \
  --input_folder <path-to-pdfs>
```

## Run the RAG agent locally

Using Prompt Flow:

```sh
pf flow serve --source ./src/llm_eval/agents/rag/ --port 8080 --host localhost
```

This opens a chat UI at `http://localhost:8080`.

## Run evaluation

### Single configuration

Evaluate one agent configuration against a test dataset:

```sh
python -m llm_eval.evaluation.runner \
  --agent_class llm_eval.agents.rag.agent.RAG \
  --eval_fn llm_eval.evaluation.metrics.eval_batch \
  --config_dir agents/rag \
  --config_file rag_agent_config.yaml \
  --eval_dataset ./src/llm_eval/evaluation/data/rag_eval.jsonl \
  --dump_output \
  --mode single
```

Parameters:
- `--agent_class`: dotted path to the agent class
- `--eval_fn`: dotted path to the evaluation function
- `--config_dir`: folder containing the agent config YAML
- `--config_file`: config file name
- `--eval_dataset`: path to JSONL test dataset
- `--dump_output`: save results to JSON files
- `--mode`: `single` or `multiple`

### Multiple configurations

Evaluate many configs at once:

```sh
python -m llm_eval.evaluation.runner \
  --agent_class llm_eval.agents.rag.agent.RAG \
  --eval_fn llm_eval.evaluation.metrics.eval_batch \
  --config_dir agents/rag/evaluation/configurations/generated \
  --eval_dataset ./src/llm_eval/evaluation/data/rag_eval.jsonl \
  --mode multiple
```

## Run the orchestrator

The orchestrator automates the full cycle: run baseline evaluation, generate prompt variants, create config variants, evaluate all of them, and pick the best one.

```sh
python -m llm_eval.orchestrator.orchestrator \
  --agent_class llm_eval.agents.rag.agent.RAG \
  --eval_fn llm_eval.evaluation.metrics.eval_batch \
  --agent_config_file_dir agents/rag \
  --agent_config_file_name rag_agent_config.yaml \
  --evaluation_dataset ./src/llm_eval/evaluation/data/rag_eval.jsonl \
  --base_variant ./examples/configs/variants.json \
  --output_dir ./agents/rag/evaluation/configurations/generated
```

## Agent configuration

Each agent uses a YAML config file that controls its behavior. See `examples/configs/rag_agent.yaml` for a full example. Key fields:

- `deployment`: model name, endpoint, API version
- `model_parameters`: temperature, seed
- `retrieval`: search type, top_k, index name, embedding model
- `intent_system_prompt`: prompt for reformulating user questions
- `chat_system_prompt`: prompt for generating answers

Changing any of these creates a new configuration version that gets tracked in logs.

## Evaluation metrics

All metrics are scored 1-5 using an LLM judge (Azure AI Evaluation SDK):

| Metric | What it measures |
|---|---|
| Relevance | Does the response answer the question accurately and completely? |
| Groundedness | Is the response consistent with the provided context? |
| Similarity | How close is the response to the expected answer? |
| Coherence | Is the response logically organized and easy to follow? |

## Test dataset format

JSONL file where each line is:

```json
{
  "session_id": "1",
  "question": "What is X?",
  "answer": "Expected answer text",
  "context": "Background information for the question"
}
```

## Monitoring

Set `APPLICATIONINSIGHTS_CONNECTION_STRING` in `.env` to send traces to Azure Application Insights. The project uses OpenTelemetry and logs agent config versions, token usage, and evaluation metrics to each trace.

For dashboards, see `infrastructure/fabric/` for Microsoft Fabric Real-Time dashboard definitions and `infrastructure/kusto/` for KQL table creation scripts.

## Services used

- Azure OpenAI: chat and embedding models
- Azure AI Search: document retrieval
- Azure Document Intelligence: document chunking
- Azure Application Insights: tracing and logs
- Microsoft Fabric: dashboards and analytics
