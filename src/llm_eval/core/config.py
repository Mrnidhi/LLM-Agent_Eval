import importlib
import logging
import os

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def load_agent_configuration(agent_folder: str, agent_config_file: str) -> dict:
    if not agent_folder or not agent_config_file:
        raise ValueError("Agent folder and agent config file are required.")

    # Resolves relative to project root (two levels up from src/llm_eval/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, agent_folder, agent_config_file)

    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("Error parsing agent config YAML: %s", e)
            raise


def import_from_path(full_path: str):
    """Dynamically import a class or function from a dotted module path like 'llm_eval.agents.rag.agent.RAG'."""
    module_path, attr_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def configure_aoai_env():
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    api_key = os.environ.get("AZURE_OPENAI_KEY")

    if not all([api_version, api_key]):
        raise EnvironmentError("AZURE_OPENAI_API_VERSION and AZURE_OPENAI_KEY must be set.")

    return {"api_key": api_key, "api_version": api_version}


def configure_embedding_env():
    endpoint = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_KEY")
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if not all([endpoint, api_key, deployment]):
        raise EnvironmentError("Embedding environment variables (AZURE_OPENAI_EMBEDDING_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_EMBEDDING_DEPLOYMENT) must be set.")

    return {
        "embedding_model_endpoint": endpoint,
        "embedding_model_api_key": api_key,
        "embedding_model_name": deployment,
    }


def configure_aisearch_env():
    endpoint = os.environ.get("AZURE_AI_SEARCH_SERVICE_ENDPOINT")
    api_key = os.environ.get("AZURE_AI_SEARCH_API_KEY")
    index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME")
    service_name = os.environ.get("AZURE_AI_SEARCH_SERVICE_NAME")

    if not all([endpoint, api_key, index_name, service_name]):
        raise EnvironmentError("Azure AI Search environment variables must be set.")

    return {
        "azure_search_endpoint": endpoint,
        "azure_search_api_key": api_key,
        "azure_search_index_name": index_name,
        "azure_search_service_name": service_name,
    }


def configure_docintell_env():
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    api_version = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION")

    if not all([endpoint, key, api_version]):
        raise EnvironmentError("Document Intelligence environment variables must be set.")

    return {
        "doc_intelligence_endpoint": endpoint,
        "doc_intelligence_key": key,
        "doc_intelligence_api_version": api_version,
    }
