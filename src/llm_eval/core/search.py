import atexit
import os
from logging import getLogger

from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
)
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from opentelemetry import trace

load_dotenv()

logger = getLogger(__name__)
tracer = trace.get_tracer(__name__)

AZURE_AI_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_AI_SEARCH_SERVICE_ENDPOINT")
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

_required_env_vars = [
    "AZURE_AI_SEARCH_SERVICE_ENDPOINT",
    "AZURE_AI_SEARCH_API_KEY",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_API_VERSION",
]

for var in _required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Environment variable {var} is not set.")


class AISearch:
    def __init__(self, embedding_deployment: str, embedding_endpoint: str, index_name: str, index_semantic_configuration_name: str) -> None:
        logger.info("Initializing AISearch: index=%s", index_name)
        self.index_name = index_name
        self.index_semantic_configuration_name = index_semantic_configuration_name

        self._embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            azure_endpoint=embedding_endpoint,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_KEY,
        )

        self._fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="metadata", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="chunk", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="text_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(self._embeddings.embed_query("Text")),
                vector_search_profile_name="myHnswProfile",
            ),
            SearchableField(name="title", type=SearchFieldDataType.String, searchable=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        ]

        try:
            self._vector_search = AzureSearch(
                azure_search_endpoint=AZURE_AI_SEARCH_SERVICE_ENDPOINT,
                azure_search_key=AZURE_AI_SEARCH_API_KEY,
                index_name=index_name,
                embedding_function=self._embeddings.embed_query,
                semantic_configuration_name=index_semantic_configuration_name or None,
                additional_search_client_options={"retry_total": 3, "logging_enable": True, "logger": logger},
                fields=self._fields,
            )
            atexit.register(self._close)
        except Exception as e:
            logger.error("Error initializing AI Search index: %s", e)
            raise RuntimeError(f"Error initializing AI Search index: {e}") from e

    def _close(self) -> None:
        logger.info("Closing Azure Search client.")

    def create_retriever(self, search_type: str, top_k=3):
        return self._vector_search.as_retriever(search_type=search_type, k=top_k)

    def ingest(self, documents: list, **kwargs) -> None:
        if not isinstance(documents, list) or not documents:
            raise ValueError("Input must be a non-empty list")
        self._vector_search.add_documents(documents)

    def search(self, query: str, search_type: str = "hybrid", top_k: int = 5) -> str:
        logger.info("Searching: query=%s", query)
        with tracer.start_as_current_span("aisearch") as span:
            if not isinstance(query, str) or not query:
                raise ValueError("Search query must be a non-empty string")
            span.set_attribute("ai_search_query", query)

            docs = self._vector_search.similarity_search(query=query, k=top_k, search_type=search_type)
            return "\t".join(doc.page_content for doc in docs)
