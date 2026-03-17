import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm_eval.core.config import load_agent_configuration
from llm_eval.core.model import AIModel
from llm_eval.core.search import AISearch
from llm_eval.core.session import SimpleInMemorySessionStore
from llm_eval.core.tracing import configure_logging, configure_tracing

load_dotenv()

logger = configure_logging()
tracer = configure_tracing("llm-eval")


class RAG:
    def __init__(self, rag_config: dict = None) -> None:
        logger.info("Initializing RAG agent")
        with tracer.start_as_current_span("RAG.__init__") as span:
            try:
                if rag_config is None:
                    rag_config = load_agent_configuration("agents/rag", "rag_agent_config.yaml")

                span.set_attribute("agent_config", str(rag_config))

                self.api_key = os.getenv("AZURE_OPENAI_KEY")
                if rag_config is None or self.api_key is None:
                    raise ValueError("agent config and api_key are required")

                self.rag_config = rag_config
                cfg = rag_config["AgentConfiguration"]

                self.aimodel = AIModel(
                    azure_deployment=cfg["deployment"]["name"],
                    openai_api_version=cfg["deployment"]["openai_api_version"],
                    azure_endpoint=cfg["deployment"]["endpoint"],
                    api_key=self.api_key,
                    model_parameters={"temperature": cfg["model_parameters"]["temperature"]},
                )

                self.aisearch = AISearch(
                    cfg["retrieval"]["deployment"]["name"],
                    cfg["retrieval"]["deployment"]["endpoint"],
                    cfg["retrieval"]["parameters"]["index_name"],
                    cfg["retrieval"]["parameters"]["index_semantic_configuration_name"],
                )

                self._session_store = SimpleInMemorySessionStore()

                # Step 1: Reformulate the user's latest question into a standalone query
                # using chat history context (handles follow-up questions like "what about its pricing?")
                self._user_intent_prompt_template = ChatPromptTemplate.from_messages([
                    ("system", cfg["intent_system_prompt"]),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                # If there's no chat_history, input goes directly to the retriever.
                # Otherwise the LLM reformulates the question first, then retrieves.
                self._history_aware_user_intent_retriever = create_history_aware_retriever(
                    self.aimodel.llm(),
                    self.aisearch.create_retriever(
                        cfg["retrieval"]["parameters"]["search_type"],
                        cfg["retrieval"]["parameters"]["top_k"],
                    ),
                    self._user_intent_prompt_template,
                )

                self._chat_prompt_template = ChatPromptTemplate.from_messages([
                    ("system", cfg["chat_system_prompt"]),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                self._question_answer_chain = create_stuff_documents_chain(
                    self.aimodel.llm(), self._chat_prompt_template
                )

                self._rag_chain = create_retrieval_chain(
                    self._history_aware_user_intent_retriever, self._question_answer_chain
                )

                # These keys must match the placeholders in the prompt templates above
                self._conversational_rag_chain = RunnableWithMessageHistory(
                    self._rag_chain,
                    self.get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
            except Exception as e:
                logger.exception("RAG init failed: %s", e)
                raise

    def __call__(self, session_id: str, question: str = " ") -> str:
        with tracer.start_as_current_span("RAG.__call__") as span:
            span.set_attribute("session_id", session_id)
            return self.chat(session_id, question)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._session_store.get_all_session_ids():
            self._session_store.create_session(session_id)
        return self._session_store.get_session(session_id)

    def chat(self, session_id, question, **kwargs):
        logger.info("chat: session_id=%s, question=%s", session_id, question)
        with tracer.start_as_current_span("RAG.chat") as span:
            try:
                cfg = self.rag_config["AgentConfiguration"]
                # These span attributes are used by Fabric dashboards to compare evaluations across config versions
                span.set_attribute("session_id", session_id)
                span.set_attribute("application_name", cfg["application_name"])
                span.set_attribute("config_version", cfg["config_version"])
                span.set_attribute("endpoint", cfg["deployment"]["endpoint"])

                response = self._conversational_rag_chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": session_id}},
                )
            except Exception as e:
                logger.error("chat error: %s", e)
                raise
            return response["answer"]


if __name__ == "__main__":
    rag = RAG()
