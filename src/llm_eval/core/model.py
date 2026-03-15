from logging import getLogger

from langchain_openai import AzureChatOpenAI

from llm_eval.core.tracing import configure_tracing

logger = getLogger(__name__)
tracer = configure_tracing("llm-eval")


class AIModel:
    def __init__(self, azure_deployment, openai_api_version, azure_endpoint, api_key, model_parameters: dict) -> None:
        logger.info("Initializing AIModel: deployment=%s", azure_deployment)

        with tracer.start_as_current_span("AIModel.__init__"):
            self._llm = AzureChatOpenAI(
                azure_deployment=azure_deployment,
                openai_api_version=openai_api_version,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                temperature=model_parameters.get("temperature", 0),
            )

    def llm(self) -> AzureChatOpenAI:
        return self._llm
