import logging
import os

from azure.ai.inference.tracing import AIInferenceInstrumentor
from azure.core.settings import settings
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry import trace
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_tracing(collection_name: str = "llm-eval"):
    settings.tracing_implementation = "opentelemetry"

    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("APPLICATIONINSIGHTS_CONNECTION_STRING environment variable is not set.")

        os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"] = connection_string
        os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true"

        configure_azure_monitor(collection_name=collection_name)
        AIInferenceInstrumentor().instrument()

        langchain_instrumentor = LangchainInstrumentor()
        if not langchain_instrumentor.is_instrumented_by_opentelemetry:
            langchain_instrumentor.instrument()

        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

        traces_exporter = AzureMonitorTraceExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(traces_exporter))

        RequestsInstrumentor().instrument()

    return trace.get_tracer(__name__)


def configure_logging():
    logger = logging.getLogger("llm_eval")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler("app.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry.attributes").setLevel(logging.ERROR)
    logging.getLogger("opentelemetry.instrumentation.instrumentor").setLevel(logging.ERROR)
    logging.getLogger("opentelemetry.trace").setLevel(logging.ERROR)
    logging.getLogger("opentelemetry.metrics").setLevel(logging.ERROR)

    return logger
