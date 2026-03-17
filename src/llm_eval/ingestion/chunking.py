import argparse
import os
from io import BytesIO
from typing import List

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureOpenAIEmbeddings

from llm_eval.core.config import configure_aoai_env, configure_docintell_env, configure_embedding_env
from llm_eval.core.search import AISearch

load_dotenv()


def process_document(doc_path, docintel_env, mode="markdown"):
    """Converts a PDF to markdown via Document Intelligence, then splits on headers."""
    loader = AzureAIDocumentIntelligenceLoader(
        file_path=doc_path,
        api_key=docintel_env["doc_intelligence_key"],
        api_endpoint=docintel_env["doc_intelligence_endpoint"],
        api_model="prebuilt-layout",
        api_version=docintel_env["doc_intelligence_api_version"],
        mode=mode,
    )
    docs = loader.load()

    # Split by H1/H2/H3 headers; add more levels here if your docs use deeper nesting
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = text_splitter.split_text(docs[0].page_content)
    print(f"Document split into {len(chunks)} chunks")
    return chunks


def get_files_from_blob_storage(storage_account_name: str, container_name: str, folder_name: str) -> List[str]:
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs(name_starts_with=folder_name)
    return [blob.name for blob in blobs]


def download_blob(storage_account_name: str, container_name: str, blob_name: str, download_path: str):
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(download_path, "wb") as f:
        f.write(blob_client.download_blob().readall())


def download_blob_to_memory(storage_account_name: str, container_name: str, blob_name: str):
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return BytesIO(blob_client.download_blob().readall())


def ingest_files_from_blob(aisearch: AISearch, docintel_env: dict, storage_account_name: str,
                           container_name: str, folder_name: str = None, mode: str = "markdown"):
    files = get_files_from_blob_storage(storage_account_name, container_name, folder_name)
    all_chunks = []
    for file in files:
        local_path = os.path.join("/tmp", os.path.basename(file))
        download_blob(storage_account_name, container_name, file, local_path)
        chunks = process_document(local_path, docintel_env, mode)
        aisearch.ingest(chunks)
        all_chunks.extend(chunks)
        os.remove(local_path)
    return all_chunks


def ingest_files_from_local_folder(aisearch: AISearch, docintel_env: dict,
                                   folder_path: str, mode: str = "markdown"):
    files = os.listdir(folder_path)
    if not files:
        print("No files found in folder")
        return []

    all_chunks = []
    for file in files:
        local_path = os.path.join(folder_path, file)
        print(f"Processing: {local_path}")
        chunks = process_document(local_path, docintel_env, mode)
        aisearch.ingest(chunks)
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    embedding_env = configure_embedding_env()
    aoai_env = configure_aoai_env()
    docintel_env = configure_docintell_env()

    parser = argparse.ArgumentParser(description="Ingest documents into AI Search index.")
    parser.add_argument("--index_name", required=True, help="Search index name")
    parser.add_argument("--index_semantic_configuration_name", required=False, help="Semantic configuration name")
    parser.add_argument("--input_folder", required=True, help="Input folder with documents")
    args = parser.parse_args()

    aisearch = AISearch(
        embedding_deployment=embedding_env["embedding_model_name"],
        embedding_endpoint=embedding_env["embedding_model_endpoint"],
        index_name=args.index_name,
        index_semantic_configuration_name=args.index_semantic_configuration_name or None,
    )

    ingest_files_from_local_folder(aisearch, docintel_env, args.input_folder)
