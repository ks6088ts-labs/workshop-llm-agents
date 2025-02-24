import logging
from os import getenv

import typer
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import (
    ChainedTokenCredential,
    ClientSecretCredential,
    DefaultAzureCredential,
)
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.azure_cosmos_db_no_sql import AzureCosmosDBNoSqlVectorSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)
logger = logging.getLogger(__name__)
app = typer.Typer()


def get_vector_embedding_policy():
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": 3072,  # for text-embedding-3-large
            }
        ]
    }


def get_indexing_policy():
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
    }


def get_cosmos_client(service_principal: bool) -> CosmosClient:
    if service_principal:
        token_credential = ChainedTokenCredential(
            ClientSecretCredential(
                client_id=getenv("AZURE_CLIENT_ID"),
                tenant_id=getenv("AZURE_TENANT_ID"),
                client_secret=getenv("AZURE_CLIENT_SECRET"),
            ),
            DefaultAzureCredential(),
        )
        return CosmosClient(
            url=getenv("AZURE_COSMOS_DB_ENDPOINT"),
            credential=token_credential,
        )
    return CosmosClient.from_connection_string(getenv("AZURE_COSMOS_DB_CONNECTION_STRING"))


def get_azure_cosmos_db_no_sql_vector_search(service_principal: bool):
    return AzureCosmosDBNoSqlVectorSearch(
        embedding=AzureOpenAIEmbeddings(
            api_key=getenv("AZURE_OPENAI_API_KEY"),
            api_version=getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
            model=getenv("AZURE_OPENAI_MODEL_EMBEDDING"),
        ),
        cosmos_client=get_cosmos_client(service_principal=service_principal),
        database_name=getenv("AZURE_COSMOS_DB_DATABASE_NAME"),
        container_name=getenv("AZURE_COSMOS_DB_CONTAINER_NAME"),
        vector_embedding_policy=get_vector_embedding_policy(),
        indexing_policy=get_indexing_policy(),
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        cosmos_database_properties={"id": getenv("AZURE_COSMOS_DB_DATABASE_NAME")},
    )


@app.command()
def insert_data(
    pdf_url: str = "https://arxiv.org/pdf/2303.08774.pdf",
    chunk_size: int = 2000,
    chunk_overlap: int = 0,
    service_principal: bool = False,
    verbose: bool = True,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load the PDF
    loader = PyMuPDFLoader(file_path=pdf_url)
    data = loader.load()

    # Split the text into chunks
    docs = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).split_documents(data)

    try:
        vector_search = get_azure_cosmos_db_no_sql_vector_search(service_principal=service_principal)
        vector_search.add_documents(docs)
    except Exception as e:
        logger.error(f"error: {e}")


@app.command()
def query_data(
    query: str = "What were the compute requirements for training GPT 4",
    service_principal: bool = False,
    verbose: bool = True,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    try:
        vector_search = get_azure_cosmos_db_no_sql_vector_search(service_principal=service_principal)
        results = vector_search.similarity_search(query=query)
        logger.info(f"got {len(results)} results")
        for idx, result in enumerate(results):
            print(f"Result {idx + 1}: {result}")
    except Exception as e:
        logger.error(f"error: {e}")


if __name__ == "__main__":
    app()
