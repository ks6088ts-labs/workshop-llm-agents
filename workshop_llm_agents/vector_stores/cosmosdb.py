from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import (
    ChainedTokenCredential,
    ClientSecretCredential,
    DefaultAzureCredential,
)
from langchain_community.vectorstores.azure_cosmos_db_no_sql import AzureCosmosDBNoSqlVectorSearch
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Microsoft Entra ID
    use_microsoft_entra_id: bool
    azure_client_id: str
    azure_client_secret: str
    azure_tenant_id: str
    # Azure Cosmos DB
    azure_cosmos_db_connection_string: str
    azure_cosmos_db_database_name: str
    azure_cosmos_db_container_name: str
    azure_cosmos_db_endpoint: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


class CosmosDBWrapper:
    def __init__(self, settings=Settings()):
        self.settings = settings

    def get_cosmos_client(self) -> CosmosClient:
        if self.settings.use_microsoft_entra_id:
            token_credential = ChainedTokenCredential(
                ClientSecretCredential(
                    client_id=self.settings.azure_client_id,
                    tenant_id=self.settings.azure_tenant_id,
                    client_secret=self.settings.azure_client_secret,
                ),
                DefaultAzureCredential(),
            )
            return CosmosClient(
                url=self.settings.azure_cosmos_db_endpoint,
                credential=token_credential,
            )
        return CosmosClient.from_connection_string(self.settings.azure_cosmos_db_connection_string)

    def get_azure_cosmos_db_no_sql_vector_search(self, embedding: AzureOpenAIEmbeddings) -> VectorStore:
        return AzureCosmosDBNoSqlVectorSearch(
            embedding=embedding,
            cosmos_client=self.get_cosmos_client(),
            database_name=self.settings.azure_cosmos_db_database_name,
            container_name=self.settings.azure_cosmos_db_container_name,
            vector_embedding_policy={
                "vectorEmbeddings": [
                    {
                        "path": "/embedding",
                        "dataType": "float32",
                        "distanceFunction": "cosine",
                        "dimensions": 3072,  # for text-embedding-3-large
                    }
                ]
            },
            indexing_policy={
                "indexingMode": "consistent",
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": '/"_etag"/?'}],
                "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
            },
            cosmos_container_properties={
                "partition_key": PartitionKey(path="/id"),
            },
            cosmos_database_properties={"id": self.settings.azure_cosmos_db_database_name},
        )
