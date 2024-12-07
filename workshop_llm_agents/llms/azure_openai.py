from azure.identity import (
    ChainedTokenCredential,
    ClientSecretCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Microsoft Entra ID
    use_microsoft_entra_id: bool
    azure_client_id: str
    azure_client_secret: str
    azure_tenant_id: str
    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_endpoint: str
    azure_openai_model_gpt: str
    azure_openai_model_embedding: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


class AzureOpenAIWrapper:
    def __init__(
        self,
        settings: Settings,
    ):
        self.settings = settings

    def get_azure_ad_token_provider(self):
        token_credential = ChainedTokenCredential(
            ClientSecretCredential(
                client_id=self.settings.azure_client_id,
                tenant_id=self.settings.azure_tenant_id,
                client_secret=self.settings.azure_client_secret,
            ),
            DefaultAzureCredential(),
        )
        return get_bearer_token_provider(
            token_credential,
            "https://cognitiveservices.azure.com/.default",
        )

    def get_azure_chat_openai(
        self,
        temperature: float = 0,
    ) -> BaseChatModel:
        if self.settings.use_microsoft_entra_id:
            return AzureChatOpenAI(
                temperature=temperature,
                azure_ad_async_token_provider=self.get_azure_ad_token_provider(),
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint,
                model=self.settings.azure_openai_model_gpt,
            )
        return AzureChatOpenAI(
            temperature=temperature,
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
            azure_endpoint=self.settings.azure_openai_endpoint,
            model=self.settings.azure_openai_model_gpt,
        )

    def get_azure_openai_embeddings(self) -> AzureOpenAIEmbeddings:
        if self.settings.use_microsoft_entra_id:
            return AzureOpenAIEmbeddings(
                azure_ad_async_token_provider=self.get_azure_ad_token_provider(),
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint,
                model=self.settings.azure_openai_model_embedding,
            )
        return AzureOpenAIEmbeddings(
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
            azure_endpoint=self.settings.azure_openai_endpoint,
            model=self.settings.azure_openai_model_embedding,
        )
