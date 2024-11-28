import json

from azure.identity import (
    ChainedTokenCredential,
    ClientSecretCredential,
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Microsoft Entra ID
    azure_client_id: str
    azure_client_secret: str
    azure_tenant_id: str
    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_endpoint: str
    azure_openai_model_gpt: str

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
        return AzureChatOpenAI(
            temperature=temperature,
            azure_ad_async_token_provider=self.get_azure_ad_token_provider(),
            api_version=self.settings.azure_openai_api_version,
            azure_endpoint=self.settings.azure_openai_endpoint,
            model=self.settings.azure_openai_model_gpt,
        )


if __name__ == "__main__":
    settings = Settings()
    wrapper = AzureOpenAIWrapper(settings)
    llm = wrapper.get_azure_chat_openai()
    response = llm.invoke(input="Hello, how are you?")
    print(
        json.dumps(
            response.model_dump(),
            indent=2,
        )
    )
