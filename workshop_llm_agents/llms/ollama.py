from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    ollama_model_chat: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


class OllamaWrapper:
    def __init__(self, settings=Settings()):
        self.settings = settings

    def get_chat_ollama(
        self,
        temperature: float = 0,
    ) -> BaseChatModel:
        return ChatOllama(
            model=self.settings.ollama_model_chat,
            temperature=temperature,
        )
