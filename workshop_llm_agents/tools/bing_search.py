from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_core.tools import BaseTool
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Bing Search
    bing_subscription_key: str
    bing_search_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


class BingSearchWrapper:
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_bing_search_tool(self, k=3) -> BaseTool:
        return BingSearchResults(
            api_wrapper=BingSearchAPIWrapper(
                bing_subscription_key=self.settings.bing_subscription_key,
                bing_search_url=self.settings.bing_search_url,
                k=k,
            ),
        )
