"""Tool for the CosmosDB search API."""

from typing import Literal

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_openai import AzureOpenAIEmbeddings

from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
from workshop_llm_agents.vector_stores.cosmosdb import CosmosDBWrapper


class CosmosDBSearchResults(BaseTool):  # type: ignore[override, override]
    """CosmosDB Search tool."""

    name: str = "cosmosdb_search_results_json"
    description: str = (
        "A wrapper around CosmosDB Search. "
        "Useful for when you need to answer questions about internal information. "
        "Input should be a search query. Output is an array of the query results."
    )
    num_results: int = 4
    """Max search results to return, default is 4."""
    api_wrapper: CosmosDBWrapper
    response_format: Literal["content_and_artifact"] = "content_and_artifact"
    embedding: AzureOpenAIEmbeddings

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, list[dict]]:
        """Use the tool."""
        try:
            vector_store = self.api_wrapper.get_azure_cosmos_db_no_sql_vector_search(
                embedding=self.embedding,
            )
            documents = vector_store.similarity_search(
                query,
                k=self.num_results,
            )
            results = []
            for document in documents:
                result = {
                    "content": document.page_content,
                    "source": document.metadata["source"],
                    "page": document.metadata["page"],
                }
                results.append(result)
            return str(results), results
        except Exception as e:
            return str(e), []


class CosmosDBSearchWrapper:
    def __init__(self, wrapper=CosmosDBWrapper()):
        self.wrapper = wrapper

    def get_cosmosdb_search_tool(self) -> BaseTool:
        return CosmosDBSearchResults(
            api_wrapper=self.wrapper,
            embedding=AzureOpenAIWrapper().get_azure_openai_embeddings(),
        )
