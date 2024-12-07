import logging

import typer
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
app = typer.Typer()


# ---
# agents
# ---
@app.command(
    help="Chatbot with tools",
)
def agents_chatbot_with_tools_run(
    verbose: bool = False,
):
    from workshop_llm_agents.agents.chatbot_with_tools import graph

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    config = {
        "configurable": {
            "thread_id": "1",
        },
    }

    while True:
        exit_code = "q"
        query = input(f"Enter a query(type '{exit_code}' to exit): ")
        if query == exit_code:
            break

        events = graph.stream(
            input={
                "messages": [
                    ("user", query),
                ]
            },
            config=config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()


@app.command(
    help="Export the graph to a PNG file",
)
def agents_chatbot_with_tools_export(
    png: str = None,
    verbose: bool = False,
):
    from workshop_llm_agents.agents.chatbot_with_tools import graph

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    print(graph.get_graph().draw_mermaid())

    if png:
        graph.get_graph().draw_mermaid_png(
            output_file_path=png,
        )


@app.command(
    help="Run documentation agent ref. https://github.com/GenerativeAgents/agent-book/tree/main/chapter10",
)
def agents_documentation_run(
    user_request: str = "スマートフォン向けの健康管理アプリを開発したい",
    k: int = 3,
    verbose: bool = True,
):
    from workshop_llm_agents.agents.documentation_agent import DocumentationAgent
    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper, Settings

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    azure_openai_wrapper = AzureOpenAIWrapper(Settings())
    llm = azure_openai_wrapper.get_azure_chat_openai()
    agent = DocumentationAgent(llm=llm, k=k)
    final_output = agent.run(user_request=user_request)
    print(final_output)


# ---
# llms
# ---
@app.command(
    help="Chat with Azure OpenAI",
)
def llms_azure_openai_chat(
    message: str = "What is the capital of Japan?",
    verbose: bool = False,
):
    import json

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper, Settings

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    azure_openai_wrapper = AzureOpenAIWrapper(Settings())
    llm = azure_openai_wrapper.get_azure_chat_openai()
    response = llm.invoke(input=message)
    print(
        json.dumps(
            response.model_dump(),
            indent=2,
        )
    )


@app.command(
    help="Embed query with Azure OpenAI",
)
def llms_azure_openai_embeddings(
    message: str = "What is the capital of Japan?",
    verbose: bool = False,
):
    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper, Settings

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    azure_openai_wrapper = AzureOpenAIWrapper(Settings())
    embeddings = azure_openai_wrapper.get_azure_openai_embeddings()
    embedding = embeddings.embed_query(message)
    print(f"Dimensions: {len(embedding)}")
    logger.info(embedding)


# ---
# tools
# ---
@app.command(
    help="Search Bing",
)
def tools_bing_search(
    query: str = "Microsoft",
    verbose: bool = False,
):
    import json

    from workshop_llm_agents.tools.bing_search import BingSearchWrapper, Settings

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    wrapper = BingSearchWrapper(Settings())
    tool = wrapper.get_bing_search_tool()
    response = tool.invoke(input=query)
    response = json.loads(response.replace("'", '"'))
    print(
        json.dumps(
            response,
            indent=2,
            ensure_ascii=False,
        )
    )


# ---
# vector_stores
# ---
@app.command(
    help="Insert data into the CosmosDB",
)
def vector_stores_cosmosdb_insert_data(
    pdf_url: str = "https://www.maff.go.jp/j/zyukyu/zikyu_ritu/attach/pdf/012-9.pdf",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    verbose: bool = False,
):
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.llms.azure_openai import Settings as AzureOpenAIWrapperSettings
    from workshop_llm_agents.vector_stores.cosmosdb import CosmosDBWrapper
    from workshop_llm_agents.vector_stores.cosmosdb import Settings as CosmosDBWrapperSettings

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    cosmosdb_wrapper = CosmosDBWrapper(CosmosDBWrapperSettings())
    vector_store = cosmosdb_wrapper.get_azure_cosmos_db_no_sql_vector_search(
        embedding=AzureOpenAIWrapper(AzureOpenAIWrapperSettings()).get_azure_openai_embeddings(),
    )

    # Load the PDF
    loader = PyMuPDFLoader(file_path=pdf_url)
    data = loader.load()

    # Split the text into chunks
    docs = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).split_documents(data)

    try:
        vector_store.add_documents(docs)
    except Exception as e:
        logger.error(f"error: {e}")


@app.command(
    help="Query data from the CosmosDB",
)
def vector_stores_cosmosdb_query_data(
    query: str = "食料自給率の長期的推移",
    verbose: bool = False,
):
    import json

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.llms.azure_openai import Settings as AzureOpenAIWrapperSettings
    from workshop_llm_agents.vector_stores.cosmosdb import CosmosDBWrapper
    from workshop_llm_agents.vector_stores.cosmosdb import Settings as CosmosDBWrapperSettings

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    cosmosdb_wrapper = CosmosDBWrapper(CosmosDBWrapperSettings())
    vector_store = cosmosdb_wrapper.get_azure_cosmos_db_no_sql_vector_search(
        embedding=AzureOpenAIWrapper(AzureOpenAIWrapperSettings()).get_azure_openai_embeddings(),
    )
    documents = vector_store.similarity_search(
        query=query,
    )
    for idx, document in enumerate(documents):
        print(f"Document {idx + 1} ---")
        print(
            json.dumps(
                document.model_dump(),
                indent=2,
                ensure_ascii=False,
            )
        )


# ---
# streamlit
# ---
@app.command(
    help="NOTE: To run the Streamlit app run `$ poetry run streamlit run main.py streamlit-app`",
)
def streamlit_app(
    verbose: bool = False,
):
    import streamlit as st

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    st.title("Code samples for Streamlit")
    st.info("Select a code sample from the sidebar to run it")


if __name__ == "__main__":
    load_dotenv(override=True)
    app()
