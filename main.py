import logging

import typer
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)
app = typer.Typer()


def set_verbosity(verbose: bool):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)


# ---
# agents
# ---
@app.command(
    help="Chatbot with tools",
)
def agents_chatbot_with_tools_run(
    verbose: bool = False,
):
    set_verbosity(verbose)
    from workshop_llm_agents.agents.chatbot_with_tools import graph

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
    set_verbosity(verbose)
    from workshop_llm_agents.agents.chatbot_with_tools import graph

    print(graph.get_graph().draw_mermaid())

    if png:
        graph.get_graph().draw_mermaid_png(
            output_file_path=png,
        )


@app.command(
    help="Run documentation agent",
)
def agents_documentation_run(
    user_request: str = "スマートフォン向けの健康管理アプリを開発したい",
    k: int = 3,
    verbose: bool = True,
):
    set_verbosity(verbose)

    from workshop_llm_agents.agents.documentation_agent import DocumentationAgent
    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper

    azure_openai_wrapper = AzureOpenAIWrapper()
    llm = azure_openai_wrapper.get_azure_chat_openai()
    agent = DocumentationAgent(llm=llm, k=k)
    final_output = agent.run(user_request=user_request)
    print(final_output)


@app.command(
    help="Run single path plan generation agent",
)
def agents_single_path_plan_generation_run(
    query: str = "スマートフォン向けの健康管理アプリを開発したい",
    verbose: bool = True,
):
    set_verbosity(verbose)
    from workshop_llm_agents.agents.single_path_plan_generation_agent import SinglePathPlanGenerationAgent
    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.tools.bing_search import BingSearchWrapper

    agent = SinglePathPlanGenerationAgent(
        llm=AzureOpenAIWrapper().get_azure_chat_openai(),
        tools=[
            BingSearchWrapper().get_bing_search_tool(),
        ],
    )
    final_output = agent.run(query=query)
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
    set_verbosity(verbose)
    import json

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper

    azure_openai_wrapper = AzureOpenAIWrapper()
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
    set_verbosity(verbose)
    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper

    azure_openai_wrapper = AzureOpenAIWrapper()
    embeddings = azure_openai_wrapper.get_azure_openai_embeddings()
    embedding = embeddings.embed_query(message)
    print(f"Dimensions: {len(embedding)}")
    logger.info(embedding)


@app.command(
    help="Chat with Ollama model",
)
def llms_ollama_chat(
    message: str = "What is the capital of Japan?",
    verbose: bool = False,
):
    set_verbosity(verbose)
    import json

    from workshop_llm_agents.llms.ollama import OllamaWrapper

    wrapper = OllamaWrapper()
    llm = wrapper.get_chat_ollama()
    response = llm.invoke(input=message)
    print(
        json.dumps(
            response.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )


# ---
# tasks
# ---
@app.command(
    help="Run the passive goal creator task",
)
def tasks_passive_goal_creator(
    query: str = "I want to learn how to cook",
    verbose: bool = False,
):
    set_verbosity(verbose)

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.tasks.passive_goal_creator import Goal, PassiveGoalCreator

    llm = AzureOpenAIWrapper().get_azure_chat_openai()
    task = PassiveGoalCreator(llm=llm)
    result: Goal = task.run(query=query)
    print(result.text)


@app.command(
    help="Run the prompt optimizer task",
)
def tasks_prompt_optimizer(
    query: str = "I want to learn how to cook",
    verbose: bool = False,
):
    set_verbosity(verbose)

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.tasks.prompt_optimizer import OptimizedGoal, PromptOptimizer

    llm = AzureOpenAIWrapper().get_azure_chat_openai()
    task = PromptOptimizer(llm=llm)
    result: OptimizedGoal = task.run(query=query)
    print(result.text)


@app.command(
    help="Run the query decomposer task",
)
def tasks_query_decomposer(
    query: str = "I want to learn how to cook",
    verbose: bool = False,
):
    set_verbosity(verbose)

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.tasks.query_decomposer import DecomposedTasks, QueryDecomposer

    llm = AzureOpenAIWrapper().get_azure_chat_openai()
    task = QueryDecomposer(llm=llm)
    result: DecomposedTasks = task.run(query=query)
    print(result)


@app.command(
    help="Run the response optimizer task",
)
def tasks_response_optimizer(
    query: str = "I want to learn how to cook",
    verbose: bool = False,
):
    set_verbosity(verbose)

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.tasks.response_optimizer import ResponseOptimizer

    llm = AzureOpenAIWrapper().get_azure_chat_openai()
    task = ResponseOptimizer(llm=llm)
    result = task.run(query=query)
    print(result)


@app.command(
    help="Run the result aggregator task",
)
def tasks_result_aggregator(
    query: str = "I want to learn how to cook",
    verbose: bool = False,
):
    set_verbosity(verbose)

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.tasks.result_aggregator import ResultAggregator

    llm = AzureOpenAIWrapper().get_azure_chat_openai()
    task = ResultAggregator(llm=llm)
    result = task.run(
        query=query,
        response_definition="Provide a summary of the search results",
        results=[
            "Learn how to use a knife",
            "Practice cooking rice",
            "Learn how to make a salad",
        ],
    )
    print(result)


@app.command(
    help="Run the task executor",
)
def tasks_task_executor(
    task: str = "What's the weather in Tokyo today?",
    verbose: bool = False,
):
    set_verbosity(verbose)

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.tasks.task_executor import TaskExecutor
    from workshop_llm_agents.tools.bing_search import BingSearchWrapper

    llm = AzureOpenAIWrapper().get_azure_chat_openai()
    task_executor = TaskExecutor(
        llm=llm,
        tools=[
            BingSearchWrapper().get_bing_search_tool(),
        ],
    )
    result = task_executor.run(task=task)
    print(result)


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
    set_verbosity(verbose)
    import json

    from workshop_llm_agents.tools.bing_search import BingSearchWrapper

    wrapper = BingSearchWrapper()
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


@app.command(
    help="Search CosmosDB",
)
def tools_cosmosdb_search(
    query: str = "Microsoft",
    verbose: bool = False,
):
    set_verbosity(verbose)

    import json

    from workshop_llm_agents.tools.cosmosdb_search import CosmosDBSearchWrapper

    wrapper = CosmosDBSearchWrapper()
    tool = wrapper.get_cosmosdb_search_tool()
    response_str = tool.invoke(input=query)
    response_json = json.loads(response_str.replace("'", '"'))
    print(
        json.dumps(
            response_json,
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
    set_verbosity(verbose)
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.vector_stores.cosmosdb import CosmosDBWrapper

    cosmosdb_wrapper = CosmosDBWrapper()
    vector_store = cosmosdb_wrapper.get_azure_cosmos_db_no_sql_vector_search(
        embedding=AzureOpenAIWrapper().get_azure_openai_embeddings(),
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
    set_verbosity(verbose)
    import json

    from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
    from workshop_llm_agents.vector_stores.cosmosdb import CosmosDBWrapper

    cosmosdb_wrapper = CosmosDBWrapper()
    vector_store = cosmosdb_wrapper.get_azure_cosmos_db_no_sql_vector_search(
        embedding=AzureOpenAIWrapper().get_azure_openai_embeddings(),
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
    verbose: bool = True,
):
    set_verbosity(verbose)
    import streamlit as st

    st.title("Code samples for Streamlit")
    st.info("Select a code sample from the sidebar to run it")


if __name__ == "__main__":
    app()
