import json
import logging
from os import getenv

import typer
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_openai import AzureChatOpenAI

load_dotenv()
logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def search(
    k: int = 4,
    query: str = "Microsoft",
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    search = BingSearchAPIWrapper(k=k)
    results = search.results(
        query=query,
        num_results=k,
    )
    for result in results:
        print(result)


@app.command()
def tool(
    k: int = 4,
    query: str = "Microsoft",
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    api_wrapper = BingSearchAPIWrapper(k=k)
    tool = BingSearchResults(api_wrapper=api_wrapper)

    # .invoke wraps utility.results
    response = tool.invoke(input=query)
    response = json.loads(response.replace("'", '"'))
    for item in response:
        print(item)


@app.command()
def chain(
    k: int = 4,
    instructions: str = """You are an assistant.""",
    query: str = "Who is the CEO of Microsoft?",
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)
    llm = AzureChatOpenAI(
        temperature=0,
        api_key=getenv("AZURE_OPENAI_API_KEY"),
        api_version=getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
        model=getenv("AZURE_OPENAI_MODEL_GPT"),
    )
    tools = [
        BingSearchResults(
            api_wrapper=BingSearchAPIWrapper(k=k),
        ),
    ]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )
    result = agent_executor.invoke({"input": query})
    print(f"Answer from agent: {result['output']}")


if __name__ == "__main__":
    load_dotenv()
    app()
