import logging
from os import getenv
from typing import Annotated

import typer
from dotenv import load_dotenv
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

load_dotenv()
logger = logging.getLogger(__name__)
app = typer.Typer()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
tools = [
    BingSearchResults(
        api_wrapper=BingSearchAPIWrapper(k=2),
    ),
]
llm = AzureChatOpenAI(
    temperature=0,
    api_key=getenv("AZURE_OPENAI_API_KEY"),
    api_version=getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
    model=getenv("AZURE_OPENAI_GPT_MODEL"),
)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")


def create_graph(
    memory: bool = False,
    interrupt: bool = False,
):
    return graph_builder.compile(
        checkpointer=MemorySaver() if memory else None,
        interrupt_before=["tools"] if interrupt else None,
    )


@app.command()
def run(
    memory: bool = False,
    interrupt: bool = False,
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    graph = create_graph(
        memory=memory,
        interrupt=interrupt,
    )
    config = {"configurable": {"thread_id": "1"}}

    while True:
        # If you type exit, the loop will break
        # If you type continue, the loop will continue
        query = input("Enter a query: ")
        if query == "exit":
            break
        events = graph.stream(
            input={"messages": [("user", query)]} if query != "continue" else None,
            config=config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
        snapshot = graph.get_state(config)
        print(snapshot.next)
        print(snapshot.values["messages"][-1].tool_calls)


@app.command()
def draw_mermaid_png(
    memory: bool = False,
    interrupt: bool = False,
    output: str = None,
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    graph = create_graph(
        memory=memory,
        interrupt=interrupt,
    )
    print(graph.get_graph().draw_mermaid())
    if output:
        graph.get_graph().draw_mermaid_png(
            output_file_path=output,
        )


if __name__ == "__main__":
    app()
