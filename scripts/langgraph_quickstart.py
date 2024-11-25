import logging
from os import getenv
from typing import Annotated

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()
logger = logging.getLogger(__name__)
app = typer.Typer()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
llm = AzureChatOpenAI(
    temperature=0,
    api_key=getenv("AZURE_OPENAI_API_KEY"),
    api_version=getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=getenv("AZURE_OPENAI_ENDPOINT"),
    model=getenv("AZURE_OPENAI_GPT_MODEL"),
)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()


@app.command()
def run(
    query: str = "what is the weather in sf",
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    final_state = graph.invoke(
        {
            "messages": [
                HumanMessage(content=query),
            ]
        },
        config={"configurable": {"thread_id": 42}},
    )
    print(final_state["messages"][-1].content)


@app.command()
def draw_mermaid_png(
    output: str = None,
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    print(graph.get_graph().draw_mermaid())
    if output:
        graph.get_graph().draw_mermaid_png(
            output_file_path=output,
        )


if __name__ == "__main__":
    app()
