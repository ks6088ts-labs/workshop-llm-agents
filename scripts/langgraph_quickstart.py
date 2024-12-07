import logging
from os import getenv
from typing import Annotated

import typer
from dotenv import load_dotenv
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from typing_extensions import TypedDict

load_dotenv(override=True)
logger = logging.getLogger(__name__)
app = typer.Typer()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # This flag is new
    ask_human: bool


class RequestAssistance(BaseModel):
    """
    Escalate the conversation to an expert.
    Use this if you are unable to assist directly or if the user requires support beyond your permissions.
    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


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
    model=getenv("AZURE_OPENAI_MODEL_GPT"),
)
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if response.tool_calls and response.tool_calls[0]["name"] == RequestAssistance.__name__:
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(create_response("No response from human.", state["messages"][-1]))
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("human", human_node)
graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", "__end__": "__end__"},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.set_entry_point("chatbot")


def create_graph(
    memory: bool = False,
    interrupt: bool = False,
):
    return graph_builder.compile(
        checkpointer=MemorySaver() if memory else None,
        interrupt_before=["human"] if interrupt else None,
    )


@app.command()
def run(
    memory: bool = False,
    interrupt: bool = False,
    replay: bool = False,
    verbose: bool = False,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    graph = create_graph(
        memory=memory,
        interrupt=interrupt,
    )
    config = {"configurable": {"thread_id": "1"}}

    query = input("Enter a query: ")
    events = graph.stream(
        input={"messages": [("user", query)]},
        config=config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    snapshot = graph.get_state(config)
    existing_message = snapshot.values["messages"][-1]
    existing_message.pretty_print()
    print(existing_message.tool_calls)
    print(snapshot.next)

    # If interrupted, type your pseudo response as AIMessage
    if snapshot.next:
        ai_message = snapshot.values["messages"][-1]
        human_response = input("Enter human response: ")
        tool_message = create_response(human_response, ai_message)
        graph.update_state(config, {"messages": [tool_message]})
        print(graph.get_state(config).values["messages"])

        # Resume the graph by invoking it with None as the inputs
        events = graph.stream(None, config, stream_mode="values")
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()

    if replay:
        to_replay = None

        # See the state history
        for state in graph.get_state_history(config):
            print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
            print("-" * 80)
            # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
            if len(state.values["messages"]) == 2:
                to_replay = state

        # Load the state from that moment and resume execution
        print(to_replay.next)
        print(to_replay.config)
        for event in graph.stream(None, to_replay.config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()


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
