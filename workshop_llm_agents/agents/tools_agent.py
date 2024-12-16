from collections.abc import Callable
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.pregel.types import StateSnapshot


class ToolsAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
    ) -> None:
        self.llm = llm.bind_tools(tools)
        self.tools = tools
        self.graph = self._create_graph()
        self.subscribers: list[Callable[[str, str], None]] = []

    def mermaid_png(self, output_file_path=None) -> bytes:
        return self.graph.get_graph().draw_mermaid_png(
            output_file_path=output_file_path,
        )

    def get_state(self, thread_id: str) -> StateSnapshot:
        return self.graph.get_state(config=self._config(thread_id))

    def subscribe(self, subscriber: Callable[[str, str], None]) -> None:
        self.subscribers.append(subscriber)

    def _notify(self, type: Literal["human", "agent"], message: str) -> None:
        for subscriber in self.subscribers:
            subscriber(type, message)

    def _config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
        }

    def _should_continue(self, state: MessagesState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def _call_model(self, state: MessagesState):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(MessagesState)

        graph.add_node("agent", self._call_model)
        graph.add_node("tools", ToolNode(self.tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
        )
        graph.add_edge("tools", "agent")

        return graph.compile(
            checkpointer=MemorySaver(),
        )

    def run(self, query: str, thread_id: str) -> str:
        self._notify("human", query)
        for event in self.graph.stream(
            input={
                "messages": [
                    ("user", query),
                ]
            },
            config=self._config(thread_id),
            stream_mode="updates",
        ):
            node = list(event.keys())[0]
            self._notify("agent", f"**{node}**: {event[node]}")
            if "messages" in event:
                event["messages"][-1].pretty_print()
        return event[node]["messages"][-1].content
