import operator
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.pregel.types import StateSnapshot
from pydantic import BaseModel, Field

from workshop_llm_agents.tools.bing_search import BingSearchWrapper
from workshop_llm_agents.tools.cosmosdb_search import CosmosDBSearchWrapper


class ResearchAgentState(BaseModel):
    human_inputs: Annotated[list[str], operator.add] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    current_task_index: int = Field(default=0)
    results: list[str] = Field(default_factory=list)


class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="List of decomposed tasks",
    )


class QueryDecomposer:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, human_inputs: list[str], latest_decomposed_tasks: list[str] | None = None) -> DecomposedTasks:
        existing_tasks = latest_decomposed_tasks if latest_decomposed_tasks else []
        formatted_tasks = "\n".join([f"  - {task}" for task in existing_tasks])
        formatted_human_inputs = "\n".join([f"  - {human_input}" for human_input in human_inputs])
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {datetime.now().strftime('%Y-%m-%d')}\n"
            "-----\n"
            "Task: Reflect the given goal or user feedback and decompose or improve it into specific, actionable tasks.\n"  # noqa: E501
            "Requirements:\n"
            "1. Achieve the goal with only the following actions. Never take actions other than those specified.\n"
            "   - Use the internet to conduct research to achieve the goal.\n"
            "   - Generate a report for the user.\n"
            "2. Since all work content will be shared with the user, there is no need to submit information to the user.\n"  # noqa: E501
            "3. Each task must be described in detail and include information that can be executed and verified independently. Do not include any abstract expressions.\n"  # noqa: E501
            "4. List the tasks in an executable order.\n"
            "5. Output the tasks in English.\n"
            "6. If there is an existing task list, maximize the reflection of user feedback to improve or supplement it.\n"  # noqa: E501
            "7. If there is user feedback, prioritize it and reflect it in the tasks.\n"
            "8. Ensure the tasks are limited to a maximum of 5.\n"
            "Existing task list:\n"
            "{existing_tasks}\n\n"
            "Goal or user feedback:\n"
            "{human_inputs}\n"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"human_inputs": formatted_human_inputs, "existing_tasks": formatted_tasks})


class TaskExecutor:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.tools = [
            BingSearchWrapper().get_bing_search_tool(),
            CosmosDBSearchWrapper().get_cosmosdb_search_tool(),
        ]

    def run(self, task: str, results: list[str]) -> str:
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(self._create_task_message(task, results))
        return result["messages"][-1].content

    @staticmethod
    def _create_task_message(task: str, results: list[str]) -> dict[str, Any]:
        context = ""
        if results:
            context = "<context>\n"
            for i, result in enumerate(results, 1):
                context += f"<result_{i}>\n{result}\n</result_{i}>\n"
            context += "</context>\n\n"

        return {
            "messages": [
                (
                    "human",
                    f"{context}"
                    f"Please execute the following task and provide a detailed response.\n\nTask: {task}\n\n"
                    "Requirements:\n"
                    "1. Use the provided tools as necessary.\n"
                    "2. Perform the execution thoroughly and comprehensively.\n"
                    "3. Provide specific facts and data as much as possible.\n"
                    "4. Clearly summarize the findings.\n"
                    "5. If the <context> tag exists, refer to the previous investigation results.\n"
                    "6. Add new information and complement or update existing information.\n",
                )
            ]
        }


class ResearchAgent:
    APPROVE_TOKEN = "[APPROVE]"

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.subscribers: list[Callable[[str, str, str], None]] = []
        self.query_decomposer = QueryDecomposer(llm)
        self.task_executor = TaskExecutor(llm)
        self.graph = self._create_graph()

    def subscribe(self, subscriber: Callable[[str, str, str], None]) -> None:
        self.subscribers.append(subscriber)

    def handle_human_message(self, human_message: str, thread_id: str) -> None:
        if self.is_next_human_approval_node(thread_id):
            self.graph.update_state(
                config=self._config(thread_id),
                values={"human_inputs": [human_message]},
                as_node="human_approval",
            )
        else:
            self.graph.update_state(
                config=self._config(thread_id),
                values={"human_inputs": [human_message]},
                as_node=START,
            )
        self._stream_events(human_message=human_message, thread_id=thread_id)

    def is_next_human_approval_node(self, thread_id: str) -> bool:
        graph_next = self._get_state(thread_id).next
        return len(graph_next) != 0 and graph_next[0] == "human_approval"

    def mermaid_png(self, output_file_path=None) -> bytes:
        return self.graph.get_graph().draw_mermaid_png(
            output_file_path=output_file_path,
        )

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(ResearchAgentState)

        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("human_approval", self._human_approval)
        graph.add_node("execute_task", self._execute_task)

        graph.add_edge(START, "decompose_query")
        graph.add_edge("decompose_query", "human_approval")
        graph.add_conditional_edges("human_approval", self._route_after_human_approval)
        graph.add_conditional_edges("execute_task", self._route_after_task_execution)

        memory = MemorySaver()

        return graph.compile(
            checkpointer=memory,
            interrupt_before=["human_approval"],
        )

    def _notify(self, type: Literal["human", "agent"], title: str, message: str) -> None:
        for subscriber in self.subscribers:
            subscriber(type, title, message)

    def _route_after_human_approval(self, state: ResearchAgentState) -> Literal["decompose_query", "execute_task"]:
        is_human_approved = state.human_inputs and state.human_inputs[-1] == ResearchAgent.APPROVE_TOKEN
        if is_human_approved:
            return "execute_task"
        else:
            return "decompose_query"

    def _route_after_task_execution(self, state: ResearchAgentState) -> Literal["execute_task", END]:
        is_task_completed = state.current_task_index >= len(state.tasks)
        if is_task_completed:
            return END
        else:
            return "execute_task"

    def _get_state(self, thread_id: str) -> StateSnapshot:
        return self.graph.get_state(config=self._config(thread_id))

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def _decompose_query(self, state: ResearchAgentState) -> dict:
        human_inputs = self._latest_human_inputs(state.human_inputs)
        # Do not refer to past task decomposition results during the initial task decomposition
        if len(human_inputs) > 1:
            latest_decomposed_tasks = state.tasks
        else:
            latest_decomposed_tasks = []

        decomposed_tasks = self.query_decomposer.run(
            human_inputs=human_inputs, latest_decomposed_tasks=latest_decomposed_tasks
        )
        return {
            "tasks": decomposed_tasks.tasks,
            "current_task_index": 0,
            "results": [],
        }

    def _human_approval(self, state: ResearchAgentState) -> dict:
        pass

    def _execute_task(self, state: ResearchAgentState) -> dict:
        result = self.task_executor.run(task=state.tasks[state.current_task_index], results=state.results)
        return {
            "results": state.results + [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _stream_events(self, human_message: str | None, thread_id: str):
        if human_message:
            self._notify("human", human_message, "")
        for event in self.graph.stream(
            input=None,
            config=self._config(thread_id),
            stream_mode="updates",
        ):
            # Retrieve information of the execution node
            node = list(event.keys())[0]
            if node in ["decompose_query", "execute_task"]:
                if node == "decompose_query":
                    title, message = self._decompose_query_message(event[node])
                else:  # execute_task
                    title, message = self._execute_task_message(event[node], thread_id)
                self._notify("agent", title, message)

    def _decompose_query_message(self, update_state: dict) -> tuple[str, str]:
        tasks = "\n".join([f"- {task}" for task in update_state["tasks"]])
        return ("Tasks have been decomposed.", tasks)

    def _execute_task_message(self, update_state: dict, thread_id: str) -> tuple[str, str]:
        current_state = self._get_state(thread_id)
        current_task_index = update_state["current_task_index"] - 1
        executed_task = current_state.values["tasks"][current_task_index]
        result = update_state["results"][-1]
        return (executed_task, result)

    def _latest_human_inputs(self, human_inputs: list[str]) -> list[str]:
        # If APPROVE_TOKEN exists, get the list after APPROVE_TOKEN
        # Otherwise, get the entire list
        if ResearchAgent.APPROVE_TOKEN in human_inputs:
            return human_inputs[human_inputs.index(ResearchAgent.APPROVE_TOKEN) + 1 :]
        else:
            return human_inputs
