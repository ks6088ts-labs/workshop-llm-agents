# https://github.com/GenerativeAgents/agent-book/blob/main/chapter12/single_path_plan_generation/main.py
import operator
from typing import Annotated, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: list[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: list[Document]
    final_summary: str


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str


class SummarizeAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        token_max: int = 1000,
    ):
        self.llm = llm
        self.map_chain = self._get_map_chain()
        self.reduce_chain = self._get_reduce_chain()
        self.token_max = token_max
        self.graph = self._create_graph()

    def _get_map_chain(self):
        map_prompt = ChatPromptTemplate.from_messages(
            [("human", "Write a concise summary of the following:\\n\\n{context}")]
        )
        return map_prompt | self.llm | StrOutputParser()

    def _get_reduce_chain(self):
        reduce_template = """
        The following is a set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary
        of the main themes.
        """

        reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

        return reduce_prompt | self.llm | StrOutputParser()

    def _length_function(self, documents: list[Document]) -> int:
        """Get number of tokens for input contents."""
        return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)

    # Here we generate a summary, given a document
    async def _generate_summary(self, state: SummaryState):
        response = await self.map_chain.ainvoke(state["content"])
        return {"summaries": [response]}

    # Here we define the logic to map out over the documents
    # We will use this an edge in the graph
    def _map_summaries(self, state: OverallState):
        # We will return a list of `Send` objects
        # Each `Send` object consists of the name of a node in the graph
        # as well as the state to send to that node
        return [Send("generate_summary", {"content": content}) for content in state["contents"]]

    def _collect_summaries(self, state: OverallState):
        return {"collapsed_summaries": [Document(summary) for summary in state["summaries"]]}

    # Add node to collapse summaries
    async def _collapse_summaries(self, state: OverallState):
        doc_lists = split_list_of_docs(state["collapsed_summaries"], self._length_function, self.token_max)
        results = []
        for doc_list in doc_lists:
            results.append(await acollapse_docs(doc_list, self.reduce_chain.ainvoke))

        return {"collapsed_summaries": results}

    # This represents a conditional edge in the graph that determines
    # if we should collapse the summaries or not
    def _should_collapse(
        self,
        state: OverallState,
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = self._length_function(state["collapsed_summaries"])
        if num_tokens > self.token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

    # Here we will generate the final summary
    async def _generate_final_summary(self, state: OverallState):
        response = await self.reduce_chain.ainvoke(state["collapsed_summaries"])
        return {"final_summary": response}

    def _create_graph(self) -> CompiledStateGraph:
        graph = StateGraph(OverallState)
        graph.add_node("generate_summary", self._generate_summary)  # same as before
        graph.add_node("collect_summaries", self._collect_summaries)
        graph.add_node("collapse_summaries", self._collapse_summaries)
        graph.add_node("generate_final_summary", self._generate_final_summary)

        # Edges:
        graph.add_conditional_edges(START, self._map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", self._should_collapse)
        graph.add_conditional_edges("collapse_summaries", self._should_collapse)
        graph.add_edge("generate_final_summary", END)

        return graph.compile()

    async def run(self, docs: list[Document]):
        async for step in self.graph.astream(
            {"contents": [doc.page_content for doc in docs]},
            {"recursion_limit": 10},
        ):
            print(list(step.keys()))

        return step["generate_final_summary"]["final_summary"]

    def mermaid_png(self, output_file_path=None) -> bytes:
        return self.graph.get_graph().draw_mermaid_png(
            output_file_path=output_file_path,
        )
