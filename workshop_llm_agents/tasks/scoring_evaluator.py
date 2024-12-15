# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/evaluation/scoring/eval_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ScoringResult(BaseModel):
    score: int = Field(..., ge=0, le=10, description="Evaluation score")
    reason: str = Field(..., description="Reason for the evaluation")

    @property
    def text(self) -> str:
        return f"Evaluation score: {self.score}/10\nReason for the evaluation: {self.reason}"


class BugReportScoringResult(BaseModel):
    explicitness: ScoringResult = Field(..., description="Is it explicitly described?")
    reproducibility: ScoringResult = Field(..., description="Are the reproduction steps specific?")

    @property
    def text(self) -> str:
        return f"Is it explicitly described?: {self.explicitness.text}\nAre the reproduction steps specific?: {self.reproducibility.text}"  # noqa E501


class ScoringEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> BugReportScoringResult:
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in evaluating bug report documents. Please evaluate the bug report document provided by the user on a scale of 1 to 10 based on the following criteria:
            - Explicitness: Is it explicitly described?
            - Reproducibility: Are the reproduction steps specific?

            Bug report document provided by the user:
            {query}
            """  # noqa E501
        )
        chain = prompt | self.llm.with_structured_output(BugReportScoringResult)
        return chain.invoke({"query": query})
