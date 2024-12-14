# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/evaluation/scoring/eval_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ScoringResult(BaseModel):
    score: int = Field(..., ge=1, le=10, description="評価スコア")
    reason: str = Field(..., description="評価の理由")

    @property
    def text(self) -> str:
        return f"評価スコア: {self.score}/10\n評価の理由: {self.reason}"


class ScoringEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> ScoringResult:
        prompt = ChatPromptTemplate.from_template(
            "You are an expert in evaluating bug report documents. Please evaluate the bug report document provided by the user on a scale of 1 to 10 based on the following criteria:\n\n"  # noqa E501
            "Evaluation Criteria:\n"
            "- Whether the model number of the hardware or the version of the software is clearly stated\n"
            "- Whether the reproduction steps are specific and detailed\n"
            "- Whether the scope of the impact of the bug is clearly stated\n"
            "- Whether the impact of the bug on the business is clearly stated\n"
            "- Whether the urgency is clearly stated\n\n"
            "Bug report document provided by the user:\n"
            "{query}\n\n"
        )
        chain = prompt | self.llm.with_structured_output(ScoringResult)
        return chain.invoke({"query": query})
