# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/evaluation/scoring/eval_chain.py
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Label(Enum):
    AZURE = "Azure"
    OPENAI = "OpenAI"
    AWS = "AWS"
    LANGCHAIN = "Langchain"
    APPLE = "Apple"
    GOOGLE = "Google"
    SONY = "Sony"


class LabelingResult(BaseModel):
    labels: list[Label] = Field(
        default_factory=list,
        min_items=0,
        max_items=3,
        description="Labels assigned to the image",
    )


class ImageLabeler:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, encoded_image: str) -> LabelingResult:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are an image labeling specialist."),
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        },
                    ]
                ),
            ]
        )
        chain = prompt | self.llm.with_structured_output(LabelingResult)
        return chain.invoke({"encoded_image": encoded_image})
