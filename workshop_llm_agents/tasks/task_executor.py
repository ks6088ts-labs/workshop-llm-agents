# https://github.com/GenerativeAgents/agent-book/blob/main/chapter12/single_path_plan_generation/main.py
from langchain_core.tools import BaseTool
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.prebuilt import create_react_agent


class TaskExecutor:
    def __init__(self, llm: BaseChatOpenAI, tools: list[BaseTool]):
        self.llm = llm
        self.tools = tools

    def run(self, task: str) -> str:
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        (
                            "次のタスクを実行し、詳細な回答を提供してください。\n\n"
                            f"タスク: {task}\n\n"
                            "要件:\n"
                            "1. 必要に応じて提供されたツールを使用してください。\n"
                            "2. 実行は徹底的かつ包括的に行ってください。\n"
                            "3. 可能な限り具体的な事実やデータを提供してください。\n"
                            "4. 発見した内容を明確に要約してください。\n"
                        ),
                    )
                ]
            }
        )
        return result["messages"][-1].content