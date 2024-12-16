from typing import Literal
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from workshop_llm_agents.agents.tools_agent import ToolsAgent
from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper
from workshop_llm_agents.tools.bing_search import BingSearchWrapper


def show_message(type: Literal["human", "agent"], message: str) -> None:
    with st.chat_message(type):
        st.markdown(message)


def app() -> None:
    load_dotenv(override=True)

    st.title("Tools Agent")

    # Save agent in st.session_state
    if "agent" not in st.session_state:
        _agent = ToolsAgent(
            llm=AzureOpenAIWrapper().get_azure_chat_openai(),
            tools=[
                BingSearchWrapper().get_bing_search_tool(),
            ],
        )
        _agent.subscribe(show_message)
        st.session_state.agent = _agent

    agent = st.session_state.agent

    # Sidebar
    with st.sidebar:
        "[Azure Portal](https://portal.azure.com/)"
        "[View the source code](https://github.com/ks6088ts-labs/workshop-llm-agents/blob/main/workshop_llm_agents/graphs/tools_agent.py)"
        st.image(
            agent.mermaid_png(
                output_file_path="docs/images/tools_agent.png",
            )
        )

    # Save thread_id in st.session_state
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    thread_id = st.session_state.thread_id
    st.write(f"thread_id: {thread_id}")

    # Accept user input
    query = st.chat_input()
    if query:
        with st.spinner():
            agent.run(query, thread_id)


app()
