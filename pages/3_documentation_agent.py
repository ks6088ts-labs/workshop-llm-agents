from typing import Literal
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from workshop_llm_agents.agents.documentation_agent import DocumentationAgent
from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper


def show_message(
    type: Literal["human", "agent"],
    title: str,
    message: str,
) -> None:
    with st.chat_message(type):
        st.markdown(f"**{title}**")
        st.markdown(message)


def app() -> None:
    load_dotenv(override=True)

    st.title("Documentation Agent")

    # Save agent in st.session_state
    if "agent" not in st.session_state:
        _agent = DocumentationAgent(
            llm=AzureOpenAIWrapper().get_azure_chat_openai(),
        )
        _agent.subscribe(show_message)
        st.session_state.agent = _agent

    agent = st.session_state.agent

    # Display graph
    with st.sidebar:
        st.image(
            agent.mermaid_png(
                output_file_path="docs/images/documentation_agent.png",
            )
        )

    # Save thread_id in st.session_state
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    thread_id = st.session_state.thread_id
    st.write(f"thread_id: {thread_id}")

    # Accept user input
    human_message = st.chat_input()
    if human_message:
        with st.spinner():
            agent.handle_human_message(human_message, thread_id)


app()
