from typing import Literal
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from workshop_llm_agents.agents.research_agent import ResearchAgent
from workshop_llm_agents.llms.azure_openai import AzureOpenAIWrapper


def show_message(type: Literal["human", "agent"], title: str, message: str) -> None:
    with st.chat_message(type):
        st.markdown(f"**{title}**")
        st.markdown(message)


def app() -> None:
    load_dotenv(override=True)

    st.title("Research Agent with Human-in-the-loop")

    # Save agent in st.session_state
    if "agent" not in st.session_state:
        _llm = AzureOpenAIWrapper().get_azure_chat_openai()
        _agent = ResearchAgent(_llm)
        _agent.subscribe(show_message)
        st.session_state.agent = _agent

    agent = st.session_state.agent

    # Display graph
    with st.sidebar:
        st.image(
            agent.mermaid_png(
                # output_file_path="docs/images/research_agent.png",
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
        # Reset approval state if there is user input
        st.session_state.approval_state = "pending"

    # Show approval button if the next node is human_approval
    if agent.is_next_human_approval_node(thread_id):
        if "approval_state" not in st.session_state:
            st.session_state.approval_state = "pending"

        if st.session_state.approval_state == "pending":
            approved = st.button("Approve")
            if approved:
                st.session_state.approval_state = "processing"
                st.rerun()
        elif st.session_state.approval_state == "processing":
            with st.spinner("Processing task..."):
                agent.handle_human_message("[APPROVE]", thread_id)
            st.session_state.approval_state = "pending"


app()
