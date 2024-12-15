import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from workshop_llm_agents.agents.chatbot_with_tools import graph as chatbot_with_tools_graph

with st.sidebar:
    st.image(chatbot_with_tools_graph.get_graph().draw_mermaid_png())
    "[Azure Portal](https://portal.azure.com/)"
    "[View the source code](https://github.com/ks6088ts-labs/workshop-llm-agents/blob/main/workshop_llm_agents/agents/chatbot_with_tools.py)"


def is_configured():
    return True


def get_session_id():
    return get_script_run_ctx().session_id


st.title("1_chatbot_with_tools")
st.write(f"Session ID: {get_session_id()}")

if not is_configured():
    st.warning("Please fill in the required fields at the sidebar.")

if prompt := st.chat_input(disabled=not is_configured()):
    events = chatbot_with_tools_graph.stream(
        input={
            "messages": [
                ("user", prompt),
            ]
        },
        config={
            "configurable": {
                "thread_id": get_session_id(),
            },
        },
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
            st.write(event["messages"][-1].content)
