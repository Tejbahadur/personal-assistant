"""
Streamlit-based Chat Interface for Personal Assistant.
Connects to LM Studio running locally.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
import json
from datetime import datetime
from agents.llm_client import LMStudioClient
from agents.orchestrator import create_assistant_graph
from agents.state import AssistantState
from config import CONFIG

def main():
    st.set_page_config(page_title="Personal Assistant - Tejbahadur", layout="wide")
    st.title("🤖 Personal Assistant - Tejbahadur")

    # Initialize session state
    if "assistant_graph" not in st.session_state:
        st.session_state.assistant_graph = create_assistant_graph()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_metadata" not in st.session_state:
        st.session_state.last_metadata = {}

    # Sidebar: Chat history
    with st.sidebar:
        st.subheader("📋 Chat History")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_metadata = {}
            st.rerun()

        if st.session_state.messages:
            for i, msg in enumerate(st.session_state.messages):
                if msg.type == "human":
                    st.write(f"**You** ({i}): {msg.content[:50]}...")
                else:
                    st.write(f"**Assistant** ({i}): {msg.content[:50]}...")

    # Main chat area
    st.subheader("💬 Chat")

    # Display chat history
    for msg in st.session_state.messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Check for UI actions from metadata
    last_metadata = st.session_state.get("last_metadata", {})
    ui_action = last_metadata.get("ui_action_required")

    # ============================================================
    # UI ACTION: Generate Documents
    # ============================================================
    if ui_action == "generate_docs":
        st.info("🎯 **Action Required:** Ready to generate tailored Cover Letter and Resume?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Generate Documents", use_container_width=True, key="btn_generate"):
                st.session_state.messages.append(HumanMessage(content="UI_ACTION: GENERATE_DOCS"))
                result = st.session_state.assistant_graph.invoke({
                    "messages": st.session_state.messages
                })

                # Update state
                st.session_state.messages.append(AIMessage(content=result["response_text"]))
                st.session_state.last_metadata = result.get("metadata", {})

                with st.chat_message("assistant"):
                    st.markdown(result["response_text"])

                st.rerun()

        with col2:
            if st.button("❌ Cancel", use_container_width=True, key="btn_cancel"):
                st.session_state.last_metadata = {}
                st.write("Cancelled. You can paste another JD anytime.")
                st.rerun()

    # ============================================================
    # UI ACTION: Approve Submission
    # ============================================================
    elif ui_action == "approve_submission":
        app_id = last_metadata.get("app_id")
        folder_path = last_metadata.get("folder_path")

        st.warning("⚠️ **Action Required:** Please review the documents in your local folder.")
        st.info(f"📂 Folder: `{folder_path}`")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Approve for Submission", use_container_width=True, key="btn_approve"):
                st.session_state.messages.append(HumanMessage(content=f"UI_ACTION: APPROVE_SUBMISSION_{app_id}"))
                result = st.session_state.assistant_graph.invoke({
                    "messages": st.session_state.messages
                })

                st.session_state.messages.append(AIMessage(content=result["response_text"]))
                st.session_state.last_metadata = result.get("metadata", {})

                with st.chat_message("assistant"):
                    st.markdown(result["response_text"])

                st.rerun()

        with col2:
            if st.button("✏️ Needs Manual Edits", use_container_width=True, key="btn_edit"):
                st.info("📝 Please edit the files in the folder manually, then approve when ready.")
                st.session_state.last_metadata = {}
                st.rerun()

        with col3:
            if st.button("↩️ Back", use_container_width=True, key="btn_back"):
                st.session_state.last_metadata = {}
                st.rerun()

    # ============================================================
    # User Input
    # ============================================================
    st.divider()
    user_input = st.chat_input("Paste a job description or ask a question...", key="chat_input")

    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))

        with st.spinner("Processing..."):
            result = st.session_state.assistant_graph.invoke({
                "messages": st.session_state.messages
            })

        st.session_state.messages.append(AIMessage(content=result["response_text"]))
        st.session_state.last_metadata = result.get("metadata", {})

        with st.chat_message("assistant"):
            st.markdown(result["response_text"])

        st.rerun()


if __name__ == "__main__":
    main()