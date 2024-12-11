import sys
import os
import uuid
import asyncio
from datetime import datetime

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.graph import chatbot_response
from agents.configuration import BaseConfiguration
from st_callable_util import get_streamlit_cb
from kg_builder.run_pipeline import process_kg
from kg_builder.kg_reset import Neo4jResetter

from utils import (
    log_feedback,
    get_conversation_export,
)

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig

st.set_page_config(
    page_title="GraphMind: Your Personal AI Research Assistant",
    page_icon="🧠",
    layout="wide",
)
st.title("🧠 GraphMind: Your Personal AI Research Assistant")

# st body content
"""
GraphMind is here to help you with your research questions! Ask anything related to your research, and I'll do my best to provide you with the information you need.

---
"""

# initialize session state for unique ID and chat history
if "user_id" not in st.session_state:
    st.session_state.user_id = uuid.uuid4()

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! How can I help you today?")]

# states for pill handling
if "selected_pill" not in st.session_state:
    st.session_state.selected_pill = None

if "first_interaction" not in st.session_state:
    st.session_state.first_interaction = False

# state for feedback handling
if "feedback_states" not in st.session_state:
    st.session_state.feedback_states = {}

# create a new container for streaming messages only, and give it context
st_callback = get_streamlit_cb(st.container())

config = {
    "configurable": {"thread_id": st.session_state.user_id},
    "callbacks": [st_callback],
}

# base_config = BaseConfiguration.from_runnable_config(config)

# sidebar setup
with st.sidebar:
    st.image("../../static/graphmind_logo.jpg")
    st.caption("Your AI-powered research assistant for multi-hop reasoning and long-context queries.")
    
    st.header("LLM Management")
    model = st.selectbox("Select LLM Model", ["llama3.1", "gpt-4o-mini", "Qwen2.5-Coder"])
    embeddings = st.selectbox("Select Embedding Model", ["text-embedding-3-small", "nomic-embed-text"])

    st.header("Graph Management")
    if st.button(
        label="Process Knowledge Graph",
        icon="🧠",
    ):
        with st.spinner("Processing knowledge graph..."):
            asyncio.run(process_kg())
        st.caption("Knowledge graph processing complete.")

    if st.button(
            label="Reset Knowledge Graph",
            icon="⚠️",
        ):
        with st.spinner("Resetting knowledge graph..."):
            try:
                resetter = Neo4jResetter()
                resetter.reset_graph()
                st.caption("Knowledge graph reset successfully.")
            except Exception as e:
                st.error(f"An error occurred while resetting the graph: {e}")

    st.header("Chat Management")
    if st.button(
        label="Clear Chat History",
        icon="🔄",
    ):
        st.session_state.messages = [AIMessage(content="Hello! How can I help you today?")]
        del st.session_state.user_id
        st.session_state.first_interaction = False
        st.rerun()
        st.caption("Chat history cleared successfully.")

    # feedback buttons
    json_conversation = get_conversation_export(st.session_state.user_id, st.session_state.messages)
    st.download_button(
        label="Save Conversation",
        data=json_conversation,
        file_name=f"chat_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        icon="📥",
    )

    st.header("Conversation ID")
    st.markdown(
            f"Conversation ID: **{st.session_state.user_id}**",
            help=f"Set URL query parameter ?convo_id={st.session_state.user_id} to continue this conversation",
        )

# display chat messages from history
for index, message in enumerate(st.session_state.messages):
    # only show feedback for AIMessage
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
            feedback_key = f"feedback_{index}"
            feedback_value = st.feedback(
                options="thumbs", 
                key=feedback_key,
            )
            # checking if feedback has changed
            if feedback_value is not None:
                if feedback_key not in st.session_state.feedback_states or st.session_state.feedback_states[feedback_key] != feedback_value:
                    st.session_state.feedback_states[feedback_key] = feedback_value
                    log_feedback(feedback_value, index)

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

sample_questions = [
    "☢️ What is deep learning and how is it used in nuclear safety research?",
    "📊 How does the YOLOv4 model compare to the YOLOv5 model?",
    "🤖 What are the key differences between supervised and unsupervised learning?",
    "💻 What are the applications of natural language processing in healthcare?",
    ]

# only show sample questions if there has been no user input yet
if not st.session_state.first_interaction:
    selected_input = st.pills(
        label="Sample Questions",
        options=sample_questions,
        selection_mode="single",
        label_visibility="collapsed",
    )
    if selected_input:
        st.session_state.selected_pill = selected_input
        st.session_state.first_interaction= True
        st.chat_message("user").markdown(selected_input)
        st.session_state.messages.append(HumanMessage(content=selected_input))

        with st.spinner(text="Thinking..."):
            bot_response = chatbot_response(input_text=selected_input, config=config)

        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append(AIMessage(content=bot_response))
        # st.rerun()

if user_input := st.chat_input("Ask me anything!"):
    st.session_state.first_interaction = True

    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.spinner(text="Thinking..."):
        bot_response = chatbot_response(input_text=user_input, config=config)

    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append(AIMessage(content=bot_response))
    # st.rerun()