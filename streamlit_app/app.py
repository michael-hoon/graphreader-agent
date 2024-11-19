import sys
import os
import uuid

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.graph import chatbot_response
from st_callable_util import get_streamlit_cb

from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="GraphMind: Your Personal AI Research Assistant",
    page_icon="🧠",
    layout="wide",
)
st.title("🧠 GraphMind: Your Personal AI Research Assistant")

with st.sidebar:
    st.image("/home/micha/Python_Files/graphreader-agent/static/graphmind_logo.jpg")
    st.caption("Your AI-powered research assistant for multi-hop reasoning and long-context queries.")
    
    st.header("Graph Management")
    if st.button("Process Knowledge Graph"):
        with st.spinner("Processing knowledge graph..."):
            st.caption("Knowledge graph processing complete.")
            pass
        pass

    st.header("LLM Management")
    llm = st.selectbox("Select LLM Model", ["llama3.1", "gpt-4o-mini", "Qwen2.5-Coder"])
    embeddings = st.selectbox("Select Embedding Model", ["text-embedding-3-small", "nomic-embed-text"])

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

if "selected_pill" not in st.session_state:
    st.session_state.selected_pill = None

if "first_interaction" not in st.session_state:
    st.session_state.first_interaction = False

# create a new container for streaming messages only, and give it context
st_callback = get_streamlit_cb(st.container())
config = {
    "configurable": {"thread_id": st.session_state.user_id},
    "callbacks": [st_callback],
}

# display chat messages from history
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
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
        st.rerun()

# user input handling
if user_input := st.chat_input("Ask me anything!"):
    st.session_state.first_interaction = True
    # display and store the user's message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # generate bot response
    with st.spinner(text="Thinking..."):
        bot_response = chatbot_response(input_text=user_input, config=config)
    # display and store the bot's response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append(AIMessage(content=bot_response))
    st.rerun()