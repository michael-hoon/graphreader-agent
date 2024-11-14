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
    page_icon="🤖",
)
st.title("🤖 GraphMind: Your Personal AI Research Assistant")

# st write magic
"""
GraphMind is here to help you with your research questions! Ask anything related to your research, and I'll do my best to provide you with the information you need.

---
"""

# initialize session state for unique ID and chat history
if "user_id" not in st.session_state:
    st.session_state.user_id = uuid.uuid4()

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! How can I help you today?")]

# create a new container for streaming messages only, and give it context
st_callback = get_streamlit_cb(st.container())
config = {
    "configurable": {"thread_id": st.session_state.user_id},
    "callbacks": [st_callback],
}

# display chat messages from history
for message in st.session_state.messages:
    if type(message) == AIMessage:
        with st.chat_message("assistant"):
            st.markdown(message.content)
    if type(message) == HumanMessage:
        with st.chat_message("user"):
            st.markdown(message.content)

# user input handling
if user_input := st.chat_input("Ask me anything!"):
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