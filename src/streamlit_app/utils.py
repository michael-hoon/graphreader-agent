import json
import logging
import uuid
from datetime import datetime
from typing import List

from langchain_core.messages import HumanMessage, AIMessage

import streamlit as st

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def log_feedback(feedback_value: int, index: int) -> None:
    """
    Logs user feedback and updates the sentiment in the corresponding AIMessage.

    Args:
        feedback_value (int): Feedback value (1 for thumbs up, 0 for thumbs down).
        index (int): Index of the AIMessage receiving feedback.
    """
    # get last two messages for logging
    current_message = st.session_state.messages[index]
    previous_message = st.session_state.messages[index - 1] if index > 0 else None

    sentiment = "positive" if feedback_value == 1 else "negative"
    
    log_messages = []
    if previous_message:
        log_messages.append(message_to_dict(previous_message))
    log_messages.append(message_to_dict(current_message))
    
    activity = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {sentiment}: {json.dumps(log_messages)}"
    logger.info(activity)
    
    st.toast("Feedback submitted successfully.", icon="👌")

    if isinstance(current_message, AIMessage):
        if "feedback" not in current_message.additional_kwargs:
            current_message.additional_kwargs["feedback"] = {}
        
        current_message.additional_kwargs["feedback"].update({
            "sentiment": sentiment,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    st.rerun()

def message_to_dict(message: HumanMessage | AIMessage) -> dict:
    """
    Converts a HumanMessage or AIMessage object to a dictionary for serialization.

    Args:
        message (MessageType): The message object to convert.

    Returns:
        dict: JSON-serializable dictionary representation of the message.
    """
    base_dict = {
        "content": message.content,
        "type": "ai" if isinstance(message, AIMessage) else "human",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if isinstance(message, AIMessage):
        base_dict.update({
            "feedback": message.additional_kwargs.get("feedback", {}),
            "metadata": message.additional_kwargs.get("metadata", {}),
        })

    return base_dict

def get_conversation_export(user_id: uuid.UUID, messages: List[HumanMessage | AIMessage]) -> str:
    """
    Prepares the conversation data for export as a JSON string.

    Args:
        user_id (uuid.UUID): Unique ID for the user session.
        messages (list): List of conversation messages.

    Returns:
        str: JSON string of the conversation data.
    """
    conversation_data = {
        "conversation_id": str(user_id),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [message_to_dict(msg) for msg in messages],
    }
    return json.dumps(conversation_data, indent=2)

def click_delete(val: int, message: List) -> None:
    """
    Deletes a message from the conversation history based on the index provided.

    Args:
        val (int): Index of the message to delete.
        message (HumanMessage | AIMessage): The message object to delete.
    """
    index_to_delete = val - 1
    if 0 <= index_to_delete < len(message):
        if isinstance(message[index_to_delete], HumanMessage):
            del message[index_to_delete]
            # check if there's an AI message to delete following the user's message
            if index_to_delete < len(message) and isinstance(message[index_to_delete], AIMessage):
                del message[index_to_delete]