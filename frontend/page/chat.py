import streamlit as st
import requests
from streamlit_chat import message
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
API_URL = f"{API_URI}/mental-health-assessment/"

def run():
    st.subheader("💬 Chat with AI")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous chat messages
    for msg in st.session_state["messages"]:
        message(msg["text"], is_user=msg["is_user"])

    # User input
    user_input = st.text_input("Type your message...")

    if st.button("Send"):
        if user_input:
            # Display user message
            st.session_state["messages"].append({"text": user_input, "is_user": True})
            message(user_input, is_user=True)

            # Send user input to API
            payload = {"message": user_input}
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                bot_response = response.json().get("assessment", "I'm here to help.")
            else:
                bot_response = "⚠️ Error: Unable to process request."

            # Display AI response
            st.session_state["messages"].append({"text": bot_response, "is_user": False})
            message(bot_response, is_user=False)

        else:
            st.warning("Please enter a message.")

