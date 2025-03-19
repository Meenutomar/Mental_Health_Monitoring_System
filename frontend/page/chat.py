import streamlit as st
import requests
from streamlit_chat import message
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
API_URL = f"{API_URI}/chat/"

def run():
    st.subheader("💬 Chat with AI")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "name" not in st.session_state:
        st.session_state["name"] = None
    if "age" not in st.session_state:
        st.session_state["age"] = None

    # Ask for name and age at the beginning
    if not st.session_state["name"] or not st.session_state["age"]:
        st.session_state["name"] = st.text_input("Enter your name:")
        st.session_state["age"] = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)

        if st.button("Start Chat"):
            if st.session_state["name"] and st.session_state["age"]:
                st.session_state["messages"].append({
                    "text": f"👋 Hi {st.session_state['name']}, let's begin!", "is_user": False
                })
                st.rerun()
            else:
                st.warning("Please enter your name and age to start.")

    else:
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
                payload = {
                    "name": st.session_state["name"],
                    "age": st.session_state["age"],
                    "message": user_input
                }
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
