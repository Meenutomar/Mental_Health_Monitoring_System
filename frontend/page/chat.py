import streamlit as st
import requests
from streamlit_chat import message
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
API_URL = f"{API_URI}/chat/"

def run(session):
    if not session:
        st.warning("Please log in to access your profile.")
        st.stop()

    st.subheader("💬 Lets Chat ")
    profile = st.session_state.get("profile")

    if profile:
        st.write(f"Welcome {st.session_state['display_name']}!")
    else:
        st.warning("No profile found. Please login.")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
  

    # Ask for name and age at the beginning
    if not st.session_state["messages"]:

        if st.button("Start Chat"):
            if profile:
                st.session_state["messages"].append({
                    "text": f"👋 Hi {st.session_state['display_name']}, let's begin!", "is_user": False
                })
                st.rerun()
            else:
                st.warning("Please enter your name and age to start.")

    else:
        # Display previous chat messages
        for i, msg in enumerate(st.session_state["messages"]):
            message(msg["text"], is_user=msg["is_user"], key=f"msg_{i}")


        # User input
        user_input = st.text_input("Type your message...")

        if st.button("Send"):
            if user_input:
                # Display user message
                st.session_state["messages"].append({"text": user_input, "is_user": True})
                #message(user_input, is_user=True)
                message(user_input, is_user=True, key=f"user_{len(st.session_state['messages'])}")
                # Send user input to API
                payload = {
                    "name": profile["name"],
                    "age": profile["age"],
                    "message": user_input
                }
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    bot_response = response.json().get("assessment", "I'm here to help.")
                else:
                    bot_response = "⚠️ Error: Unable to process request."

                # Display AI response
                st.session_state["messages"].append({"text": bot_response, "is_user": False})
                #message(bot_response, is_user=False)
                message(bot_response, is_user=False, key=f"bot_{len(st.session_state['messages']) + 1}")

            else:
                st.warning("Please enter a message.")
