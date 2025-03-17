import streamlit as st
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Use API URL from .env
API_URI = os.getenv("API_URI")


API_URL = f"{API_URI}/mental-health-assessment/"
def run():
    st.subheader("💬 Chat with AI")
    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)
    user_response = st.text_area("How are you feeling today?")

    if st.button("Submit"):
        if name and age and user_response:
            payload = {"name": name, "age": age, "message": user_response}
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                st.write(f"🤖 AI: {response.json()['assessment']}")
            else:
                st.error("Error in response.")
        else:
            st.warning("Please fill in all fields.")
