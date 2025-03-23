import streamlit as st
import requests
from streamlit_chat import message
from dotenv import load_dotenv
import os
from datetime import datetime, timezone

# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
API_URL = f"{API_URI}/chat/"
SAVE_SESSION_URL = f"{API_URI}/session/"  # <-- new endpoint for saving full chat

def run(session):
    if not session:
        st.warning("Please log in to access your profile.")
        st.stop()

    st.subheader("💬 Let's Chat")
    profile = st.session_state.get("profile")

    if profile:
        st.write(f"Welcome {st.session_state['display_name']}!")
    else:
        st.warning("No profile found. Please login.")

    # --- Session state initialization ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "session_start_time" not in st.session_state:
        st.session_state["session_start_time"] = datetime.utcnow()

    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = f"session_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    # --- Chat logic ---
    if not st.session_state["messages"]:
        if st.button("Start Chat"):
            if profile:
                welcome_text = f"👋 Hi {st.session_state['display_name']}, let's begin!"
                st.session_state["messages"].append({"text": welcome_text, "is_user": False})
                st.session_state["chat_history"].append({"text": welcome_text, "is_user": False})

                st.rerun()
            else:
                st.warning("Please log in to start.")
    else:
        
        # --- End Session Button ---
        if st.button("Save & End Session"):
            st.info("📤 Saving session...")

            now = datetime.now(timezone.utc)
            conversation_date = now.date().isoformat()  # 'YYYY-MM-DD'

            save_payload = {
                "session_id": st.session_state["chat_session_id"],
                "user_email": session["user"]["email"],
                "conversation": st.session_state["chat_history"],
                "started_at": str(st.session_state["session_start_time"]),
                "ended_at": str(now),
            }
            try:
                #print("Sending payload:", save_payload)
                # Grab token from the session
                access_token = session.get("access_token") 

                # Send token in Authorization header
                headers = {"Authorization": f"Bearer {access_token}"}

                save_response = requests.post(SAVE_SESSION_URL, json=save_payload, headers=headers)

              
            except Exception as e:
                print("Error::", e)
            if save_response.status_code == 201:
                st.success("✅ Session saved successfully!")
                # Reset session
                st.session_state["messages"] = []
                st.session_state["chat_history"] = []
            else:
                st.error("❌ Failed to save session. Please try again.")
        
        # Display past messages
        for i, msg in enumerate(st.session_state["messages"]):
            message(msg["text"], is_user=msg["is_user"], key=f"msg_{i}")

        # Input box
        user_input = st.text_input("Type your message...")

        if st.button("Send"):
            if user_input:
                # Save user message
                st.session_state["messages"].append({"text": user_input, "is_user": True})
                st.session_state["chat_history"].append({"text": user_input, "is_user": True})

                message(user_input, is_user=True, key=f"user_{len(st.session_state['messages'])}")

                # API call to backend
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


                # Save bot response
                st.session_state["messages"].append({"text": bot_response, "is_user": False})
                st.session_state["chat_history"].append({"text": bot_response, "is_user": False})

                message(bot_response, is_user=False, key=f"bot_{len(st.session_state['messages']) + 1}")
            else:
                st.warning("Please enter a message.")

       


