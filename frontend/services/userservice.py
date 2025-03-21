import streamlit as st
from PIL import Image
from streamlit_supabase_auth import login_form
import requests
import io
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")  # FastAPI backend URL
SUPABASE_URL = os.getenv("SUPABASE_URL")

def fetch_profile(email, token):
    """Fetch user profile from FastAPI backend."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URI}/profile/{email}", headers=headers)
    #print('Fetch Profile:', response.json())
    if response.status_code == 200:
        profile = response.json()
        if "profile" not in st.session_state:
            st.session_state.profile = profile
        return profile
    return None

def save_profile(profile_data, token):
    """Save or update user profile via FastAPI backend with auth token."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URI}/profile", json=profile_data, headers=headers)
    return response.status_code == 200

def upload_image_to_supabase(image_data, email, token):
    """Upload image to Supabase Storage via FastAPI backend."""
    file_name = f"{email}.png"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    files = {
        "file": ("file", image_data, "image/png")
    }

    response = requests.post(
        f"{API_URI}/upload_image/{file_name}",
        files=files,
        headers=headers  # Don't manually set Content-Type
    )

    if response.status_code == 200:
        return response.json().get("url")
    
    print("Upload Failed:", response.status_code, response.text)
    return None
