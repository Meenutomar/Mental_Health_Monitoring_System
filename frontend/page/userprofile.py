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


def fetch_profile(email):
    """Fetch user profile from FastAPI backend."""
    response = requests.get(f"{API_URI}/profile/{email}")
    print('Fetch Profile:', response.json())
    if response.status_code == 200:
        return response.json()
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
        "Authorization": f"Bearer {token}",
        "Content-Type": "image/png"
    }

    response = requests.post(
        f"{API_URI}/upload_image/{file_name}",
        files={"file": ("file", image_data, "image/png")},
        headers=headers
    )

    if response.status_code == 200:
        return response.json().get("url")  # Get uploaded image URL
    return None


def run():
    # 1. Authenticate user
    session = login_form()

    if not session:
        st.warning("Please log in to access your profile.")
        st.stop()

    token = session['access_token']
    user_email = session['user']['email']

    st.title("My Profile")

    # 2. Fetch existing profile
    existing_profile = fetch_profile(user_email)
    print('Existing Profile:', existing_profile)

    # 3. Profile Pic Columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Show existing or default profile pic
    profile_pic_url = existing_profile["profile_pic_url"] if existing_profile else "./assets/default_profile.png"

    with col1:
        st.markdown("**Profile Picture**")
        st.image(profile_pic_url, width=120)

    # Upload new image
    with col2:
        st.markdown("**Upload Image**")
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    # Capture new image from camera
    with col3:
        st.markdown("**Take Picture**")
        captured_img = st.camera_input("Capture Image")

    # 4. User Profile Form
    st.markdown("### Personal Details")

    name = st.text_input("Full Name", value=existing_profile["name"] if existing_profile else "")
    age = st.number_input("Age", min_value=18, max_value=100,
                          value=existing_profile["age"] if existing_profile else 30, step=1)
    phone = st.text_input("Phone Number", value=existing_profile["phone"] if existing_profile else "")
    address = st.text_area("Communication Address", value=existing_profile["address"] if existing_profile else "")
    city = st.text_input("City", value=existing_profile["city"] if existing_profile else "")
    state = st.text_input("State", value=existing_profile["state"] if existing_profile else "")
    country = st.text_input("Country", value=existing_profile["country"] if existing_profile else "")
    zip_code = st.text_input("PIN Code / Zip Code", value=existing_profile["zip_code"] if existing_profile else "")

    # 5. Save Button
    if st.button("Save Profile"):
        profile_pic_url = existing_profile["profile_pic_url"] if existing_profile else ""

        # Upload image if available
        img_data = uploaded_file or captured_img
        if img_data:
            img_bytes = img_data.getvalue()
            profile_pic_url = upload_image_to_supabase(io.BytesIO(img_bytes), user_email, token)

        # Prepare profile payload
        profile_data = {
            "email": user_email,
            "name": name,
            "age": age,
            "phone": phone,
            "address": address,
            "city": city,
            "state": state,
            "country": country,
            "zip_code": zip_code,
            "profile_pic_url": profile_pic_url,
        }
        print('Save Profile: ', profile_data)
        if save_profile(profile_data, token):
            st.success("Profile saved successfully!")
        else:
            st.error("Error saving profile")
