import streamlit as st
import io
from services.userservice import fetch_profile, save_profile, upload_image_to_supabase


def run(session):
    

    if not session:
        st.warning("Please log in to access your profile.")
        st.stop()

    token = session['access_token']
    user_email = session['user']['email']

    st.markdown('<h1><i class="bi bi-person"></i> My Profile</h1>', unsafe_allow_html=True)



    # 2. Fetch existing profile
    existing_profile = fetch_profile(user_email,token)
    #print('Existing Profile:', existing_profile)

    # 3. Profile Pic Columns
    col1, col2, col3 = st.columns([1, 2, 1])

    # Show existing or default profile pic
    profile_pic_url = existing_profile.get("profile_pic_url") if existing_profile else None
    profile_pic_url = profile_pic_url if profile_pic_url else "./assets/default_profile.png"
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

    name = st.text_input("Full Name", value=existing_profile.get("name", "") if existing_profile else "")
    age = st.number_input("Age", min_value=18, max_value=100, value=existing_profile.get("age", 30) if existing_profile else 30, step=1)
    phone = st.text_input("Phone Number", value=existing_profile.get("phone", "") if existing_profile else "")
    address = st.text_area("Communication Address", value=existing_profile.get("address", "") if existing_profile else "")
    city = st.text_input("City", value=existing_profile.get("city", "") if existing_profile else "")
    state = st.text_input("State", value=existing_profile.get("state", "") if existing_profile else "")
    country = st.text_input("Country", value=existing_profile.get("country", "") if existing_profile else "")
    zip_code = st.text_input("PIN Code / Zip Code", value=existing_profile.get("zip_code", "") if existing_profile else "")

    # 5. Save Button
    if st.button("Save Profile"):
        profile_pic_url = existing_profile.get("profile_pic_url", "") if existing_profile else ""

        # Upload image if available - camera takes priority
        img_data = captured_img or uploaded_file
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
