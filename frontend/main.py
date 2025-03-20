import streamlit as st
from streamlit_supabase_auth import logout_button
from streamlit_option_menu import option_menu
import base64
import page.chat  
import page.image_upload
import page.live_audio
import page.live_video
import page.speech


# Define colors
bg_color = "#262730"  # Sidebar background
text_color = "white"  
selected_bg_color = "#139262"  # Darker orange for selection
header_bg = "#1E1E1E"  # Dark Gray header (modern look)
border_color = "#a887d6"  # Orange bottom border

# ✅ Convert logo.png to base64 for embedding in HTML
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

logo_base64 = get_base64_image("./assets/logo.png")



def show_dashboard():
 
    if not st.session_state.get("authenticated", False):
        st.switch_page("login.py")  # Redirect to login if not authenticated

     # Sidebar
    st.sidebar.image("./assets/logo.png", width=80)
    st.sidebar.markdown("***AI-powered mental health chatbot***")

    with st.sidebar:
        # Sidebar Menu
        selected_page = option_menu(
            menu_title="",
            options=["Chat", "Image Upload", "Live Audio", "Live Video", "Speech"],
            icons=["chat-dots", "cloud-upload", "mic", "camera-video", "soundwave"],
            menu_icon="list",
            default_index=0,
            styles={
            "container": {"padding": "10px", "background-color": bg_color},
            "icon": {"color": text_color, "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "padding": "10px",
                "border-radius": "5px",
                "color": text_color,
                "background-color": bg_color,
                "border-color": border_color,
                "border": 4
            },
            "nav-link-selected": {"background-color": selected_bg_color, "color": "white", "font-weight": "bold"},
        },
        )
      
        st.write(f"Logged in as: {st.session_state.user_email}")
        
        if logout_button():  # SINGLE Logout Button ✅
            st.session_state.clear()  # Clear all session variables
            st.session_state.authenticated = False  # Explicitly reset authentication
            st.switch_page("login.py")  # Redirect to login

        st.write("---")
    # Render the selected page
    if selected_page == "Chat":
        page.chat.run()
    elif selected_page == "Image Upload":
        page.image_upload.run()
    elif selected_page == "Live Audio":
        page.live_audio.run()
    elif selected_page == "Live Video":
        page.live_video.run()
    elif selected_page == "Speech":
        page.speech.run()
