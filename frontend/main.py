import streamlit as st
st.set_page_config(page_title="Mental Health Diagnostic Tool", layout="wide")

from streamlit_option_menu import option_menu
import page.chat  
import page.image_upload
import page.live_audio
import page.live_video
import page.speech



# Detect Streamlit theme using CSS (Trick)
is_dark_theme = st.get_option("theme.base") == "dark"

# Define colors based on theme
bg_color = "#262730" if is_dark_theme else "#f8f9fa"  # Dark for dark mode, Light for light mode
text_color = "white" if is_dark_theme else "black"
selected_bg_color = "#FF8C00"  # Orange for selection

st.markdown("<h1 style='text-align: center; color: cyan;'>💙 Mental Health AI 💙</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3468/3468089.png", width=100)
st.sidebar.markdown("### AI-powered mental health chatbot & mood analysis.")

# Sidebar Menu
with st.sidebar:
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
                "color": text_color,  # Adjust text color
                "background-color": bg_color,  # Adjust background color
            },
            "nav-link-selected": {"background-color": selected_bg_color, "color": "white", "font-weight": "bold"},
        },
    )

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
