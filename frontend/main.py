import streamlit as st
import base64
from streamlit_option_menu import option_menu
import page.chat  
import page.image_upload
import page.live_audio
import page.live_video
import page.speech

st.set_page_config(page_title="RoboMH - The Mental Health Diagnostic Tool", layout="wide")

# Define colors
bg_color = "#262730"  # Sidebar background
text_color = "white"  
selected_bg_color = "#FF4500"  # Darker orange for selection
header_bg = "#1E1E1E"  # Dark Gray header (modern look)
border_color = "#FF4500"  # Orange bottom border

# ✅ Convert logo.png to base64 for embedding in HTML
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

logo_base64 = get_base64_image("./assets/logo.png")

# ✅ Inject Header with Orange Bottom Border
st.markdown(f"""
    <style>
        /* Custom Header */
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 70px;
            background-color: {header_bg};  /* Dark gray */
            padding: 10px 20px;
            display: flex;
            align-items: center;
           
            font-size: 12px;
            font-weight: bold;
            color: white;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
            border-bottom: 5px solid {border_color};  /* Orange Bottom Border */
            z-index: 10000;
        }}

        .header img {{
            height: 55px;
            margin-right: 10px;
        }}

        /* Push content down */
        .main-content {{
            margin-top: 85px;
        }}

        /* Hide Streamlit Default Toolbar */
        #MainMenu, header, footer {{
            visibility: hidden;
        }}
    </style>
    <div class="header">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
       
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("./assets/logo.png", width=80)
st.sidebar.markdown("***AI-powered mental health chatbot***")

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
                "color": text_color,
                "background-color": bg_color,
            },
            "nav-link-selected": {"background-color": selected_bg_color, "color": "white", "font-weight": "bold"},
        },
    )

# Add spacing for main content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)
