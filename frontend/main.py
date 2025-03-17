import streamlit as st
import page.chat  
import page.image_upload
import page.live_audio
import page.live_video
import page.speech

st.set_page_config(page_title="Mental Health Diagnostic Tool", layout="wide")


# Injecting CSS for Sidebar Styling
st.markdown(
    """
    <style>
        /* Sidebar Navigation */
        .sidebar-nav {
            font-size: 22px;
            font-weight: bold;
            padding-left: 10px;
            margin-bottom: 15px;
            
        }

        /* Sidebar Menu - Flexbox for Equal Width */
        .sidebar-menu {
            display: flex;
            flex-direction: column;
            gap: 10px;  /* Space between buttons */
        }

        /* Menu Buttons - Same Size */
        .sidebar-menu button {
            width: 100%;
            height: 45px;  /* Fixed height */
            background: none;
            border: none;
            text-align: left;
            padding: 10px;
            font-size: 16px;
            font-weight: normal;
            color: #4F8BF9;  /* Blue color */
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            border-bottom: 2px solid transparent;
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }

        /* Hover Effect */
        .sidebar-menu button:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }

        /* Active Button Styling */
        .sidebar-menu .active {
            color: #FF8C00;  /* Orange */
            border-bottom: 3px solid #FF8C00;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Title
st.sidebar.markdown("<div class='sidebar-nav'>📌 Navigation</div>", unsafe_allow_html=True)

# Define Pages with Icons
menu_options = {
    "Chat": ("💬", "chat"),
    "Image Upload": ("📷", "image_upload"),
    "Live Audio": ("🎵", "live_audio"),
    "Live Video": ("📹", "live_video"),
    "Speech": ("🎙️", "speech"),
}

# Store the selected page in session state
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Chat"

# Render Sidebar Menu
st.sidebar.markdown("<div class='sidebar-menu'>", unsafe_allow_html=True)

for name, (icon, key) in menu_options.items():
    button_label = f"{icon} {name}"
    if st.sidebar.button(button_label, key=name):
        st.session_state["selected_page"] = name

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Render the selected page
if st.session_state["selected_page"] == "Chat":
    page.chat.run()
elif st.session_state["selected_page"] == "Image Upload":
    page.image_upload.run()
elif st.session_state["selected_page"] == "Live Audio":
    page.live_audio.run()
elif st.session_state["selected_page"] == "Live Video":
    page.live_video.run()
elif st.session_state["selected_page"] == "Speech":
    page.speech.run()
