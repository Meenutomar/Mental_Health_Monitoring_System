import page.my_sessions
import page.userprofile
import streamlit as st
from streamlit_supabase_auth import logout_button, login_form
from streamlit_option_menu import option_menu
import base64
import page.chat  
import page.image_upload
import page.report
import page.my_sessions
from services.userservice import fetch_profile

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
""", unsafe_allow_html=True)


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
   
    # 1. Authenticate user
    session = login_form()

    if not session:
        st.warning("Please log in to access your profile.")
        st.stop()

    token = session['access_token']
    user_email = session['user']['email']


    # 2. Fetch existing profile
    profile = fetch_profile(user_email,token)
    if profile is None or profile.get('status') == 'not_found':
        display_name = st.session_state.user_email
        profile_pic =  "./assets/default_profile.png"
    else:
        display_name =  profile.get('name', st.session_state.user_email)
        profile_pic = profile.get("profile_pic_url", "./assets/default_profile.png")

    if "display_name" not in st.session_state:
        st.session_state["display_name"] = display_name
    if "profile_pic" not in st.session_state:
        st.session_state["profile_pic"] = profile_pic

   # ✅ Inject Header with Orange Bottom Border
    st.markdown(f"""
    <style>
        /* Hide Streamlit default header & footer */
        #MainMenu, header, footer {{
            visibility: hidden;
        }}

        /* Custom Header */
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 70px;
            background-color: {header_bg};
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 16px;
            font-weight: bold;
            color: white;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
            border-bottom: 5px solid {border_color};
            z-index: 10000;
        }}

        .header-left {{
            display: flex;
            align-items: center;
        }}

        .header-left img {{
            height: 50px;
            margin-right: 10px;
        }}

        .header-right {{
            display: flex;
            align-items: center;
        }}

        .header-right p {{
            margin: 0 10px 0 0;
        }}

        .header-right img {{
            height: 45px;
            width: 45px;
            border-radius: 50%;
            border: 2px solid white;
        }}
         /* Fixed Footer */
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: {border_color};
            color: white;
            text-align: right;
            padding: 10px;
            font-size: 14px;
            z-index: 1000;
        }}

        /* Push content below the fixed header */
        .block-container {{
            padding-top: 85px !important;
        }}
    </style>

    <div class="header">
        <div class="header-left">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo">
            <span>RoboMH</span>
        </div>
        <div class="header-right">
            <p>Welcome {display_name}!</p>
            <img src="{profile_pic}" alt="Profile Pic">
        </div>
    </div>
    """, unsafe_allow_html=True)

     # Sidebar
    st.sidebar.image("./assets/logo.png", width=80)
    st.sidebar.markdown("***AI-powered mental health chatbot***")

    with st.sidebar:
        # Sidebar Menu
        selected_page = option_menu(
            menu_title="",
            options=["Chat",  "My Profile", "My Sessions", "Reports"],
            icons=["chat-dots", "person", "inboxes", "bar-chart"],
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
        page.chat.run(session)
    elif selected_page == "Image Upload":
        page.image_upload.run()
    elif selected_page == "My Profile":
        page.userprofile.run(session)
    elif selected_page == "My Sessions":
        page.my_sessions.run(session)
    elif selected_page == "Reports":
        page.report.run(session)

    # Inject fixed footer
    st.markdown('<div class="footer">© 2025 RoboMH | All Rights Reserved</div>', unsafe_allow_html=True)
