import streamlit as st
from login import show_login
from main import show_dashboard

#st.set_page_config(page_title="RoboMH", layout="wide")
st.markdown("""
   <style>
        /* Change color of all buttons */
        button, [data-testid="baseButton"] {
            background-color: #139262 !important;  /* Green color */
            color: white !important;
            border-radius: 5px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            padding: 8px 16px;
        }

        button:hover, [data-testid="baseButton"]:hover {
            background-color: #218838 !important; /* Darker green */
        }
    </style>
""", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    show_dashboard()  # If logged in, show the dashboard
else:
    show_login()  # If not logged in, show login page
