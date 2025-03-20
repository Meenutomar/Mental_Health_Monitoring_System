import streamlit as st
from streamlit_supabase_auth import login_form

def show_login():
    st.title("Login to RoboMH")

    session = login_form()  # Supabase authentication

    if session:  # If login is successful
        st.session_state.authenticated = True
        st.session_state.user_email = session['user']['email']
        st.success(f"Welcome, {st.session_state.user_email}!")
        st.rerun()  # Redirect to dashboard
