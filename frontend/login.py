import streamlit as st
from streamlit_supabase_auth import login_form
import base64

# ✅ Convert logo.png to base64 for embedding in HTML
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

logo_base64 = get_base64_image("./assets/logo.png")

def show_login():
    # ✅ Custom CSS for Styling
    st.markdown("""
        <style>
            .container {
                display: flex;
                height: 90vh;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .left-section {
                flex: 4;  /* Left side is now 3x wider */
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                padding: 60px; /* More padding */
                text-align: center;
                color: white;
                background-color: #1E1E1E;
                border-radius: 10px 0 0 10px;
            }
            .left-section h1 {
                font-size: 42px;
                color: #FF4500;
            }
            .left-section p {
                font-size: 22px;
                max-width: 70%;
                color: #ddd;
            }
            .right-section {
                flex: 3; /* Wider than before */
                background-color: #2B2D42;
                padding: 50px;
                border-radius: 0 10px 10px 0;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            .right-section h2 {
                text-align: center;
                color: white;
                font-size: 28px;
                margin-bottom: 20px;
            }
            div[data-testid="stForm"] button {
                background-color: #28a745 !important;
                color: white !important;
                font-size: 20px;
                font-weight: bold;
                border-radius: 5px;
                padding: 12px 18px;
                width: 100%;
            }
            div[data-testid="stForm"] button:hover {
                background-color: #218838 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Layout: Branding (Left) | Login Form (Right)
    col1, col2 = st.columns([3, 2.5])  # Left column is 3x wider, right is 1.5x wider

    with col1:
        st.markdown(
            f"""
            <div class='left-section'>
                <img src="data:image/png;base64,{logo_base64}" alt="Logo">
                <p>Your AI-powered mental health companion. Start your journey towards self-care.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    with col2:
        st.markdown("<div class='right-section'>", unsafe_allow_html=True)
        st.header("Sign In to Continue")

        session = login_form()  # Supabase authentication

        if session:
            st.session_state.authenticated = True
            st.session_state.user_email = session['user']['email']
            st.success(f"Welcome, {st.session_state.user_email}!")
            st.rerun()  # Redirect to dashboard

        st.markdown("</div>", unsafe_allow_html=True)


