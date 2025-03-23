import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import requests
from streamlit_chat import message
from dotenv import load_dotenv
from datetime import datetime, timezone
import streamlit as st
import requests
import pandas as pd
import report_pdf
import altair as alt

# Load environment variables
load_dotenv()
API_URI = os.getenv("API_URI")
ANALYZE_SESSION_URL = f"{API_URI}/analysis" 

def run(session):
    if not session:
        st.warning("Please log in to access your profile.")
        st.stop()
    
    st.subheader("📚 My Reports")

    profile = st.session_state.get("profile")

    if not profile:
        st.warning("No profile found. Please login.")
    else:
        user = profile['email']
        analyses = fetch_user_analyses(session, user)
        print('Analyses:', analyses)
        if analyses:
            df = pd.DataFrame(analyses)
            df['created_at'] = pd.to_datetime(df['created_at'])

            st.subheader("📈 Recovery Over Time")
            # Sort by created_at to ensure proper order
            df = df.sort_values('created_at')

            # Bar chart with timestamp (datetime) on X-axis
            bar_chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('created_at:T', title='Session Time'),
                y=alt.Y('score:Q', title='Recovery Score'),
                tooltip=['created_at:T', 'score:Q', 'tone:N']
            ).properties(
                title='Recovery Score per Session',
                width=700,
                height=400
            )

            st.altair_chart(bar_chart, use_container_width=True)


            st.subheader("🧠 Session Insights")
            for i, row in df.iterrows():
                with st.expander(f"Session on {row['created_at'].strftime('%Y-%m-%d')}"):
                    st.markdown(f"**Emotions Detected:** {row['tone']}")
                    st.markdown(f"**Insights:** {row['change']}")
                    st.markdown(f"**Score:** {row['score']}")
                    st.markdown(f"**Analysis:** {row['summary']}")
            
            if st.button("📥 Download Progress Report"):
                pdf_buffer = report_pdf.generate_pdf(profile, analyses, logo_path='./assets/logo.png')
                st.download_button(
                    label="Download PDF",
                    data=pdf_buffer,
                    file_name="mental_health_report.pdf",
                    mime="application/pdf"
                )
        else:
            st.info("No pre-existing reports found.")
            if st.button("🔄 Generate Fresh"):
                analysis_payload = {
                    "session_id": st.session_state["chat_session_id"],
                    "user_id": profile["email"]
                }
                # Grab token from the session
                access_token = session.get("access_token") 

                # Send token in Authorization header
                headers = {"Authorization": f"Bearer {access_token}"}

                # After session saved to Supabase from frontend
                response = requests.post(ANALYZE_SESSION_URL, json=analysis_payload, headers=headers)
                if response.status_code == 200:
                    st.success("Analysis completed successfully.")
                    st.rerun()
                else:
                    st.error("Failed to analyze session.")
                
               
    
def fetch_user_analyses(session, user):
    if not session:
        st.warning("Please log in to access your profile.")
    
    access_token = session.get("access_token") 
    headers = {
                "Authorization": f"Bearer {access_token}"
        }

    response = requests.get(f"{ANALYZE_SESSION_URL}/user/{user}", headers=headers)
 
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch analysis data.")
        return []
