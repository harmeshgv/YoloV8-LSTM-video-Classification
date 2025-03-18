import streamlit as st
import requests

st.title("Video Violence Detection")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Process Video"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/process-video/", files=files)

        if response.status_code == 200:
            st.video(response.content)
        else:
            st.error("Video processing failed.")