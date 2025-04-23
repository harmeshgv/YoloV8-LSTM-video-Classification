import streamlit as st
import requests
import os
import tempfile
import time

# Use backend service name in Docker network
BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000')

st.title("Video Violence Detection")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Process Video"):
        with st.spinner('Processing video...'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                with open(tmp_file_path, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(f"{BACKEND_URL}/process-video/", files=files)

                if response.status_code == 200:
                    processed_video_path = "processed_output.mp4"
                    with open(processed_video_path, "wb") as f:
                        f.write(response.content)

                    st.success("Video processed successfully!")
                    st.video(processed_video_path)

                    csv_response = requests.get(f"{BACKEND_URL}/get-results/")
                    if csv_response.status_code == 200:
                        st.download_button(
                            label="Download Analysis Results",
                            data=csv_response.content,
                            file_name="violence_analysis_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.error(f"Processing failed: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend service. Please try again later.")
            finally:
                time.sleep(0.5)
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
