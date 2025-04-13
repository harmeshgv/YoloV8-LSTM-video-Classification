import streamlit as st
import requests
import os
import tempfile
import time

st.title("Video Violence Detection")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Process Video"):
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Open the temp file and send it to the backend
            with open(tmp_file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post("http://localhost:8000/process-video/", files=files)

            if response.status_code == 200:
                # Save and display the processed video
                processed_video_path = "processed_output.mp4"
                with open(processed_video_path, "wb") as f:
                    f.write(response.content)

                st.success("Video processed successfully!")
                st.video(processed_video_path)

                # Offer download link for the CSV results
                csv_response = requests.get("http://localhost:8000/get-results/")
                if csv_response.status_code == 200:
                    st.download_button(
                        label="Download Analysis Results",
                        data=csv_response.content,
                        file_name="violence_analysis_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Failed to fetch analysis results.")
            else:
                st.error(f"Video processing failed. Status code: {response.status_code}")

        finally:
            # Optional delay to ensure file handle is released on Windows
            time.sleep(0.5)
            os.unlink(tmp_file_path)
