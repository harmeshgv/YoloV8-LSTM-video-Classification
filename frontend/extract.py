# frontend/app.py
import streamlit as st
import requests
import os
import tempfile
import time
from io import BytesIO

# Backend configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')  # Change to 'http://backend:8000' for Docker

# Page setup
st.set_page_config(page_title="Violence Detection System", layout="wide")
st.title("🎥 Violence Detection in Videos")
st.markdown("""
    Upload a video file to analyze for violent content using AI detection models.
    The system will identify potential violent interactions, weapons, and aggressive behaviors.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    show_processed_video = st.checkbox("Show processed video", value=True)
    st.markdown("---")
    st.markdown("**Note:** Processing may take several minutes depending on video length.")

# File upload section
uploaded_file = st.file_uploader(
    "Choose a video file (MP4, AVI, MOV)",
    type=["mp4", "avi", "mov"],
    accept_multiple_files=False
)

# Display original video
if uploaded_file is not None:
    st.header("Original Video")
    st.video(uploaded_file)

# Processing section
if uploaded_file and st.button("🚀 Process Video", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner('Initializing processing...'):
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Stage 1: Upload and process
            status_text.text("Uploading video to backend...")
            progress_bar.progress(10)
            
            with open(tmp_path, 'rb') as f:
                files = {'file': ('uploaded_video.mp4', f, 'video/mp4')}
                response = requests.post(
                    f"{BACKEND_URL}/process-video/",
                    files=files,
                    timeout=60
                )
            
            progress_bar.progress(30)
            
            # Stage 2: Handle response
            if response.status_code == 200:
                status_text.text("Processing complete! Retrieving results...")
                progress_bar.progress(70)
                
                # Show processed video if enabled
                if show_processed_video:
                    st.header("Processed Video")
                    st.video(BytesIO(response.content))
                
                # Get CSV results
                csv_response = requests.get(f"{BACKEND_URL}/get-results/", timeout=30)
                
                if csv_response.status_code == 200:
                    status_text.text("Displaying analysis results...")
                    progress_bar.progress(90)
                    
                    # Show sample data
                    st.header("Analysis Results")
                    sample_response = requests.get(f"{BACKEND_URL}/sample-results/")
                    if sample_response.status_code == 200:
                        st.json(sample_response.json())
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Full Analysis (CSV)",
                        data=csv_response.content,
                        file_name="violence_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.success("✅ Processing completed successfully!")
                    progress_bar.progress(100)
                else:
                    st.error(f"Failed to get results: {csv_response.text}")
            else:
                st.error(f"Processing failed: {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("🔌 Could not connect to the backend service. Please ensure it's running.")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The video might be too long or the server is busy.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            progress_bar.empty()
            status_text.empty()

# Docker instructions
with st.expander("ℹ️ How to run this system with Docker"):
    st.markdown("""
    **docker-compose.yml setup:**
    ```yaml
    version: '3.8'
    
    services:
      backend:
        build: ./backend
        ports:
          - "8000:8000"
        volumes:
          - ./extracted_feature_data:/extracted_feature_data
        environment:
          - CUDA_VISIBLE_DEVICES=0  # Enable GPU if available
    
      frontend:
        build: ./frontend
        ports:
          - "8501:8501"
        depends_on:
          - backend
        environment:
          - BACKEND_URL=http://backend:8000
    ```
    
    **To run:**
    ```bash
    docker-compose up --build
    ```
    
    Access the frontend at http://localhost:8501
    """)