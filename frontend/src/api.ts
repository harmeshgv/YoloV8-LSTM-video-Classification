// src/api.ts
import axios from "axios";

const BASE_URL = "https://harmesh95-vio.hf.space"; // FastAPI backend

// Create axios instance with better error handling
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

// ✅ Health check
export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await api.get("/");
    console.log("Health check response:", response.data);
    return response.data.status === "ok";
  } catch (error: any) {
    console.error("❌ Backend health check failed:", error.message);
    return false;
  }
};

// ✅ Upload video for either extraction or prediction
export const analyzeVideo = async (mode: "extract" | "predict", file: File) => {
  const formData = new FormData();
  formData.append("mode", mode);
  formData.append("file", file);

  try {
    console.log("Uploading video...", file.name, "Mode:", mode);
    const response = await api.post("/analyze", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      timeout: 120000, // 2 minute timeout for video processing
    });
    console.log("Upload successful:", response.data);
    return response.data;
  } catch (error: any) {
    console.error("Upload failed:", error);
    if (error.response) {
      throw new Error(error.response.data.detail || `Server error: ${error.response.status}`);
    } else if (error.request) {
      throw new Error("Cannot connect to server. Make sure the backend is running on port 8000.");
    } else {
      throw new Error("Request failed: " + error.message);
    }
  }
};