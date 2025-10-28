import React, { useState, useEffect } from "react";
import { checkHealth, analyzeVideo } from "./api";
import "./App.css";

const App: React.FC = () => {
  const [backendUp, setBackendUp] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"extract" | "predict">("predict");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    (async () => {
      const healthy = await checkHealth();
      setBackendUp(healthy);
    })();
  }, []);

  const handleUpload = async () => {
    if (!videoFile) return alert("Please select a video file!");
    setLoading(true);
    setResult(null);

    try {
      const response = await analyzeVideo(mode, videoFile);
      setResult(response);
    } catch (error) {
      alert("Failed to process video. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div id="root">
      <div className="main-card">
        <div className="header-title">🎥 Violence Detection System</div>

        <div className={`status ${backendUp ? "status-ok" : "status-fail"}`}>
          {backendUp ? "✅ Backend Connected" : "❌ Backend Not Reachable"}
        </div>

        <div className="form-group">
          <input
            type="file"
            accept="video/*"
            onChange={(e) =>
              e.target.files && setVideoFile(e.target.files[0])
            }
          />
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as "extract" | "predict")}
          >
            <option value="extract">Extract Features Only</option>
            <option value="predict">Predict Violence</option>
          </select>
          <button
            onClick={handleUpload}
            disabled={loading || !backendUp}
          >
            {loading ? "Processing..." : "Upload & Analyze"}
          </button>
        </div>

        {result && (
          <div className="result-card">
            <div className="result-title">📊 Result</div>
            <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
