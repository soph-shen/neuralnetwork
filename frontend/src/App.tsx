import React, { useState } from "react";
import Canvas from "./Canvas";
import "./App.css";

const GRID_SIZE = 64;
const PIXEL_SIZE = 10;

function App() {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (pixels: number[][]) => {
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);

      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixels }),
      });

      // If FastAPI returned an error, try to read the message
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try {
          const data = await res.json();
          if (data?.detail) msg = data.detail;
        } catch {
          const text = await res.text();
          if (text) msg = text;
        }
        throw new Error(msg);
      }

      const data = await res.json();
      setPrediction(data.predicted_digit);
    } catch (e: any) {
      console.error("Prediction error:", e);
      setError(e?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#f5f5f7",
      }}
    >
      <div
        style={{
          padding: 24,
          borderRadius: 16,
          background: "#ffffff",
          boxShadow: "0 10px 30px rgba(0,0,0,0.08)",
          textAlign: "center",
        }}
      >
        <h1 style={{ marginBottom: 16 }}>Draw a digit</h1>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 12,
          }}
        >
          <Canvas
            gridSize={GRID_SIZE}
            pixelSize={PIXEL_SIZE}
            handlePredict={handlePredict}
            onClear={handleClear}  
          />

          <div style={{ marginTop: 8 }}>
            {loading && <p>Predicting…</p>}
            {error && <p style={{ color: "red" }}>{error}</p>}
            {prediction !== null && (
              <p style={{ fontSize: 20 }}>
                Predicted digit: <strong>{prediction}</strong>
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;