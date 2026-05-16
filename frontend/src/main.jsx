import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";

const Dashboard = () => (
  <main style={{ fontFamily: "sans-serif", padding: "2rem", background: "#0b1220", color: "#e2e8f0", minHeight: "100vh" }}>
    <h1>Train-LM</h1>
    <p>Offline-first local fine-tuning and inference platform scaffold.</p>
    <ul>
      <li>Dataset upload + validation pipeline</li>
      <li>LoRA / QLoRA training orchestration</li>
      <li>Model registry + GGUF export flow</li>
      <li>Local inference + Telegram integration</li>
    </ul>
    <nav>
      <Link to="/datasets" style={{ color: "#93c5fd" }}>Datasets</Link>
    </nav>
  </main>
);

const Datasets = () => (
  <main style={{ fontFamily: "sans-serif", padding: "2rem" }}>
    <h2>Datasets</h2>
    <p>Dataset management UI module placeholder.</p>
    <Link to="/">Back</Link>
  </main>
);

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/datasets" element={<Datasets />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
