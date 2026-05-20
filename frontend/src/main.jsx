import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

const apiFetch = async (path, { body, headers, ...options } = {}) => {
  const token = localStorage.getItem("trainlm_token");
  const init = {
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
    ...options,
  };
  if (body && !(body instanceof FormData)) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  } else if (body) {
    init.body = body;
  }
  const response = await fetch(`${API_BASE}${path}`, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data?.detail?.message || data?.detail || response.statusText;
    throw new Error(detail);
  }
  return data;
};

const PageShell = ({ children }) => (
  <main style={{ fontFamily: "Inter, sans-serif", padding: "2rem", minHeight: "100vh", background: "#0b1220", color: "#e2e8f0" }}>
    <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
      <h1 style={{ margin: 0 }}>Train-LM</h1>
      <nav style={{ display: "flex", gap: "1rem" }}>
        <Link to="/" style={{ color: "#93c5fd" }}>Dashboard</Link>
        <Link to="/datasets" style={{ color: "#93c5fd" }}>Datasets</Link>
        <Link to="/training" style={{ color: "#93c5fd" }}>Training</Link>
        <Link to="/models" style={{ color: "#93c5fd" }}>Models</Link>
        <Link to="/chat" style={{ color: "#93c5fd" }}>Chat</Link>
        <Link to="/auth" style={{ color: "#93c5fd" }}>Auth</Link>
      </nav>
    </header>
    {children}
  </main>
);

const Dashboard = () => (
  <PageShell>
    <section style={{ maxWidth: 720 }}>
      <h2>Offline-first LLM Studio</h2>
      <p>
        Upload datasets, run LoRA/QLoRA fine-tuning, track models, and chat with local adapters.
      </p>
      <ul>
        <li>Dataset validation + storage</li>
        <li>Fine-tuning jobs with progress and logs</li>
        <li>Model registry and inference endpoint</li>
      </ul>
    </section>
  </PageShell>
);

const Auth = () => {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [message, setMessage] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setMessage("");
    try {
      if (mode === "register") {
        await apiFetch("/auth/register", { method: "POST", body: form });
      }
      const data = await apiFetch("/auth/login", {
        method: "POST",
        body: { username: form.username, password: form.password },
      });
      localStorage.setItem("trainlm_token", data.data.access_token);
      setMessage("Authenticated.");
    } catch (err) {
      setMessage(err.message);
    }
  };

  return (
    <PageShell>
      <section style={{ maxWidth: 480 }}>
        <h2>{mode === "login" ? "Login" : "Register"}</h2>
        <form onSubmit={submit} style={{ display: "grid", gap: "1rem" }}>
          <input
            placeholder="Username"
            value={form.username}
            onChange={(e) => setForm({ ...form, username: e.target.value })}
          />
          {mode === "register" && (
            <input
              placeholder="Email"
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
            />
          )}
          <input
            placeholder="Password"
            type="password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
          />
          <button type="submit">{mode === "login" ? "Login" : "Register"}</button>
        </form>
        <button onClick={() => setMode(mode === "login" ? "register" : "login")} style={{ marginTop: "1rem" }}>
          Switch to {mode === "login" ? "Register" : "Login"}
        </button>
        {message && <p style={{ marginTop: "1rem" }}>{message}</p>}
      </section>
    </PageShell>
  );
};

const Datasets = () => {
  const [datasets, setDatasets] = useState([]);
  const [message, setMessage] = useState("");
  const [file, setFile] = useState(null);

  const load = async () => {
    try {
      const data = await apiFetch("/datasets");
      setDatasets(data.data.items);
    } catch (err) {
      setMessage(err.message);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const upload = async (e) => {
    e.preventDefault();
    if (!file) {
      setMessage("Select a dataset file.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    try {
      await apiFetch("/datasets/upload", { method: "POST", body: formData });
      setFile(null);
      setMessage("Uploaded.");
      load();
    } catch (err) {
      setMessage(err.message);
    }
  };

  return (
    <PageShell>
      <section style={{ maxWidth: 800 }}>
        <h2>Datasets</h2>
        <form onSubmit={upload} style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
          <input type="file" onChange={(e) => setFile(e.target.files[0])} />
          <button type="submit">Upload</button>
        </form>
        {message && <p>{message}</p>}
        <ul>
          {datasets.map((ds) => (
            <li key={ds.id}>
              {ds.name} — {ds.row_count} rows ({Math.round(ds.size_bytes / 1024)} KB)
            </li>
          ))}
        </ul>
      </section>
    </PageShell>
  );
};

const Training = () => {
  const [jobs, setJobs] = useState([]);
  const [form, setForm] = useState({
    run_name: "",
    base_model: "",
    dataset_id: "",
    method: "lora",
  });
  const [message, setMessage] = useState("");

  const load = async () => {
    try {
      const data = await apiFetch("/training/jobs");
      setJobs(data.data.items);
    } catch (err) {
      setMessage(err.message);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const submit = async (e) => {
    e.preventDefault();
    setMessage("");
    try {
      await apiFetch("/training/jobs", {
        method: "POST",
        body: {
          ...form,
          dataset_id: form.dataset_id || null,
        },
      });
      setForm({ ...form, run_name: "" });
      load();
    } catch (err) {
      setMessage(err.message);
    }
  };

  return (
    <PageShell>
      <section style={{ maxWidth: 900 }}>
        <h2>Training Jobs</h2>
        <form onSubmit={submit} style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
          <input
            placeholder="Run name"
            value={form.run_name}
            onChange={(e) => setForm({ ...form, run_name: e.target.value })}
          />
          <input
            placeholder="Base model (local path or HF id)"
            value={form.base_model}
            onChange={(e) => setForm({ ...form, base_model: e.target.value })}
          />
          <input
            placeholder="Dataset ID (optional)"
            value={form.dataset_id}
            onChange={(e) => setForm({ ...form, dataset_id: e.target.value })}
          />
          <select value={form.method} onChange={(e) => setForm({ ...form, method: e.target.value })}>
            <option value="lora">LoRA</option>
            <option value="qlora">QLoRA</option>
          </select>
          <button type="submit">Queue Training Job</button>
        </form>
        {message && <p>{message}</p>}
        <ul>
          {jobs.map((job) => (
            <li key={job.id}>
              {job.run_name} — {job.status} ({Math.round(job.progress * 100)}%)
            </li>
          ))}
        </ul>
      </section>
    </PageShell>
  );
};

const Models = () => {
  const [models, setModels] = useState([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    apiFetch("/models")
      .then((data) => setModels(data.data.items))
      .catch((err) => setMessage(err.message));
  }, []);

  return (
    <PageShell>
      <section style={{ maxWidth: 800 }}>
        <h2>Models</h2>
        {message && <p>{message}</p>}
        <ul>
          {models.map((model) => (
            <li key={model.id}>
              {model.name} — {model.base_model}
            </li>
          ))}
        </ul>
      </section>
    </PageShell>
  );
};

const Chat = () => {
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState("");

  const loadModels = useCallback(async () => {
    try {
      const data = await apiFetch("/inference/models");
      setModels(data.data.items);
      if (data.data.items.length && !modelId) {
        setModelId(data.data.items[0].model_id);
      }
    } catch (err) {
      setError(err.message);
    }
  }, [modelId]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const send = async (e) => {
    e.preventDefault();
    setError("");
    const currentPrompt = prompt.trim();
    if (!currentPrompt || !modelId) return;
    setMessages((prev) => [...prev, { role: "user", text: currentPrompt }]);
    setPrompt("");
    try {
      const data = await apiFetch("/inference/chat", {
        method: "POST",
        body: { model_id: modelId, prompt: currentPrompt },
      });
      setMessages((prev) => [...prev, { role: "assistant", text: data.data.completion }]);
    } catch (err) {
      setError(err.message);
    }
  };

  const modelOptions = useMemo(
    () =>
      models.map((m) => (
        <option key={m.model_id} value={m.model_id}>
          {m.model_name} (v{m.version})
        </option>
      )),
    [models]
  );

  return (
    <PageShell>
      <section style={{ maxWidth: 900 }}>
        <h2>Chat</h2>
        <label>
          Model:
          <select value={modelId} onChange={(e) => setModelId(e.target.value)} style={{ marginLeft: "0.5rem" }}>
            {modelOptions}
          </select>
        </label>
        <div style={{ marginTop: "1rem", background: "#0f172a", padding: "1rem", borderRadius: "8px", minHeight: "200px" }}>
          {messages.map((msg, idx) => (
            <p key={idx} style={{ color: msg.role === "assistant" ? "#a7f3d0" : "#f8fafc" }}>
              <strong>{msg.role}:</strong> {msg.text}
            </p>
          ))}
        </div>
        <form onSubmit={send} style={{ marginTop: "1rem", display: "flex", gap: "0.5rem" }}>
          <input
            style={{ flex: 1 }}
            placeholder="Ask something..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <button type="submit">Send</button>
        </form>
        {error && <p>{error}</p>}
      </section>
    </PageShell>
  );
};

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/training" element={<Training />} />
        <Route path="/models" element={<Models />} />
        <Route path="/chat" element={<Chat />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
