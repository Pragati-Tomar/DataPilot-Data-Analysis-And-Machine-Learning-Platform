import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, ArrowRight, BarChart3, Bot, BrainCircuit, Check, Database, Eraser,
  FileText, Gauge, GitBranch, GitCompareArrows, Menu, Play, Plus, RefreshCw, Search,
  ShieldCheck, Sparkles, Target, Trash2, Upload, UserRound, X
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";
import "./styles.css";

const navGroups = [
  { label: "WORKSPACE", items: [["Overview", Gauge], ["Datasets", Database], ["Data Cleaning", Eraser], ["EDA", BarChart3], ["Feature Engineering", GitBranch]] },
  { label: "", items: [["ML Studio", BrainCircuit], ["Model Comparison", GitCompareArrows], ["Explainability", Target]] },
  { label: "", items: [["AI Assistant", Bot], ["Reports", FileText]] }
];

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";
const AUTH_STORAGE_KEY = "datapilot.session";
let activeAccessToken = sessionStorage.getItem(AUTH_STORAGE_KEY) || "";

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (activeAccessToken) headers.set("Authorization", `Bearer ${activeAccessToken}`);
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Backend request failed");
  return data;
}

function fileToForm(file) {
  const form = new FormData();
  form.append("file", file);
  return form;
}

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const register = mode === "register";

  const submit = async event => {
    event.preventDefault();
    setError("");
    setPending(true);
    try {
      const result = await api(register ? "/auth/register" : "/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password })
      });
      onAuthenticated(result);
    } catch (requestError) {
      setError(requestError.message || "Could not sign in. Try again.");
    } finally {
      setPending(false);
    }
  };

  return <main className="auth-screen">
    <section className="auth-card">
      <div className="auth-brand"><svg className="brand-logo" viewBox="0 0 40 40" aria-hidden="true"><rect x="1" y="1" width="38" height="38" rx="11" fill="#eff6ff"/><rect x="7" y="25" width="6" height="9" rx="2.5" fill="#78a6ff"/><rect x="16" y="19" width="6" height="15" rx="2.5" fill="#2563eb"/><rect x="25" y="14" width="6" height="20" rx="2.5" fill="#4f7dff"/><path d="m20 8 14-4-6.7 12.6-2.7-5.1L20 8Z" fill="#1d4ed8"/></svg><div><strong>DATAPILOT</strong><small>DATA TO DECISIONS</small></div></div>
      <p className="auth-eyebrow">YOUR DATA WORKSPACE</p>
      <h1>{register ? "Create your account" : "Welcome back"}</h1>
      <p className="auth-intro">{register ? "Set up an account to keep your datasets in your workspace." : "Sign in to continue to your datasets and analysis."}</p>
      <form onSubmit={submit} className="auth-form">
        {register && <label>Full name<input autoComplete="name" value={name} onChange={event => setName(event.target.value)} required minLength={2} maxLength={120} placeholder="Your name"/></label>}
        <label>Email address<input type="email" autoComplete="email" value={email} onChange={event => setEmail(event.target.value)} required maxLength={254} placeholder="you@example.com"/></label>
        <label>Password<input type="password" autoComplete={register ? "new-password" : "current-password"} value={password} onChange={event => setPassword(event.target.value)} required minLength={register ? 8 : undefined} maxLength={128} placeholder={register ? "At least 8 characters" : "Your password"}/></label>
        {error && <div className="auth-error" role="alert">{error}</div>}
        <button className="auth-submit" type="submit" disabled={pending}>{pending ? "Please wait…" : register ? "Create account" : "Sign in"}</button>
      </form>
      <p className="auth-switch">{register ? "Already have an account?" : "New to DataPilot?"} <button type="button" onClick={() => { setMode(register ? "login" : "register"); setError(""); }}> {register ? "Sign in" : "Create an account"}</button></p>
      <div className="auth-security"><span/> Your workspace is protected by account sign-in</div>
    </section>
  </main>;
}

function App() {
  const [authToken, setAuthToken] = useState(() => sessionStorage.getItem(AUTH_STORAGE_KEY) || "");
  const [authUser, setAuthUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(() => !sessionStorage.getItem(AUTH_STORAGE_KEY));
  const [page, setPage] = useState("Overview");
  const [mobileNav, setMobileNav] = useState(false);
  const [workspaceOpen, setWorkspaceOpen] = useState(true);
  const [assistantOpen, setAssistantOpen] = useState(true);
  const [toast, setToast] = useState("");
  const [backendConnected, setBackendConnected] = useState(false);
  const [llmConfigured, setLlmConfigured] = useState(null);
  const [datasetCount, setDatasetCount] = useState(0);
  const [dataset, setDataset] = useState({ id: "", name: "No dataset loaded", rows: 0, columns: 0, data: [], columnsInfo: [], missing: 0, duplicates: 0 });
  const [features, setFeatures] = useState([]);
  const [edaComplete, setEdaComplete] = useState(false);
  const [training, setTraining] = useState(false);
  const [copilotPending, setCopilotPending] = useState(false);
  const copilotPendingRef = useRef(false);
  const [trained, setTrained] = useState(false);
  const [explanationComplete, setExplanationComplete] = useState(false);
  const [copilotComplete, setCopilotComplete] = useState(false);
  const [reportComplete, setReportComplete] = useState(false);
  const [trainingProblem, setTrainingProblem] = useState("");
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedCustomer, setSelectedCustomer] = useState("");
  const [chat, setChat] = useState([{ role: "ai", text: "Ask a question about the current dataset or model results." }]);
  const [question, setQuestion] = useState("");

  const notify = (message) => {
    setToast(message);
    window.clearTimeout(window.__dpToast);
    window.__dpToast = window.setTimeout(() => setToast(""), 2500);
  };
  const go = (target) => { setPage(target); setMobileNav(false); window.scrollTo({ top: 0, behavior: "smooth" }); };

  const loadDataset = async (data) => {
    setDataset(data);
    const firstId = data.data?.[0]?.customerID || data.data?.[0]?.customerId || data.data?.[0]?.id || "";
    setSelectedCustomer(firstId);
    setFeatures(data.features || []);
  };

  const invalidateDownstream = () => {
    setEdaComplete(false); setTrained(false); setTrainingProblem(""); setModels([]); setSelectedModel("");
    setExplanationComplete(false); setCopilotComplete(false); setReportComplete(false);
    setChat([{ role: "ai", text: "Ask a question about the current dataset or model results." }]);
  };

  const acceptSession = session => {
    activeAccessToken = session.access_token;
    sessionStorage.setItem(AUTH_STORAGE_KEY, session.access_token);
    setAuthToken(session.access_token);
    setAuthUser(session.user);
    setAuthChecked(false);
  };

  const signOut = () => {
    activeAccessToken = "";
    sessionStorage.removeItem(AUTH_STORAGE_KEY);
    setAuthToken("");
    setAuthUser(null);
    setAuthChecked(true);
    setDatasetCount(0);
    setDataset({ id: "", name: "No dataset loaded", rows: 0, columns: 0, data: [], columnsInfo: [], missing: 0, duplicates: 0 });
    invalidateDownstream();
  };

  useEffect(() => {
    if (!authToken) { setAuthUser(null); setAuthChecked(true); return; }
    activeAccessToken = authToken;
    let active = true;
    setAuthChecked(false);
    (async () => {
      try {
        const user = await api("/auth/me");
        if (!active) return;
        setAuthUser(user);
      } catch {
        if (active) signOut();
        return;
      }
      try {
        const status = await api("/health");
        if (!active) return;
        setBackendConnected(true);
        setLlmConfigured(Boolean(status.llm_configured));
        const list = await api("/datasets");
        if (!active) return;
        setDatasetCount(list.datasets?.length || 0);
        if (list.datasets?.length) await loadDataset(await api(`/datasets/${list.datasets[0].id}`));
      } catch {
        if (active) { setBackendConnected(false); notify("Could not load your workspace. Check the backend connection."); }
      } finally {
        if (active) setAuthChecked(true);
      }
    })();
    return () => { active = false; };
  }, [authToken]);

  const runTraining = async (config) => {
    if (training || !dataset.id) return;
    setModels([]); setTrained(false); setSelectedModel(""); setExplanationComplete(false); setCopilotComplete(false); setReportComplete(false);
    setTraining(true);
    notify("Training pipeline started.");
    try {
      const result = await api(`/datasets/${dataset.id}/train`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config) });
      const rows = (result.models || []).filter(item => item.score !== null).map(item => item.metrics);
      setModels(rows);
      setTrainingProblem(result.problem_type || config.problem_type);
      setSelectedModel(result.selected || rows[0]?.[0] || "");
      setTrained(rows.length > 0);
      setExplanationComplete(false); setCopilotComplete(false); setReportComplete(false);
      const excluded = result.excluded_features || [];
      notify(`Trained ${rows.length} model(s) on ${result.train_rows} rows; evaluated on ${result.test_rows}.${excluded.length ? ` Excluded target-derived features: ${excluded.join(", ")}.` : ""}`);
      go("Model Comparison");
    } catch (error) {
      notify(error.message);
    } finally {
      setTraining(false);
    }
  };
  const chooseModel = (name) => {
    if (name !== selectedModel) { setExplanationComplete(false); setCopilotComplete(false); }
    setSelectedModel(name);
  };
  const sendQuestion = async (text = question) => {
    const q = String(text).trim(); if (!q || !dataset.id || copilotPendingRef.current) return;
    setChat(c => [...c, { role: "user", text: q }]); setQuestion("");
    if (llmConfigured === false) {
      setChat(c => [...c, { role: "ai", text: "AI Assistant needs the configured Ollama service and model. Start Ollama, confirm the model in backend/.env, then refresh this page." }]);
      return;
    }
    copilotPendingRef.current = true;
    setCopilotPending(true);
    try {
      const row = Math.max(0, dataset.data.findIndex((item, index) => String(item.customerID || item.customerId || item.id || index) === String(selectedCustomer)));
      const result = await api("/copilot", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q, dataset_id: dataset.id, model: selectedModel, row }) });
      setChat(c => [...c, { role: "ai", text: result.answer, sources: result.sources || [] }]);
      setCopilotComplete(true);
    } catch (error) { setChat(c => [...c, { role: "ai", text: `AI Assistant could not answer: ${error.message}` }]); }
    finally { copilotPendingRef.current = false; setCopilotPending(false); }
  };

  const handleFile = async (file) => {
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["csv","json","xlsx","parquet"].includes(ext)) { notify("Use CSV, JSON, XLSX or Parquet."); return; }
    try {
      const data = await api("/datasets/upload", { method: "POST", body: fileToForm(file) });
      invalidateDownstream(); await loadDataset(data); setDatasetCount(count => count + 1); notify(`${file.name} loaded successfully.`); go("Datasets");
    } catch (error) { notify(error.message); }
  };

  const applyCleaning = async (operation = "all") => {
    if (!dataset.id) return false;
    try { const data = await api(`/datasets/${dataset.id}/clean`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ operation, missing: true, duplicates: true }) }); invalidateDownstream(); await loadDataset(data); notify(`${operation} completed successfully.`); return true; }
    catch (error) { notify(error.message); return false; }
  };

  const undoCleaning = async () => {
    if (!dataset.id) return;
    try { const data = await api(`/datasets/${dataset.id}/undo`, { method: "POST" }); invalidateDownstream(); await loadDataset(data); notify("Last dataset operation was undone."); } catch (error) { notify(error.message); }
  };

  const addFeature = async (name, expression) => {
    if (!dataset.id) return;
    try { const data = await api(`/datasets/${dataset.id}/features`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, expression }) }); invalidateDownstream(); await loadDataset(data); notify(`${name} created successfully.`); }
    catch (error) { notify(error.message); }
  };

  const removeFeature = async (name) => {
    if (!dataset.id) return;
    try { const data = await api(`/datasets/${dataset.id}/features/${encodeURIComponent(name)}`, { method: "DELETE" }); invalidateDownstream(); await loadDataset(data); notify(`${name} removed successfully.`); }
    catch (error) { notify(error.message); }
  };

  const validateDataset = async () => { if (!dataset.id) return; try { const r = await api(`/datasets/${dataset.id}/schema`); notify(r.valid ? `Schema valid: ${r.columns.length} columns checked.` : "Schema validation failed."); } catch (e) { notify(e.message); } };

  if (!authChecked) return <main className="auth-screen"><div className="auth-loading"><span className="auth-spinner"/>Loading your workspace…</div></main>;
  if (!authUser) return <AuthScreen onAuthenticated={acceptSession}/>;

  return <div className="app">
    <aside className={`sidebar ${mobileNav ? "open" : ""} ${workspaceOpen ? "" : "closed"}`}>
      <div className="brand"><svg className="brand-logo" viewBox="0 0 40 40" aria-hidden="true"><defs><linearGradient id="dp-mark-blue" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stopColor="#2563eb"/><stop offset="1" stopColor="#4f7dff"/></linearGradient></defs><rect x="1" y="1" width="38" height="38" rx="11" fill="#eff6ff"/><rect x="7" y="25" width="6" height="9" rx="2.5" fill="#78a6ff"/><rect x="16" y="19" width="6" height="15" rx="2.5" fill="url(#dp-mark-blue)"/><rect x="25" y="14" width="6" height="20" rx="2.5" fill="url(#dp-mark-blue)"/><path d="m20 8 14-4-6.7 12.6-2.7-5.1L20 8Z" fill="#1d4ed8" stroke="#eff6ff" strokeWidth="1.1" strokeLinejoin="round"/><path d="m20 8 7.1 3.5" fill="none" stroke="#fff" strokeWidth="1.1" strokeLinecap="round"/></svg><div><strong>DATAPILOT</strong><small>DATA TO DECISIONS</small></div></div>
      <nav>{navGroups.map((group, index) => <div className="nav-group" key={group.label || `nav-group-${index}`}>{group.label && <div className="side-label">{group.label}</div>}{group.items.map(([label, Icon]) => <button key={label} className={page === label ? "nav-item active" : "nav-item"} onClick={() => go(label)}>{Icon && <Icon size={16}/>}<span>{label}</span></button>)}</div>)}</nav>
      <div className="sidebar-bottom"><div className="sidebar-dataset"><span>ACTIVE DATASET</span><b title={dataset.name}>{dataset.id ? dataset.name : "No dataset loaded"}</b><small>{dataset.id ? `${dataset.rows.toLocaleString()} rows · ${dataset.columns} columns` : "Upload data to begin"}</small></div></div>
    </aside>
    <main className={`main ${workspaceOpen ? "" : "workspace-collapsed"}`}>
      <header className="topbar"><button className="icon-btn workspace-toggle" onClick={() => { if (window.matchMedia("(max-width: 800px)").matches) setMobileNav(v => !v); else setWorkspaceOpen(v => !v); }} aria-label={(window.matchMedia("(max-width: 800px)").matches ? mobileNav : workspaceOpen) ? "Close Workspace navigation" : "Open Workspace navigation"} aria-expanded={window.matchMedia("(max-width: 800px)").matches ? mobileNav : workspaceOpen}><Menu size={19}/></button><div className="breadcrumbs"><span>DataPilot</span><span className="crumb-separator">/</span><b>{page}</b></div><div className="top-actions"><div className="top-dataset"><span>DATASET</span><b>{dataset.id ? dataset.name : "No dataset loaded"}</b><small>{dataset.id ? `${dataset.rows.toLocaleString()} rows · ${dataset.columns} columns` : "Workspace ready"}</small></div><div className="top-profile"><span><UserRound size={16}/></span><b>{authUser.name || authUser.email}</b></div><button className="logout-button" onClick={signOut}>Sign out</button></div></header>
      <div className="content">
        {page === "Overview" && <Overview setPage={go} dataset={dataset} trained={trained} models={models} modelProblem={trainingProblem} edaComplete={edaComplete} explanationComplete={explanationComplete} copilotComplete={copilotComplete} reportComplete={reportComplete} training={training} datasetCount={datasetCount} chat={chat} question={question} setQuestion={setQuestion} sendQuestion={sendQuestion} llmConfigured={llmConfigured} copilotPending={copilotPending} selectedModel={selectedModel} assistantOpen={assistantOpen} setAssistantOpen={setAssistantOpen}/>} 
        {page === "Datasets" && <Datasets dataset={dataset} onFile={handleFile} onValidate={validateDataset}/>} 
        {page === "Data Cleaning" && <Cleaning notify={notify} applyCleaning={applyCleaning} undoCleaning={undoCleaning} dataset={dataset}/>} 
        {page === "EDA" && <EDA dataset={dataset} notify={notify} onComplete={() => setEdaComplete(true)}/>} 
        {page === "Feature Engineering" && <Features features={features} setFeatures={setFeatures} notify={notify} addFeature={addFeature} removeFeature={removeFeature} dataset={dataset}/>} 
        {page === "ML Studio" && <MLStudio training={training} runTraining={runTraining} notify={notify} dataset={dataset}/>} 
        {page === "Model Comparison" && <ModelComparison models={models} selectedModel={selectedModel} setSelectedModel={chooseModel} setPage={go} problem={trainingProblem}/>} 
        {page === "Explainability" && <Explainability selectedCustomer={selectedCustomer} setSelectedCustomer={setSelectedCustomer} dataset={dataset} notify={notify} selectedModel={selectedModel} onComplete={() => setExplanationComplete(true)}/>} 
        {page === "AI Assistant" && <Copilot chat={chat} question={question} setQuestion={setQuestion} sendQuestion={sendQuestion} dataset={dataset} llmConfigured={llmConfigured} selectedModel={selectedModel} pending={copilotPending}/>} 
        {page === "Reports" && <Reports notify={notify} dataset={dataset} trained={trained} onComplete={() => setReportComplete(true)}/>} 
      </div>
    </main>
    {toast && <div className="toast"><Check size={16}/>{toast}</div>}
  </div>;
}
function PageHeader({ title, subtitle, actions }) { return <div className="page-head"><div><h1>{title}</h1>{subtitle && <p>{subtitle}</p>}</div><div className="head-actions">{actions}</div></div>; }
function Button({ children, primary = false, onClick, icon, disabled = false }) { return <button disabled={disabled} className={primary ? "btn primary" : "btn"} onClick={onClick}>{icon}{children}</button>; }
function Kpi({ label, value, note }) { return <div className="card kpi"><div className="kpi-top"><span>{label}</span></div><strong>{value}</strong>{note && <small>{note}</small>}</div>; }

function Overview({ setPage, dataset, trained, models, modelProblem, edaComplete, explanationComplete, copilotComplete, reportComplete, training, datasetCount, chat, question, setQuestion, sendQuestion, llmConfigured, copilotPending, selectedModel, assistantOpen, setAssistantOpen }) {
  const [eda, setEda] = useState(null);
  const [edaLoading, setEdaLoading] = useState(false);
  const messagesRef = useRef(null);
  useEffect(() => {
    if (!dataset.id) { setEda(null); return; }
    let active = true;
    setEdaLoading(true);
    api(`/datasets/${dataset.id}/eda`).then(result => { if (active) setEda(result); }).catch(() => { if (active) setEda(null); }).finally(() => { if (active) setEdaLoading(false); });
    return () => { active = false; };
  }, [dataset.id, dataset.rows, dataset.columns, dataset.missing, dataset.duplicates]);
  useEffect(() => { if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight; }, [chat, copilotPending]);
  const best = trained && models.length ? models.reduce((a,b) => Number(String(a[1]).replace("%","")) >= Number(String(b[1]).replace("%","")) ? a : b) : null;
  const steps = [
    { name: "Dataset", page: "Datasets", done: Boolean(dataset.id) },
    { name: "Cleaning", page: "Data Cleaning", done: Boolean(dataset.id) && (Boolean(dataset.cleaningOperations?.length) || (dataset.missing === 0 && dataset.duplicates === 0)) },
    { name: "EDA", page: "EDA", done: edaComplete },
    { name: "Features", page: "Feature Engineering", done: Boolean(dataset.features?.length) || trained },
    { name: "ML", page: "ML Studio", done: trained, running: training },
    { name: "Explainability", page: "Explainability", done: explanationComplete },
    { name: "AI Assistant", page: "AI Assistant", done: copilotComplete },
    { name: "Report", page: "Reports", done: reportComplete }
  ];
  const nextIndex = steps.findIndex(step => !step.done);
  const completedCount = steps.filter(step => step.done).length;
  const pipeline = steps.map((step, index) => ({ ...step, status: step.done ? "done" : index === nextIndex ? "run" : "pending" }));
  const nextStep = nextIndex >= 0 ? steps[nextIndex] : null;
  const cleanName = dataset.name.replace(/\.[^.]+$/, "").replace(/\s*\(\d+\)$/, "").replace(/[_-]+/g, " ").trim();
  const numericCount = eda?.numeric_columns ?? dataset.columnsInfo.filter(c => /number|integer|float/.test(String(c[1]))).length;
  const categoricalCount = eda?.categorical_columns ?? Math.max(0, dataset.columns - numericCount);
  const totalCells = (eda?.rows ?? dataset.rows) * (eda?.columns ?? dataset.columns);
  const missingPct = eda?.missing_percentage ?? (totalCells ? (dataset.missing ?? 0) / totalCells * 100 : 0);
  const duplicatePct = dataset.rows ? ((eda?.duplicates ?? dataset.duplicates ?? 0) / dataset.rows) * 100 : 0;
  const missingByColumn = [...(eda?.missing_by_column || [])].sort((a,b) => b.missing - a.missing).slice(0,7);
  const distribution = eda?.numeric_distributions?.[0];
  const number = value => Number(value || 0).toLocaleString();
  const recent = (eda?.insights || []).slice(0,3);
  return <div className={`overview-layout ${assistantOpen ? "" : "assistant-closed"}`}><section className="overview-main">
    <div className="overview-welcome"><div><div className="overview-eyebrow">DATA TO DECISIONS</div><h1>{dataset.id ? "Welcome!" : "Welcome to DataPilot!"}</h1><p>{dataset.id ? `Explore ${cleanName || dataset.name}, track analysis progress, and review dataset insights.` : "Upload a dataset to begin your analysis."}</p></div><div className="overview-welcome-actions">{!assistantOpen && <Button icon={<Bot size={15}/>} onClick={() => setAssistantOpen(true)}>Show AI Assistant</Button>}<Button primary icon={<Upload size={15}/>} onClick={() => setPage("Datasets")}>{dataset.id ? "Add dataset" : "Upload dataset"}</Button></div></div>
    {!dataset.id ? <section className="card overview-empty"><Database size={24}/><h3>No dataset loaded</h3><p>Upload a CSV, Excel, JSON, or Parquet file to see its overview here.</p><Button primary onClick={() => setPage("Datasets")}>Choose dataset</Button></section> : <>
      <div className="overview-kpis">
        <button className="overview-kpi" onClick={() => setPage("Datasets")}><span className="overview-kpi-icon"><Database size={17}/></span><span className="overview-kpi-label">Datasets</span><strong>{number(datasetCount)}</strong><small>View datasets <ArrowRight size={12}/></small></button>
        <div className="overview-kpi"><span className="overview-kpi-icon"><BarChart3 size={17}/></span><span className="overview-kpi-label">Rows</span><strong>{number(eda?.rows ?? dataset.rows)}</strong><small>{cleanName || dataset.name}</small></div>
        <div className="overview-kpi"><span className="overview-kpi-icon"><GitBranch size={17}/></span><span className="overview-kpi-label">Columns</span><strong>{number(eda?.columns ?? dataset.columns)}</strong><small>{number(numericCount)} numeric · {number(categoricalCount)} categorical</small></div>
        <button className="overview-kpi" onClick={() => setPage("Model Comparison")}><span className="overview-kpi-icon"><BrainCircuit size={17}/></span><span className="overview-kpi-label">Models trained</span><strong>{number(models.length)}</strong><small>{best ? `${best[0]} · ${best[1]}` : "Train a model to compare"}</small></button>
      </div>
      <section className="card overview-pipeline"><div className="section-title"><div><h3>Analysis progress</h3><p>{nextStep ? `Next up: ${nextStep.name}` : "All stages complete"}</p></div><span className={`status ${completedCount === steps.length ? "success" : ""}`}>{completedCount}/{steps.length}</span></div><div className="pipeline-progress"><span style={{width:`${completedCount / steps.length * 100}%`}}/></div><div className="pipeline">{pipeline.map(({name,page:target,status,done,running},i)=><React.Fragment key={name}><button className={`pipeline-step ${status}`} onClick={() => setPage(target)}><div className="step-icon">{done ? <Check size={14}/> : running ? <RefreshCw size={14}/> : status === "run" ? <Activity size={14}/> : <span>{i + 1}</span>}</div><b>{name}</b><small>{done ? "Complete" : running ? "Running" : status === "run" ? "Next" : "Waiting"}</small></button>{i < pipeline.length - 1 && <ArrowRight className="pipeline-arrow" size={13}/>}</React.Fragment>)}</div></section>
      <section className="card overview-summary"><div className="section-title"><div><h3>Dataset overview</h3><p>{edaLoading ? "Refreshing dataset statistics…" : dataset.name}</p></div><Button onClick={() => setPage("Datasets")}>Inspect data</Button></div><div className="overview-summary-grid"><div className="overview-facts"><div><span>Rows</span><b>{number(eda?.rows ?? dataset.rows)}</b></div><div><span>Columns</span><b>{number(eda?.columns ?? dataset.columns)}</b></div><div><span>Missing values</span><b>{number(eda?.missing ?? dataset.missing)} <small>({missingPct.toFixed(2)}%)</small></b></div><div><span>Duplicate rows</span><b>{number(eda?.duplicates ?? dataset.duplicates)} <small>({duplicatePct.toFixed(2)}%)</small></b></div><div><span>File</span><b>{dataset.name.split(".").pop()?.toUpperCase() || "—"}</b></div></div><div className="overview-type-chart"><div className="overview-chart-title"><b>Column types</b><span><i className="type-dot numeric"/>Numeric <i className="type-dot categorical"/>Categorical</span></div><ResponsiveContainer width="100%" height={155}><BarChart data={[{name:"Numeric",count:numericCount},{name:"Categorical",count:categoricalCount}]} margin={{top:10,right:8,left:-24,bottom:0}}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="name" tick={{fontSize:10}}/><YAxis allowDecimals={false} tick={{fontSize:10}}/><Tooltip/><Bar dataKey="count" radius={[4,4,0,0]}><Cell fill="#3b82f6"/><Cell fill="#8b5cf6"/></Bar></BarChart></ResponsiveContainer></div></div></section>
      <div className="overview-charts"><section className="card overview-chart-card"><div className="section-title"><div><h3>Missing values</h3><p>Per-column missing cell count</p></div></div>{missingByColumn.length ? <ResponsiveContainer width="100%" height={178}><BarChart data={missingByColumn} margin={{top:12,right:7,left:-24,bottom:0}}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="name" tick={{fontSize:9}} interval={0}/><YAxis allowDecimals={false} tick={{fontSize:9}}/><Tooltip/><Bar dataKey="missing" fill="#3b82f6" radius={[3,3,0,0]}/></BarChart></ResponsiveContainer> : <div className="overview-chart-empty">{edaLoading ? "Loading dataset statistics…" : "No missing values detected."}</div>}</section>
        <section className="card overview-chart-card"><div className="section-title"><div><h3>{distribution ? `Distribution · ${distribution.name}` : "Numeric distribution"}</h3><p>{distribution ? "Observed values grouped into bins" : "Numeric columns with varying values"}</p></div></div>{distribution?.bins?.length ? <ResponsiveContainer width="100%" height={178}><BarChart data={distribution.bins} margin={{top:12,right:7,left:-24,bottom:0}}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="label" tick={{fontSize:8}} interval="preserveStartEnd"/><YAxis allowDecimals={false} tick={{fontSize:9}}/><Tooltip/><Bar dataKey="count" fill="#5596f5" radius={[2,2,0,0]}/></BarChart></ResponsiveContainer> : <div className="overview-chart-empty">{edaLoading ? "Loading dataset statistics…" : "No numeric distribution available."}</div>}</section></div>
      <section className="card overview-insights"><div className="section-title"><div><h3>Recent insights</h3><p>Computed from the active dataset</p></div><button className="overview-text-link" onClick={() => setPage("EDA")}>Open EDA <ArrowRight size={12}/></button></div>{recent.length ? recent.map((insight,i)=><div className="overview-insight" key={`${i}-${insight}`}><span className="insight-badge">{i === 0 ? <BarChart3 size={15}/> : <Sparkles size={15}/>}</span><p>{insight}</p><small>EDA</small></div>) : <div className="overview-chart-empty">{edaLoading ? "Analyzing dataset…" : "No insights available yet."}</div>}{best && <div className="overview-insight"><span className="insight-badge model"><BrainCircuit size={15}/></span><p><b>{best[0]}</b> leads the current model comparison with {best[1]}.</p><small>Modeling</small></div>}</section>
    </>}
  </section><aside className="card overview-copilot" aria-hidden={!assistantOpen}><div className="overview-copilot-head"><div><h2>AI Assistant</h2><small>{copilotPending ? "Analyzing dataset…" : llmConfigured === null ? "Checking connection" : llmConfigured ? "Ready to help" : "LLM connection required"}</small></div><button className="panel-close" onClick={() => setAssistantOpen(false)} aria-label="Close AI Assistant panel"><X size={16}/></button></div><div className="overview-copilot-context">{dataset.id ? `${dataset.name} · ${number(dataset.rows)} rows · ${number(dataset.columns)} columns` : "Upload a dataset to ask questions"}</div><div className="overview-chat-messages" ref={messagesRef} aria-live="polite">{chat.slice(-4).map((message,index)=><div className={`overview-chat-message ${message.role}`} key={`${chat.length}-${index}`}><span className="overview-chat-avatar">{message.role === "ai" ? <Sparkles size={14}/> : "You"}</span><div><p>{message.text}</p>{message.sources?.length > 0 && <small>Sources: {message.sources.slice(0,3).join(", ")}</small>}</div></div>)}{copilotPending && <div className="overview-chat-message ai" role="status"><span className="overview-chat-avatar"><Sparkles size={14}/></span><div className="overview-typing"><i/><i/><i/> Preparing an answer…</div></div>}</div><div className="overview-related"><b>Ask about</b><button onClick={() => setQuestion("Summarize this dataset")} disabled={!dataset.id || copilotPending}>Dataset summary <ArrowRight size={12}/></button><button onClick={() => setQuestion("Which columns have the most missing values?")} disabled={!dataset.id || copilotPending}>Missing values <ArrowRight size={12}/></button><button onClick={() => setQuestion("Which model has the strongest evaluation score?")} disabled={!dataset.id || copilotPending || !trained}>Model comparison <ArrowRight size={12}/></button></div><form className="overview-chat-input" onSubmit={event => { event.preventDefault(); sendQuestion(); }}><input value={question} onChange={event => setQuestion(event.target.value)} disabled={!dataset.id || copilotPending} placeholder={copilotPending ? "AI Assistant is thinking…" : "Ask about your data…"}/><button type="submit" aria-label="Send question" disabled={!dataset.id || copilotPending || !question.trim()}><ArrowRight size={17}/></button></form></aside></div>;
}
function Datasets({ dataset, onFile, onValidate }) {
  const input = useRef(null);
  const [previewCount, setPreviewCount] = useState(5);
  const [filter, setFilter] = useState("");
  const rows = dataset.data.filter(row => Object.values(row).some(v => String(v).toLowerCase().includes(filter.toLowerCase()))).slice(0, previewCount);
  const previewColumns = Object.keys(rows[0] || {}).slice(0, 8);
  return <div><PageHeader title="Datasets" subtitle="Upload data or inspect the current file" actions={<Button primary icon={<Plus size={15}/>} onClick={() => input.current?.click()}>Upload dataset</Button>}/>
    <input ref={input} hidden type="file" accept=".csv,.json,.xlsx,.parquet" onChange={e => { onFile(e.target.files?.[0]); e.target.value = ""; }}/>
    <div className="upload card" onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); onFile(e.dataTransfer.files?.[0]); }}><div className="upload-icon"><Upload size={21}/></div><h2>Drop a file to upload</h2><p>CSV, XLSX, JSON or Parquet</p><Button primary onClick={() => input.current?.click()}>Browse files</Button></div>
    <section className="card dataset-preview"><div className="section-title"><div><h3>{dataset.id ? dataset.name : "Dataset preview"}</h3><p>{dataset.id ? `${dataset.rows.toLocaleString()} rows · ${dataset.columns} columns` : "Upload a file to inspect its rows and columns."}</p></div>{dataset.id && <span className="status success">Loaded</span>}</div>{dataset.id && <><div className="mini-stats"><span><b>{dataset.rows.toLocaleString()}</b> Rows</span><span><b>{dataset.columns}</b> Columns</span><span><b>{dataset.columnsInfo.filter(c => /number|integer|float/.test(String(c[1]))).length}</b> Numeric</span><span><b>{dataset.columnsInfo.filter(c => /category|string/.test(String(c[1]))).length}</b> Categorical</span><span><b>{dataset.missing ?? 0}</b> Missing</span><span><b>{dataset.duplicates ?? 0}</b> Duplicates</span></div><div className="dataset-toolbar"><div className="top-search"><Search size={15}/><input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Filter rows"/></div><Button onClick={() => setPreviewCount(v => v >= 20 ? 5 : v + 5)}>{previewCount >= 20 ? "Show fewer" : "Show more"}</Button><Button onClick={onValidate}><ShieldCheck size={15}/>Check schema</Button></div><div className="table-wrap"><table><thead><tr>{previewColumns.map(k => <th key={k}>{k}</th>)}</tr></thead><tbody>{rows.length ? rows.map((row,i)=><tr key={i}>{previewColumns.map(k => <td key={k}>{String(row[k] ?? "")}</td>)}</tr>) : <tr><td className="table-empty" colSpan={Math.max(previewColumns.length,1)}>{filter ? "No rows match this filter." : "No preview rows available."}</td></tr>}</tbody></table></div></>}</section>
  </div>;
}

function Cleaning({ notify, applyCleaning, undoCleaning, dataset }) {
  const [history, setHistory] = useState([]);
  const missing = dataset.missing ?? 0;
  const duplicates = dataset.duplicates ?? 0;
  const actions = [
    ["Handle missing values", `${missing.toLocaleString()} cells`, "Median / mode imputation", "missing"],
    ["Remove duplicate rows", `${duplicates.toLocaleString()} rows`, "Keep first occurrence", "duplicates"],
    ["Encode categories", "Categorical columns", "One-hot encoding", "encode"],
    ["Review numeric outliers", "Numeric columns", "IQR detection", "outliers"]
  ];
  const apply = async (title, operation) => { if (await applyCleaning(operation)) setHistory(h => [title, ...h]); };
  const preview = async (operation, title) => {
    if (!dataset.id) return;
    try {
      const result = await api(`/datasets/${dataset.id}/clean/preview?operation=${encodeURIComponent(operation)}`);
      notify(`${title}: ${result.rows_removed} rows removed, ${result.missing_removed} missing cells resolved, ${result.columns_added} columns added.`);
    } catch (error) { notify(error.message); }
  };

  return <div>
    <PageHeader title="Data Cleaning" subtitle={dataset.id ? `${dataset.name} · ${dataset.rows.toLocaleString()} rows · ${dataset.columns} columns` : undefined} actions={dataset.id && <><Button onClick={() => undoCleaning()}><RefreshCw size={15}/>Undo</Button><Button primary onClick={() => preview("all", "Full cleaning plan")}>Preview all</Button></>}/>
    {!dataset.id ? <section className="card empty-state"><h3>No dataset loaded</h3><p>Upload a dataset to inspect missing values, duplicates, and cleaning options.</p></section> : <>
    <section className="card cleaning-summary">
      <div className="section-title"><div><h3>Dataset quality</h3><p>{missing || duplicates ? "Review the detected issues before training." : "No missing values or duplicate rows detected."}</p></div><span className={`status ${missing || duplicates ? "warning" : "success"}`}>{missing || duplicates ? "Needs review" : "Clear"}</span></div>
      <div className="mini-stats"><span><b>{dataset.rows.toLocaleString()}</b> Rows</span><span><b>{dataset.columns}</b> Columns</span><span><b>{missing.toLocaleString()}</b> Missing cells</span><span><b>{duplicates.toLocaleString()}</b> Duplicate rows</span></div>
      <div className="quality-issues">
        {missing > 0 && <div className="quality-issue"><span className="issue-mark">!</span><div><b>Missing values</b><small>{missing.toLocaleString()} empty cells across the dataset</small></div><Button onClick={() => preview("missing", "Missing-value plan")}>Review</Button></div>}
        {duplicates > 0 && <div className="quality-issue"><span className="issue-mark">!</span><div><b>Duplicate records</b><small>{duplicates.toLocaleString()} duplicate rows</small></div><Button onClick={() => preview("duplicates", "Duplicate-removal plan")}>Review</Button></div>}
        {!missing && !duplicates && <div className="quality-issue clear"><Check size={16}/><div><b>Dataset checks passed</b><small>No missing cells or duplicate rows were found.</small></div></div>}
      </div>
    </section>
    <section className="card cleaning-actions">
      <div className="section-title"><div><h3>Cleaning operations</h3><p>Preview the effect before applying an operation.</p></div>{history.length > 0 && <span className="status">{history.length} applied this visit</span>}</div>
      <div className="table-wrap cleaning-action-table"><table><thead><tr><th>Operation</th><th>Scope</th><th>Method</th><th>Actions</th></tr></thead><tbody>{actions.map(([title,scope,method,operation]) => <tr key={operation}><td><strong>{title}</strong></td><td>{scope}</td><td>{method}</td><td><div className="action-buttons"><Button onClick={() => preview(operation, title)}>Preview</Button><Button primary onClick={() => apply(title, operation)}>Apply</Button></div></td></tr>)}</tbody></table></div>
    </section>
    </>}
  </div>;
}

function EDA({ dataset, notify, onComplete }) {
  const [eda, setEda] = useState(null);
  const [numericColumn, setNumericColumn] = useState("");
  const [categoricalColumn, setCategoricalColumn] = useState("");
  const [loading, setLoading] = useState(false);

  const load = async () => {
    if (!dataset.id) return;

    setLoading(true);

    try {
      const result = await api(`/datasets/${dataset.id}/eda`);
      setEda(result);
      onComplete?.();

      if (!numericColumn && result.numeric_stats?.length) {
        setNumericColumn(result.numeric_stats[0].name);
      }

      if (!categoricalColumn && result.categorical_stats?.length) {
        setCategoricalColumn(result.categorical_stats[0].name);
      }

      notify("EDA analysis refreshed.");
    } catch (error) {
      notify(error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!dataset.id) return;

    setLoading(true);

    api(`/datasets/${dataset.id}/eda`)
      .then(result => {
        setEda(result);
        onComplete?.();

        if (result.numeric_stats?.length) {
          setNumericColumn(result.numeric_stats[0].name);
        }

        if (result.categorical_stats?.length) {
          setCategoricalColumn(result.categorical_stats[0].name);
        }
      })
      .catch(error => notify(error.message))
      .finally(() => setLoading(false));
  }, [dataset.id]);

  const numericStats = eda?.numeric_stats || [];
  const numericDistributions = eda?.numeric_distributions || [];
  const categoricalStats = eda?.categorical_stats || [];

  const selectedNumeric =
    numericDistributions.find(
      item => item.name === numericColumn
    ) || numericDistributions[0];

  const selectedNumericStats =
    numericStats.find(
      item => item.name === numericColumn
    ) || numericStats[0];

  const selectedCategorical =
    categoricalStats.find(
      item => item.name === categoricalColumn
    ) || categoricalStats[0];

  const categoricalData =
    selectedCategorical?.top_values || [];

  const missingData =
    eda?.missing_by_column || [];

  const outlierData =
    numericStats
      .filter(item => item.outliers > 0)
      .sort((a, b) => b.outliers - a.outliers)
      .slice(0, 12);

  const correlationData =
    eda?.correlations?.slice(0, 12) || [];
  const categoricalNumericComparisons = eda?.categorical_numeric_comparisons || [];
  const selectedComparison = categoricalNumericComparisons.find(
    item => item.category === categoricalColumn && item.measure === numericColumn
  ) || categoricalNumericComparisons[0];

  return (
    <div>
      <PageHeader
        title="Exploratory Analysis"
        subtitle="Distributions, data quality, and relationships"
        actions={
          <Button onClick={load} disabled={loading}>
            <RefreshCw size={15} />
            {loading ? "Analyzing…" : "Refresh Analysis"}
          </Button>
        }
      />

      {!eda ? (
        <section className="card">
          <div className="empty-state">
            <BarChart3 size={32} />
            <h3>{dataset.id ? "Loading analysis" : "Upload a dataset to begin"}</h3>
            <p>{dataset.id ? "Analyzing the current dataset." : "Upload a CSV, JSON, XLSX or Parquet file from Datasets."}</p>
          </div>
        </section>
      ) : (
        <>
          <div className="stats-strip card">
            <span>
              <b>{eda.rows?.toLocaleString() ?? "—"}</b>
              <small>Rows</small>
            </span>

            <span>
              <b>{eda.columns ?? "—"}</b>
              <small>Columns</small>
            </span>

            <span>
              <b>{eda.numeric_columns ?? "—"}</b>
              <small>Numeric</small>
            </span>

            <span>
              <b>{eda.categorical_columns ?? "—"}</b>
              <small>Categorical</small>
            </span>

            <span>
              <b>{eda.missing ?? 0}</b>
              <small>Missing cells</small>
            </span>

            <span>
              <b>{eda.duplicates ?? 0}</b>
              <small>Duplicate rows</small>
            </span>
          </div>

          <div className="grid-2">
            <section className="card chart-card">
              <div className="section-title">
                <div>
                    <h3>Numeric distributions</h3>
                </div>

                {numericDistributions.length > 0 && (
                  <select
                    value={numericColumn}
                    onChange={e =>
                      setNumericColumn(e.target.value)
                    }
                  >
                    {numericDistributions.map(item => (
                      <option
                        key={item.name}
                        value={item.name}
                      >
                        {item.name}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {selectedNumeric?.bins?.length ? (
                <>
                  <ResponsiveContainer
                    width="100%"
                    height={320}
                  >
                    <BarChart
                      data={selectedNumeric.bins}
                      margin={{
                        top: 10,
                        right: 15,
                        left: 0,
                        bottom: 55
                      }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        vertical={false}
                      />

                      <XAxis
                        dataKey="label"
                        angle={-35}
                        textAnchor="end"
                        height={70}
                        interval={0}
                      />

                      <YAxis
                        allowDecimals={false}
                        label={{
                          value: "Frequency",
                          angle: -90,
                          position: "insideLeft"
                        }}
                      />

                      <Tooltip
                        formatter={(value, name) => [
                          value,
                          name === "count"
                            ? "Records"
                            : name
                        ]}
                        labelFormatter={label =>
                          `Range: ${label}`
                        }
                      />

                      <Bar
                        dataKey="count"
                        name="Records"
                        fill="#2563eb"
                        radius={[5, 5, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>

                  {selectedNumericStats && (
                    <div className="mini-stats">
                      <span>
                        <b>
                          {selectedNumericStats.mean}
                        </b>
                        Mean
                      </span>

                      <span>
                        <b>
                          {selectedNumericStats.median}
                        </b>
                        Median
                      </span>

                      <span>
                        <b>
                          {selectedNumericStats.std}
                        </b>
                        Std Dev
                      </span>

                      <span>
                        <b>
                          {selectedNumericStats.min}
                        </b>
                        Min
                      </span>

                      <span>
                        <b>
                          {selectedNumericStats.max}
                        </b>
                        Max
                      </span>

                      <span>
                        <b>
                          {selectedNumericStats.outliers}
                        </b>
                        Outliers
                      </span>
                    </div>
                  )}
                </>
              ) : (
                <div className="empty-state">
                  <BarChart3 size={28} />
                  <p>No numerical distribution available.</p>
                </div>
              )}
            </section>

            <section className="card chart-card">
              <div className="section-title">
                <div>
                    <h3>Category counts</h3>
                </div>

                {categoricalStats.length > 0 && (
                  <select
                    value={categoricalColumn}
                    onChange={e =>
                      setCategoricalColumn(e.target.value)
                    }
                  >
                    {categoricalStats.map(item => (
                      <option
                        key={item.name}
                        value={item.name}
                      >
                        {item.name}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {categoricalData.length > 0 ? (
                <ResponsiveContainer
                  width="100%"
                  height={320}
                >
                  <BarChart
                    layout="vertical"
                    data={categoricalData}
                    margin={{
                      top: 5,
                      right: 25,
                      left: 20,
                      bottom: 5
                    }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      horizontal={false}
                    />

                    <XAxis
                      type="number"
                      allowDecimals={false}
                    />

                    <YAxis
                      type="category"
                      dataKey="value"
                      width={120}
                    />

                    <Tooltip
                      formatter={(value, name, props) => {
                        if (name === "count") {
                          return [
                            `${value} records`,
                            "Frequency"
                          ];
                        }

                        return [value, name];
                      }}
                    />

                    <Bar
                      dataKey="count"
                      name="Frequency"
                      fill="#2563eb"
                      radius={[0, 5, 5, 0]}
                      label={{
                        position: "right"
                      }}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state">
                  <BarChart3 size={28} />
                  <p>No categorical distribution available.</p>
                </div>
              )}

              {selectedCategorical && (
                <div className="mini-stats">
                  <span>
                    <b>{selectedCategorical.unique}</b>
                    Unique values
                  </span>

                  <span>
                    <b>{selectedCategorical.missing}</b>
                    Missing
                  </span>

                  <span>
                    <b>
                      {selectedCategorical.high_cardinality
                        ? "Yes"
                        : "No"}
                    </b>
                    High cardinality
                  </span>
                </div>
              )}
            </section>
          </div>

          <div className="grid-2">
            <section className="card chart-card">
              <div className="section-title">
                <div>
                  <h3>Missing Values by Column</h3>
                  <p>
                    Columns with the highest number of missing values
                  </p>
                </div>
              </div>

              {missingData.length > 0 ? (
                <ResponsiveContainer
                  width="100%"
                  height={300}
                >
                  <BarChart
                    layout="vertical"
                    data={missingData.slice(0, 12)}
                    margin={{
                      top: 5,
                      right: 30,
                      left: 20,
                      bottom: 5
                    }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      horizontal={false}
                    />

                    <XAxis
                      type="number"
                      allowDecimals={false}
                    />

                    <YAxis
                      type="category"
                      dataKey="name"
                      width={120}
                    />

                    <Tooltip
                      formatter={(value, name) => [
                        value,
                        name === "missing"
                          ? "Missing values"
                          : name
                      ]}
                    />

                    <Bar
                      dataKey="missing"
                      name="Missing"
                      fill="#e879f9"
                      radius={[0, 5, 5, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state">
                  <Check size={28} />
                  <h3>No missing values</h3>
                  <p>
                    All columns currently contain complete values.
                  </p>
                </div>
              )}
            </section>

            <section className="card chart-card">
              <div className="section-title">
                <div>
                  <h3>Numeric Outliers</h3>
                  <p>
                    Potential outliers detected using the IQR rule
                  </p>
                </div>
              </div>

              {outlierData.length > 0 ? (
                <ResponsiveContainer
                  width="100%"
                  height={300}
                >
                  <BarChart
                    layout="vertical"
                    data={outlierData}
                    margin={{
                      top: 5,
                      right: 30,
                      left: 20,
                      bottom: 5
                    }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      horizontal={false}
                    />

                    <XAxis
                      type="number"
                      allowDecimals={false}
                    />

                    <YAxis
                      type="category"
                      dataKey="name"
                      width={120}
                    />

                    <Tooltip
                      formatter={(value, name) => [
                        value,
                        name === "outliers"
                          ? "Potential outliers"
                          : name
                      ]}
                    />

                    <Bar
                      dataKey="outliers"
                      name="Outliers"
                      fill="#f59e0b"
                      radius={[0, 5, 5, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state">
                  <Check size={28} />
                  <h3>No potential outliers detected</h3>
                  <p>
                    No numeric columns currently contain IQR outliers.
                  </p>
                </div>
              )}
            </section>
          </div>

          <section className="card chart-card">
            <div className="section-title">
              <div>
                <h3>Numeric Correlations</h3>
                <p>
                  Strongest relationships between numeric variables
                </p>
              </div>

            </div>

            {correlationData.length > 0 ? (
              <ResponsiveContainer
                width="100%"
                height={340}
              >
                <BarChart
                  layout="vertical"
                  data={correlationData}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 100,
                    bottom: 5
                  }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    horizontal={false}
                  />

                  <XAxis
                    type="number"
                    domain={[-1, 1]}
                    tickCount={5}
                  />

                  <YAxis
                    type="category"
                    dataKey="x"
                    width={100}
                  />

                  <Tooltip
                    formatter={(value) => [
                      value,
                      "Correlation"
                    ]}
                    labelFormatter={(label, payload) => {
                      const item = payload?.[0]?.payload;

                      return item
                        ? `${item.x} × ${item.y}`
                        : label;
                    }}
                  />

                  <Bar
                    dataKey="value"
                    name="Correlation"
                    fill="#2563eb"
                    radius={[0, 5, 5, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                <BarChart3 size={28} />
                <p>
                  At least two numeric columns are required
                  for correlation analysis.
                </p>
              </div>
            )}
          </section>

          {selectedComparison && (
            <section className="card chart-card">
              <div className="section-title">
                <div>
                  <h3>{selectedComparison.category} vs {selectedComparison.measure}</h3>
                  <p>Group means calculated from the uploaded rows; this shows association, not causation.</p>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={selectedComparison.groups} margin={{top:10,right:20,left:10,bottom:10}}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false}/>
                  <XAxis dataKey="name"/>
                  <YAxis/>
                  <Tooltip formatter={(value, name) => [value, name === "mean" ? "Mean" : name]}/>
                  <Bar dataKey="mean" name="Mean" fill="#2563eb" radius={[5,5,0,0]}/>
                </BarChart>
              </ResponsiveContainer>
            </section>
          )}

          <section className="card">
            <div className="section-title">
              <div>
                <h3>Observations</h3>
              </div>
            </div>

            <div className="insight-list">
              {(eda.insights || []).map((insight, index) => (
                <div className="large-insight" key={index}>
                  <div>
                    <p>{insight}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

function Features({ features, setFeatures, notify, addFeature, removeFeature, dataset }) {
  const columns = dataset.columnsInfo || [];

  const numericColumns = columns
    .filter(c => /number|integer|float|double|int/i.test(String(c[1])))
    .map(c => c[0]);

  const allColumns = columns.map(c => c[0]);

  const [featureName, setFeatureName] = useState("");
  const [type, setType] = useState("copy");
  const [source, setSource] = useState(numericColumns[0] || allColumns[0] || "");
  const [secondSource, setSecondSource] = useState(numericColumns[1] || numericColumns[0] || "");
  const [bins, setBins] = useState("4");

  useEffect(() => {
    if (!source && allColumns.length) {
      setSource(numericColumns[0] || allColumns[0]);
    }
  }, [dataset.id]);

  const create = async () => {
    const name = featureName.trim();

    if (!name) {
      notify("Enter a feature name.");
      return;
    }

    if (!source) {
      notify("Select a source column.");
      return;
    }

    let expression = "";

    if (type === "copy") {
      expression = `copy:${source}`;
    }

    if (type === "ratio") {
      if (!secondSource) {
        notify("Select the second column.");
        return;
      }

      expression = `ratio:${source}/${secondSource}`;
    }

    if (type === "multiply") {
      if (!secondSource) {
        notify("Select the second column.");
        return;
      }

      expression = `multiply:${source}*${secondSource}`;
    }

    if (type === "log1p") {
      expression = `log1p:${source}`;
    }

    if (type === "bin") {
      expression = `bin:${source}:${bins}`;
    }

    await addFeature(name, expression);
    setFeatureName("");
  };

  const addRecommended = (name, expression) => {
    if (features.includes(name)) {
      removeFeature(name);
      return;
    }

    addFeature(name, expression);
  };

  const recommended = numericColumns.slice(0, 6).flatMap(column => [
    { name: `${column}_log1p`, type: "numeric", expression: `log1p:${column}`, description: `Log transform for ${column}`, available: true },
    { name: `${column}_bins`, type: "category", expression: `bin:${column}:4`, description: `Four quantile groups for ${column}`, available: true }
  ]);

  return (
    <div>
      <PageHeader
        title="Feature Engineering"
        subtitle="Create and review derived columns"
      />

      <section className="card">
        <div className="section-title">
          <div>
            <h3>Create Feature</h3>
            <p>Build a new feature from existing dataset columns.</p>
          </div>
        </div>

        <div className="feature-builder">
          <input
            value={featureName}
            onChange={e => setFeatureName(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter") {
                create();
              }
            }}
            placeholder="Feature name"
          />

          <select
            value={type}
            onChange={e => setType(e.target.value)}
          >
            <option value="copy">Copy column</option>
            <option value="ratio">Ratio</option>
            <option value="multiply">Multiply</option>
            <option value="log1p">Log transform</option>
            <option value="bin">Quantile bins</option>
          </select>

          <select
            value={source}
            onChange={e => setSource(e.target.value)}
          >
            {allColumns.map(column => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>

          {(type === "ratio" || type === "multiply") && (
            <select
              value={secondSource}
              onChange={e => setSecondSource(e.target.value)}
            >
              {numericColumns.map(column => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          )}

          {type === "bin" && (
            <select
              value={bins}
              onChange={e => setBins(e.target.value)}
            >
              <option value="2">2 bins</option>
              <option value="3">3 bins</option>
              <option value="4">4 bins</option>
              <option value="5">5 bins</option>
              <option value="6">6 bins</option>
              <option value="8">8 bins</option>
              <option value="10">10 bins</option>
            </select>
          )}

          <Button
            primary
            onClick={create}
          >
            <Plus size={15} />
            Create Feature
          </Button>
        </div>
      </section>

      <section className="card">
        <div className="section-title">
          <div>
            <h3>Recommended Features</h3>
            <p>Useful transformations detected from the current dataset.</p>
          </div>

          <span className="status success">
            <Check size={13} />
            {features.length} active
          </span>
        </div>

        {recommended.map(item => (
          <div
            className="feature-row"
            key={item.name}
          >
            <div className="feature-icon">
              <GitBranch size={17} />
            </div>

            <div>
              <b>{item.name}</b>
              <span>
                {item.type} · {item.description}
              </span>
            </div>

            <span className="importance high">
              {item.available ? "Available" : "Unavailable"}
            </span>

            <Button
              disabled={!item.available}
              onClick={() =>
                addRecommended(
                  item.name,
                  item.expression
                )
              }
            >
              {features.includes(item.name)
                ? "Remove"
                : "Add"}
            </Button>
          </div>
        ))}

        {features
          .filter(
            feature =>
              !recommended.some(
                item => item.name === feature
              )
          )
          .map(feature => (
            <div
              className="feature-row"
              key={feature}
            >
              <div className="feature-icon">
                <GitBranch size={17} />
              </div>

              <div>
                <b>{feature}</b>
                <span>
                  custom · active engineered feature
                </span>
              </div>

              <span className="importance high">
                Active
              </span>

              <Button
                onClick={() =>
                  removeFeature(feature)
                }
              >
                <Trash2 size={15} />
                Remove
              </Button>
            </div>
          ))}
      </section>
    </div>
  );
}

function MLStudio({ training, runTraining, notify, dataset }) {
  const columnInfo = dataset.columnsInfo || [];

  const columnNames = columnInfo.map(
    c => c[0]
  );

  const numericNames = columnInfo
    .filter(c =>
      /number|integer|float|double|int/i.test(
        String(c[1])
      )
    )
    .map(c => c[0]);

  const engineeredNames = new Set(dataset.features || []);
  const defaultTarget = columnNames.find(c => /target|label|outcome|class/i.test(c)) ||
    [...numericNames].reverse().find(column => {
      const info = columnInfo.find(item => item[0] === column);
      return !engineeredNames.has(column) && Number(info?.[4] || 0) > Math.max(20, dataset.rows * 0.05);
    }) || columnNames[columnNames.length - 1] || "";

  const inferProblemType = column => {
    const info = columnInfo.find(item => item[0] === column);
    const unique = Number(info?.[4] || 0);
    const numeric = /number|integer|float|double|int/i.test(String(info?.[1] || ""));
    return numeric && unique > Math.max(20, dataset.rows * 0.05) ? "Regression" : "Classification";
  };

  const [target, setTarget] =
    useState(defaultTarget);

  const [problem, setProblem] =
    useState(() => inferProblemType(defaultTarget));

  const classificationModels = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "SVM",
    "KNN"
  ];

  const regressionModels = [
    "Linear Regression",
    "Ridge",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "SVR"
  ];

  const models =
    problem === "Regression"
      ? regressionModels
      : classificationModels;

  const [selected, setSelected] =
    useState(classificationModels);

  useEffect(() => {
    setTarget(defaultTarget);
    setProblem(inferProblemType(defaultTarget));
  }, [dataset.id]);

  useEffect(() => {
    setSelected(
      problem === "Regression"
        ? regressionModels
        : classificationModels
    );
  }, [problem]);

  const toggle = model => {
    setSelected(current =>
      current.includes(model)
        ? current.filter(x => x !== model)
        : [...current, model]
    );
  };

  const train = () => {
    if (!target) {
      notify("Select a target variable.");
      return;
    }

    if (!selected.length) {
      notify("Select at least one model.");
      return;
    }

    runTraining({
      target,
      problem_type: problem,
      models: selected
    });
  };

  return (
    <div>
      <PageHeader
        title="ML Studio"
        subtitle="Choose a target and estimators for the current dataset"
        actions={
          <Button
            primary
            icon={training ? <RefreshCw size={15} className="loading-spin" /> : <Play size={15} />}
            onClick={train}
            disabled={
              training ||
              !target ||
              !selected.length
            }
          >
            {training
              ? "Training…"
              : "Train Models"}
          </Button>
        }
      />

      <div className="stats-strip card">
        <span>
          <b>{dataset.rows?.toLocaleString() || "0"}</b>
          <small>Training rows</small>
        </span>

        <span>
          <b>{dataset.columns || 0}</b>
          <small>Features</small>
        </span>

        <span>
          <b>{numericNames.length}</b>
          <small>Numeric</small>
        </span>

        <span>
          <b>
            {columnNames.length -
              numericNames.length}
          </b>
          <small>Categorical</small>
        </span>
      </div>

      <div className="ml-layout">
        <section className="card config-card">
          <h3>Training Configuration</h3>

          <label>Target variable</label>

          <select
            value={target}
            onChange={e => {
              setTarget(e.target.value);
              setProblem(inferProblemType(e.target.value));
            }}
          >
            {columnNames.map(column => (
              <option
                key={column}
                value={column}
              >
                {column}
              </option>
            ))}
          </select>

          <label>Problem type</label>

          <div className="segmented">
            <button
              className={
                problem === "Classification"
                  ? "selected"
                  : ""
              }
              onClick={() =>
                setProblem("Classification")
              }
            >
              Classification
            </button>

            <button
              className={
                problem === "Regression"
                  ? "selected"
                  : ""
              }
              onClick={() =>
                setProblem("Regression")
              }
            >
              Regression
            </button>
          </div>

          <label>Models</label>

          {models.map(model => (
            <label
              className="check"
              key={model}
            >
              <input
                type="checkbox"
                checked={selected.includes(model)}
                onChange={() =>
                  toggle(model)
                }
              />

              <span>{model}</span>
            </label>
          ))}

          <div className="config-summary">
            <b>{selected.length}</b>{" "}
            models selected

            <br />

            <small>
              Target: {target || "—"} ·{" "}
              {problem}
            </small>
          </div>
        </section>

        <section>
          <div className="training-banner" aria-live="polite" aria-busy={training}>
            <div>
              <div className="eyebrow">
                MODEL TRAINING
              </div>

              <h2>
                {training
                  ? "Training models…"
                  : "Ready to train"}
              </h2>

              <p>
                {training
                  ? "FastAPI is preprocessing the dataset and evaluating each selected model. This can take a little while."
                  : "Train the selected models on this dataset."}
              </p>
              {training && <div className="training-progress" role="progressbar" aria-label="Training selected models" aria-valuetext="Training in progress"><span /></div>}
            </div>

            <div
              className={`big-progress ${
                training ? "active" : ""
              }`}
            >
              <span>
                {training
                  ? "RUN"
                  : "READY"}
              </span>
            </div>
          </div>

          <section className="card">
            <h3>Evaluation Metrics</h3>

            {problem === "Classification" ? (
              <div className="mini-stats">
                <span>
                  <b>Accuracy</b>
                  Overall correct predictions
                </span>

                <span>
                  <b>Precision</b>
                  Correct positive predictions
                </span>

                <span>
                  <b>Recall</b>
                  Positive cases detected
                </span>

                <span>
                  <b>F1</b>
                  Precision/recall balance
                </span>

                <span>
                  <b>ROC-AUC</b>
                  Ranking quality
                </span>
              </div>
            ) : (
              <div className="mini-stats">
                <span>
                  <b>R²</b>
                  Explained variance
                </span>

                <span>
                  <b>MAE</b>
                  Mean absolute error
                </span>

                <span>
                  <b>RMSE</b>
                  Root mean squared error
                </span>
              </div>
            )}
          </section>
        </section>
      </div>
    </div>
  );
} 

function ModelComparison({ models, selectedModel, setSelectedModel, setPage, problem }) {
  const isRegression = String(problem).toLowerCase() === "regression";
  const selected = models.find(row => row[0] === selectedModel) || models[0];
  const best = models.reduce((a, b) => Number(String(a?.[1]).replace("%", "")) >= Number(String(b?.[1]).replace("%", "")) ? a : b, models[0]);
  const metricLabels = isRegression
    ? ["R²", "MAE", "RMSE", "R² score", "ROC-AUC", "Training time"]
    : ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Training time"];

  return <div>
    <PageHeader title="Model Comparison" subtitle="Held-out evaluation metrics" actions={<Button primary disabled={!models.length} onClick={() => setPage("Explainability")}>Explain selected model <ArrowRight size={15}/></Button>}/>
    {models.length ? <>
      <div className="kpis">
        <Kpi label="Models tested" value={models.length}/>
        <Kpi label={isRegression ? "Best R²" : "Best accuracy"} value={best?.[1] || "—"} note={best?.[0]}/>
        <Kpi label={isRegression ? "Best MAE" : "Best F1"} value={isRegression ? best?.[2] || "—" : best?.[4] || "—"} note="Held-out data"/>
        <Kpi label={isRegression ? "Best RMSE" : "Best ROC-AUC"} value={isRegression ? best?.[3] || "—" : best?.[5] || "—"} note="Held-out data"/>
      </div>
      <section className="card">
        <div className="section-title"><div><h3>Model performance</h3><p>Select a model to use for explanation and AI Assistant.</p></div></div>
        <div className="table-wrap"><table><thead><tr><th>Model</th>{metricLabels.map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>
          {models.map(row => <tr key={row[0]} tabIndex={0} onClick={() => setSelectedModel(row[0])} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") setSelectedModel(row[0]); }} className={selectedModel === row[0] ? "selected-row" : ""}><td><strong>{row[0]}</strong>{row[0] === best[0] && <span className="best-tag">Best</span>}</td>{row.slice(1, 7).map((value, i) => <td key={i}>{value}</td>)}</tr>)}
        </tbody></table></div>
      </section>
      {selected && <section className="card"><div className="section-title"><div><h3>Selected model: {selected[0]}</h3></div></div><div className="mini-stats">{metricLabels.map((label, i) => <span key={label}><b>{selected[i + 1] || "—"}</b> {label}</span>)}</div></section>}
    </> : <section className="card empty-state"><h3>No trained models</h3><p>Choose a target and train one or more models to compare their held-out metrics.</p><Button primary onClick={() => setPage("ML Studio")}><Play size={15}/>Open ML Studio</Button></section>}
  </div>;
}

function Explainability({ selectedCustomer, setSelectedCustomer, dataset, notify, selectedModel, onComplete }) {
  const [explanation, setExplanation] = useState(null); const [globalExplanation, setGlobalExplanation] = useState(null);
  const customerIds = dataset.data.map((x, i) => x.customerID || x.customerId || x.id || String(i)).filter(Boolean).slice(0, 50);
  const row = Math.max(0, customerIds.indexOf(selectedCustomer));
  useEffect(() => {
    if (!dataset.id || !selectedModel) return;
    Promise.all([api(`/datasets/${dataset.id}/explain?model=${encodeURIComponent(selectedModel)}&row=${row}`), api(`/datasets/${dataset.id}/explain/global?model=${encodeURIComponent(selectedModel)}`)]).then(([individual, global]) => { setExplanation(individual); setGlobalExplanation(global); onComplete?.(); }).catch(() => { setExplanation(null); setGlobalExplanation(null); });
  }, [dataset.id, selectedModel, row]);
  const globalImportance = globalExplanation?.features || [];
  const individualImportance = explanation?.features || [];
  const isRegression = explanation?.problem_type === "regression"; const probability = explanation?.probability != null ? Math.round(explanation.probability * 100) : 0;
  const predictionDisplay = isRegression && explanation?.prediction != null && Number.isFinite(Number(explanation.prediction))
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(Number(explanation.prediction))
    : explanation?.prediction ?? "Train a model";
  const refresh = () => {
    if (!dataset.id || !selectedModel) return;
    Promise.all([
      api(`/datasets/${dataset.id}/explain?model=${encodeURIComponent(selectedModel)}&row=${row}`),
      api(`/datasets/${dataset.id}/explain/global?model=${encodeURIComponent(selectedModel)}`)
    ]).then(([individual, global]) => {
      setExplanation(individual); setGlobalExplanation(global); onComplete?.(); notify(`${global.method} explanation refreshed.`);
    }).catch(error => notify(error.message));
  };
  return <div>
    <PageHeader title="Model Explainability" subtitle={selectedModel || "Train a model to view its explanations"} actions={<Button onClick={refresh} disabled={!selectedModel}><RefreshCw size={15}/>Refresh</Button>}/>
    {!selectedModel ? <section className="card empty-state"><h3>No model selected</h3><p>Train a model, then select it in Model Comparison to inspect global importance and individual predictions.</p></section> : <div className="grid-2">
      <section className="card chart-card">
        <div className="section-title"><div><h3>Global feature importance</h3><p>{globalExplanation?.method || "Model-derived importance"} · relative scale</p></div></div>
        <ResponsiveContainer width="100%" height={350}><BarChart layout="vertical" data={globalImportance} margin={{left:20,right:20}}><CartesianGrid strokeDasharray="3 3" horizontal={false}/><XAxis type="number"/><YAxis dataKey="name" type="category" width={140}/><Tooltip/><Bar dataKey="value" fill="#2563eb" radius={[0,5,5,0]}/></BarChart></ResponsiveContainer>
      </section>
      <section className="card prediction">
        <div className="section-title"><div><h3>Individual prediction</h3><p>Customer: {selectedCustomer || "—"}</p></div><select value={selectedCustomer} onChange={e => setSelectedCustomer(e.target.value)}>{customerIds.map(x => <option key={x}>{x}</option>)}</select></div>
        <div className="risk"><div className="risk-ring">{isRegression ? "≈" : probability}<small>{isRegression ? "estimate" : "%"}</small></div><div><span>{isRegression ? `Predicted ${explanation?.target || "value"}` : "Prediction"}</span><h2>{predictionDisplay}</h2><p>Model: {selectedModel || "—"}</p></div></div>
        <h4>Individual feature contributions</h4>
        {individualImportance.slice(0,4).map((feature,i) => <div className={`contributor ${feature.direction === "decreases" ? "negative" : "positive"}`} key={feature.name}><span>{feature.direction === "decreases" ? "−" : feature.direction === "increases" ? "+" : "•"}</span><div><b>{feature.name}</b><small>{feature.direction === "influences" ? "Model sensitivity" : `${feature.direction} prediction`} · relative influence {Math.abs(feature.value)}</small></div><em>{i === 0 ? "High" : "Medium"}</em></div>)}
        <div className="explanation"><p>{explanation ? "Importance scores show relative influence. They are not prediction units." : "Train a model to generate an explanation."}</p></div>
      </section>
    </div>}
  </div>;
}
function Copilot({ chat, question, setQuestion, sendQuestion, dataset, llmConfigured, selectedModel, pending }) {
  const suggestions = ["Summarize this dataset", "Which columns have the most missing values?", "What patterns are visible in the numeric columns?", "Which model has the strongest evaluation score?"];
  const sourceColumns = dataset.columnsInfo?.map(column => column[0]) || [];
  const messagesRef = useRef(null);
  const copilotStatus = llmConfigured === null ? "Checking connection" : llmConfigured ? "API key configured" : "API key required";

  useEffect(() => {
    const messages = messagesRef.current;
    if (messages) messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
  }, [chat, pending]);

  return <div>
    <PageHeader title="AI Assistant" subtitle={dataset.id ? `Ask about ${dataset.name}` : "Upload a dataset to start"} />
    <div className="copilot-layout">
      <section className="card chat-panel">
        <div className="chat-head"><div className="copilot-avatar"><Bot size={19}/></div><div><b>AI Assistant</b><small>{pending ? "Analyzing…" : copilotStatus}</small></div><span className="online-dot" style={{background: llmConfigured === false ? "#d69737" : undefined}}/></div>
        <div className="messages" ref={messagesRef} aria-live="polite">
          {!dataset.id ? <div className="message ai"><div className="message-avatar"><Database size={14}/></div><div><p>Upload a dataset before asking questions.</p></div></div> : chat.map((m,i) => <div key={i} className={`message ${m.role}`}><div className="message-avatar">{m.role === "ai" ? <Bot size={14}/> : "PT"}</div><div><p>{m.text}</p>{m.sources?.length > 0 && <><small className="source-label">Dataset columns</small><div className="source-tags">{m.sources.slice(0,4).map(source => <span key={source}>{source}</span>)}</div></>}</div></div>)}
          {pending && <div className="message ai pending-message" role="status" aria-label="AI Assistant is preparing an answer"><div className="message-avatar"><Sparkles size={14}/></div><div className="typing-bubble"><span className="typing-dots" aria-hidden="true"><i/><i/><i/></span><span>Analyzing dataset and preparing an answer</span></div></div>}
        </div>
        <div className="suggestions">{suggestions.map(s => <button key={s} disabled={pending || !dataset.id} onClick={() => sendQuestion(s)}>{s}</button>)}</div>
        <div className="chat-input"><input value={question} disabled={pending || !dataset.id} onChange={e => setQuestion(e.target.value)} onKeyDown={e => e.key === "Enter" && !pending && sendQuestion()} placeholder={pending ? "Waiting for AI Assistant…" : dataset.id ? "Ask about your analysis…" : "Upload a dataset first"}/><button disabled={pending || !dataset.id || !question.trim()} onClick={() => sendQuestion()} aria-label={pending ? "Waiting for AI Assistant" : "Send question"}><ArrowRight size={18}/></button></div>
      </section>
      <aside className="card reliability"><h3>Analysis context</h3><p>Evidence included with each question</p><div className="context-item"><b>{dataset.name}</b><small>{dataset.rows.toLocaleString()} rows · {dataset.columns} columns</small></div><div className="context-item"><b>Exploratory analysis</b><small>Column summaries, missing values, correlations</small></div><div className="context-item"><b>Selected model</b><small>{selectedModel || "None selected"}</small></div>{llmConfigured === false && <div className="guardrail-note">Add OPENAI_API_KEY to backend/.env and restart the API to enable answers.</div>}<hr/><h4>Dataset columns</h4>{sourceColumns.slice(0,10).map(s => <div className="source-line" key={s}><Database size={14}/><code>{s}</code></div>)}</aside>
    </div>
  </div>;
}

function Reports({ notify, dataset, trained, onComplete }) {
  const [generated, setGenerated] = useState(false); const [selected, setSelected] = useState(["Dataset Overview","Data Quality","EDA Insights","Model Performance","Explainability"]);
  const sections = ["Dataset Overview","Data Quality","Cleaning Operations","EDA Insights","Feature Engineering","Model Performance","Explainability","Business Recommendations"];
  const toggle = s => setSelected(x => x.includes(s) ? x.filter(v => v !== s) : [...x,s]);
  const generate = async () => { if (!dataset.id) return; try { const result = await api(`/datasets/${dataset.id}/report`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sections:selected})}); const blob=new Blob([result.content],{type:"text/html;charset=utf-8"}); const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download=result.filename; a.click(); URL.revokeObjectURL(url); setGenerated(true); onComplete?.(); notify("Report generated and downloaded."); } catch(e){notify(e.message);} };
  return <div><PageHeader title="Reports" subtitle="Build a report from this analysis" actions={<Button primary disabled={!dataset.id || !selected.length} onClick={generate}><FileText size={15}/>Generate report</Button>}/><div className="report-layout"><section className="card report-builder"><h3>Sections</h3>{sections.map(s => <label className="report-check" key={s}><input type="checkbox" checked={selected.includes(s)} onChange={() => toggle(s)}/><span>{s}</span><Check size={15}/></label>)}<div className="report-note"><span>{selected.length} sections selected · {generated ? "Report generated" : dataset.id ? "Ready" : "Upload a dataset first"}</span></div></section><section className="card report-preview"><div className="report-paper"><div className="report-logo">DataPilot</div><span className="eyebrow">ANALYSIS REPORT</span><h1>{dataset.name}</h1><p className="muted">{generated ? "Generated from current analysis" : "Preview"} · {trained ? "Models trained" : "Training pending"}</p><div className="report-kpis"><b>{dataset.rows.toLocaleString()}<small>Rows</small></b><b>{dataset.columns}<small>Columns</small></b><b>{dataset.missing ?? 0}<small>Missing</small></b><b>{dataset.duplicates ?? 0}<small>Duplicates</small></b></div><h3>Included sections</h3><ul>{selected.map(s => <li key={s}>{s}</li>)}</ul></div></section></div></div>;
}


createRoot(document.getElementById("root")).render(<App />);
