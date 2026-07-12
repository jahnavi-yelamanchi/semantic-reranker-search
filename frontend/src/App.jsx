import { useEffect, useMemo, useState } from "react";
import { BarChart3, FileUp, Gauge, Loader2, Search } from "lucide-react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

const MODES = [
  { id: "bm25", label: "BM25" },
  { id: "base", label: "Base model" },
  { id: "finetuned", label: "ONNX INT8" },
];

function formatSize(sizeMb) {
  if (sizeMb == null) return "-";
  if (sizeMb > 0 && sizeMb < 0.01) return "<0.01 MB";
  return `${sizeMb} MB`;
}

function App() {
  const [title, setTitle] = useState("Benefits FAQ");
  const [text, setText] = useState(
    "Our benefits plan includes medical, dental, and vision coverage. Employees can enroll during onboarding or open enrollment. The 401k match starts after 90 days. Remote employees receive a monthly workspace stipend."
  );
  const [query, setQuery] = useState("What benefits do remote employees receive?");
  const [mode, setMode] = useState("bm25");
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [status, setStatus] = useState("Ready");
  const [artifactStatus, setArtifactStatus] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/metrics`).then((res) => res.json()).then(setMetrics).catch(() => setMetrics([]));
    fetch(`${API_BASE}/artifacts`).then((res) => res.json()).then(setArtifacts).catch(() => setArtifacts([]));
  }, []);

  const selectedMode = useMemo(() => MODES.find((item) => item.id === mode), [mode]);
  const readyArtifacts = useMemo(() => artifacts.filter((item) => item.present), [artifacts]);

  async function ingestDocument(event) {
    event.preventDefault();
    setLoading(true);
    setStatus("Indexing document");
    try {
      const response = await fetch(`${API_BASE}/documents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text }),
      });
      const payload = await response.json();
      setStatus(`Indexed ${payload.chunk_count} chunk${payload.chunk_count === 1 ? "" : "s"}`);
    } catch {
      setStatus("Could not index document");
    } finally {
      setLoading(false);
    }
  }

  async function loadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const body = await file.text();
    setTitle(file.name);
    setText(body);
    setStatus(`Loaded ${file.name}`);
  }

  async function runSearch(event) {
    event.preventDefault();
    setLoading(true);
    setStatus(`Searching with ${selectedMode?.label ?? mode}`);
    try {
      const response = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, mode, top_k: topK }),
      });
      const payload = await response.json();
      setResults(payload.results ?? []);
      setArtifactStatus(payload.artifact_status ?? "");
      setStatus(`Returned ${payload.results?.length ?? 0} ranked result${payload.results?.length === 1 ? "" : "s"}`);
    } catch {
      setStatus("Search failed");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Search size={18} /></div>
          <div>
            <h1>Reranker Search</h1>
            <p>BM25 vs embeddings vs trained INT8</p>
          </div>
        </div>
        <div className="side-stat">
          <Gauge size={18} />
          <span>{status}</span>
        </div>
        <div className="side-stat">
          <BarChart3 size={18} />
          <span>{metrics.length || 3} benchmark rows</span>
        </div>
        <div className="side-stat">
          <Gauge size={18} />
          <span>{readyArtifacts.length || 0} model artifact{readyArtifacts.length === 1 ? "" : "s"} ready</span>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h2>Semantic search workbench</h2>
            <p>Upload product docs, FAQs, or job listings and compare retrieval strategies.</p>
          </div>
          {loading ? <Loader2 className="spin" size={22} /> : null}
        </header>

        <div className="grid">
          <form className="panel document-panel" onSubmit={ingestDocument}>
            <div className="panel-heading">
              <FileUp size={18} />
              <h3>Document input</h3>
            </div>
            <label>
              Title
              <input value={title} onChange={(event) => setTitle(event.target.value)} />
            </label>
            <label>
              Upload text file
              <input type="file" accept=".txt,.md,.csv,.json,text/*" onChange={loadFile} />
            </label>
            <label>
              Text
              <textarea value={text} onChange={(event) => setText(event.target.value)} />
            </label>
            <button type="submit" disabled={loading}>
              <FileUp size={16} />
              Index document
            </button>
          </form>

          <form className="panel search-panel" onSubmit={runSearch}>
            <div className="panel-heading">
              <Search size={18} />
              <h3>Search</h3>
            </div>
            <label>
              Query
              <input value={query} onChange={(event) => setQuery(event.target.value)} />
            </label>
            <div className="mode-row" role="tablist" aria-label="Ranking mode">
              {MODES.map((item) => (
                <button
                  type="button"
                  className={mode === item.id ? "mode selected" : "mode"}
                  key={item.id}
                  onClick={() => setMode(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <label>
              Top K
              <input type="number" min="1" max="20" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
            </label>
            <button type="submit" disabled={loading}>
              <Search size={16} />
              Search corpus
            </button>
            {artifactStatus ? <p className="artifact">{artifactStatus}</p> : null}
          </form>
        </div>

        <section className="results">
          <div className="section-heading">
            <h3>Ranked results</h3>
            <span>{selectedMode?.label}</span>
          </div>
          {results.length === 0 ? (
            <div className="empty">Run a query to compare ranked snippets.</div>
          ) : (
            <div className="result-list">
              {results.map((result) => (
                <article className="result-card" key={result.chunk_id}>
                  <div className="rank">#{result.rank}</div>
                  <div>
                    <h4>{result.title}</h4>
                    <p>{result.snippet}</p>
                    <span>score {result.score}</span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="metrics">
          <div className="section-heading">
            <h3>Benchmark summary</h3>
            <span>from artifacts/metrics.json</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Recall@5</th>
                <th>P95 latency</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{row.recall_at_5.toFixed ? row.recall_at_5.toFixed(3) : row.recall_at_5}</td>
                  <td>{row.p95_latency_ms} ms</td>
                  <td>{formatSize(row.size_mb)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
