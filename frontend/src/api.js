const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function ask(query, sessionId = "demo") {
  const r = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json(); // { answer, citations, route, plan, trace, context, planner_notes }
}

export async function ingest(text) {
  const r = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(text)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function uploadFiles(files) {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const r = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getDbInfo() {
  const r = await fetch(`${API_BASE}/dbinfo`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function listDocs(page=1, pageSize=10) {
  const offset = (page-1)*pageSize;
  const r = await fetch(`${API_BASE}/docs?offset=${offset}&limit=${pageSize}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getModels() {
  const r = await fetch(`${API_BASE}/models`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getMemory(sessionId="demo") {
  const r = await fetch(`${API_BASE}/memory/${encodeURIComponent(sessionId)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function clearAll() {
  const r = await fetch(`${API_BASE}/clear`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function rerankDebug(q) {
  const r = await fetch(`${API_BASE}/rerank-debug?q=${encodeURIComponent(q)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
