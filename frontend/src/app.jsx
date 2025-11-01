import React, { useEffect, useRef, useState } from "react"
import {
  ask, ingest, uploadFiles, getDbInfo, listDocs,
  getModels, getMemory, clearAll, rerankDebug
} from "./api"

/** SIDEBAR */
function Sidebar({ tab, setTab }) {
  const items = [
    { key:"chat",     label:"Chat", icon:"💬" },
    { key:"upload",   label:"Upload", icon:"📤" },
    { key:"docs",     label:"Documents", icon:"📚" },
    { key:"db",       label:"DB Info", icon:"🗄️" },
    { key:"rerank",   label:"Rerank Debug", icon:"🔎" },
    { key:"memory",   label:"Memory", icon:"🧠" },
    { key:"settings", label:"Settings", icon:"⚙️" },
  ]
  return (
    <aside className="sidebar">
      <div className="brand">Agentic RAG</div>
      <div className="muted" style={{fontSize:12}}>Dashboard</div>
      <nav className="nav" style={{display:"grid", gap:6, marginTop:8}}>
        {items.map(it => (
          <a key={it.key} onClick={()=>setTab(it.key)}
             className={tab===it.key ? "active" : ""} style={{cursor:"pointer"}}>
            <span>{it.icon}</span><span>{it.label}</span>
          </a>
        ))}
      </nav>
      <div style={{marginTop:"auto"}} className="muted">
        <div className="card" style={{fontSize:12}}>
          <div style={{fontWeight:600, color:"var(--white)"}}>Tips</div>
          Upload PDFs/DOCX, then ask:
          <ul><li>“Summarize the docs I uploaded”</li><li>“Show last 5 memories (SQL)”</li></ul>
        </div>
      </div>
    </aside>
  )
}

/** TOPBAR */
function Topbar({ right=null }) {
  return (
    <div className="topbar">
      <div className="muted">Build your perfect Agentic RAG</div>
      <div>{right}</div>
    </div>
  )
}

/* CHAT PAGE — ChatGPT-style layout */
function ChatPage() {
  const [q, setQ] = useState("")
  const [busy, setBusy] = useState(false)
  const [log, setLog] = useState([])

  const endRef = useRef(null)

  async function onAsk(e){
    e.preventDefault()
    if(!q.trim()) return
    setBusy(true)
    try{
      const res = await ask(q)
      setLog(l => [...l, { role:"user", text:q }, { role:"assistant", ...res }])
      setQ("")
    }catch(e){ alert(e.message) }
    finally{ setBusy(false) }
  }

  useEffect(()=>{ endRef.current?.scrollIntoView({ behavior:"smooth" }) }, [log])

  return (
    <div className="content" style={{
      display:"flex",
      flexDirection:"column",
      height:"100%",
      background:"var(--bg)"
    }}>
      <Topbar right={<span className="badge">Chat</span>} />

      {/* chat messages area */}
      <div style={{
        flex:1,
        overflowY:"auto",
        padding:"20px 16px",
        display:"flex",
        flexDirection:"column",
        gap:"12px"
      }}>
        {log.map((m,i)=>(
          <div key={i} style={{
            alignSelf: m.role==="user" ? "flex-end" : "flex-start",
            maxWidth:"80%",
            background: m.role==="user" ? "var(--accent)" : "var(--card)",
            color: m.role==="user" ? "#fff" : "var(--white)",
            padding:"12px 16px",
            borderRadius:"14px",
            boxShadow:"0 1px 3px rgba(0,0,0,0.1)"
          }}>
            {m.role==="assistant" && (
              <div style={{fontSize:12,marginBottom:4,color:"var(--muted)"}}>
                {m.route ? `🧠 ${m.route}` : "Assistant"}
              </div>
            )}
            <div style={{whiteSpace:"pre-wrap",lineHeight:1.5}}>
              {m.answer || m.text}
            </div>
            {m.citations?.length ? (
              <details style={{marginTop:6,fontSize:12}}>
                <summary style={{cursor:"pointer",color:"var(--muted)"}}>Sources</summary>
                <ul style={{margin:0,paddingLeft:16}}>
                  {m.citations.map((c,idx)=>(
                    <li key={idx} style={{color:"var(--muted)"}}>
                      <code>{c.doc_id}</code>
                      {c.meta?.source ? ` — ${c.meta.source}` : ""}
                    </li>
                  ))}
                </ul>
              </details>
            ):null}
          </div>
        ))}
        <div ref={endRef}/>
      </div>

      {/* bottom chat input bar */}
      <form onSubmit={onAsk}
        style={{
          display:"flex",
          gap:"10px",
          padding:"14px",
          borderTop:"1px solid var(--border)",
          background:"var(--panel)",
          position:"sticky",
          bottom:0
        }}>
        <input
          value={q}
          onChange={e=>setQ(e.target.value)}
          placeholder="Send a message..."
          style={{
            flex:1,
            border:"1px solid var(--border)",
            borderRadius:"12px",
            padding:"12px",
            fontSize:"15px",
            background:"#fff",
            color:"#111"
          }}
        />
        <button className="btn primary" disabled={busy}>
          {busy ? "Thinking..." : "Send"}
        </button>
      </form>
    </div>
  )
}


/** UPLOAD PAGE */
function UploadPage(){
  const ref = useRef(null)
  const [last, setLast] = useState(null)

  async function onUpload(){
    const files = ref.current?.files
    if(!files || !files.length) return alert("Choose files first.")
    try{
      const res = await uploadFiles(files)
      setLast(res)
      alert(`Uploaded: ${res.uploaded}, Ingested: ${res.ingested}`)
      ref.current.value = ""
    }catch(e){ alert(e.message) }
  }

  return (
    <div className="content">
      <Topbar right={<span className="badge">Upload</span>} />
      <div style={{padding:18}} className="grid cols-2">
        <div className="panel">
          <h3 style={{marginTop:0,color:"var(--white)"}}>Upload & Ingest</h3>
          <p className="muted">TXT, MD, CSV, LOG, PDF, DOCX</p>
          <input ref={ref} multiple type="file" accept=".txt,.md,.csv,.log,.pdf,.docx"/>
          <div style={{marginTop:10,display:"flex",gap:8}}>
            <button className="btn primary" onClick={onUpload}>Upload & Ingest</button>
            <button className="btn" onClick={async ()=>{
              const text = prompt("Quick ingest text:")
              if(text){ await ingest(text); alert("Ingested!") }
            }}>Quick Ingest</button>
          </div>
        </div>

        <div className="panel">
          <h3 style={{marginTop:0,color:"var(--white)"}}>Last Upload</h3>
          <pre style={{whiteSpace:"pre-wrap",margin:0}} className="muted">
            {last ? JSON.stringify(last,null,2) : "No uploads yet."}
          </pre>
        </div>
      </div>
    </div>
  )
}

/** DOCUMENTS PAGE */
function DocsPage(){
  const [page,setPage] = useState(1)
  const [size,setSize] = useState(10)
  const [data,setData] = useState({total:0,items:[]})

  async function load(){
    const res = await listDocs(page, size)
    setData(res)
  }
  useEffect(()=>{ load() }, [page,size])

  const totalPages = Math.max(1, Math.ceil(data.total / size))

  return (
    <div className="content">
      <Topbar right={<span className="badge">Documents</span>} />
      <div style={{padding:18}} className="grid">
        <div className="panel">
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
            <h3 style={{margin:0,color:"var(--white)"}}>Documents</h3>
            <div className="muted">Total: {data.total}</div>
          </div>

          <div style={{display:"flex",gap:8,alignItems:"center",margin:"10px 0"}}>
            <span className="muted">Page</span>
            <button className="btn" onClick={()=>setPage(p=>Math.max(1,p-1))}>Prev</button>
            <span className="badge">{page}/{Math.max(1,totalPages)}</span>
            <button className="btn" onClick={()=>setPage(p=>Math.min(totalPages,p+1))}>Next</button>
            <span className="muted" style={{marginLeft:10}}>Page size</span>
            <select value={size} onChange={e=>{setPage(1);setSize(parseInt(e.target.value))}}>
              {[5,10,20,50].map(n=> <option key={n} value={n}>{n}</option>)}
            </select>
          </div>

          <table>
            <thead>
              <tr><th style={{width:220}}>Doc ID</th><th>Source</th><th>Snippet</th></tr>
            </thead>
            <tbody>
              {data.items.map(it=>(
                <tr key={it.id}>
                  <td><code>{it.id}</code></td>
                  <td>{it.source || <span className="muted">unknown</span>}</td>
                  <td className="muted">{it.snippet}</td>
                </tr>
              ))}
            </tbody>
          </table>

        </div>
      </div>
    </div>
  )
}

/** DB INFO */
function DbInfoPage(){
  const [info,setInfo] = useState(null)
  useEffect(()=>{ (async ()=>{ setInfo(await getDbInfo()) })() },[])
  return (
    <div className="content">
      <Topbar right={<span className="badge">DB Info</span>} />
      <div style={{padding:18}} className="grid cols-2">
        <div className="panel">
          <h3 style={{marginTop:0,color:"var(--white)"}}>Vector DB</h3>
          {info ? (
            <div className="grid">
              <div className="card"><div className="muted">Name</div><div>{info.vector_db.name}</div></div>
              <div className="card"><div className="muted">Collection</div><div>{info.vector_db.collection}</div></div>
              <div className="card"><div className="muted">Path</div><div><code>{info.vector_db.path}</code></div></div>
              <div className="card"><div className="muted">Docs</div><div className="pill">{info.vector_db.doc_count ?? "?"}</div></div>
            </div>
          ): <div className="muted">Loading…</div>}
        </div>

        <div className="panel">
          <h3 style={{marginTop:0,color:"var(--white)"}}>Sparse DB</h3>
          {info ? (
            <div className="grid">
              <div className="card"><div className="muted">Name</div><div>{info.sparse_db.name}</div></div>
              <div className="card"><div className="muted">Tables</div><div>{info.sparse_db.tables.join(", ")}</div></div>
              <div className="card"><div className="muted">Path</div><div><code>{info.sparse_db.path}</code></div></div>
              <div className="card"><div className="muted">Docs</div><div className="pill">{info.sparse_db.doc_count ?? "?"}</div></div>
            </div>
          ): <div className="muted">Loading…</div>}
        </div>
      </div>
    </div>
  )
}

/** RERANK DEBUG */
function RerankPage(){
  const [q,setQ] = useState("")
  const [data,setData] = useState(null)
  return (
    <div className="content">
      <Topbar right={<span className="badge">Rerank Debug</span>} />
      <div style={{padding:18}} className="grid">
        <div className="panel">
          <form onSubmit={(e)=>{e.preventDefault(); (async()=>setData(await rerankDebug(q)))()}} style={{display:"flex",gap:10}}>
            <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Type a query to debug ranking…" style={{flex:1}}/>
            <button className="btn primary">Run</button>
          </form>
        </div>
        {data && (
          <div className="grid cols-2">
            <div className="panel">
              <h3 style={{marginTop:0,color:"var(--white)"}}>Dense</h3>
              <ul className="muted">{data.dense.map((d,i)=><li key={i}><code>{d.id}</code> — {d.src || "?"}<br/>{d.text}</li>)}</ul>
            </div>
            <div className="panel">
              <h3 style={{marginTop:0,color:"var(--white)"}}>Fused (RRF)</h3>
              <ul className="muted">{data.fused.map((d,i)=><li key={i}><code>{d.id}</code> — {d.src || "?"} — RRF {d.rrf?.toFixed(4)}<br/>{d.text}</li>)}</ul>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/** MEMORY */
function MemoryPage(){
  const [session,setSession] = useState("demo")
  const [rows,setRows] = useState([])
  async function load(){ const res = await getMemory(session); setRows(res.rows||[]) }
  useEffect(()=>{ load() }, [])
  return (
    <div className="content">
      <Topbar right={<span className="badge">Memory</span>} />
      <div style={{padding:18}} className="grid">
        <div className="panel" style={{display:"flex",gap:10}}>
          <input value={session} onChange={e=>setSession(e.target.value)} placeholder="session id" style={{flex:1}}/>
          <button className="btn" onClick={load}>Load</button>
        </div>
        <div className="panel">
          <table>
            <thead><tr><th style={{width:180}}>Time</th><th>User</th><th>Assistant (first 400 chars)</th></tr></thead>
            <tbody>
              {rows.map((r,i)=>(
                <tr key={i}>
                  <td className="muted">{r[0]}</td>
                  <td>{r[1]}</td>
                  <td className="muted" style={{maxWidth:700, overflow:"hidden", textOverflow:"ellipsis"}}>{r[2]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length && <div className="muted">No rows</div>}
        </div>
      </div>
    </div>
  )
}

/** SETTINGS */
function SettingsPage(){
  const [models,setModels] = useState(null)
  useEffect(()=>{ (async()=> setModels(await getModels()))() }, [])
  return (
    <div className="content">
      <Topbar right={
        <div>
          <button className="btn" onClick={async()=>{
            if(!confirm("This will clear vector store + sqlite docs + memories")) return
            const res = await clearAll(); alert(res.cleared ? "Cleared" : "Failed")
          }}>Clear All</button>
          <span style={{marginLeft:10}} className="badge">Settings</span>
        </div>
      } />
      <div style={{padding:18}} className="grid cols-2">
        <div className="panel">
          <h3 style={{marginTop:0,color:"var(--white)"}}>Active Models</h3>
          {models ? (
            <div className="grid">
              {Object.entries(models.models || {}).map(([k,v])=>(
                <div key={k} className="card">
                  <div className="muted">{k.toUpperCase()}</div>
                  <div>{v}</div>
                </div>
              ))}
            </div>
          ) : <div className="muted">Loading…</div>}
          <div className="muted" style={{marginTop:10}}>Change via backend env: <code>LLM_RAG</code>, <code>LLM_CODE</code>, <code>LLM_SQL</code>.</div>
        </div>
        <div className="panel">
          <h3 style={{marginTop:0,color:"var(--white)"}}>About</h3>
          <div className="muted">Hybrid search (Chroma dense + BM25 sparse) with Reciprocal Rank Fusion. File uploads feed both stores.</div>
        </div>
      </div>
    </div>
  )
}

export default function App(){
  const [tab,setTab] = useState("chat")
  return (
    <div className="app">
      <Sidebar tab={tab} setTab={setTab}/>
      {tab==="chat" && <ChatPage/>}
      {tab==="upload" && <UploadPage/>}
      {tab==="docs" && <DocsPage/>}
      {tab==="db" && <DbInfoPage/>}
      {tab==="rerank" && <RerankPage/>}
      {tab==="memory" && <MemoryPage/>}
      {tab==="settings" && <SettingsPage/>}
    </div>
  )
}
