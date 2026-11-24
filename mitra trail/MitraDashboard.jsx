import { useState, useEffect, useCallback, useRef } from "react";

// ─────────────────────────────────────────────────────────────
// CONFIG — set this to your Raspberry Pi's local IP address
// e.g.  const PI_IP = "192.168.1.42"
// Find it on Pi by running:  hostname -I
// ─────────────────────────────────────────────────────────────
const PI_IP   = "192.168.1.100";   // ← CHANGE THIS to your Pi's IP
const API     = `http://${PI_IP}:5050/api`;
const POLL_MS = 3000;              // refresh every 3 seconds

// ── MOCK DATA — used when Pi is unreachable (for preview) ───────────────
const MOCK_STATUS = {
  pi_online: true, child_name: "Arjun",
  session_start: new Date(Date.now() - 42 * 60000).toISOString(),
  distress_count: 1, distress_threshold: 3, internet_ok: true,
};
const MOCK_SESSION = {
  session_start: new Date(Date.now() - 42 * 60000).toISOString(),
  duration: "42m 18s", child_name: "Arjun", dominant_mood: "happy",
  emotion_counts: { happy: 4, sad: 1, excited: 3 },
  rhymes_played: ["Twinkle Twinkle Little Star", "Baa Baa Black Sheep"],
  languages_taught: ["Hindi", "French"],
  call_count: 0, auto_call_count: 0, distress_count: 1, total_events: 14,
};
const MOCK_EVENTS = [
  { id:14, time:"15:44", type:"emotion",   label:"😶 Emotion: Excited",          detail:"excited",  severity:"info"    },
  { id:13, time:"15:40", type:"language",  label:"🌍 Language: French",           detail:"French",   severity:"info"    },
  { id:12, time:"15:31", type:"distress",  label:"⚠️ Distress count: 1/3",        detail:"sad #1",   severity:"warning" },
  { id:11, time:"15:28", type:"emotion",   label:"😶 Emotion: Sad",               detail:"sad",      severity:"warning" },
  { id:10, time:"15:20", type:"rhyme",     label:"🎵 Rhyme played: Baa Baa Black Sheep", detail:"Baa Baa Black Sheep", severity:"info" },
  { id:9,  time:"15:11", type:"emotion",   label:"😶 Emotion: Happy",             detail:"happy",    severity:"info"    },
  { id:8,  time:"14:58", type:"language",  label:"🌍 Language: Hindi",            detail:"Hindi",    severity:"info"    },
  { id:7,  time:"14:44", type:"rhyme",     label:"🎵 Rhyme played: Twinkle Twinkle Little Star", detail:"Twinkle Twinkle Little Star", severity:"info" },
  { id:6,  time:"14:33", type:"emotion",   label:"😶 Emotion: Excited",           detail:"excited",  severity:"info"    },
  { id:5,  time:"14:22", type:"emotion",   label:"😶 Emotion: Excited",           detail:"excited",  severity:"info"    },
  { id:4,  time:"14:15", type:"emotion",   label:"😶 Emotion: Happy",             detail:"happy",    severity:"info"    },
  { id:3,  time:"14:08", type:"emotion",   label:"😶 Emotion: Happy",             detail:"happy",    severity:"info"    },
  { id:2,  time:"14:04", type:"emotion",   label:"😶 Emotion: Happy",             detail:"happy",    severity:"info"    },
  { id:1,  time:"14:02", type:"session_start", label:"🟢 Session started",        detail:"",         severity:"info"    },
];

// ── helpers ──────────────────────────────────────────────────────────────
function fmtTime(iso) {
  if (!iso || iso === "—") return "—";
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch { return "—"; }
}

function timeSince(iso) {
  if (!iso || iso === "—") return "—";
  try {
    const secs = Math.floor((Date.now() - new Date(iso)) / 1000);
    if (secs < 60)  return `${secs}s ago`;
    if (secs < 3600) return `${Math.floor(secs/60)}m ago`;
    return `${Math.floor(secs/3600)}h ago`;
  } catch { return "—"; }
}

const MOOD_EMOJI = {
  happy:"😊", excited:"🤩", neutral:"😐",
  sad:"😢", scared:"😨", lonely:"🥺", missing:"💔",
  bedtime:"😴", praise:"🌟", morning:"🌅",
};
const MOOD_COLOR = {
  happy:"#22c55e", excited:"#f59e0b", neutral:"#94a3b8",
  sad:"#60a5fa", scared:"#f97316", lonely:"#a78bfa",
  missing:"#f472b6", bedtime:"#818cf8", praise:"#fbbf24", morning:"#fb923c",
};
const SEV_STYLES = {
  alert:   { bg:"#3b0a0a", border:"#ef4444", dot:"#ef4444", label:"ALERT" },
  warning: { bg:"#2d1c04", border:"#f59e0b", dot:"#f59e0b", label:"WARN"  },
  info:    { bg:"#0f1923", border:"#1e3a5f", dot:"#3b82f6", label:""      },
};
const TYPE_ICON = {
  emotion:"🎭", rhyme:"🎵", language:"🌍", call:"📞",
  auto_call:"🚨", distress:"⚠️", session_start:"🟢", session_end:"🔴",
};

// ── Styles ────────────────────────────────────────────────────────────────
const S = {
  app: {
    minHeight:"100vh", background:"#070d14",
    fontFamily:"'DM Sans', 'Segoe UI', sans-serif",
    color:"#e2e8f0", padding:"0",
  },
  header: {
    background:"linear-gradient(135deg,#0d1f35 0%,#0a1628 100%)",
    borderBottom:"1px solid #1e3a5f",
    padding:"20px 28px", display:"flex", alignItems:"center",
    justifyContent:"space-between", flexWrap:"wrap", gap:"12px",
  },
  headerLeft: { display:"flex", alignItems:"center", gap:"14px" },
  logoRing: {
    width:"44px", height:"44px", borderRadius:"50%",
    background:"linear-gradient(135deg,#3b82f6,#8b5cf6)",
    display:"flex", alignItems:"center", justifyContent:"center",
    fontSize:"20px", boxShadow:"0 0 16px rgba(59,130,246,0.4)",
  },
  headerTitle: { margin:0, fontSize:"22px", fontWeight:"700",
    letterSpacing:"-0.5px", color:"#f1f5f9" },
  headerSub: { margin:0, fontSize:"13px", color:"#64748b", marginTop:"2px" },
  piStatus: {
    display:"flex", alignItems:"center", gap:"8px",
    background:"#0f1923", border:"1px solid #1e3a5f",
    borderRadius:"20px", padding:"6px 14px", fontSize:"13px",
  },
  dot: (on) => ({
    width:"8px", height:"8px", borderRadius:"50%",
    background: on ? "#22c55e" : "#ef4444",
    boxShadow: on ? "0 0 6px #22c55e" : "0 0 6px #ef4444",
    flexShrink:0,
  }),
  main: { padding:"20px 28px", maxWidth:"1200px", margin:"0 auto" },
  grid: { display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr",
    gap:"14px", marginBottom:"20px" },
  gridTablet: { gridTemplateColumns:"1fr 1fr" },
  statCard: {
    background:"#0d1f35", border:"1px solid #1e3a5f",
    borderRadius:"14px", padding:"18px 20px",
  },
  statLabel: { fontSize:"11px", fontWeight:"600", textTransform:"uppercase",
    letterSpacing:"1px", color:"#64748b", marginBottom:"6px" },
  statValue: { fontSize:"28px", fontWeight:"700", lineHeight:"1",
    color:"#f1f5f9", marginBottom:"4px" },
  statSub: { fontSize:"12px", color:"#475569" },
  twoCol: { display:"grid", gridTemplateColumns:"1fr 340px", gap:"20px",
    alignItems:"start" },
  panel: {
    background:"#0d1f35", border:"1px solid #1e3a5f",
    borderRadius:"14px", overflow:"hidden",
  },
  panelHead: {
    padding:"14px 18px", borderBottom:"1px solid #1e3a5f",
    display:"flex", alignItems:"center", justifyContent:"space-between",
  },
  panelTitle: { fontSize:"14px", fontWeight:"600", color:"#94a3b8",
    textTransform:"uppercase", letterSpacing:"0.8px", margin:0 },
  feedScroll: { maxHeight:"520px", overflowY:"auto" },
  feedItem: (sev) => ({
    padding:"12px 18px",
    borderBottom:"1px solid #0f1923",
    borderLeft:`3px solid ${SEV_STYLES[sev]?.border || "#1e3a5f"}`,
    background: SEV_STYLES[sev]?.bg || "transparent",
    display:"flex", gap:"12px", alignItems:"flex-start",
    transition:"background 0.15s",
  }),
  feedDot: (sev) => ({
    width:"8px", height:"8px", borderRadius:"50", marginTop:"5px",
    background: SEV_STYLES[sev]?.dot || "#3b82f6",
    flexShrink:0,
  }),
  feedLabel: { fontSize:"14px", color:"#e2e8f0", fontWeight:"500",
    lineHeight:"1.4" },
  feedTime: { fontSize:"11px", color:"#475569", marginTop:"3px" },
  feedBadge: (sev) => ({
    fontSize:"10px", fontWeight:"700", letterSpacing:"0.5px",
    padding:"2px 6px", borderRadius:"4px", marginLeft:"6px",
    background: sev==="alert" ? "#ef444420" : sev==="warning" ? "#f59e0b20" : "transparent",
    color: sev==="alert" ? "#ef4444" : sev==="warning" ? "#f59e0b" : "transparent",
    display: SEV_STYLES[sev]?.label ? "inline" : "none",
  }),
  wellnessCard: {
    background:"#0d1f35", border:"1px solid #1e3a5f",
    borderRadius:"14px", padding:"18px", marginBottom:"14px",
  },
  wLabel: { fontSize:"11px", fontWeight:"600", textTransform:"uppercase",
    letterSpacing:"1px", color:"#64748b", marginBottom:"10px" },
  moodRow: { display:"flex", gap:"8px", flexWrap:"wrap" },
  moodChip: (emotion, count) => ({
    display:"flex", alignItems:"center", gap:"5px",
    background:`${MOOD_COLOR[emotion] || "#3b82f6"}18`,
    border:`1px solid ${MOOD_COLOR[emotion] || "#3b82f6"}40`,
    borderRadius:"20px", padding:"4px 10px", fontSize:"13px",
    color: MOOD_COLOR[emotion] || "#3b82f6",
  }),
  tagRow: { display:"flex", gap:"6px", flexWrap:"wrap" },
  tag: (color) => ({
    background:`${color}15`, border:`1px solid ${color}35`,
    borderRadius:"6px", padding:"4px 10px", fontSize:"12px",
    color: color, fontWeight:"500",
  }),
  distressBar: {
    height:"8px", borderRadius:"4px", background:"#1e3a5f",
    overflow:"hidden", marginTop:"8px",
  },
  distressFill: (pct) => ({
    height:"100%", borderRadius:"4px",
    width:`${Math.min(pct * 100, 100)}%`,
    background: pct >= 1 ? "#ef4444" : pct >= 0.6 ? "#f59e0b" : "#3b82f6",
    transition:"width 0.4s ease",
  }),
  alertBanner: {
    background:"#3b0a0a", border:"1px solid #ef4444",
    borderRadius:"10px", padding:"12px 16px",
    display:"flex", gap:"10px", alignItems:"center",
    marginBottom:"14px", fontSize:"14px", color:"#fca5a5",
  },
  refreshBtn: {
    background:"#1e3a5f", border:"none", borderRadius:"8px",
    padding:"6px 12px", fontSize:"12px", color:"#94a3b8",
    cursor:"pointer", display:"flex", alignItems:"center", gap:"5px",
  },
  emptyFeed: { padding:"40px 18px", textAlign:"center",
    color:"#475569", fontSize:"14px" },
  spinner: {
    width:"18px", height:"18px", border:"2px solid #1e3a5f",
    borderTop:"2px solid #3b82f6", borderRadius:"50%",
    animation:"spin 0.8s linear infinite",
  },
  ipInput: {
    background:"#0f1923", border:"1px solid #1e3a5f", borderRadius:"8px",
    padding:"6px 12px", fontSize:"13px", color:"#e2e8f0",
    width:"160px", outline:"none",
  },
};

// ── Main component ────────────────────────────────────────────────────────
export default function MitraDashboard() {
  const [piIP, setPiIP]         = useState(PI_IP);
  const [editIP, setEditIP]     = useState(false);
  const [inputIP, setInputIP]   = useState(PI_IP);

  const [status,  setStatus]  = useState(null);
  const [session, setSession] = useState(null);
  const [events,  setEvents]  = useState([]);
  const [loading, setLoading] = useState(true);
  const [offline, setOffline] = useState(false);
  const [lastRefresh, setLastRefresh] = useState(null);

  const prevAlertCount = useRef(0);
  const audioCtx = useRef(null);

  const api = useCallback((path) => `http://${piIP}:5050/api${path}`, [piIP]);

  // Tiny chime for new alerts
  const chime = useCallback(() => {
    try {
      if (!audioCtx.current) audioCtx.current = new AudioContext();
      const ctx = audioCtx.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain); gain.connect(ctx.destination);
      osc.type = "sine"; osc.frequency.value = 880;
      gain.gain.setValueAtTime(0, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.05);
      gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.4);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.4);
    } catch {}
  }, []);

  const fetchAll = useCallback(async () => {
    try {
      const [s, se, ev] = await Promise.all([
        fetch(api("/status")).then(r => r.json()),
        fetch(api("/session")).then(r => r.json()),
        fetch(api("/events")).then(r => r.json()),
      ]);
      setStatus(s); setSession(se); setEvents(ev);
      setOffline(false); setLastRefresh(new Date());

      // Chime on new alert-severity events
      const alerts = ev.filter(e => e.severity === "alert").length;
      if (alerts > prevAlertCount.current) chime();
      prevAlertCount.current = alerts;
    } catch {
      setOffline(true);
      // Fall back to mock data so the dashboard is still useful as preview
      if (!status) {
        setStatus(MOCK_STATUS); setSession(MOCK_SESSION); setEvents(MOCK_EVENTS);
      }
    } finally { setLoading(false); }
  }, [api, chime, status]);

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, POLL_MS);
    return () => clearInterval(id);
  }, [fetchAll]);

  // Counts
  const alertCount   = events.filter(e => e.severity === "alert").length;
  const warnCount    = events.filter(e => e.severity === "warning").length;
  const rhymeCount   = (session?.rhymes_played || []).length;
  const langCount    = (session?.languages_taught || []).length;
  const distressPct  = status ? (status.distress_count / status.distress_threshold) : 0;

  return (
    <div style={S.app}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width:6px; }
        ::-webkit-scrollbar-track { background:#0d1f35; }
        ::-webkit-scrollbar-thumb { background:#1e3a5f; border-radius:3px; }
        @keyframes spin { to { transform:rotate(360deg); } }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        .feed-item:hover { background:#12253d !important; }
        .refresh-btn:hover { background:#243d5a !important; color:#e2e8f0 !important; }
        .stat-card { transition: border-color 0.2s; }
        .stat-card:hover { border-color:#2d5080 !important; }
      `}</style>

      {/* ── HEADER ── */}
      <header style={S.header}>
        <div style={S.headerLeft}>
          <div style={S.logoRing}>🤖</div>
          <div>
            <p style={S.headerTitle}>Mitra Parent Dashboard</p>
            <p style={S.headerSub}>
              {status?.child_name && status.child_name !== "—"
                ? `Watching over ${status.child_name} · Session started ${fmtTime(status?.session_start)}`
                : "Waiting for session…"}
            </p>
          </div>
        </div>

        <div style={{ display:"flex", gap:"10px", alignItems:"center", flexWrap:"wrap" }}>
          {/* Pi IP editor */}
          {editIP
            ? <form onSubmit={e => { e.preventDefault(); setPiIP(inputIP); setEditIP(false); }}
                style={{ display:"flex", gap:"6px" }}>
                <input value={inputIP} onChange={e => setInputIP(e.target.value)}
                  style={S.ipInput} placeholder="192.168.x.x" autoFocus />
                <button type="submit" style={{...S.refreshBtn, background:"#1e4d3a", color:"#4ade80"}}>✓</button>
                <button type="button" onClick={() => setEditIP(false)} style={S.refreshBtn}>✕</button>
              </form>
            : <button onClick={() => { setInputIP(piIP); setEditIP(true); }}
                style={S.refreshBtn} className="refresh-btn">
                🌐 {piIP}
              </button>}

          {/* Last refresh */}
          <span style={{ fontSize:"11px", color:"#475569" }}>
            {lastRefresh ? `Updated ${timeSince(lastRefresh?.toISOString())}` : "Connecting…"}
          </span>

          {loading
            ? <div style={S.spinner} />
            : <button onClick={fetchAll} style={S.refreshBtn} className="refresh-btn">↻ Refresh</button>}

          {/* Pi online pill */}
          <div style={S.piStatus}>
            <div style={S.dot(!offline)} />
            <span style={{ color: offline ? "#ef4444" : "#22c55e" }}>
              {offline ? "Pi Offline" : "Pi Online"}
            </span>
          </div>
        </div>
      </header>

      <main style={S.main}>

        {/* ── ALERT BANNER (only shows if there's a 🚨 auto-call or alert event) ── */}
        {alertCount > 0 && (
          <div style={S.alertBanner}>
            <span style={{ fontSize:"20px" }}>🚨</span>
            <div>
              <strong style={{ color:"#ef4444" }}>
                {alertCount} alert{alertCount > 1 ? "s" : ""} this session
              </strong>
              <span style={{ color:"#fca5a5", marginLeft:"8px" }}>
                — Auto-call was triggered. Check the feed below for details.
              </span>
            </div>
          </div>
        )}

        {/* ── STAT CARDS ── */}
        <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:"14px", marginBottom:"20px" }}>
          {[
            { label:"Session Duration", value: session?.duration || "—", sub: fmtTime(session?.session_start) + " start", icon:"⏱" },
            { label:"Alerts / Warnings", value: `${alertCount} / ${warnCount}`,
              sub: alertCount > 0 ? "action needed" : "all clear", icon:"🔔",
              accent: alertCount > 0 ? "#ef4444" : warnCount > 0 ? "#f59e0b" : "#22c55e" },
            { label:"Rhymes Played",  value: rhymeCount, sub: (session?.rhymes_played||[]).slice(-1)[0] || "none yet", icon:"🎵" },
            { label:"Languages",      value: langCount, sub: (session?.languages_taught||[]).join(", ") || "none yet", icon:"🌍" },
          ].map(c => (
            <div key={c.label} style={{ ...S.statCard, borderColor: c.accent ? `${c.accent}50` : "#1e3a5f" }}
              className="stat-card">
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
                <p style={S.statLabel}>{c.label}</p>
                <span style={{ fontSize:"18px", opacity:0.6 }}>{c.icon}</span>
              </div>
              <p style={{ ...S.statValue, color: c.accent || "#f1f5f9" }}>{c.value}</p>
              <p style={S.statSub}>{c.sub}</p>
            </div>
          ))}
        </div>

        {/* ── MAIN 2-COL LAYOUT ── */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 340px", gap:"20px", alignItems:"start" }}>

          {/* ── LEFT: ALERT + EVENT FEED ── */}
          <div style={S.panel}>
            <div style={S.panelHead}>
              <p style={S.panelTitle}>📋 Activity Feed</p>
              <span style={{ fontSize:"12px", color:"#475569" }}>
                {events.length} events · newest first
              </span>
            </div>
            <div style={S.feedScroll}>
              {events.length === 0
                ? <p style={S.emptyFeed}>No events yet — waiting for Mitra to start a session…</p>
                : events.map(evt => {
                    const sev = evt.severity || "info";
                    return (
                      <div key={evt.id} style={S.feedItem(sev)} className="feed-item">
                        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:"4px", flexShrink:0 }}>
                          <div style={S.feedDot(sev)} />
                          <span style={{ fontSize:"16px", lineHeight:1 }}>{TYPE_ICON[evt.type] || "•"}</span>
                        </div>
                        <div style={{ flex:1, minWidth:0 }}>
                          <div style={{ display:"flex", alignItems:"baseline", gap:"4px", flexWrap:"wrap" }}>
                            <span style={S.feedLabel}>{evt.label}</span>
                            {SEV_STYLES[sev]?.label &&
                              <span style={S.feedBadge(sev)}>{SEV_STYLES[sev].label}</span>}
                          </div>
                          {evt.detail && evt.detail !== evt.label &&
                            !["rhyme","language","emotion"].includes(evt.type) && (
                            <p style={{ ...S.feedTime, fontSize:"12px", color:"#64748b" }}>
                              {evt.detail}
                            </p>
                          )}
                          <p style={S.feedTime}>{evt.time}</p>
                        </div>
                      </div>
                    );
                  })}
            </div>
          </div>

          {/* ── RIGHT: WELLNESS SIDEBAR ── */}
          <div>
            {/* Distress tracker */}
            <div style={S.wellnessCard}>
              <p style={S.wLabel}>⚠️ Distress Tracker</p>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:"4px" }}>
                <span style={{ fontSize:"13px", color:"#94a3b8" }}>
                  Sad / scared triggers this session
                </span>
                <span style={{ fontSize:"18px", fontWeight:"700",
                  color: distressPct >= 1 ? "#ef4444" : distressPct >= 0.6 ? "#f59e0b" : "#3b82f6" }}>
                  {status?.distress_count ?? 0} / {status?.distress_threshold ?? 3}
                </span>
              </div>
              <div style={S.distressBar}>
                <div style={S.distressFill(distressPct)} />
              </div>
              <p style={{ fontSize:"11px", color:"#475569", marginTop:"6px" }}>
                {distressPct >= 1
                  ? "🚨 Threshold reached — auto-call triggered"
                  : distressPct >= 0.6
                  ? "⚠️ Getting close — keep an eye on it"
                  : "✅ All good — within normal range"}
              </p>
            </div>

            {/* Mood breakdown */}
            <div style={S.wellnessCard}>
              <p style={S.wLabel}>🎭 Mood This Session</p>
              {Object.keys(session?.emotion_counts || {}).length === 0
                ? <p style={{ fontSize:"13px", color:"#475569" }}>No emotions detected yet</p>
                : <>
                    <div style={{ ...S.moodRow, marginBottom:"12px" }}>
                      {Object.entries(session?.emotion_counts || {})
                        .sort((a,b) => b[1]-a[1])
                        .map(([emotion, count]) => (
                          <div key={emotion} style={S.moodChip(emotion, count)}>
                            <span>{MOOD_EMOJI[emotion] || "😶"}</span>
                            <span style={{ fontWeight:"600" }}>{emotion}</span>
                            <span style={{ opacity:0.7, fontSize:"11px" }}>×{count}</span>
                          </div>
                        ))}
                    </div>
                    <div style={{ fontSize:"12px", color:"#64748b" }}>
                      Dominant mood:{" "}
                      <strong style={{ color: MOOD_COLOR[session?.dominant_mood] || "#f1f5f9" }}>
                        {MOOD_EMOJI[session?.dominant_mood]} {session?.dominant_mood}
                      </strong>
                    </div>
                  </>}
            </div>

            {/* Rhymes */}
            <div style={S.wellnessCard}>
              <p style={S.wLabel}>🎵 Rhymes Played</p>
              {(session?.rhymes_played || []).length === 0
                ? <p style={{ fontSize:"13px", color:"#475569" }}>No rhymes yet</p>
                : <div style={S.tagRow}>
                    {session.rhymes_played.map((r,i) => (
                      <span key={i} style={S.tag("#a78bfa")}>{r}</span>
                    ))}
                  </div>}
            </div>

            {/* Languages */}
            <div style={S.wellnessCard}>
              <p style={S.wLabel}>🌍 Languages Explored</p>
              {(session?.languages_taught || []).length === 0
                ? <p style={{ fontSize:"13px", color:"#475569" }}>No languages yet</p>
                : <div style={S.tagRow}>
                    {session.languages_taught.map((l,i) => (
                      <span key={i} style={S.tag("#34d399")}>{l}</span>
                    ))}
                  </div>}
            </div>

            {/* Call log */}
            <div style={S.wellnessCard}>
              <p style={S.wLabel}>📞 Call Summary</p>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"8px" }}>
                {[
                  { label:"Manual calls", value: session?.call_count ?? 0, color:"#60a5fa" },
                  { label:"Auto calls",   value: session?.auto_call_count ?? 0,
                    color: (session?.auto_call_count ?? 0) > 0 ? "#ef4444" : "#94a3b8" },
                ].map(c => (
                  <div key={c.label} style={{ background:"#0f1923", borderRadius:"8px",
                    padding:"10px 12px", border:"1px solid #1e3a5f" }}>
                    <p style={{ ...S.statLabel, marginBottom:"4px" }}>{c.label}</p>
                    <p style={{ fontSize:"22px", fontWeight:"700", color:c.color, margin:0 }}>
                      {c.value}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* How to use hint */}
            <div style={{ background:"#080f1a", border:"1px solid #1e3a5f",
              borderRadius:"10px", padding:"14px 16px", fontSize:"12px", color:"#475569",
              lineHeight:"1.6" }}>
              <strong style={{ color:"#64748b", display:"block", marginBottom:"4px" }}>
                💡 How this works
              </strong>
              Dashboard auto-refreshes every 3s from your Pi at{" "}
              <code style={{ color:"#3b82f6" }}>{piIP}:5050</code>.
              Change the IP above if needed. All data stays on your home WiFi — nothing reaches the internet.
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
