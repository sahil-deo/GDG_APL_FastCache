"""
FastCache – Streamlit UI  (light-mode, editorial precision)
Run:  streamlit run app.py
"""
from __future__ import annotations
import sys, time, hashlib
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from fastcache import SemanticCache
from fastcache.config import CacheConfig
from fastcache.embedders.gemini import GeminiEmbedder
import urllib.request
import json

class GeminiAPIError(Exception):
    pass

QUERY_GROUPS = [
    {
        "topic": "Python Basics",
        "queries": [
            "What is Python?",
            "Explain Python programming.",
            "Can you tell me about the Python language?",
            "How do I define a variable in Python?"
        ]
    },
    {
        "topic": "Machine Learning",
        "queries": [
            "What is Machine Learning?",
            "Explain ML briefly.",
            "Define machine learning.",
            "What is the difference between AI and ML?"
        ]
    },
    {
        "topic": "REST APIs",
        "queries": [
            "What is a REST API?",
            "How do REST APIs work?",
            "Explain RESTful architecture.",
            "Give me an example of a REST endpoint."
        ]
    }
]

class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                tokens = len(prompt.split()) + len(text.split())
                cost = tokens * 0.000001
                llm_ms = (time.perf_counter() - t0) * 1000
                return text, tokens, llm_ms, cost
        except Exception as e:
            raise GeminiAPIError(str(e))

class DemoCache(SemanticCache):
    @property
    def threshold(self): return self.config.threshold
    @threshold.setter
    def threshold(self, value): self.config.threshold = value
    @property
    def hit_rate(self): return self.stats.hit_rate
    @property
    def total_queries(self): return self.stats.total_queries
    @property
    def cache_hits(self): return self.stats.cache_hits
    @property
    def avg_latency_ms(self): return self.stats.avg_total_latency_ms
    @property
    def avg_llm_latency_ms(self): return self.stats.avg_miss_latency_ms
    @property
    def avg_cache_latency_ms(self): return self.stats.avg_hit_latency_ms
    @property
    def tokens_saved(self): return self.stats.tokens_saved
    @property
    def size(self): return self.stats.total_entries
    @property
    def entries(self): 
        from fastcache.stores.memory import InMemoryStore
        if isinstance(self.store, InMemoryStore):
            class UIEntry:
                def __init__(self, e):
                    self.query = e.query
                    self.response = e.response
                    self.hits = e.hit_count
                    self.vector = e.vector
            return [UIEntry(e) for ns in self.store._entries.values() for e in ns]
        return []

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FastCache",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS – clean light + editorial type
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Geist', -apple-system, sans-serif !important;
    color: #EAEAEA !important;
}

/* ── Canvas ── */
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at 50% -10%, #1a1a24 0%, #0a0a0c 100%) !important; }
[data-testid="stHeader"]           { background: transparent !important; display:none; }
[data-testid="stSidebar"]          {
    background: rgba(15, 15, 15, 0.3) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1280px !important; }

/* ── Topbar ── */
.fc-topbar {
    background: rgba(15, 15, 15, 0.5);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 0 32px;
    height: 52px;
    display: flex;
    align-items: center;
    gap: 20px;
    position: sticky;
    top: 0;
    z-index: 100;
    margin: 0 -3rem 2rem;
}
.fc-wordmark {
    font-family: 'Geist Mono', monospace;
    font-size: 13px; font-weight: 500;
    color: #FFF;
    letter-spacing: -0.01em; display: flex; align-items: center; gap: 7px;
}
.fc-dot {
    width: 6px; height: 6px; background: #2DD4BF; border-radius: 50%;
    box-shadow: 0 0 8px #2DD4BF;
    animation: fc-blink 2.4s ease-in-out infinite;
}
@keyframes fc-blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.fc-tb-divider { width:1px; height:18px; background:rgba(255,255,255,0.1); }
.fc-tb-pill {
    font-family: 'Geist Mono', monospace; font-size: 10px; color: #AAA;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 4px; padding: 2px 8px; letter-spacing: 0.04em;
}
.fc-tb-pill.active { background: rgba(45,212,191,0.1); border-color: rgba(45,212,191,0.3); color: #2DD4BF; }
.fc-status { margin-left: auto; }

/* ── Sidebar ── */
.fc-sidebar-label {
    font-size: 10px; font-weight: 600; color: #888;
    letter-spacing: 0.1em; text-transform: uppercase; margin: 20px 0 6px;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 13px !important; font-weight: 600 !important; color: #FFF !important;
    letter-spacing: 0 !important; text-transform: none !important; margin: 0 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.02) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
    transition: transform 0.2s, border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: rgba(255,255,255,0.15) !important; transform: translateY(-1px); }
[data-testid="metric-container"] label {
    font-family: 'Geist Mono', monospace !important; font-size: 10px !important;
    color: #888 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 600 !important; color: #FFF !important; letter-spacing: -0.02em !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid rgba(255,255,255,0.1) !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] {
    font-size: 12px !important; font-weight: 500 !important; color: #888 !important;
    padding: 8px 18px !important; background: transparent !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #FFF !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #FFF !important; border-bottom-color: #2DD4BF !important; }

/* ── Inputs & buttons ── */
[data-testid="stTextInput"] input, [data-testid="stTextInput"] input:focus {
    background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important; font-family: 'Geist Mono', monospace !important;
    font-size: 13px !important; color: #FFF !important; padding: 9px 12px !important;
}
[data-testid="stTextInput"] input:focus { border-color: #2DD4BF !important; box-shadow: 0 0 0 1px rgba(45,212,191,0.5) !important; }

.stButton > button {
    border-radius: 6px !important; font-family: 'Geist', sans-serif !important;
    font-size: 12px !important; font-weight: 500 !important; padding: 8px 16px !important;
    background: rgba(255,255,255,0.05) !important; color: #FFF !important; border: 1px solid rgba(255,255,255,0.1) !important;
}
.stButton > button:hover { background: rgba(255,255,255,0.1) !important; }
.stButton > button[kind="primary"] { background: #2DD4BF !important; color: #000 !important; border-color: #2DD4BF !important; font-weight: 600 !important; }
.stButton > button[kind="primary"]:hover { background: #14B8A6 !important; box-shadow: 0 0 12px rgba(45,212,191,0.4) !important; }
[data-testid="stSelectbox"] > div > div { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.1) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #FFF !important; }

/* ── Chat bubbles ── */
.fc-messages { display:flex; flex-direction:column; gap:20px; margin-bottom:24px; }
.fc-row { display:flex; gap:10px; align-items:flex-start; }
.fc-row.user { flex-direction:row-reverse; }

.fc-avatar {
    width: 28px; height: 28px; flex-shrink: 0; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Geist Mono', monospace; font-size: 9px; font-weight: 600; margin-top: 2px;
}
.fc-avatar.user { background:#2DD4BF; color:#000; box-shadow: 0 0 10px rgba(45,212,191,0.2); }
.fc-avatar.bot  { background:#FFF; color:#000; box-shadow: 0 0 10px rgba(255,255,255,0.1); }

.fc-bubble-wrap { display:flex; flex-direction:column; gap:4px; max-width:72%; }
.fc-bubble { padding: 12px 16px; border-radius: 8px; font-size: 13px; line-height: 1.6; border: 1px solid transparent; }
.fc-bubble.user { background: rgba(45,212,191,0.15); border-color: rgba(45,212,191,0.3); color: #2DD4BF; font-family: 'Geist Mono', monospace; font-size: 12px; border-top-right-radius: 2px; }
.fc-bubble.bot { background: rgba(255,255,255,0.03); backdrop-filter: blur(10px); border-color: rgba(255,255,255,0.08); color: #EAEAEA; border-top-left-radius: 2px; }

/* result metadata strip */
.fc-result-meta { display: flex; align-items: center; gap: 8px; font-family: 'Geist Mono', monospace; font-size: 10px; }
.fc-tag { padding: 2px 8px; border-radius: 4px; font-size: 9px; font-weight: 600; letter-spacing: 0.08em; border: 1px solid; }
.fc-tag.hit  { color:#2DD4BF; border-color:rgba(45,212,191,0.3); background:rgba(45,212,191,0.1); }
.fc-tag.miss { color:#F59E0B; border-color:rgba(245,158,11,0.3); background:rgba(245,158,11,0.1); }
.fc-tag.exact{ color:#818CF8; border-color:rgba(129,140,248,0.3); background:rgba(129,140,248,0.1); }

.fc-meta-stat { color:#888; font-size:10px; }
.fc-meta-stat .v { color:#CCC; }

/* latency bar */
.fc-lat-bar { height: 2px; background:rgba(255,255,255,0.05); border-radius:1px; margin-top:8px; max-width:280px; }
.fc-lat-fill { height:100%; border-radius:1px; }
.fc-lat-fill.fast { background:#2DD4BF; }
.fc-lat-fill.mid  { background:#F59E0B; }
.fc-lat-fill.slow { background:#EF4444; }

/* ── LOADING bubble ── */
.fc-loading-bubble {
    background: rgba(255,255,255,0.03); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; border-bottom-left-radius: 2px;
    padding: 14px 16px; max-width: 380px; display: flex; flex-direction: column; gap: 10px;
}
.fc-pipeline { display: flex; align-items: center; gap: 0; font-family: 'Geist Mono', monospace; font-size: 10px; flex-wrap: wrap; row-gap: 4px; }
.fc-pipe-step { color: #555; display:flex; align-items:center; gap:4px; transition:color .3s; }
.fc-pipe-step.done   { color: #2DD4BF; }
.fc-pipe-step.active { color: #F59E0B; }
.fc-pipe-arrow { color: #444; margin: 0 5px; }
.fc-pipe-step.done + .fc-pipe-arrow   { color: rgba(45,212,191,0.5); }
.fc-pipe-step.active + .fc-pipe-arrow { color: rgba(245,158,11,0.5); }

.fc-dots span {
    display: inline-block; width: 5px; height: 5px; background: #444; border-radius: 50%;
    margin-right: 4px; animation: fc-wave 1.1s ease-in-out infinite;
}
.fc-dots span:nth-child(1) { animation-delay: 0s; }
.fc-dots span:nth-child(2) { animation-delay: .15s; }
.fc-dots span:nth-child(3) { animation-delay: .30s; }
@keyframes fc-wave {
    0%,60%,100%{ transform:translateY(0); background:#444; }
    30%        { transform:translateY(-4px); background:#F59E0B; }
}
.fc-loading-label { font-family: 'Geist Mono', monospace; font-size: 10px; color: #888; display: flex; align-items: center; gap: 8px; }
.fc-loading-label .elapsed { color: #555; }
.fc-loading-stage-badge {
    font-family: 'Geist Mono', monospace; font-size: 9px; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #F59E0B; border: 1px solid rgba(245,158,11,0.3);
    background: rgba(245,158,11,0.1); border-radius: 4px; padding: 2px 8px;
    animation: fc-pulse .9s ease-in-out infinite;
}
@keyframes fc-pulse { 0%,100%{opacity:1} 50%{opacity:.45} }

/* ── PROCESSING / LOGS ── */
.fc-processing-bar {
    background: rgba(245,158,11,0.05); border: 1px solid rgba(245,158,11,0.2);
    border-radius: 6px; padding: 10px 14px; font-family: 'Geist Mono', monospace;
    font-size: 11px; color: #FCD34D; margin-bottom: 12px;
}

/* ── Example buttons ── */
.fc-example-btn {
    display: inline-block; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px; padding: 5px 14px; font-size: 12px; color: #CCC;
    cursor: pointer; margin: 0 4px 6px 0; transition: all .15s; font-family: 'Geist', sans-serif;
}
.fc-example-btn:hover { background: rgba(255,255,255,0.1); border-color:rgba(255,255,255,0.2); color:#FFF; }

/* ── Benchmark log ── */
.fc-bench-log {
    font-family: 'Geist Mono', monospace; font-size: 11px;
    background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px; padding: 16px 18px; max-height: 380px; overflow-y: auto;
    line-height: 1.9; color: #AAA;
}
.fc-bench-log .log-hit  { color: #2DD4BF; }
.fc-bench-log .log-miss { color: #F59E0B; }
.fc-bench-log .log-head { color: #FFF; font-weight: 600; }
.fc-bench-log::-webkit-scrollbar { width: 4px; }
.fc-bench-log::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

/* ── Cache table ── */
.fc-cache-table { width:100%; border-collapse:collapse; font-size:12px; }
.fc-cache-table thead th {
    background: rgba(0,0,0,0.2); border-bottom: 1px solid rgba(255,255,255,0.08);
    font-family: 'Geist Mono', monospace; font-size: 10px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: .08em; padding: 8px 12px; text-align: left;
}
.fc-cache-table tbody td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #CCC; vertical-align: top; }
.fc-cache-table tbody tr:hover td { background: rgba(255,255,255,0.02); }
.fc-cache-table .q-text { font-weight:500; color:#FFF; margin-bottom:2px; }
.fc-cache-table .r-text { color:#888; font-size:11px; }
.fc-hits-badge {
    font-family:'Geist Mono',monospace; font-size:11px; font-weight:600;
    display:inline-block; padding:2px 8px; border-radius:4px; border:1px solid;
}
.fc-hits-badge.zero { color:#888; border-color:rgba(255,255,255,0.1); background:rgba(255,255,255,0.05); }
.fc-hits-badge.some { color:#2DD4BF; border-color:rgba(45,212,191,0.3); background:rgba(45,212,191,0.1); }

/* ── Similarity matrix ── */
.fc-matrix-wrap { overflow-x:auto; }
.fc-matrix { border-collapse:collapse; font-family:'Geist Mono',monospace; font-size:10px; }
.fc-matrix th {
    padding:4px 2px; color:#888; font-weight:500; font-size:9px;
    max-width:40px; overflow:hidden; white-space:nowrap;
    writing-mode:vertical-rl; transform:rotate(180deg);
    height:72px; text-align:left; vertical-align:bottom;
}
.fc-matrix td.label { color:#888; font-size:9px; white-space:nowrap; max-width:180px; overflow:hidden; text-overflow:ellipsis; padding-right:8px; }
.fc-matrix td.cell {
    width:36px; height:36px; text-align:center;
    font-size:9px; font-weight:500;
    border:1px solid rgba(255,255,255,0.1);
    border-radius:3px;
}

/* ── Similarity card (right panel) ── */
.fc-sim-card { background:rgba(255,255,255,0.02); backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:14px 16px; }
.fc-sim-val { font-family:'Geist Mono',monospace; font-size:22px; font-weight:600; line-height:1; margin-bottom:2px; color:#FFF; }
.fc-sim-lbl { font-family:'Geist Mono',monospace; font-size:9px; color:#888; letter-spacing:.1em; text-transform:uppercase; margin-bottom:8px; }
.fc-sim-bar-wrap { height:3px; background:rgba(255,255,255,0.05); border-radius:2px; margin-bottom:6px; }
.fc-sim-bar-fill { height:100%; border-radius:2px; }
.fc-sim-foot { display:flex; justify-content:space-between; font-family:'Geist Mono',monospace; font-size:9px; color:#888; }

/* ── Dividers ── */
hr { border-color:rgba(255,255,255,0.1) !important; margin:16px 0 !important; }

/* Spinner override */
[data-testid="stSpinner"] { display:none !important; }

/* Misc cleanup */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] { gap:0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    cache=None, client=None,
    chat_history=[],
    bench_log=[], bench_done=False,
    api_key="",
    model="gemini-3.1-flash-lite",
    threshold=0.85,
    pending_prompt=None,
    query_start_ts=None,
    last_similarity=None,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _start_dashboard(_cache):
    try:
        _cache.serve_dashboard(port=5555, background=True)
        return True
    except Exception as e:
        print(f"Failed to start admin dashboard: {e}")
        return False

def get_cache() -> SemanticCache:
    if st.session_state.cache is None:
        embedder = GeminiEmbedder(api_key=st.session_state.api_key, model="gemini-embedding-2") if st.session_state.api_key else None
        config = CacheConfig(threshold=st.session_state.threshold)
        st.session_state.cache = DemoCache(config=config, embedder=embedder, dashboard=True)
    
    # Always ensure the dashboard points to the current active cache
    try:
        import fastcache.dashboard.app as dash_app
        dash_app._global_cache = st.session_state.cache
    except Exception:
        pass

    _start_dashboard(st.session_state.cache)
    return st.session_state.cache

def get_client() -> LLMClient | None:
    if not st.session_state.api_key:
        return None
    if st.session_state.client is None:
        try:
            st.session_state.client = LLMClient(
                api_key=st.session_state.api_key,
                model=st.session_state.model,
            )
        except Exception:
            return None
    return st.session_state.client

def handle_query(query: str) -> dict:
    cache = get_cache()
    client = get_client()
    if client is None:
        return {"error": "No API key configured."}
    
    t0 = time.perf_counter()
    hits_before = cache.stats.cache_hits
    
    last_hit_info = {}
    def on_hit_cb(p, r, sim, meta):
        last_hit_info['similarity'] = sim
        last_hit_info['original_query'] = meta.get('namespace', 'semantic match')
        return True
    cache.on_hit = on_hit_cb
    
    fallback_info = {}
    def fallback(p):
        response, tokens, llm_ms, cost = client.complete(p)
        fallback_info['tokens'] = tokens
        fallback_info['cost'] = cost
        return response
        
    try:
        response = cache.query(query, fallback=fallback)
    except Exception as e:
        return {"error": str(e)}
        
    latency_ms = (time.perf_counter() - t0) * 1000
    hits_after = cache.stats.cache_hits
    hit = (hits_after > hits_before)
    
    if hit and not last_hit_info:
        last_hit_info['similarity'] = 1.0
        last_hit_info['original_query'] = "exact match"
        
    if hit:
        with cache._stats_lock:
            toks = len(query.split()) + len(response.split())
            cache._stats.tokens_saved += toks
            
    return dict(
        response=response, 
        hit=hit, 
        similarity=last_hit_info.get('similarity', 0.0),
        latency_ms=latency_ms, 
        tokens=fallback_info.get('tokens', 0), 
        cost=fallback_info.get('cost', 0.0),
        original_query=last_hit_info.get('original_query', query) if hit else query
    )

def ms_class(ms: float) -> str:
    return "fast" if ms < 120 else ("slow" if ms > 700 else "mid")

def sim_color(sim: float, threshold: float) -> str:
    if sim >= 0.99: return "#16A34A"
    if sim >= threshold: return "#2563EB"
    if sim >= 0.80: return "#D97706"
    return "#DC2626"

def matrix_bg(v: float) -> str:
    if v >= 0.99: return ("#DCFCE7", "#166534")
    if v >= 0.92: return ("#DBEAFE", "#1D4ED8")
    if v >= 0.80: return ("#FEF9C3", "#92400E")
    if v >= 0.60: return ("#FEE2E2", "#991B1B")
    return ("#F9FAFB", "#CCC")


# ─────────────────────────────────────────────────────────────────────────────
# Topbar
# ─────────────────────────────────────────────────────────────────────────────
cache = get_cache()
model_short = st.session_state.model.replace("gemini-","")

st.markdown(f"""
<div class="fc-topbar">
  <div class="fc-wordmark"><div class="fc-dot"></div>FastCache</div>
  <div class="fc-tb-divider"></div>
  <span class="fc-tb-pill">{model_short}</span>
  <span class="fc-tb-pill">gemini-embedding-2</span>
  <span class="fc-tb-pill">cosine · {st.session_state.threshold:.2f}</span>
  <span class="fc-tb-pill active">● live</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### FastCache")
    st.markdown('<div style="font-size:11px;color:#999;margin-bottom:16px;">Semantic LLM caching framework</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="fc-sidebar-label">API Key</div>', unsafe_allow_html=True)
    st.caption("[Get a free key →](https://aistudio.google.com/apikey)")
    api_key_input = st.text_input("key", value=st.session_state.api_key, type="password",
                                   placeholder="AIza...", label_visibility="collapsed")
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.session_state.cache = None
        st.session_state.client = None

    st.markdown('<div class="fc-sidebar-label">Model</div>', unsafe_allow_html=True)
    MODELS = ["gemini-3.1-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
    model_choice = st.selectbox("model", MODELS,
        index=MODELS.index(st.session_state.model) if st.session_state.model in MODELS else 0,
        label_visibility="collapsed")
    if model_choice != st.session_state.model:
        st.session_state.model = model_choice
        st.session_state.client = None

    st.markdown('<div class="fc-sidebar-label">Similarity Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("thr", min_value=0.70, max_value=0.99, step=0.01,
                          value=st.session_state.threshold, label_visibility="collapsed")
    hint_txt = "strict" if threshold > 0.95 else ("balanced" if threshold > 0.87 else "permissive")
    st.markdown(f'<div style="font-family:\'Geist Mono\',monospace;font-size:10px;color:#999;margin-top:-6px;">{threshold:.2f} · {hint_txt}</div>', unsafe_allow_html=True)
    if threshold != st.session_state.threshold:
        st.session_state.threshold = threshold
        if st.session_state.cache:
            st.session_state.cache.threshold = threshold

    st.divider()
    if st.button("Clear cache & history", use_container_width=True):
        st.session_state.cache = None
        st.session_state.chat_history = []
        st.session_state.bench_log = []
        st.session_state.bench_done = False
        st.session_state.pending_prompt = None
        st.session_state.last_similarity = None
        st.rerun()

    st.divider()
    # Live mini-stats in sidebar
    hr = cache.hit_rate * 100
    llm_calls = cache.total_queries - cache.cache_hits
    st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:8px;">
  <div style="display:flex;justify-content:space-between;font-size:11px;">
    <span style="color:#999;">Hit rate</span>
    <span style="font-family:'Geist Mono',monospace;font-weight:600;color:{'#2DD4BF' if hr>50 else '#FFF'}">{hr:.0f}%</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:11px;">
    <span style="color:#999;">Cache hits</span>
    <span style="font-family:'Geist Mono',monospace;font-weight:600;">{cache.cache_hits}</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:11px;">
    <span style="color:#999;">LLM calls</span>
    <span style="font-family:'Geist Mono',monospace;font-weight:600;">{llm_calls}</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:11px;">
    <span style="color:#999;">Avg latency</span>
    <span style="font-family:'Geist Mono',monospace;font-weight:600;">{cache.avg_latency_ms:.0f}ms</span>
  </div>
</div>
""".replace('\n', ''), unsafe_allow_html=True)
    
    st.markdown("<br><div style='text-align:center'><a href='http://localhost:5555' target='_blank' style='font-size:11px;color:#2DD4BF;text-decoration:none;'>↗ Open Admin Dashboard</a></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics bar
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Queries", cache.total_queries)
with c2:
    hr = cache.hit_rate * 100
    st.metric("Hit Rate", f"{hr:.1f}%", delta=f"{cache.cache_hits} hits" if cache.cache_hits else None)
with c3:
    st.metric("Avg Latency", f"{cache.avg_latency_ms:.0f} ms",
              delta=f"vs {cache.avg_llm_latency_ms:.0f}ms LLM" if cache.avg_llm_latency_ms else None,
              delta_color="inverse")
with c4: st.metric("Tokens Saved", f"{cache.tokens_saved:,}")
with c5:
    llm_calls = cache.total_queries - cache.cache_hits
    st.metric("LLM Calls", f"{llm_calls}", delta=f"{cache.cache_hits} avoided" if cache.cache_hits else None)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_bench, tab_explorer = st.tabs(["Chat", "Benchmark", "Cache Explorer"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – CHAT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    is_loading = st.session_state.pending_prompt is not None

    if not st.session_state.api_key:
        st.markdown("""
<div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:5px;padding:12px 16px;font-size:12px;color:#FCD34D;display:flex;align-items:center;gap:8px;">
  <span>→</span> Enter your Gemini API key in the sidebar to start querying.
</div>
""", unsafe_allow_html=True)

    chat_col, info_col = st.columns([3, 1])

    with chat_col:
        # ── Render messages ──
        if not st.session_state.chat_history and not is_loading:
            st.markdown("""
<div style="text-align:center;padding:56px 0 40px;color:#888;">
  <div style="font-family:'Geist Mono',monospace;font-size:11px;line-height:2.2;margin-bottom:14px;color:#CCC;">
    MISS  → embed → vector lookup → LLM → write cache<br>
    HIT   → embed → cosine ≥ threshold → return cached<br>
    EXACT → hash match → return instantly
  </div>
  <div style="font-size:12px;color:#555;">Ask anything — then rephrase it to see the cache activate.</div>
</div>
""", unsafe_allow_html=True)
        else:
            msgs = ""
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    msgs += f"""
<div class="fc-row user">
  <div class="fc-bubble-wrap" style="max-width:72%">
    <div class="fc-bubble user">{msg["content"]}</div>
  </div>
  <div class="fc-avatar user">YOU</div>
</div>"""
                else:
                    hit = msg.get("hit", False)
                    ms = msg.get("latency_ms", 0)
                    sim = msg.get("similarity", 0)
                    tok = msg.get("tokens", 0)
                    orig = msg.get("original_query", "")
                    ms_c = ms_class(ms)
                    spark_pct = min(100, (ms / 2000) * 100)

                    if hit and orig and orig != msg.get("_query", orig):
                        tag = '<span class="fc-tag hit">HIT</span>'
                    elif hit:
                        tag = '<span class="fc-tag exact">EXACT</span>'
                    else:
                        tag = '<span class="fc-tag miss">MISS</span>'

                    sim_disp = f'<span class="fc-meta-stat">sim <span class="v">{sim:.4f}</span></span>' if sim else ""
                    orig_disp = f'<span class="fc-meta-stat">matched <span class="v">"{orig[:32]}{"…" if len(orig)>32 else ""}"</span></span>' if hit and orig else ""

                    safe_resp = (msg['content']
                        .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                        .replace("\n","<br/>"))

                    msgs += f"""
<div class="fc-row">
  <div class="fc-avatar bot">FC</div>
  <div class="fc-bubble-wrap">
    <div class="fc-result-meta">
      {tag}
      <span class="fc-meta-stat"><span class="v {'fast' if ms<120 else ('slow' if ms>700 else '')}">{ms:.0f}ms</span></span>
      {sim_disp}
      {orig_disp}
    </div>
    <div class="fc-bubble bot">
      {safe_resp}
      <div class="fc-lat-bar"><div class="fc-lat-fill {ms_c}" style="width:{spark_pct:.0f}%"></div></div>
    </div>
  </div>
</div>"""

            # Loading bubble
            if is_loading:
                pending = st.session_state.pending_prompt
                elapsed = time.time() - (st.session_state.query_start_ts or time.time())
                safe_p = pending.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                msgs += f"""
<div class="fc-row user">
  <div class="fc-bubble-wrap">
    <div class="fc-bubble user">{safe_p}</div>
  </div>
  <div class="fc-avatar user">YOU</div>
</div>
<div class="fc-row">
  <div class="fc-avatar bot">FC</div>
  <div class="fc-bubble-wrap">
    <div class="fc-result-meta">
      <span class="fc-loading-stage-badge">PROCESSING</span>
      <span style="font-family:'Geist Mono',monospace;font-size:10px;color:#CCC;">{elapsed:.1f}s</span>
    </div>
    <div class="fc-loading-bubble">
      <div class="fc-pipeline">
        <span class="fc-pipe-step done">embed</span>
        <span class="fc-pipe-arrow">→</span>
        <span class="fc-pipe-step active">lookup</span>
        <span class="fc-pipe-arrow">→</span>
        <span class="fc-pipe-step">llm</span>
        <span class="fc-pipe-arrow">→</span>
        <span class="fc-pipe-step">store</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <div class="fc-dots"><span></span><span></span><span></span></div>
        <span style="font-family:'Geist Mono',monospace;font-size:10px;color:#CCC;">querying fastcache…</span>
      </div>
    </div>
  </div>
</div>"""

            st.markdown(f'<div class="fc-messages">{msgs}</div>'.replace('\n', ''), unsafe_allow_html=True)

        # ── Input area ──
        if is_loading:
            st.markdown("""
<div class="fc-processing-bar">
  <span style="animation:fc-pulse .9s ease-in-out infinite;display:inline-block">◌</span>
  Processing — please wait…
</div>
""", unsafe_allow_html=True)
        else:
            # Example prompts
            st.markdown('<div style="margin-bottom:10px;font-size:11px;color:#BBB;">Try these:</div>', unsafe_allow_html=True)
            ex_cols = st.columns(3)
            examples = [
                "What is Python?", "Explain machine learning", "What is a REST API?",
                "Tell me about Python", "Define machine learning", "How do REST APIs work?",
            ]
            for i, ex in enumerate(examples):
                with ex_cols[i % 3]:
                    if st.button(ex, key=f"ex_{i}", use_container_width=True):
                        st.session_state.pending_prompt = ex
                        st.session_state.query_start_ts = time.time()
                        st.rerun()

            with st.form("chat_form", clear_on_submit=True):
                ci, cb = st.columns([6, 1])
                with ci:
                    user_input = st.text_input("q", placeholder="Ask anything…", label_visibility="collapsed")
                with cb:
                    submitted = st.form_submit_button("Send →", type="primary", use_container_width=True)

            if submitted and user_input.strip():
                st.session_state.pending_prompt = user_input.strip()
                st.session_state.query_start_ts = time.time()
                st.rerun()

    # ── Info sidebar col ──
    with info_col:
        sim = st.session_state.last_similarity
        if sim is not None:
            thr = st.session_state.threshold
            color = sim_color(sim, thr)
            verdict = "HIT" if sim >= thr else "MISS"
            tag_cls = "hit" if sim >= thr else "miss"
            st.markdown(f"""
            <div class="fc-sim-card">
              <div style="font-family:'Geist Mono',monospace;font-size:9px;color:#BBB;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;">Last similarity</div>
              <div class="fc-sim-val" style="color:{color}">{sim:.4f}</div>
              <div class="fc-sim-lbl">cosine score</div>
              <div class="fc-sim-bar-wrap"><div class="fc-sim-bar-fill" style="width:{sim*100:.0f}%;background:{color}"></div></div>
              <div class="fc-sim-foot">
                <span>threshold {thr:.2f}</span>
                <span class="fc-tag {tag_cls}" style="font-size:9px;padding:1px 6px;">{verdict}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Cache size widget
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        c = get_cache()
        st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.05);border-radius:6px;padding:14px 16px;">
  <div style="font-family:'Geist Mono',monospace;font-size:9px;color:#888;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">Cache</div>
  <div style="font-family:'Geist Mono',monospace;font-size:22px;font-weight:600;color:#FFF;line-height:1;margin-bottom:3px;">{c.size}</div>
  <div style="font-family:'Geist Mono',monospace;font-size:9px;color:#888;text-transform:uppercase;letter-spacing:.08em;">entries stored</div>
</div>
""", unsafe_allow_html=True)


# ── Execute pending query (two-rerun state machine) ──
if st.session_state.pending_prompt and "chat_form" not in str(st.session_state.get("_pending_source","")):
    pending = st.session_state.pending_prompt
    result = handle_query(pending)
    if "error" in result:
        st.session_state.chat_history.append({"role":"user","content":pending})
        st.session_state.chat_history.append({"role":"assistant","content":f"Error: {result['error']}","hit":False})
    else:
        st.session_state.last_similarity = result.get("similarity")
        st.session_state.chat_history.append({"role":"user","content":pending,"_query":pending})
        st.session_state.chat_history.append({
            "role":"assistant",
            "content":result["response"],
            "hit":result["hit"],
            "similarity":result["similarity"],
            "latency_ms":result["latency_ms"],
            "tokens":result["tokens"],
            "cost":result["cost"],
            "original_query":result["original_query"],
            "_query":pending,
        })
    st.session_state.pending_prompt = None
    st.session_state.query_start_ts = None
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown('<div style="font-size:12px;color:#999;margin-bottom:20px;">Replay fixed query groups. Each group\'s first query calls the LLM; rephrases should hit the cache.</div>', unsafe_allow_html=True)

    bcol1, bcol2 = st.columns([4, 1])
    with bcol1:
        selected_groups = st.multiselect(
            "Groups",
            options=[g["topic"] for g in QUERY_GROUPS],
            default=[g["topic"] for g in QUERY_GROUPS],
            label_visibility="collapsed",
        )
    with bcol2:
        run_bench = st.button("Run", type="primary", use_container_width=True)

    with st.expander("Preview queries", expanded=False):
        for group in QUERY_GROUPS:
            if group["topic"] in selected_groups:
                st.markdown(f"**{group['topic']}**")
                for q in group["queries"]:
                    st.markdown(f"<span style='font-family:Geist Mono,monospace;font-size:11px;color:#555'>— {q}</span>", unsafe_allow_html=True)

    if run_bench:
        if not st.session_state.api_key:
            st.error("Add your Gemini API key in the sidebar.")
        else:
            groups = [g for g in QUERY_GROUPS if g["topic"] in selected_groups]
            total  = sum(len(g["queries"]) for g in groups)
            st.session_state.bench_log = []
            progress = st.progress(0, text="Running…")
            done = 0
            for group in groups:
                st.session_state.bench_log.append(f'<span class="log-head">▸ {group["topic"]}</span>')
                for query in group["queries"]:
                    res = handle_query(query)
                    done += 1
                    progress.progress(done / total, text=f"{done}/{total}")
                    cls  = "log-hit" if res.get("hit") else "log-miss"
                    tag  = "HIT " if res.get("hit") else "MISS"
                    sim  = res.get("similarity", 0)
                    lat  = res.get("latency_ms", 0)
                    tok  = res.get("tokens", 0)
                    st.session_state.bench_log.append(
                        f'<span class="{cls}">[{tag}  sim={sim:.4f}  {lat:5.0f}ms  {tok:4d} tok]</span>  {query}')
                st.session_state.bench_log.append("")
            progress.empty()
            st.session_state.bench_done = True
            st.rerun()

    if st.session_state.bench_done and st.session_state.bench_log:
        c = get_cache()
        b1, b2, b3, b4 = st.columns(4)
        with b1: st.metric("Hit Rate", f"{c.hit_rate*100:.1f}%")
        with b2: st.metric("Cache Latency", f"{c.avg_cache_latency_ms:.0f} ms")
        with b3: st.metric("LLM Latency",   f"{c.avg_llm_latency_ms:.0f} ms")
        with b4:
            speedup = (c.avg_llm_latency_ms / c.avg_cache_latency_ms) if c.avg_cache_latency_ms > 0 else 0
            st.metric("Speedup", f"{speedup:.1f}×")
        log_html = "<br>".join(st.session_state.bench_log)
        st.markdown(f'<div class="fc-bench-log">{log_html}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – CACHE EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    c = get_cache()

    if c.size == 0:
        st.markdown("""
<div style="text-align:center;padding:56px 0;color:#666;">
  <div style="font-size:11px;font-family:'Geist Mono',monospace;margin-bottom:8px;">CACHE EMPTY</div>
  <div style="font-size:12px;">Run queries in Chat or Benchmark to populate the cache.</div>
</div>
""", unsafe_allow_html=True)
    else:
        exp_left, exp_right = st.columns([3, 2])

        with exp_left:
            st.markdown(f'<div style="font-size:12px;color:#999;margin-bottom:14px;font-family:\'Geist Mono\',monospace;">{c.size} entries · cosine index</div>', unsafe_allow_html=True)
            search = st.text_input("search", placeholder="Filter by query or response text…", label_visibility="collapsed")

            entries = c.entries
            if search:
                entries = [e for e in entries if search.lower() in e.query.lower() or search.lower() in e.response.lower()]

            rows_html = ""
            for i, entry in enumerate(entries):
                q_safe = entry.query.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                r_safe = (entry.response[:110] + ("…" if len(entry.response)>110 else "")).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                badge_cls = "some" if entry.hits > 0 else "zero"
                rows_html += f"""
                <tr>
                  <td style="font-family:'Geist Mono',monospace;font-size:10px;color:#CCC;width:28px">{i+1}</td>
                  <td>
                    <div class="q-text">{q_safe}</div>
                    <div class="r-text">{r_safe}</div>
                  </td>
                  <td style="text-align:center;width:60px">
                    <span class="fc-hits-badge {badge_cls}">{entry.hits}</span>
                  </td>
                </tr>"""

            st.markdown(f"""
<table class="fc-cache-table">
  <thead><tr>
    <th>#</th><th>Query / Response</th><th style="text-align:center">Hits</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

        with exp_right:
            if c.size > 1 and c.size <= 20:
                st.markdown('<div style="font-size:11px;color:#999;margin-bottom:12px;font-family:\'Geist Mono\',monospace;">Similarity matrix — cosine between all cached vectors</div>', unsafe_allow_html=True)
                vecs   = np.stack([e.vector for e in c.entries])
                norms  = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
                normed = vecs / norms
                sim_mx = normed @ normed.T

                labels = [e.query[:22] + ("…" if len(e.query)>22 else "") for e in c.entries]
                n = len(labels)

                tbl  = '<div class="fc-matrix-wrap"><table class="fc-matrix"><thead><tr><td></td>'
                for lbl in labels:
                    tbl += f'<th>{lbl}</th>'
                tbl += '</tr></thead><tbody>'
                for i, row_lbl in enumerate(labels):
                    tbl += f'<tr><td class="label">{row_lbl}</td>'
                    for j in range(n):
                        v = float(sim_mx[i, j])
                        bg, fg = matrix_bg(v)
                        tbl += f'<td class="cell" title="{v:.4f}" style="background:{bg};color:{fg}">{v:.2f}</td>'
                    tbl += '</tr>'
                tbl += '</tbody></table>'

                # Legend
                tbl += """
                <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;font-size:10px;font-family:'Geist Mono',monospace;color:#999;">
                  <span><span style="display:inline-block;width:10px;height:10px;background:#DCFCE7;border:1px solid #BBF7D0;border-radius:2px;vertical-align:middle;margin-right:4px"></span>≥0.99</span>
                  <span><span style="display:inline-block;width:10px;height:10px;background:#DBEAFE;border:1px solid #BFDBFE;border-radius:2px;vertical-align:middle;margin-right:4px"></span>≥0.92 hit</span>
                  <span><span style="display:inline-block;width:10px;height:10px;background:#FEF9C3;border:1px solid #FEF08A;border-radius:2px;vertical-align:middle;margin-right:4px"></span>≥0.80</span>
                  <span><span style="display:inline-block;width:10px;height:10px;background:#FEE2E2;border:1px solid #FECACA;border-radius:2px;vertical-align:middle;margin-right:4px"></span>≥0.60</span>
                </div></div>"""

                st.markdown(tbl, unsafe_allow_html=True)
            elif c.size > 20:
                st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:11px;color:#CCC;">Matrix hidden for caches > 20 entries.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-family:\'Geist Mono\',monospace;font-size:11px;color:#CCC;">Add more entries to see the similarity matrix.</div>', unsafe_allow_html=True)