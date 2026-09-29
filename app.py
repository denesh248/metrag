import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import streamlit as st
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables (.env locally or st.secrets on Streamlit Cloud)
load_dotenv()
try:
    if hasattr(st, "secrets"):
        for k in ["GROQ_API_KEY", "OPENAI_API_KEY", "GROQ_MODEL"]:
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = st.secrets[k]
except Exception:
    pass

from utils.audio_processor import process_input, DOWNLOAD_DIR
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# ─── Page Configuration ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MetRAG • Intelligent Video & Meeting Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Modern Cyber-Glass Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #08090d;
    --bg-card: #0f1118;
    --bg-card-hover: #151824;
    --bg-glass: rgba(18, 20, 31, 0.75);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(139, 92, 246, 0.4);
    
    --primary: #8b5cf6;
    --primary-light: #a78bfa;
    --primary-glow: rgba(139, 92, 246, 0.25);
    --cyan: #06b6d4;
    --cyan-glow: rgba(6, 182, 212, 0.2);
    --emerald: #10b981;
    --rose: #f43f5e;
    --amber: #f59e0b;
    
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

/* Global resets */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

/* Subtle background glowing radial effects */
.stApp::before {
    content: '';
    position: fixed;
    top: -20%; left: 15%;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.07) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -10%; right: 10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0b0c13 !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Hero Section */
.hero-container {
    padding: 1.5rem 0 1.2rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
}
.brand-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.75rem;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--primary-light);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 10%, #c4b5fd 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 0.5rem;
}

/* Glassmorphism Cards */
.v-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.v-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 8px 30px var(--primary-glow);
    transform: translateY(-2px);
}
.v-card-accent {
    border-left: 3px solid var(--primary);
}
.v-card-cyan {
    border-left: 3px solid var(--cyan);
}
.v-card-emerald {
    border-left: 3px solid var(--emerald);
}
.v-card-rose {
    border-left: 3px solid var(--rose);
}

.card-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}
.card-label {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.card-body {
    color: var(--text-primary);
    font-size: 0.92rem;
    line-height: 1.7;
}

/* Pill Badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-purple { background: rgba(139, 92, 246, 0.15); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.3); }
.badge-cyan { background: rgba(6, 182, 212, 0.15); color: #67e8f9; border: 1px solid rgba(6, 182, 212, 0.3); }
.badge-green { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }

/* Stepper Status Bars */
.step-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.7rem 0.9rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    margin-bottom: 0.45rem;
    border: 1px solid var(--border-subtle);
    font-size: 0.82rem;
}
.step-indicator {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}
.step-pending { background: #334155; }
.step-active {
    background: var(--primary);
    box-shadow: 0 0 10px var(--primary);
    animation: pulse-glow 1.5s infinite;
}
.step-done {
    background: var(--emerald);
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.4);
}
@keyframes pulse-glow {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
}

/* Chat Styling */
.chat-window {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 480px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
.chat-bubble-user {
    background: rgba(139, 92, 246, 0.14);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1.1rem;
    margin: 0.5rem 0 0.75rem auto;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-bubble-ai {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(6, 182, 212, 0.25);
    border-radius: 12px 12px 12px 2px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem auto 0.75rem 0;
    max-width: 85%;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Input Fields & Buttons */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #11131c !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px var(--primary-glow) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
}

/* Tab navigation styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.02);
    padding: 6px;
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 16px;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(139, 92, 246, 0.2) !important;
    color: #ffffff !important;
    border-bottom: 2px solid var(--primary) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Initialize Session State ────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = {}
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "execution_time" not in st.session_state:
    st.session_state.execution_time = 0.0

def set_step(step_name: str, status: str):
    st.session_state.pipeline_steps[step_name] = status

def get_step_class(step_name: str) -> str:
    s = st.session_state.pipeline_steps.get(step_name, "pending")
    if s == "active":
        return "step-active"
    elif s == "done":
        return "step-done"
    return "step-pending"

# ─── Sidebar Configuration ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem">
        <span style="font-size:1.8rem">⚡</span>
        <div>
            <div style="font-size:1.25rem; font-weight:800; letter-spacing:-0.02em; background:linear-gradient(135deg, #fff, #a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">METRAG AI</div>
            <div style="font-size:0.68rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.1em">Meeting & Video Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Ingestion Mode Selection
    st.markdown('<div class="card-label">📁 Ingestion Source</div>', unsafe_allow_html=True)
    source_type = st.radio("Choose input method:", ["YouTube URL", "Upload Media File"], label_visibility="collapsed")

    source_path = ""
    if source_type == "YouTube URL":
        source_path = st.text_input(
            "Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Supports YouTube videos, podcasts, and talks.",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload Audio/Video",
            type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"],
            help="Upload your recorded meeting file.",
        )
        if uploaded_file is not None:
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            saved_path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source_path = saved_path
            st.success(f"File loaded: `{uploaded_file.name}`")

    st.markdown("---")
    st.markdown('<div class="card-label">⚙️ Engine & Architecture</div>', unsafe_allow_html=True)

    whisper_engine = st.selectbox(
        "Whisper STT Engine",
        ["groq (whisper-large-v3, Cloud Ultra-Fast)", "local (openai-whisper, Offline)", "openai (whisper-1, Cloud)"],
        index=0,
    )
    selected_whisper_engine = "groq"
    if "local" in whisper_engine:
        selected_whisper_engine = "local"
    elif "openai" in whisper_engine:
        selected_whisper_engine = "openai"

    groq_model = st.selectbox(
        "Groq LLM Reasoner",
        [
            "qwen/qwen3.8-27b (Verified Active & Ultra-Fast)",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "Custom Model ID...",
        ],
        index=0,
        help="qwen/qwen3.8-27b is the active reasoning model on your Groq key."
    )
    if "Custom" in groq_model:
        selected_groq_model = st.text_input("Enter custom Groq model ID:", value="qwen/qwen3.8-27b").strip()
    else:
        selected_groq_model = groq_model.split(" ")[0]
    os.environ["GROQ_MODEL"] = selected_groq_model

    st.markdown("---")
    run_button = st.button("🚀 Process & Analyze Meeting", use_container_width=True)

    # Pipeline Live Indicators
    st.markdown('<div class="card-label" style="margin-top:1.5rem">📊 Live Pipeline Monitor</div>', unsafe_allow_html=True)
    pipeline_stages = [
        ("audio", "🔊", "Audio Ingestion & Slicing"),
        ("transcription", "📝", "Whisper Speech-to-Text"),
        ("title", "🏷️", "AI Meeting Titling"),
        ("summary", "📋", "Executive Summarization"),
        ("extraction", "🔍", "Action & Decision Extraction"),
        ("rag", "🧠", "ChromaDB RAG Indexing"),
    ]
    for key, icon, label in pipeline_stages:
        status_cls = get_step_class(key)
        st.markdown(f"""
        <div class="step-item">
            <div class="step-indicator {status_cls}"></div>
            <span>{icon} {label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:var(--text-muted); line-height:1.5">
        🔒 <strong>Hardware Accelerated</strong><br>
        Inference executed via Groq LPU™ Tensor Processing Engine & OpenAI Whisper STT.
    </div>
    """, unsafe_allow_html=True)

# ─── Main Body ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="brand-pill">⚡ Powered by Groq LPU™ & Whisper</div>
    <h1 class="hero-title">AI Video & Meeting Intelligence</h1>
    <div class="hero-desc">
        Automate meeting intelligence: Transcribe recordings, generate executive syntheses, extract committed action items, and interrogate transcripts in real-time with RAG.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Execution Pipeline ──────────────────────────────────────────────────────────
if run_button:
    if not source_path.strip():
        st.error("⚠️ Please provide a valid YouTube URL or upload a media file.")
    else:
        st.session_state.pipeline_running = True
        st.session_state.chat_history = []
        st.session_state.result = None
        for key, _, _ in pipeline_stages:
            set_step(key, "pending")

        progress_msg = st.empty()
        start_time = time.time()

        try:
            with progress_msg.container():
                st.info("⚡ Executing pipeline... monitor live status in the left sidebar.")

            # 1. Audio Processing
            set_step("audio", "active")
            chunks = process_input(source_path)
            set_step("audio", "done")

            # 2. Transcription
            set_step("transcription", "active")
            transcript = transcribe_all(chunks, engine=selected_whisper_engine)
            set_step("transcription", "done")

            # 3. Title Generation
            set_step("title", "active")
            meeting_title = generate_title(transcript)
            set_step("title", "done")

            # 4. Summarization
            set_step("summary", "active")
            meeting_summary = summarize(transcript)
            set_step("summary", "done")

            # 5. Extraction
            set_step("extraction", "active")
            action_items = extract_action_items(transcript)
            key_decisions = extract_key_decisions(transcript)
            open_questions = extract_questions(transcript)
            set_step("extraction", "done")

            # 6. RAG Engine
            set_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            set_step("rag", "done")

            total_elapsed = round(time.time() - start_time, 2)
            st.session_state.execution_time = total_elapsed
            st.session_state.result = {
                "title": meeting_title,
                "transcript": transcript,
                "summary": meeting_summary,
                "action_items": action_items,
                "key_decisions": key_decisions,
                "open_questions": open_questions,
                "rag_chain": rag_chain,
                "chunks_count": len(chunks),
                "source": source_path,
            }
            progress_msg.success(f"🎉 Complete analysis finished in {total_elapsed}s!")
            time.sleep(0.8)
            progress_msg.empty()
            st.rerun()

        except Exception as exc:
            for key, _, _ in pipeline_stages:
                if st.session_state.pipeline_steps.get(key) == "active":
                    set_step(key, "pending")
            progress_msg.error(f"❌ Execution encountered an error: {exc}")

# ─── Dashboard Results View ──────────────────────────────────────────────────────
if st.session_state.result:
    res = st.session_state.result

    # Meeting Header Ribbon
    st.markdown(f"""
    <div class="v-card v-card-accent" style="margin-bottom:1.5rem">
        <div class="card-header-bar">
            <span class="card-label">📌 Executive Meeting Brief</span>
            <div style="display:flex; gap:0.5rem">
                <span class="badge badge-purple">{res.get('chunks_count', 1)} Audio Chunk(s)</span>
                <span class="badge badge-cyan">Inference: {st.session_state.execution_time}s</span>
                <span class="badge badge-green">LPU Accelerated</span>
            </div>
        </div>
        <div style="font-size:1.6rem; font-weight:800; color:#ffffff; margin-bottom:0.4rem">
            {res['title']}
        </div>
        <div style="font-size:0.8rem; color:var(--text-muted); font-family:'JetBrains Mono', monospace">
            Source: {os.path.basename(res['source'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Multi-Tab Intelligence Center
    tab_summary, tab_actions, tab_decisions, tab_questions, tab_chat, tab_transcript, tab_export = st.tabs([
        "📋 Executive Summary",
        "✅ Action Items",
        "🔑 Key Decisions",
        "❓ Open Questions",
        "💬 Interactive RAG Chat",
        "📝 Full Transcript",
        "📤 Export & Share",
    ])

    with tab_summary:
        st.markdown(f"""
        <div class="v-card v-card-cyan">
            <div class="card-label">Executive Brief & Key Discussion Themes</div>
            <div class="card-body" style="white-space: pre-line">
{res['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_actions:
        st.markdown(f"""
        <div class="v-card v-card-emerald">
            <div class="card-label">Delegated Responsibilities & Milestones</div>
            <div class="card-body" style="white-space: pre-line">
{res['action_items']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_decisions:
        st.markdown(f"""
        <div class="v-card v-card-accent">
            <div class="card-label">Consensus & Strategic Alignments</div>
            <div class="card-body" style="white-space: pre-line">
{res['key_decisions']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_questions:
        st.markdown(f"""
        <div class="v-card v-card-rose">
            <div class="card-label">Unresolved Topics & Pending Clarifications</div>
            <div class="card-body" style="white-space: pre-line">
{res['open_questions']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_chat:
        st.markdown("""
        <div style="margin-bottom:0.75rem">
            <span style="font-size:1.1rem; font-weight:700">Conversational Knowledge Retrieval (RAG)</span>
            <div style="font-size:0.82rem; color:var(--text-secondary)">
                Ask specific questions. Answers are cross-referenced strictly against the Chroma vector database of the meeting transcript.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Question Pills
        st.markdown('<span style="font-size:0.75rem; color:var(--text-muted); font-weight:600">SUGGESTED QUESTIONS:</span>', unsafe_allow_html=True)
        q_cols = st.columns(3)
        suggested_q = None
        with q_cols[0]:
            if st.button("📌 What were the top 3 priorities discussed?", use_container_width=True):
                suggested_q = "What were the top 3 priorities discussed?"
        with q_cols[1]:
            if st.button("⏰ What deadlines or commitments were agreed?", use_container_width=True):
                suggested_q = "What deadlines or commitments were agreed upon?"
        with q_cols[2]:
            if st.button("⚠️ Were any budget, risk, or technical blockers raised?", use_container_width=True):
                suggested_q = "Were any budget, risk, or technical blockers raised?"

        # Chat history container using native Streamlit chat components
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant", avatar="⚡"):
                        st.markdown(message["content"])
        else:
            st.info("💡 Transcript indexed. Ask any question below or click a suggested prompt above.")

        # Input row
        user_input_val = st.chat_input("Ask a question about this meeting or video...")
        prompt_to_run = suggested_q or user_input_val

        if prompt_to_run:
            st.session_state.chat_history.append({"role": "user", "content": prompt_to_run})
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            with st.chat_message("assistant", avatar="⚡"):
                with st.spinner("Searching transcript & querying Groq..."):
                    answer = ask_question(res["rag_chain"], prompt_to_run)
                    st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    with tab_transcript:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem">
            <span style="font-size:1.1rem; font-weight:700">Verbatim Meeting Transcript</span>
            <span class="badge badge-purple">{len(res['transcript'].split())} Words • {len(res['transcript'])} Characters</span>
        </div>
        """, unsafe_allow_html=True)

        transcript_search = st.text_input("🔍 Search transcript:", placeholder="Type a keyword to filter...")
        display_transcript = res["transcript"]
        if transcript_search.strip():
            count = display_transcript.lower().count(transcript_search.lower())
            st.caption(f"Found {count} occurrence(s) of '{transcript_search}'")

        st.text_area(
            "Complete Transcript",
            value=display_transcript,
            height=380,
            disabled=True,
            label_visibility="collapsed",
        )

        st.download_button(
            label="📥 Download Transcript (.txt)",
            data=res["transcript"],
            file_name=f"{res['title'].replace(' ', '_')}_transcript.txt",
            mime="text/plain",
        )

    with tab_export:
        st.markdown("""
        <div style="font-size:1.1rem; font-weight:700; margin-bottom:0.5rem">Executive Report Generation</div>
        <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:1.5rem">
            Export the synthesized meeting documentation in structured markdown format.
        </div>
        """, unsafe_allow_html=True)

        # Markdown Report Generation
        report_md = f"""# Executive Meeting Report: {res['title']}
*Generated by MetRAG AI Meeting Intelligence Engine*
- **Source:** {res['source']}
- **Audio Chunks:** {res.get('chunks_count', 1)}
- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 Executive Summary
{res['summary']}

---

## ✅ Action Items & Owners
{res['action_items']}

---

## 🔑 Key Decisions
{res['key_decisions']}

---

## ❓ Open Questions & Risks
{res['open_questions']}

---

## 📝 Verbatim Transcript Excerpt
{res['transcript'][:1500]}... [Full transcript available in MetRAG AI]
"""
        st.markdown(f"""
        <div class="v-card">
            <div class="card-label">Preview Markdown Report</div>
            <pre style="background:rgba(0,0,0,0.3); padding:1rem; border-radius:8px; font-size:0.8rem; max-height:250px; overflow-y:auto; color:var(--text-secondary)">{report_md}</pre>
        </div>
        """, unsafe_allow_html=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Download Markdown Report (.md)",
                data=report_md,
                file_name=f"{res['title'].replace(' ', '_')}_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                label="📥 Download JSON Summary (.json)",
                data=json.dumps({
                    "title": res["title"],
                    "summary": res["summary"],
                    "action_items": res["action_items"],
                    "key_decisions": res["key_decisions"],
                    "open_questions": res["open_questions"],
                    "source": res["source"],
                }, indent=2),
                file_name=f"{res['title'].replace(' ', '_')}_data.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    # Empty State Landing
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:4.5rem 1rem; text-align:center">
        <div style="font-size:3.5rem; margin-bottom:1rem">⚡</div>
        <div style="font-size:1.6rem; font-weight:800; color:#ffffff; margin-bottom:0.5rem">
            Ready for Ingestion
        </div>
        <div style="color:var(--text-secondary); font-size:0.92rem; max-width:480px; line-height:1.7; margin-bottom:1.75rem">
            Select a <strong>YouTube URL</strong> or <strong>Upload a meeting recording</strong> from the left sidebar, then click <strong>Process & Analyze Meeting</strong>.
        </div>
        <div style="display:flex; gap:0.75rem; flex-wrap:wrap; justify-content:center">
            <span class="badge badge-purple">⚡ Groq LPU™ Powered</span>
            <span class="badge badge-cyan">🎙️ Whisper Speech-to-Text</span>
            <span class="badge badge-green">🧠 ChromaDB Vector Retrieval</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Grid
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown("""
        <div class="v-card v-card-accent">
            <div style="font-size:1.5rem; margin-bottom:0.5rem">⚡</div>
            <div style="font-weight:700; font-size:0.95rem; margin-bottom:0.3rem">Sub-Second Groq Reasoning</div>
            <div style="font-size:0.82rem; color:var(--text-secondary); line-height:1.6">
                Powered by LLaMA 3.3 70B on Groq LPUs delivering 500+ tokens/sec for instantaneous executive summaries.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f_col2:
        st.markdown("""
        <div class="v-card v-card-cyan">
            <div style="font-size:1.5rem; margin-bottom:0.5rem">🎙️</div>
            <div style="font-weight:700; font-size:0.95rem; margin-bottom:0.3rem">Whisper Audio Slicing</div>
            <div style="font-size:0.82rem; color:var(--text-secondary); line-height:1.6">
                Automated audio chunking and speech-to-text transcription with auto-language detection and high fidelity.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f_col3:
        st.markdown("""
        <div class="v-card v-card-emerald">
            <div style="font-size:1.5rem; margin-bottom:0.5rem">🧠</div>
            <div style="font-weight:700; font-size:0.95rem; margin-bottom:0.3rem">Conversational RAG</div>
            <div style="font-size:0.82rem; color:var(--text-secondary); line-height:1.6">
                Vectorized search via ChromaDB and all-MiniLM embeddings with strict anti-hallucination guardrails.
            </div>
        </div>
        """, unsafe_allow_html=True)