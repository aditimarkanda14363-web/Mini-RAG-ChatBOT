"""
Mini RAG-Based AI Chatbot
Streamlit Application Frontend & User Interface
Course: Prompt Engineering
"""

from pathlib import Path
import time
from typing import Generator
import streamlit as st

from src.config import DEFAULT_CONFIG, RAGConfig, SAMPLE_DOCS_DIR
from src.llm_client import OllamaClient
from src.rag_pipeline import QueryContext, RAGPipeline, SourceCitation


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mini RAG Chatbot | Prompt Engineering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Modern CSS Design & Animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #CBD5E1;
        margin: 0 0 16px 0;
        line-height: 1.5;
    }

    /* Badges Bar */
    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    .tag-badge {
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.08);
        color: #E2E8F0;
        border: 1px solid rgba(255, 255, 255, 0.14);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        backdrop-filter: blur(4px);
    }
    .tag-badge.accent {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .tag-badge.ollama {
        background: rgba(99, 102, 241, 0.2);
        color: #C7D2FE;
        border-color: rgba(99, 102, 241, 0.35);
    }

    /* Pulsing status indicator */
    .pulse-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        display: inline-block;
    }
    .pulse-green {
        background: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulseGreen 1.8s infinite;
    }
    .pulse-red {
        background: #EF4444;
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
        animation: pulseRed 1.8s infinite;
    }
    @keyframes pulseGreen {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulseRed {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(239, 68, 68, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Status Banner for Grounded vs Refusal */
    .grounded-pill {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 10px;
    }
    .refusal-pill {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 10px;
    }

    /* Modern Citation Cards */
    .source-card {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 14px 18px;
        margin-top: 10px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .source-card:hover {
        box-shadow: 0 6px 16px -4px rgba(15, 23, 42, 0.12);
        transform: translateY(-1px);
    }
    .source-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid #F1F5F9;
    }
    .source-title {
        font-weight: 700;
        font-size: 0.88rem;
        color: #1E293B;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .source-page-badge {
        background: #EFF6FF;
        color: #2563EB;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .score-badge-high {
        background: linear-gradient(135deg, #10B981, #059669);
        color: #FFFFFF;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .score-badge-med {
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        color: #FFFFFF;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .score-badge-low {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: #FFFFFF;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .source-snippet {
        font-family: 'JetBrains Mono', monospace, Consolas;
        font-size: 0.82rem;
        color: #334155;
        background: #F8FAFC;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Suggestion Chips */
    .suggestion-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 24px;
    }
    .suggestion-header {
        font-size: 0.9rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Code & Telemetry blocks */
    .telemetry-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        background: #0F172A;
        color: #E2E8F0;
        padding: 14px;
        border-radius: 8px;
        overflow-x: auto;
    }

    /* Sidebar cards */
    .sidebar-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline Initialization (Cached Singleton)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_pipeline() -> RAGPipeline:
    """Initializes and caches the singleton RAGPipeline instance."""
    return RAGPipeline()


pipeline = get_rag_pipeline()
llm_client = OllamaClient()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ---------------------------------------------------------------------------
# Sidebar: Control Panel & Ingestion Engine
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ RAG Control Panel")
    
    # 1. System Health Live Pulse
    health = llm_client.check_health()
    if health["status"]:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span class="pulse-dot pulse-green"></span>
            <span style="font-weight:700; color:#065F46; font-size:0.88rem;">Local Ollama Online</span>
        </div>
        """, unsafe_allow_html=True)
        available_models = health.get("models", ["llama3.2:3b"])
    else:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span class="pulse-dot pulse-red"></span>
            <span style="font-weight:700; color:#991B1B; font-size:0.88rem;">Ollama Daemon Offline</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Tip: Start Ollama in PowerShell with `ollama serve`")
        available_models = ["llama3.2:3b"]

    # Model Selector
    selected_model = st.selectbox(
        "Active Local LLM:",
        options=available_models if available_models else ["llama3.2:3b"],
        index=0,
        help="Local LLM model served by Ollama (Default: llama3.2:3b)"
    )

    st.markdown("---")

    # 2. Document Ingestion Section
    st.markdown("#### 📂 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files:",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload multiple documents. Text and page numbers will be extracted automatically."
    )

    if uploaded_files:
        if st.button("⚡ Process & Index Documents", use_container_width=True, type="primary"):
            progress_bar = st.progress(0, text="Initializing ingestion...")
            start_time = time.time()
            total_chunks = 0
            
            for idx, uf in enumerate(uploaded_files):
                pct = int((idx / len(uploaded_files)) * 90)
                progress_bar.progress(pct, text=f"Chunking & embedding {uf.name}...")
                res = pipeline.ingest_document(uf, filename=uf.name)
                total_chunks += res.get("chunks", 0)
                if uf.name not in st.session_state.indexed_files:
                    st.session_state.indexed_files.append(uf.name)
            
            progress_bar.progress(100, text="Indexing complete!")
            time.sleep(0.4)
            progress_bar.empty()
            elapsed = time.time() - start_time
            st.success(f"Indexed {len(uploaded_files)} file(s) ({total_chunks} chunks) in {elapsed:.1f}s!")

    # Course Sample Documents Button
    st.markdown("##### Or Load Assignment Samples:")
    if st.button("📚 Load Course Sample Documents", use_container_width=True):
        with st.spinner("Indexing AI Overview PDF & Prompt Guide TXT..."):
            samples = [
                SAMPLE_DOCS_DIR / "artificial_intelligence_overview.pdf",
                SAMPLE_DOCS_DIR / "prompt_engineering_guide.txt"
            ]
            loaded_count = 0
            for sample_path in samples:
                if sample_path.exists():
                    pipeline.ingest_document(sample_path, filename=sample_path.name)
                    if sample_path.name not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append(sample_path.name)
                    loaded_count += 1
            st.success(f"Indexed {loaded_count} course sample documents into FAISS!")
            st.rerun()

    # 3. Vector Database Metrics
    doc_count = pipeline.vector_store.count()
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Vector Chunks", doc_count)
    with col_m2:
        st.metric("Documents", len(st.session_state.indexed_files))

    if st.session_state.indexed_files:
        with st.expander("📄 Active Document Inventory", expanded=False):
            for fname in st.session_state.indexed_files:
                icon = "📕" if fname.lower().endswith(".pdf") else "📝"
                st.caption(f"{icon} `{fname}`")

    st.markdown("---")

    # 4. Prompt Engineering Hyperparameters
    with st.expander("🔬 Retrieval & Prompt Settings"):
        top_k = st.slider("Top-K Passages:", min_value=1, max_value=8, value=3)
        sim_threshold = st.slider(
            "Relevance Similarity Cutoff:",
            min_value=0.10,
            max_value=0.75,
            value=0.35,
            step=0.05,
            help="Queries with top cosine similarity below this score trigger epistemic refusal."
        )
        temperature = st.slider(
            "LLM Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Lower values produce deterministic, grounded answers without hallucinations."
        )
        pipeline.config.similarity_threshold = sim_threshold

    # 5. Clear / Reset Options
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()
    with col_b2:
        if st.button("🗑️ Reset All", use_container_width=True):
            pipeline.reset()
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

    # 6. Export Chat Transcript
    if st.session_state.messages:
        transcript_lines = ["# RAG Chatbot Conversation Transcript\n"]
        for m in st.session_state.messages:
            role = "User" if m["role"] == "user" else "Assistant"
            transcript_lines.append(f"### {role}\n{m['content']}\n")
            if m.get("sources"):
                transcript_lines.append("**Sources Cited:**")
                for s in m["sources"]:
                    transcript_lines.append(f"- {s['source']} (Page {s['page']}, Similarity: {s['score']:.3f})")
                transcript_lines.append("\n")
        
        st.download_button(
            "📥 Export Transcript (.md)",
            data="\n".join(transcript_lines),
            file_name="rag_chat_transcript.md",
            mime="text/markdown",
            use_container_width=True
        )


# ---------------------------------------------------------------------------
# Main Chat Header & Hero Card
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🤖 Mini RAG-Based AI Chatbot</span>
    </div>
    <div class="hero-subtitle">
        Practical Assignment for <b>Prompt Engineering</b> &mdash; Offline Retrieval-Augmented Generation featuring page-accurate citations and strict epistemic refusal.
    </div>
    <div class="badge-bar">
        <span class="tag-badge accent">🛡️ 100% Offline & Private</span>
        <span class="tag-badge ollama">🦙 Ollama: llama3.2:3b</span>
        <span class="tag-badge">⚡ FAISS Cosine Index</span>
        <span class="tag-badge">🧠 all-MiniLM-L6-v2 (384-dim)</span>
        <span class="tag-badge">📄 PyMuPDF Page Tracking</span>
        <span class="tag-badge">🚫 Zero Paid APIs</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper function to render citation badges
# ---------------------------------------------------------------------------
def render_citation_card(source: SourceCitation):
    badge_class = "score-badge-high" if source.score >= 0.70 else ("score-badge-med" if source.score >= 0.45 else "score-badge-low")
    icon = "📕" if source.source.lower().endswith(".pdf") else "📝"
    
    st.markdown(f"""
    <div class="source-card">
        <div class="source-card-header">
            <span class="source-title">
                {icon} <b>{source.source}</b>
                <span class="source-page-badge">Page {source.page}</span>
            </span>
            <span class="{badge_class}">Cosine Match: {source.score:.3f}</span>
        </div>
        <div class="source-snippet">{source.snippet}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Welcome / Interactive Suggestion Chips (when chat is empty)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div class="suggestion-box">
        <div class="suggestion-header">
            <span>💡 <b>Prompt Engineering Quick-Test Prompts</b> (Click any chip to test instantly):</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🏛️ Who coined Artificial Intelligence at Dartmouth in 1956?", use_container_width=True):
            st.session_state.pending_prompt = "Who coined Artificial Intelligence at Dartmouth in 1956?"
            st.rerun()
        if st.button("📐 What is the self-attention formula from Transformers?", use_container_width=True):
            st.session_state.pending_prompt = "What is the self-attention formula from Attention Is All You Need?"
            st.rerun()
    with col_s2:
        if st.button("🔄 What are the 4 phases of a RAG pipeline?", use_container_width=True):
            st.session_state.pending_prompt = "What are the four phases of a RAG pipeline?"
            st.rerun()
        if st.button("🍪 How do you bake a chocolate cake? (Refusal Test)", use_container_width=True):
            st.session_state.pending_prompt = "How do you bake a chocolate cake?"
            st.rerun()


# ---------------------------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------------------------
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Status Badge
            if msg.get("sufficient_info", True):
                st.markdown('<span class="grounded-pill">✅ Factually Grounded</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="refusal-pill">⚠️ Epistemic Refusal (Insufficient Context)</span>', unsafe_allow_html=True)

            # Response Content
            st.markdown(msg["content"])

            # Sources & Telemetry Tabs
            sources = msg.get("sources", [])
            has_telemetry = bool(msg.get("prompt_used") and msg["prompt_used"] != "N/A")
            
            if sources or has_telemetry:
                tab_titles = []
                if sources:
                    tab_titles.append(f"📚 Retrieved Sources ({len(sources)})")
                if has_telemetry:
                    tab_titles.append("🔬 Prompt Engineering Inspector")
                
                tabs = st.tabs(tab_titles)
                tab_idx = 0

                # Sources Tab
                if sources:
                    with tabs[tab_idx]:
                        for src_data in sources:
                            c_obj = SourceCitation(
                                source=src_data["source"],
                                page=src_data["page"],
                                score=src_data["score"],
                                snippet=src_data["snippet"],
                                chunk_id=src_data.get("chunk_id", "")
                            )
                            render_citation_card(c_obj)
                    tab_idx += 1

                # Prompt Engineering Telemetry Tab
                if has_telemetry:
                    with tabs[tab_idx]:
                        sub_t1, sub_t2 = st.tabs(["🏷️ System Persona & Directives", "📜 Full Delimited Prompt"])
                        with sub_t1:
                            st.caption("Instructions constraining the local model to strict grounding & honest refusal:")
                            st.code(msg.get("system_prompt", ""), language="markdown")
                        with sub_t2:
                            st.caption("Exact context passages & query injected into the LLM:")
                            st.code(msg.get("prompt_used", ""), language="markdown")
        else:
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Handle User Input (via chat input or suggestion chips)
# ---------------------------------------------------------------------------
user_text_input = st.chat_input("Ask a question about your uploaded documents...")
active_query = user_text_input or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if active_query:
    # 1. Append User Message
    st.session_state.messages.append({
        "role": "user",
        "content": active_query,
        "timestamp": time.time()
    })

    with st.chat_message("user"):
        st.markdown(active_query)

    # 2. Assistant Response with Streamlit Real-Time Token Streaming
    with st.chat_message("assistant"):
        # Fast retrieval step
        query_ctx = pipeline.prepare_query(
            question=active_query,
            top_k=top_k,
            model_name=selected_model,
            temperature=temperature,
            chat_history=st.session_state.messages[:-1]
        )

        full_answer = ""
        sufficient_info = True

        if not query_ctx.is_ready:
            # Case 1: Empty Index
            st.markdown('<span class="refusal-pill">⚠️ Epistemic Refusal (Empty Index)</span>', unsafe_allow_html=True)
            full_answer = query_ctx.immediate_answer
            st.info(full_answer)
            sufficient_info = False

        elif not query_ctx.is_relevant:
            # Case 2: Relevance Cutoff Refusal (Fast Refusal < 50ms)
            st.markdown('<span class="refusal-pill">⚠️ Epistemic Refusal (Insufficient Information)</span>', unsafe_allow_html=True)
            full_answer = query_ctx.immediate_answer
            st.markdown(full_answer)
            sufficient_info = False

        else:
            # Case 3: Grounded Answer with Real-Time Token Streaming
            st.markdown('<span class="grounded-pill">✅ Factually Grounded</span>', unsafe_allow_html=True)
            
            def token_generator() -> Generator[str, None, None]:
                stream = pipeline.llm_client.generate_stream(
                    prompt=query_ctx.prompt,
                    system_prompt=query_ctx.system_prompt,
                    model=query_ctx.active_model,
                    temperature=query_ctx.temperature
                )
                for chunk in stream:
                    yield chunk

            # Stream tokens to UI in real-time
            try:
                full_answer = st.write_stream(token_generator)
            except Exception as e:
                full_answer = f"⚠️ LLM Inference Error: {str(e)}"
                st.error(full_answer)
                sufficient_info = False

            # Check if LLM generated an explicit refusal inside the stream
            if full_answer and ("cannot find sufficient information" in full_answer.lower() or "does not contain" in full_answer.lower()):
                sufficient_info = False

        # 3. Interactive Sources & Telemetry Tabs for Live Message
        if query_ctx.citations or (query_ctx.prompt and query_ctx.prompt != "N/A"):
            live_tabs = []
            if query_ctx.citations:
                live_tabs.append(f"📚 Retrieved Sources ({len(query_ctx.citations)})")
            if query_ctx.prompt and query_ctx.prompt != "N/A":
                live_tabs.append("🔬 Prompt Engineering Inspector")
            
            rendered_tabs = st.tabs(live_tabs)
            r_idx = 0

            if query_ctx.citations:
                with rendered_tabs[r_idx]:
                    for cit in query_ctx.citations:
                        render_citation_card(cit)
                r_idx += 1

            if query_ctx.prompt and query_ctx.prompt != "N/A":
                with rendered_tabs[r_idx]:
                    t_sub1, t_sub2 = st.tabs(["🏷️ System Persona & Directives", "📜 Full Delimited Prompt"])
                    with t_sub1:
                        st.caption("System prompt enforcing context boundaries & honesty:")
                        st.code(query_ctx.system_prompt, language="markdown")
                    with t_sub2:
                        st.caption("Delimited context and question sent to Ollama:")
                        st.code(query_ctx.prompt, language="markdown")

    # 4. Save Assistant Message into History
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": [
            {
                "source": s.source,
                "page": s.page,
                "score": s.score,
                "snippet": s.snippet,
                "chunk_id": s.chunk_id
            }
            for s in query_ctx.citations
        ],
        "sufficient_info": sufficient_info,
        "system_prompt": query_ctx.system_prompt,
        "prompt_used": query_ctx.prompt,
        "timestamp": time.time()
    })
