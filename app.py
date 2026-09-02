"""
app.py — RAG-LAW: AI Legal Assistant (single Streamlit entry point).

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    Main file path : app.py
    Secrets        : GROQ_API_KEY = "gsk_..."
"""

from __future__ import annotations

import streamlit as st

from rag_pipeline import (
    AVAILABLE_MODELS,
    DEFAULT_LLM_MODEL,
    answer_query,
    build_index_from_upload,
    build_preloaded_index,
    discover_pdfs,
    get_groq_api_key,
    load_embedding_model,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG-LAW · AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "chat_history": [],          # list[dict]: {role, content, citations}
        "active_vectorstore": None,
        "active_source_label": None,
        "selected_model": DEFAULT_LLM_MODEL,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚖️ RAG-LAW")
    st.caption("Retrieval-Augmented Generation over legal documents")
    st.divider()

    # ── API key status ──────────────────────────────────────────────────────
    api_key = get_groq_api_key()
    if api_key:
        st.success("✅ Groq API key detected", icon="🔑")
    else:
        st.error(
            "❌ **GROQ_API_KEY not set.**\n\n"
            "Add it to `.streamlit/secrets.toml` (local) or the "
            "**Secrets** panel on Streamlit Community Cloud.",
            icon="🔑",
        )

    st.divider()

    # ── Model selector ──────────────────────────────────────────────────────
    st.subheader("🤖 LLM Model")
    st.session_state.selected_model = st.selectbox(
        "Groq model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_model)
        if st.session_state.selected_model in AVAILABLE_MODELS
        else 0,
        label_visibility="collapsed",
        help="All models are served via Groq's ultra-fast inference API.",
    )

    st.divider()

    # ── How it works ────────────────────────────────────────────────────────
    st.markdown(
        """
**How it works**

1. Load a preloaded legal doc or upload your own PDF.
2. Type a question in the chat box.
3. **FAISS** retrieves the most relevant passages.
4. **Groq LLM** streams a grounded answer with citations.

**Embeddings:** `all-MiniLM-L6-v2` (local, CPU-only)
"""
    )

    st.divider()

    # ── Clear chat ──────────────────────────────────────────────────────────
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ Answers are grounded in the indexed documents only. "
        "This is a demo — **not legal advice**."
    )

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.title("⚖️ AI Legal Assistant")
st.write(
    "Ask questions about legal documents. "
    "The assistant retrieves the most relevant passages and streams "
    "a grounded answer with source citations."
)

# ---------------------------------------------------------------------------
# Warm up the embedding model in the background (avoids first-query lag)
# ---------------------------------------------------------------------------

with st.spinner("⚙️ Loading embedding model…"):
    load_embedding_model()

# ---------------------------------------------------------------------------
# Document selection — tabs
# ---------------------------------------------------------------------------

tab_preloaded, tab_upload = st.tabs(["📂 Preloaded Documents", "📤 Upload Your PDF"])

# ── Tab 1: Preloaded PDFs ───────────────────────────────────────────────────
with tab_preloaded:
    st.subheader("Preloaded Legal Documents")
    st.write(
        "These documents are bundled with the app and indexed at startup. "
        "Click **Load** to activate them."
    )

    pdfs = discover_pdfs()
    if not pdfs:
        st.warning("⚠️ No PDF files found at the repository root.")
    else:
        # Show available documents as an info block
        pdf_list_md = "\n".join(f"- 📄 `{p.name}`" for p in pdfs)
        st.markdown(f"**Available documents:**\n{pdf_list_md}")

        col_load, col_status = st.columns([2, 3])
        with col_load:
            load_btn = st.button(
                "📥 Load All Preloaded Documents",
                key="load_preloaded",
                type="primary",
                use_container_width=True,
            )
        with col_status:
            if st.session_state.active_source_label:
                st.info(f"🟢 Active: **{st.session_state.active_source_label}**")

        if load_btn:
            with st.spinner("🔍 Loading FAISS index…"):
                vs = build_preloaded_index()
            if vs:
                st.session_state.active_vectorstore = vs
                st.session_state.active_source_label = (
                    f"{len(pdfs)} preloaded document(s)"
                )
                st.session_state.chat_history = []
                st.success(
                    f"✅ Loaded **{len(pdfs)} document(s)** — ready to query!",
                    icon="⚖️",
                )
            else:
                st.error("❌ Failed to build index. Check that the PDFs are readable.")

# ── Tab 2: Upload PDF ───────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload Your Own Legal PDF")
    st.write("Upload any PDF document to index and query it with the RAG pipeline.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            index_btn = st.button(
                "📥 Index Uploaded PDF",
                key="index_upload",
                type="primary",
                use_container_width=True,
            )
        with col_info:
            st.write(f"**File:** `{uploaded_file.name}`")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

        if index_btn:
            with st.spinner(f"⚙️ Indexing `{uploaded_file.name}`… this may take a moment."):
                vs = build_index_from_upload(
                    uploaded_file.read(), uploaded_file.name
                )
            if vs:
                st.session_state.active_vectorstore = vs
                st.session_state.active_source_label = uploaded_file.name
                st.session_state.chat_history = []
                st.success(
                    f"✅ Indexed **{uploaded_file.name}** — ready to query!",
                    icon="📄",
                )
            else:
                st.error(
                    "❌ Indexing failed. Make sure the PDF contains extractable text "
                    "(not a scanned image-only PDF)."
                )

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

st.divider()

if st.session_state.active_vectorstore is None:
    st.info(
        "👆 **Load a document above** to start asking legal questions.",
        icon="ℹ️",
    )
else:
    st.subheader(f"💬 Querying: {st.session_state.active_source_label}")
    st.caption(
        f"Model: `{st.session_state.selected_model}` via Groq · "
        "Change model in the sidebar."
    )

    # ── Render past messages ────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📄 Sources used", expanded=False):
                    for cit in msg["citations"]:
                        st.markdown(
                            f"**{cit['source']}** — Page {cit['page']}\n\n"
                            f"> {cit['snippet']}…"
                        )

    # ── Chat input ──────────────────────────────────────────────────────────
    question = st.chat_input("Ask a legal question…")

    if question:
        # Display user bubble
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append(
            {"role": "user", "content": question, "citations": []}
        )

        # Stream assistant response
        with st.chat_message("assistant"):
            stream_gen, citations = answer_query(
                st.session_state.active_vectorstore,
                question,
                model_name=st.session_state.selected_model,
            )
            answer_text = st.write_stream(stream_gen)

            if citations:
                with st.expander("📄 Sources used", expanded=False):
                    for cit in citations:
                        st.markdown(
                            f"**{cit['source']}** — Page {cit['page']}\n\n"
                            f"> {cit['snippet']}…"
                        )

        # Persist to history
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer_text,
                "citations": citations,
            }
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "RAG-LAW · Built with "
    "[Streamlit](https://streamlit.io), "
    "[LangChain](https://langchain.com), "
    "[Groq](https://groq.com), "
    "and [FAISS](https://github.com/facebookresearch/faiss). "
    "**Not legal advice.**"
)
