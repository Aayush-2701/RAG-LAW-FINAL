"""
app.py — RAG-LAW: AI Legal Assistant (single Streamlit entry point).

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    Main file path: app.py
    Secrets: GROQ_API_KEY = "your-groq-api-key"
"""

from __future__ import annotations

import streamlit as st

from rag_pipeline import (
    answer_query,
    build_index_from_upload,
    build_preloaded_index,
    get_groq_api_key,
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
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚖️ RAG-LAW")
    st.caption("Retrieval-Augmented Generation over legal documents")

    st.divider()

    # API key status indicator
    api_key = get_groq_api_key()
    if api_key:
        st.success("✅ Groq API key found", icon="🔑")
    else:
        st.error(
            "❌ **GROQ_API_KEY not set.**\n\n"
            "Add it to `.streamlit/secrets.toml` (local) or the "
            "**Secrets** panel on share.streamlit.io (cloud).",
            icon="🔑",
        )

    st.divider()
    st.markdown(
        """
**How it works**

1. Index a PDF (preloaded or uploaded).
2. Type a legal question.
3. The app retrieves the most relevant passages via FAISS similarity search.
4. Groq's LLM synthesises an answer grounded in those passages.

**Model:** `deepseek-r1-distill-llama-70b` via Groq  
**Embeddings:** `all-MiniLM-L6-v2` (local, CPU)
"""
    )

    st.divider()
    st.caption(
        "Answers are based solely on the indexed documents. "
        "This is a demo — not legal advice."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("⚖️ AI Legal Assistant")
st.write(
    "Ask questions about legal documents. The assistant retrieves relevant "
    "passages and grounds its answers in them."
)

# Session-state keys
if "active_vectorstore" not in st.session_state:
    st.session_state.active_vectorstore = None
if "active_source_label" not in st.session_state:
    st.session_state.active_source_label = None

# ---------------------------------------------------------------------------
# Tabs: preloaded PDFs vs. file upload
# ---------------------------------------------------------------------------

tab_preloaded, tab_upload = st.tabs(["📂 Preloaded Documents", "📤 Upload Your PDF"])

# --- Tab 1: Preloaded PDFs ---------------------------------------------------
with tab_preloaded:
    st.subheader("Preloaded Legal Documents")
    st.write(
        "These PDFs are bundled with the app. Click **Load** to build "
        "(or reuse a cached) FAISS index."
    )

    with st.spinner("🔍 Scanning for preloaded PDFs…"):
        # Trigger the cached index build (no-op if already cached)
        preloaded_vs = build_preloaded_index()

    if preloaded_vs is None:
        st.warning(
            "No PDFs found in the `pdfs/` directory or repo root. "
            "Add PDFs to `pdfs/` and redeploy, or use the Upload tab.",
            icon="📭",
        )
    else:
        from pathlib import Path
        import glob

        root = Path(__file__).parent
        pdf_paths = list(root.glob("*.pdf")) + list((root / "pdfs").glob("*.pdf"))
        seen, unique = set(), []
        for p in pdf_paths:
            r = p.resolve()
            if r not in seen:
                seen.add(r)
                unique.append(p.name)

        st.markdown("**Available documents:**")
        for name in unique:
            st.markdown(f"- 📄 {name}")

        if st.button("✅ Use Preloaded Documents", type="primary", key="btn_preloaded"):
            st.session_state.active_vectorstore = preloaded_vs
            st.session_state.active_source_label = f"{len(unique)} preloaded document(s)"
            st.success(
                f"Index loaded! {len(unique)} document(s) are ready to query.",
                icon="✅",
            )

# --- Tab 2: User upload ------------------------------------------------------
with tab_upload:
    st.subheader("Upload a PDF")
    st.write("Upload any legal PDF. The index is built on-the-fly and stored for this session.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Basic sanity check
        if not uploaded_file.name.lower().endswith(".pdf"):
            st.error(
                f"'{uploaded_file.name}' does not appear to be a PDF file. "
                "Please upload a valid PDF.",
                icon="🚫",
            )
        else:
            st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

            if st.button("⚙️ Build Index from Upload", type="primary", key="btn_upload"):
                with st.spinner(
                    f"Indexing **{uploaded_file.name}** — this may take up to 30s…"
                ):
                    try:
                        pdf_bytes = uploaded_file.read()
                        vectorstore = build_index_from_upload(
                            pdf_bytes, uploaded_file.name
                        )
                        st.session_state.active_vectorstore = vectorstore
                        st.session_state.active_source_label = uploaded_file.name
                        st.success(
                            f"✅ Index built for **{uploaded_file.name}**! Ready to query.",
                            icon="✅",
                        )
                    except ValueError as exc:
                        st.error(
                            f"Could not extract text from the PDF: {exc}\n\n"
                            "Make sure the file is not password-protected or scanned-only.",
                            icon="🚫",
                        )
                    except Exception as exc:
                        st.error(
                            f"Unexpected error while indexing: {exc}",
                            icon="🚫",
                        )

# ---------------------------------------------------------------------------
# Query interface — always visible below the tabs
# ---------------------------------------------------------------------------

st.divider()

active_label = st.session_state.get("active_source_label")
active_vs = st.session_state.get("active_vectorstore")

if active_label:
    st.info(f"🗂️ **Active index:** {active_label}", icon="📂")

st.subheader("Ask a Legal Question")

user_question = st.text_area(
    "Your question:",
    height=120,
    placeholder=(
        "e.g. What rights does Article 12 of the UDHR guarantee? "
        "What are the provisions for bail under the BNSS?"
    ),
    label_visibility="collapsed",
)

ask_button = st.button("🔍 Ask AI Lawyer", type="primary", disabled=(not user_question.strip()))

if ask_button:
    # Guard: question must not be empty
    if not user_question.strip():
        st.warning("Please enter a question before clicking Ask.", icon="⚠️")

    # Guard: must have an active index
    elif active_vs is None:
        st.error(
            "No documents are indexed yet. Please load the preloaded documents "
            "or upload a PDF first.",
            icon="📭",
        )

    # Guard: must have a Groq API key
    elif not get_groq_api_key():
        st.error(
            "**GROQ_API_KEY is missing.** Add it to `.streamlit/secrets.toml` "
            "(local dev) or the Streamlit Cloud Secrets panel.",
            icon="🔑",
        )

    else:
        with st.spinner("🤖 Thinking… (retrieving passages and calling Groq)"):
            try:
                answer, sources = answer_query(user_question, active_vs)
            except ValueError as exc:
                st.error(str(exc), icon="🔑")
                st.stop()
            except RuntimeError as exc:
                st.error(
                    f"The AI call failed. Check your Groq API key and try again.\n\n"
                    f"Details: {exc}",
                    icon="🚫",
                )
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error: {exc}", icon="🚫")
                st.stop()

        # --- Display answer ---
        st.markdown("### 💬 Answer")
        # Strip <think> reasoning tags that DeepSeek-R1 sometimes emits
        import re
        clean_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        st.markdown(clean_answer)

        # --- Display source chunks ---
        if sources:
            with st.expander("📑 Source Passages Retrieved", expanded=False):
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(
                        f"**Chunk {i}** — `{source_name}` (page {page})"
                    )
                    st.markdown(
                        f"> {doc.page_content[:600]}{'…' if len(doc.page_content) > 600 else ''}"
                    )
                    if i < len(sources):
                        st.markdown("---")
