"""
rag_pipeline.py — Backend RAG module for RAG-LAW.

Responsibilities:
  - Load and cache the HuggingFace embedding model (all-MiniLM-L6-v2).
  - Load the pre-built FAISS index from Vectorstore/ for instant cold-starts.
  - Build a fresh FAISS index from user-supplied PDF bytes (per-upload).
  - Retrieve relevant document chunks via similarity search.
  - Stream answers using ChatGroq (deepseek-r1-distill-llama-70b by default).
  - Return source citations (source filename + page number) alongside answers.

No Streamlit UI code lives here — only pure backend logic imported by app.py.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator, List, Optional

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "deepseek-r1-distill-llama-70b"

AVAILABLE_MODELS: list[str] = [
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama3-70b-8192",
]

# Repository root — PDFs and the pre-built Vectorstore live here
REPO_ROOT = Path(__file__).parent
VECTORSTORE_DIR = REPO_ROOT / "Vectorstore"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5  # chunks to retrieve per query

SYSTEM_PROMPT = """\
You are an expert AI legal assistant. Use ONLY the pieces of information \
provided in the Context below to answer the user's question.

Rules:
- If the answer is not contained in the context, respond with exactly: \
"I don't have enough information in the provided documents to answer that question."
- Do NOT fabricate statutes, case law, or facts.
- Be concise, well-structured, and cite the source document and page number.
- Use numbered lists for multi-part answers.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Groq API key helper
# ---------------------------------------------------------------------------


def get_groq_api_key() -> Optional[str]:
    """Return the Groq API key from st.secrets or environment, or None."""
    try:
        key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    except Exception:
        key = os.environ.get("GROQ_API_KEY")
    return key or None


# ---------------------------------------------------------------------------
# Embedding model (loaded once per container lifetime)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache the sentence-transformer embedding model (CPU-only)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------


def discover_pdfs() -> list[Path]:
    """Return all PDF files found at the repository root (alphabetically)."""
    return sorted(REPO_ROOT.glob("*.pdf"))


# ---------------------------------------------------------------------------
# Pre-built Vectorstore loader
# ---------------------------------------------------------------------------


def _load_prebuilt_vectorstore() -> Optional[FAISS]:
    """
    Load the pre-built FAISS index from Vectorstore/ if available.
    This avoids re-embedding on every cold start.
    """
    index_file = VECTORSTORE_DIR / "index.faiss"
    pkl_file = VECTORSTORE_DIR / "index.pkl"
    if not (index_file.exists() and pkl_file.exists()):
        return None
    try:
        embeddings = load_embedding_model()
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def build_preloaded_index() -> Optional[FAISS]:
    """
    Return a FAISS index over all preloaded PDFs at the repo root.

    Strategy:
      1. Try loading the committed Vectorstore/ index (instant).
      2. Fallback: embed all root-level PDFs from scratch.
    """
    vs = _load_prebuilt_vectorstore()
    if vs is not None:
        return vs

    pdfs = discover_pdfs()
    if not pdfs:
        return None
    return _build_index_from_paths([str(p) for p in pdfs])


def build_index_from_upload(pdf_bytes: bytes, filename: str) -> Optional[FAISS]:
    """Build a fresh FAISS index from user-uploaded PDF bytes."""
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        return _build_index_from_paths([tmp_path])
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _build_index_from_paths(pdf_paths: list[str]) -> Optional[FAISS]:
    """Core helper: parse PDFs → split → embed → return FAISS vectorstore."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    all_docs: list[Document] = []

    for path in pdf_paths:
        try:
            loader = PDFPlumberLoader(path)
            raw_docs = loader.load()
            # Ensure source metadata is set to the bare filename
            for doc in raw_docs:
                doc.metadata.setdefault("source", Path(path).name)
            chunks = splitter.split_documents(raw_docs)
            all_docs.extend(chunks)
        except Exception as exc:
            st.warning(f"⚠️ Could not process `{Path(path).name}`: {exc}")

    if not all_docs:
        return None

    embeddings = load_embedding_model()
    return FAISS.from_documents(all_docs, embeddings)


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


def retrieve_chunks(vectorstore: FAISS, query: str, k: int = TOP_K) -> list[Document]:
    """Return the top-k most relevant document chunks for the query."""
    return vectorstore.similarity_search(query, k=k)


def _format_context(chunks: list[Document]) -> str:
    """Concatenate chunks into a single context string for the LLM."""
    parts: list[str] = []
    for i, doc in enumerate(chunks, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] Source: {source}, Page {page}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def format_citations(chunks: list[Document]) -> list[dict]:
    """Return deduplicated citation dicts for display in the UI."""
    citations: list[dict] = []
    seen: set[tuple] = set()
    for doc in chunks:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append(
                {
                    "source": source,
                    "page": page,
                    "snippet": doc.page_content[:350].strip(),
                }
            )
    return citations


# ---------------------------------------------------------------------------
# Answer generation (streaming)
# ---------------------------------------------------------------------------


def answer_query(
    vectorstore: FAISS,
    question: str,
    model_name: str = DEFAULT_LLM_MODEL,
) -> tuple[Generator, list[dict]]:
    """
    Retrieve relevant chunks and stream an LLM answer via Groq.

    Returns
    -------
    (stream_generator, citations)
        stream_generator : yields str tokens — pass to ``st.write_stream``
        citations        : list of citation dicts for the "Sources" expander
    """
    api_key = get_groq_api_key()
    if not api_key:
        def _no_key() -> Generator:
            yield (
                "❌ **No Groq API key found.**\n\n"
                "Add `GROQ_API_KEY` to your `.streamlit/secrets.toml` (local) "
                "or the **Secrets** panel on Streamlit Community Cloud."
            )
        return _no_key(), []

    chunks = retrieve_chunks(vectorstore, question)
    context = _format_context(chunks)
    citations = format_citations(chunks)

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT)])
    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        streaming=True,
        temperature=0.1,
        max_retries=2,
    )
    chain = prompt | llm

    def _stream() -> Generator:
        try:
            for chunk in chain.stream({"context": context, "question": question}):
                yield chunk.content
        except Exception as exc:
            yield f"\n\n❌ **Groq API error:** `{exc}`"

    return _stream(), citations
