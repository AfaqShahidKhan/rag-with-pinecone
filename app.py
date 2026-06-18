"""
app.py — Streamlit RAG interface.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

from typing import Generator

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports after page config ────────────────────────────────────────────
from src.config.settings import settings
from src.generation.prompt_builder import build_prompt
from src.retrieval.retriever import search
import ollama


# ── Session state defaults ────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],
        "ingestion_status": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ── Streaming generator ───────────────────────────────────────────────────────
def _stream_answer(prompt: str, model: str) -> Generator[str, None, None]:
    for chunk in ollama.generate(model=model, prompt=prompt, stream=True):
        token = chunk.get("response", "")
        if token:
            yield token


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Model")
    generation_model = st.selectbox(
        "Generation model",
        options=["gemma3", "llama3", "deepseek-r1", "mistral"],
        index=0,
        help="Must be available in your local Ollama instance.",
    )
    embed_model = st.text_input(
        "Embedding model",
        value=settings.ollama.embed_model,
        disabled=True,
        help="Set in .env — changing requires re-ingestion.",
    )

    st.divider()

    st.subheader("Retrieval")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=15, value=settings.retrieval.top_k)
    score_threshold = st.slider(
        "Min similarity score",
        min_value=0.0, max_value=1.0,
        value=0.0, step=0.05,
        help="Filter out chunks below this cosine similarity score.",
    )
    show_sources = st.toggle("Show sources", value=True)

    st.divider()

    st.subheader("📄 Ingestion")
    st.caption(f"PDF directory: `{settings.data_raw}`")

    if st.button("▶ Run Ingestion", use_container_width=True):
        with st.spinner("Ingesting PDFs — this may take a minute..."):
            try:
                from src.ingestion.pipeline import run_ingestion_pipeline
                total = run_ingestion_pipeline()
                st.session_state.ingestion_status = f"✅ Done — {total} vectors indexed."
            except Exception as e:
                st.session_state.ingestion_status = f"❌ Error: {e}"

    if st.session_state.ingestion_status:
        if st.session_state.ingestion_status.startswith("✅"):
            st.success(st.session_state.ingestion_status)
        else:
            st.error(st.session_state.ingestion_status)

    st.divider()

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Index: `{settings.pinecone.index_name}` · {settings.pinecone.region}")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("📚 RAG Assistant")
st.caption("Ask questions about your ingested documents.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)", expanded=False):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(
                        f"**[{i+1}]** `{src['source']}` — Page {src['page']} "
                        f"*(score: {src['score']:.4f})*"
                    )
                    st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))
                    if i < len(msg["sources"]) - 1:
                        st.divider()

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        threshold = score_threshold if score_threshold > 0 else None

        with st.spinner("Searching documents..."):
            results = search(question, top_k=top_k, score_threshold=threshold)

        if not results:
            answer = "I couldn't find any relevant information in the documents."
            st.markdown(answer)
        else:
            package = build_prompt(question, results)
            answer = st.write_stream(_stream_answer(package.prompt, generation_model))

        # Show sources inline
        if show_sources and results:
            with st.expander(f"📎 Sources ({len(package.sources)} chunks)", expanded=False):
                for i, src in enumerate(package.sources):
                    st.markdown(
                        f"**[{i+1}]** `{src.source}` — Page {src.page} "
                        f"*(score: {src.score:.4f})*"
                    )
                    st.caption(src.text[:300] + ("..." if len(src.text) > 300 else ""))
                    if i < len(package.sources) - 1:
                        st.divider()

        # Persist to history
        sources_serialized = [
            {
                "source": s.source,
                "page": s.page,
                "score": s.score,
                "text": s.text,
            }
            for s in package.sources
        ] if results else []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_serialized,
        })