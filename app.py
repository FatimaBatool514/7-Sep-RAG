import os
import re
from io import BytesIO

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


st.set_page_config(
    page_title="PDF RAG with Groq",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or your environment."
        )
    return Groq(api_key=api_key)


def extract_pdf_text(uploaded_file):
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page": page_number, "text": text})

    return pages


def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "page": page["page"],
                    }
                )

            if end >= len(text):
                break

            start = max(end - overlap, start + 1)

    return chunks


def build_faiss_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve(query, index, chunks, embedding_model, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            }
        )

    return results


def generate_answer(question, retrieved_chunks):
    client = get_groq_client()

    context_parts = []
    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Context {i} | Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful RAG assistant.
Answer the user's question using only the supplied document context.
If the answer is not supported by the context, say that the answer was not
found in the uploaded document. Do not invent facts.
When possible, mention the relevant page number(s)."""

    user_prompt = f"""Document context:

{context}

Question:
{question}
"""

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


def reset_document():
    for key in ["index", "chunks", "filename", "page_count", "chunk_count"]:
        st.session_state.pop(key, None)


st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload a PDF → extract text → chunk → embed → FAISS retrieval → "
    "answer with an open-weight model served by Groq."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks", 2, 10, TOP_K)
    st.write(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.write(f"LLM: `{LLM_MODEL}`")

    if st.button("Clear document", use_container_width=True):
        reset_document()
        st.rerun()

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="The PDF is processed in memory and is not saved to the GitHub repository.",
)

if uploaded_file is not None:
    if st.session_state.get("filename") != uploaded_file.name:
        with st.spinner("Processing PDF and building FAISS index..."):
            try:
                pages = extract_pdf_text(uploaded_file)

                if not pages:
                    st.error(
                        "No extractable text was found. "
                        "This app currently expects a text-based PDF."
                    )
                    st.stop()

                chunks = chunk_text(pages)
                embedding_model = load_embedding_model()
                index = build_faiss_index(chunks, embedding_model)

                st.session_state["index"] = index
                st.session_state["chunks"] = chunks
                st.session_state["filename"] = uploaded_file.name
                st.session_state["page_count"] = len(pages)
                st.session_state["chunk_count"] = len(chunks)

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")
                st.stop()

    st.success(
        f"Indexed **{st.session_state['filename']}** — "
        f"{st.session_state['page_count']} pages, "
        f"{st.session_state['chunk_count']} chunks."
    )

    question = st.chat_input("Ask a question about the uploaded PDF")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                embedding_model = load_embedding_model()

                results = retrieve(
                    question,
                    st.session_state["index"],
                    st.session_state["chunks"],
                    embedding_model,
                    top_k=top_k,
                )

                if not results:
                    st.warning("No relevant context was retrieved.")
                    st.stop()

                answer = generate_answer(question, results)
                st.markdown(answer)

                with st.expander("Retrieved context"):
                    for i, item in enumerate(results, start=1):
                        st.markdown(
                            f"**Context {i} — Page {item['page']} — "
                            f"Similarity: {item['score']:.3f}**"
                        )
                        st.write(item["text"])

            except Exception as exc:
                st.error(f"Error while generating the answer: {exc}")

else:
    st.info("Upload a PDF to build the RAG index.")
