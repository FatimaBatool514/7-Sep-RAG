import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Updated imports pointing to langchain_classic
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Page configuration
st.set_page_config(page_title="RAG PDF Q&A Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF (powered by Groq)")

# Sidebar for API key input
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not groq_api_key:
    groq_api_key = os.environ.get("GROQ_API_KEY", "")

# Load HuggingFace Embeddings Model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file and groq_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Processing PDF document..."):
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        os.remove(tmp_file_path)

    st.success("PDF processed successfully! Ask your questions below.")

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the following context provided.
    Provide a clear and concise answer. If you cannot find the answer in the context, 
    simply state "I couldn't find the answer in the provided document."

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_query = st.text_input("Ask a question about your document:")

    if user_query:
        with st.spinner("Searching document and generating answer..."):
            response = retrieval_chain.invoke({"input": user_query})
            st.subheader("Answer:")
            st.write(response["answer"])

            with st.expander("View Source Document Chunks"):
                for idx, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {idx+1}:**")
                    st.write(doc.page_content)
                    st.divider()

elif not groq_api_key:
    st.info("Please enter your Groq API key in the sidebar to proceed.")
else:
    st.info("Please upload a PDF file to begin.")
