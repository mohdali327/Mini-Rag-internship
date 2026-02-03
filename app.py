import streamlit as st
import time
import os
import tempfile

# Core LangChain & Integration Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, AIMessage
from pinecone import Pinecone

# Internal Engine Import
from final_engine import get_rag_response

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="RAG Intern Bot | Resume Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Llama-3 Resume RAG Assistant")
st.markdown("---")

# ------------------ SIDEBAR: DOCUMENT MANAGEMENT ------------------
with st.sidebar:
    st.header("📂 Document Manager")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF (Resume, Notes, Paper)",
        type="pdf"
    )

    if uploaded_file:
        st.success("File received!")

        if st.button("Process & Index Document"):
            with st.spinner("Processing and indexing..."):
                try:
                    # 1. Save uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        pdf_path = tmp.name

                    # 2. Load PDF using PyPDFLoader (Reliable for most PDFs)
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    if not docs:
                        st.error("Text extraction failed. Is this a scanned image PDF?")
                        st.stop()
                    
                    st.info(f"📄 Extracted {len(docs)} pages.")

                    # 3. Chunking (Important for precise retrieval)
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=600,
                        chunk_overlap=100
                    )
                    chunks = splitter.split_documents(docs)
                    st.info(f"✂️ Created {len(chunks)} text chunks.")

                    # 4. Initialize Embeddings
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                    # 5. SAFE NAMESPACE MANAGEMENT (Wipe old data if exists)
                    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                    index_name = "mini-rag-index"
                    index = pc.Index(index_name)
                    stats = index.describe_index_stats()
                    
                    target_namespace = "current_resume"
                    
                    if target_namespace in stats.get('namespaces', {}):
                        st.info(f"🧹 Clearing existing data in '{target_namespace}'...")
                        index.delete(delete_all=True, namespace=target_namespace)
                    
                    # 6. Index new document into the namespace
                    PineconeVectorStore.from_documents(
                        chunks,
                        embeddings,
                        index_name=index_name,
                        namespace=target_namespace
                    )

                    # Clean up temp file
                    os.remove(pdf_path)
                    st.success("✅ Document indexed successfully!")

                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()
    st.header("⚙️ System Status")
    st.success("Pinecone: Connected")
    st.success("Groq: LLaMA-3.1 Ready")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ------------------ CHAT STATE (MEMORY) ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ DISPLAY CHAT HISTORY ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ CHAT INPUT & EXECUTION ------------------
if question := st.chat_input("Ask something about the uploaded document..."):
    
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        with st.spinner("Analyzing context..."):
            try:
                # 1. Format chat history for LangChain Memory
                chat_history_objs = []
                for m in st.session_state.messages[:-1]: # exclude current question
                    if m["role"] == "user":
                        chat_history_objs.append(HumanMessage(content=m["content"]))
                    else:
                        chat_history_objs.append(AIMessage(content=m["content"]))

                # 2. Call Engine (Returns answer AND docs for eval/source tracking)
                answer, retrieved_docs = get_rag_response(question, chat_history_objs)

                # 3. Show Answer
                placeholder.markdown(answer)
                
                # 4. Professional Touch: Show Sources
                with st.expander("📚 View Document Sources"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.caption(doc.page_content)

                # 5. Save to session state
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Ensure your API keys are correct in the .env file.")

