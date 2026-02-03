import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Updated for Path 2
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

def ingest_document(file_path):
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    print(f"✅ Loaded {len(raw_docs)} pages.")

    # 2. Chunking (Requirement: 800-1200 tokens)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"✅ Created {len(chunks)} chunks.")

    # 3. Local Embeddings (No API Key or Quota needed!)
    # This model will download once (~80MB) and run on your CPU
    print("⏳ Initializing local HuggingFace model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Upload to Pinecone (Ensure dimensions = 384 in dashboard)
    index_name = "mini-rag-index"
    
    print("🚀 Uploading to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    print("✨ Ingestion complete! Your data is ready in the cloud.")
    return vectorstore

if __name__ == "__main__":
    # Ensure this matches your sidebar filename
    my_pdf_file = "sample_data.pdf" 
    ingest_document(my_pdf_file)