import os
from dotenv import load_dotenv

# Mac / Apple Silicon fix
os.environ["GRPC_DNS_RESOLVER"] = "native"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq

load_dotenv()

def get_rag_response(query: str, chat_history: list = None):
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Vector Store with Namespace
    vectorstore = PineconeVectorStore(
        index_name="mini-rag-index", 
        embedding=embeddings,
        namespace="current_resume"
    )

    # 3. Initial Retrieval
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10, "namespace": "current_resume"}
    )
    docs = retriever.invoke(query)

    if not docs:
        return "I couldn't find any relevant details in the provided resume.", []

    # 4. Reranking (Phase 2 Roadmap)
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)
    docs = reranker.compress_documents(docs, query)

    # 5. Build context string
    context = "\n\n".join(doc.page_content for doc in docs)

    # 6. LLM Setup
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # 7. Prompt (Phase 3 Roadmap - Groundedness)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional resume assistant. Answer questions ONLY using the provided context. If the info is not there, say 'I don't know'."),
        ("system", f"Context:\n{context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({"question": query, "chat_history": chat_history or []})
    
    # IMPORTANT: Return both for Evaluation purposes
    return answer, docs