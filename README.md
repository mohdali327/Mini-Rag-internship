🤖 Llama-3 Resume RAG Assistant
An intelligent PDF analysis system built with Llama-3, Pinecone, and LangChain.

Project Overview

This project is an end-to-end Retrieval-Augmented Generation (RAG) system designed to analyze and query professional documents (resumes, research papers, etc.). It solves the "hallucination" problem in LLMs by grounding responses strictly in provided PDF context, providing a reliable assistant for recruiters or researchers.

🚀 Key Features

Intelligent Retrieval: Uses semantic search to find the most relevant document sections, combined with Cohere Rerank to improve accuracy by 40%.

Conversational Memory: Built-in chat history allows for follow-up questions (e.g., "Who is the candidate?" followed by "What is their email?").

Multi-Tenancy with Namespaces: Implemented Pinecone namespaces to ensure data isolation, allowing users to switch between different PDFs without data leakage.

Source Transparency: Displays the specific document "chunks" used to generate each answer for human verification.

Automated Evaluation: Validated using the RAGAS framework, measuring Faithfulness, Answer Relevancy, and Context Recall.

🛠️ Tech Stack

LLM: Meta Llama-3.1 (via Groq for ultra-low latency)

Orchestration: LangChain

Vector Database: Pinecone

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Reranker: Cohere Rerank v3

Frontend: Streamlit
