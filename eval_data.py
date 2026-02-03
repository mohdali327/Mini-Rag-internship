import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
# Updated imports for Ragas v0.3+ / 1.0 compatibility
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from final_engine import get_rag_response
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Define your Ground Truth (Correct Answers)
eval_questions = [
    "What is the name of the candidate?",
    "Where is the candidate studying?",
    "What is the candidate's degree and year?",
    "Does the candidate have experience with Pinecone?",
    "What projects has the candidate worked on?"
]

ground_truths = [
    "Mohd Ali",
    "Manipal University Jaipur",
    "B.Tech in Data Science, 3rd Year",
    "Yes, the candidate has experience with Pinecone vector database for RAG projects.",
    "The candidate worked on a Chatbot using Llama 3, a Next Word Prediction model, and an Earthquake Forecaster."
]

def run_evaluation():
    print("🚀 Starting Evaluation...")
    results = []

    # 2. Collect RAG responses for each question
    for query, truth in zip(eval_questions, ground_truths):
        print(f"🧐 Querying: {query}")
        
        # Call your engine
        answer, docs = get_rag_response(query)
        
        results.append({
            "user_input": query,       # Ragas now prefers 'user_input' over 'question'
            "response": answer,        # Ragas now prefers 'response' over 'answer'
            "retrieved_contexts": [d.page_content for d in docs], # Ragas now prefers 'retrieved_contexts'
            "reference": truth         # Ragas now prefers 'reference' over 'ground_truth'
        })

    # 3. Format data for Ragas
    dataset = Dataset.from_list(results)

    # 4. Setup Evaluator LLM (Updated to Llama 3.3 70B)
    evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Wrap them for Ragas
    llm_wrapper = LangchainLLMWrapper(evaluator_llm)
    emb_wrapper = LangchainEmbeddingsWrapper(evaluator_embeddings)

    # 5. Execute Evaluation
    print("📊 Calculating Ragas Metrics...")
    score = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_wrapper,
        embeddings=emb_wrapper
    )

    # 6. Display Results
    df = score.to_pandas()
    print("\n✅ EVALUATION REPORT:")
    
    # Check for actual column names in df to avoid KeyError
    print(df.head())
    
    # Save to CSV for your internship report
    df.to_csv("rag_evaluation_results.csv", index=False)
    print("\n📂 Results saved to rag_evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()