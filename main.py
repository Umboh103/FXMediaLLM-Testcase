import os
import time
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from google import genai
from dotenv import load_dotenv

# Load environment variables dari file .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("api_usage.log"), logging.StreamHandler()]
)

app = FastAPI(
    title="FXMedia Assessment - Final Project",
    description="Full Backend Service: LLM Wrapper, Vector DB, and RAG Implementation.",
    version="1.0.0"
)

# Initialize Google GenAI Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Initialize ChromaDB 
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge_base")


# ==========================================
# Pydantic Schemas (Validasi Data)
# ==========================================
class QueryRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str

class RagResponse(BaseModel):
    query: str
    context_used: List[str]
    final_answer: str


# ==========================================
# EVENT STARTUP: Auto-Seed 5 Dokumen (Test Case 2)
# ==========================================
@app.on_event("startup")
async def startup_event():
    """Inserts 5 documents into the Vector DB on server startup."""
    logging.info("Starting the document seeding process into ChromaDB...")
    
    documents = [
        "LangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs). It was developed by Harrison Chase.",
        "Machine learning is a subset of artificial intelligence (AI) that focuses on building systems that learn from data.",
        "Supervised learning is a type of machine learning where the model is trained on labeled data.",
        "Vector databases store high-dimensional vectors representing data, allowing for incredibly fast similarity searches.",
        "Retrieval-Augmented Generation (RAG) is a technique that grants LLMs access to external knowledge bases to provide accurate answers."
    ]
    
    metadatas = [
        {"title": "LangChain Framework", "source": "AI Tech Docs"},
        {"title": "Introduction to ML", "source": "Machine Learning Foundations"},
        {"title": "Supervised Learning", "source": "Data Science Basics"},
        {"title": "Vector DB 101", "source": "Advanced Architectures"},
        {"title": "Understanding RAG", "source": "Tech Trends 2026"}
    ]
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info("ChromaDB successfully seeded with 5 documents!")
    except Exception as e:
        logging.error(f"Failed to seed the Vector DB: {str(e)}")


# ==========================================
# TEST CASE 1: LLM API Wrapper
# ==========================================
@app.post("/api/ask")
async def api_ask(request: QueryRequest):
    start_time = time.time()
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key is not configured.")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.question,
        )
        process_time = time.time() - start_time
        logging.info(f"[/api/ask] Prompt: '{request.question}' | Waktu: {process_time:.2f}s")
        return {"answer": response.text}
    except Exception as e:
        logging.error(f"LLM API Error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Failed to connect to the LLM API: {str(e)}")


# ==========================================
# TEST CASE 2: Vector Database Integration
# ==========================================
@app.post("/api/search")
async def api_search(request: SearchRequest):
    try:
        results = collection.query(
            query_texts=[request.query],
            n_results=3
        )
        
        formatted_results = []
        if results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            
            for i in range(len(docs)):
                similarity_score = round(1 / (1 + distances[i]), 4)
                formatted_results.append({
                    "text": docs[i],
                    "score": similarity_score,
                    "metadata": metas[i]
                })
                
        return {
            "query": request.query,
            "results": formatted_results
        }
    except Exception as e:
        logging.error(f"Vector Search Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to perform vector search: {str(e)}")

# ==========================================
# TEST CASE 3: Simple RAG Backend
# ==========================================
@app.post("/api/rag", response_model=RagResponse)
async def api_rag(request: QueryRequest):
    """RAG endpoint combining Vector Retrieval and LLM Generation."""
    start_time = time.time()
    try:
        search_results = collection.query(
            query_texts=[request.question],
            n_results=3
        )
        
        retrieved_docs = search_results["documents"][0] if search_results["documents"] else []
        
       # Combining the retrieved documents into a single text string for context
        context_str = "\n".join([f"- {doc}" for doc in retrieved_docs]) if retrieved_docs else "No context available."
        
       # Strict prompt to force the exact text output
        prompt_template = (
            f"You are a strict data extractor. Your task is to answer the Question based ONLY on the Context provided.\n"
            f"CRITICAL RULE: You MUST reply by copying the EXACT text/sentence from the Context that answers the question. "
            f"Do NOT rephrase, do NOT summarize, and do NOT add any extra conversational words.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question:\n{request.question}\n\n"
            f"Answer:"
        )
        
      
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_template,
        )
         
        process_time = time.time() - start_time
        logging.info(f"[/api/rag] Question: '{request.question}' | Success in {process_time:.2f}s")
        
        return RagResponse(
            query=request.question,
            context_used=retrieved_docs,
            final_answer=response.text.strip()
        )
        
    except Exception as e:
        logging.error(f"An error occurred in the RAG system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred in the RAG system: {str(e)}")