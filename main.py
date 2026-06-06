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

# Setup Logging (Bonus Test Case 1)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("api_usage.log"), logging.StreamHandler()]
)

# Initialize FastAPI App
app = FastAPI(
    title="FXMedia Assessment - Test Case 1 & 2",
    description="LLM API Wrapper and Vector Database Integration (Using ChromaDB Default Embedding)."
)

# Initialize Google GenAI Client (Untuk Test Case 1)
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Inisialisasi ChromaDB (Menggunakan Default Embedding Function: all-MiniLM-L6-v2)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge_base")


# ==========================================
# Pydantic Schemas
# ==========================================
class QueryRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str


# ==========================================
# EVENT STARTUP: Auto-Seed 5 Dokumen (Test Case 2)
# ==========================================
@app.on_event("startup")
async def startup_event():
    """Menjalankan proses memasukkan 5 dokumen ke Vector DB."""
    logging.info("Memulai proses seeding dokumen ke ChromaDB...")
    
    documents = [
        "LangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs).",
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
        logging.info("ChromaDB berhasil di-seed dengan 5 dokumen menggunakan model bawaan!")
    except Exception as e:
        logging.error(f"Gagal melakukan seeding ke Vector DB: {str(e)}")


# ==========================================
# TEST CASE 1: LLM API Wrapper
# ==========================================
@app.post("/api/ask")
async def api_ask(request: QueryRequest):
    start_time = time.time()
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key belum dikonfigurasi.")

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
        raise HTTPException(status_code=502, detail=f"Gagal terhubung dengan LLM API: {str(e)}")


# ==========================================
# TEST CASE 2: Vector Database Integration
# ==========================================
@app.post("/api/search")
async def api_search(request: SearchRequest):
    """Endpoint untuk mencari dokumen yang mirip berdasarkan vektor embedding."""
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
        raise HTTPException(status_code=500, detail=f"Gagal melakukan pencarian vektor: {str(e)}")