import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# Load environment variables dari file .env
load_dotenv()

# Setup Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("api_usage.log"), logging.StreamHandler()]
)

# Initialize FastAPI App
app = FastAPI(
    title="FXMedia Assessment - Test Case 1",
    description="LLM API Wrapper using FastAPI and Gemini."
)

# Initialize Google GenAI Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.warning("GEMINI_API_KEY is not set in the environment variables!")
client = genai.Client()


class QueryRequest(BaseModel):
    question: str

# ==========================================
# TEST CASE 1: LLM API Wrapper
# ==========================================
@app.post("/api/ask")
async def api_ask(request: QueryRequest):
    """Endpoint untuk menembak LLM API secara langsung."""
    start_time = time.time()
    
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured on server.")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=request.question,
        )
        
        process_time = time.time() - start_time
        logging.info(f"[/api/ask] Prompt: '{request.question}' | Model: gemini-2.5-flash | Time: {process_time:.2f}s")
        
        return {"answer": response.text}
        
    except Exception as e:
        logging.error(f"LLM API Error: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Failed to communicate with LLM API: {str(e)}")