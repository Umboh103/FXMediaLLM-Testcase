# FXMedia LLM Developer Assessment

This repository contains the backend implementation for the LLM Developer Intern technical assessment at FXMedia. It is built using **Python (FastAPI)**, **ChromaDB** (Vector Database), and the **Gemini API**.

---

## Features Completed
- [x] **Test Case 1:** LLM API Wrapper (`/api/ask`) with request logging and error handling.
- [x] **Test Case 2:** Vector DB Integration (`/api/search`) using ChromaDB with auto-seeding on startup and similarity score calculation.
- [x] **Test Case 3:** Simple RAG Backend (`/api/rag`) combining retrieval and generative context with strict constraints.

*(Note: Test Case 4 regarding Graph RAG is attached as a separate PDF/document in the submission email).*

---

## Tech Stack
* **Framework:** FastAPI (Python)
* **LLM Provider:** Google GenAI (Gemini 2.5 Flash)
* **Vector Database:** ChromaDB (Local In-Memory)

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone [https://github.com/Umboh103/testcase-LLM-FXMedia.git](https://github.com/Umboh103/testcase-LLM-FXMedia.git)
cd testcase-LLM-FXMedia
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell / Command Prompt)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory of the project and insert your Gemini API Key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
> **Note:** The `.env` file and `api_usage.log` are strictly ignored by `.gitignore` to prevent any accidental exposure of sensitive credentials to the public repository.

### 5. Start the Application Server
Run the FastAPI application locally using Uvicorn:
```bash
uvicorn main:app --reload
```

Upon successful startup, the server will automatically seed the vector database, and you will see this confirmation in your terminal console:
```text
2026-06-08 20:15:00 [INFO] ChromaDB successfully seeded with 5 documents!
```

---

## API Testing & Documentation

FastAPI automatically generates interactive documentation for testing. While the server is running, open your web browser and navigate to:

👉 **Swagger UI Interface:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Available Endpoints:

1. **`POST /api/ask` (LLM API Wrapper)**
   * **Payload:** `{"question": "Your question here"}`
2. **`POST /api/search` (Vector Similarity Search)**
   * **Payload:** `{"query": "Your search query here"}`
3. **`POST /api/rag` (Simple RAG Backend)**
   * **Payload:** `{"question": "Your question here"}`