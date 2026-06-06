# FXMedia LLM Developer Assessment

This repository contains the backend implementation for the LLM Developer Intern technical assessment at FXMedia. It is built using **Python (FastAPI)**, **ChromaDB** (Vector Database), and the **Gemini API**.

## Features Completed
- [x] **Test Case 1:** LLM API Wrapper (`/api/ask`) with request logging and error handling.
- [x] **Test Case 2:** Vector DB Integration (`/api/search`) using ChromaDB with auto-seeding on startup and similarity score calculation.
- [x] **Test Case 3:** Simple RAG Backend (`/api/rag`) combining retrieval and generative context.

*(Note: Test Case 4 regarding Graph RAG is attached as a separate PDF/document in the submission email).*

## Tech Stack
* **Framework:** FastAPI (Python)
* **LLM Provider:** Google GenAI (Gemini 2.5 Flash & text-embedding-004)
* **Vector Database:** ChromaDB (Local In-Memory)

## How to Run the Project

**1. Clone the repository**
```bash
git clone <your-github-repo-link>
cd fxmedia-llm-assessment