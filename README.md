# Presidio AI - Legal Case Similarity Interface

A modern web application for finding similar legal cases using AI. This application takes a legal text input, finds similar cases from a database using Legal-BERT embeddings, and displays a comparison of case facts and judgements.

## 🚀 How to Run

### Prerequisites
- Python 3.13+
- Node.js 18+
- `uv` package manager for Python

### 0. UV Setup

```bash
# Install uv
pip install uv

# Download environment
uv sync

# Activate environment
source .venv/bin/activate
```

### 1. Backend Setup

The backend is a FastAPI application that handles the logic and database interactions.

```bash
# Start the server
uvicorn backend.main:app --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

The frontend is a React application built with Vite and Tailwind CSS.

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

## 📂 Project Structure

- `backend/`: FastAPI application and logic.
  - `main.py`: API entry point.
  - `logic.py`: Core logic for embeddings and search.
- `frontend/`: React application.
  - `src/components/`: UI components (`InputSection`, `ResultsTable`).
  - `src/api.js`: API client.
- `scotus_cases.db`: SQLite database containing legal cases.