# Presidio AI - Legal Case Similarity Interface

Are you facing a legal challenge? Do you have no idea where to start?

Fear not!

PresidioAI will help you find Supreme Court cases where justices wrestled with situations similar to yours. All you have to do is type in a brief description of your scenario. Our web app will then help you find the precedent you need! 

Don't be lost. Get started today, and learn how the courts have evaluated cases like yours. Use that knowledge to evaluate whether you have a shot & start planning your argument.

*this tool is provided as an academic exercise for informational purposes only.  for legal advice, please seek the counsel of a lawyer.*

Powered by your choice of SBERT, nlpaueb’s LEGAL-BERT, and CaseHold's Legal-BERT.

## How to Run

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

## Project Structure

- `backend/`: FastAPI application and logic.
  - `main.py`: API entry point.
  - `logic.py`: Core logic for embeddings and search.
- `frontend/`: React application.
  - `src/components/`: UI components (`InputSection`, `ResultsTable`).
  - `src/api.js`: API client.
- `scotus_cases.db`: SQLite database containing legal cases.