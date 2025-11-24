# Presidio AI - Legal Case Similarity Interface

A modern web application for finding similar legal cases using AI. This application takes a legal text input, finds similar cases from a database using Legal-BERT embeddings, and displays a comparison of case facts and judgements.

## 🚀 How to Run

### Prerequisites
- Python 3.13+
- Node.js 18+
- `uv` (optional, but recommended for Python dependency management)

### 1. Backend Setup

The backend is a FastAPI application that handles the logic and database interactions.

```bash
# Install dependencies
pip install fastapi uvicorn scikit-learn numpy torch transformers

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

Access the application at `http://localhost:5173`.

## 🛠️ Mock API & Customization

The application currently uses a **Mock Model** to demonstrate the "multiple models" feature, as the underlying database (`scotus_cases.db`) only contains a single source of truth for case facts and judgements.

### Where is the Mock Logic?

The logic for generating the response is located in:
`backend/logic.py`

Specifically, inside the `find_similar_cases` function:

```python
# backend/logic.py

result_item = {
    "case_name": case['case_name'],
    "similarity_score": float(score),
    "case_facts": {
        "Database Source": case['case_facts'],
        # MOCK DATA HERE
        "Mock Summarizer": f"[MOCK] Summary of facts for {case['case_name']}..."
    },
    "judgement": {
        "Database Source": case['decision'],
        # MOCK DATA HERE
        "Mock Summarizer": f"[MOCK] Alternative judgement summary..."
    }
}
```

### How to Replace with Real Models

To integrate real AI models (e.g., GPT-4, Claude, or a custom fine-tuned LLM):

1.  **Open `backend/logic.py`**.
2.  **Import your model client** (e.g., OpenAI API, HuggingFace pipeline).
3.  **Locate the `find_similar_cases` function**.
4.  **Replace the Mock strings** with actual calls to your model.

**Example Replacement:**

```python
# Pseudo-code example
from my_ai_models import summarize_facts, generate_judgement

# ... inside find_similar_cases loop ...

result_item = {
    # ...
    "case_facts": {
        "Database Source": case['case_facts'],
        "AI Summarizer": summarize_facts(case['case_facts']) # Real call
    },
    "judgement": {
        "Database Source": case['decision'],
        "AI Analysis": generate_judgement(case['decision']) # Real call
    }
}
```

## 📂 Project Structure

- `backend/`: FastAPI application and logic.
  - `main.py`: API entry point.
  - `logic.py`: Core logic for embeddings and search.
- `frontend/`: React application.
  - `src/components/`: UI components (`InputSection`, `ResultsTable`).
  - `src/api.js`: API client.
- `scotus_cases.db`: SQLite database containing legal cases.