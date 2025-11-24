from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from . import logic

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data and model on startup
    logic.initialize_case_data()
    yield
    # Clean up if needed

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, specify frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str
    model: str = "legal-bert"

@app.post("/analyze")
async def analyze_case(request: AnalyzeRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    try:
        results = logic.find_similar_cases(request.text, request.model)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
