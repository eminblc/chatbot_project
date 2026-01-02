"""FastAPI REST API for TV Series Chatbot with multi-series support."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, List, Dict
from src.core.multi_series_service import MultiSeriesService
from src.utils.logging import setup_logging, get_logger
from src.utils.validators import validate_query
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()
setup_logging()
logger = get_logger(__name__)
multi_series_service = MultiSeriesService()

app = FastAPI(
    title="Series Chatbot API",
    description="RAG-based chatbot for TV series content",
    version="1.2"
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from config.paths import get_series_paths
    _, _, default_db_dir = get_series_paths("stranger_things")
    if not os.path.exists(default_db_dir):
        logger.warning("Default series database not found. Process data first.")
    logger.info("API initialized successfully")
except (ImportError, ValueError, FileNotFoundError, OSError) as e:
    logger.error("Failed to initialize API: %s", e, exc_info=True)


class QueryRequest(BaseModel):
    """Request model for /ask endpoint."""
    query: str
    series: str = "stranger_things"
    use_local: bool = True
    season: Optional[int] = None
    episode: Optional[int] = None
    
    @validator('query')
    def validate_query_field(cls, v):  # pylint: disable=no-self-argument
        validate_query(v)
        return v
    
    @validator('series')
    def validate_series_field(cls, v):  # pylint: disable=no-self-argument
        allowed = ["stranger_things", "breaking_bad", "all"]
        if v not in allowed:
            raise ValueError(f"Series must be one of: {', '.join(allowed)}")
        return v


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Process query and return answer with sources. Supports single/multi-series queries."""
    logger.info("Processing query: %s... (series: %s, local: %s)", 
               request.query[:50], request.series, request.use_local)
    
    if request.series == "all":
        return multi_series_service.query_all_series(
            query=request.query,
            season=request.season,
            episode=request.episode,
            use_local=request.use_local
        )
    
    result = multi_series_service.query_single_series(
        series_name=request.series,
        query=request.query,
        season=request.season,
        episode=request.episode,
        use_local=request.use_local
    )
    
    return {
        "status": "success",
        "original_query": request.query,
        "optimized_query": result.optimized_query,
        "answer": result.answer,
        "sources": result.sources,
        "source_count": len(result.sources)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Series Chatbot API"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Series Chatbot API",
        "version": "1.2",
        "endpoints": {
            "/ask": "POST - Query the chatbot",
            "/evaluate": "POST - Run RAGAS evaluation on test set",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation (Swagger UI)"
        },
        "supported_series": ["stranger_things", "breaking_bad", "all"]
    }


class EvaluationRequest(BaseModel):
    """Request model for /evaluate endpoint."""
    test_set_path: str = "data/test/test_set.json"
    save_results: bool = True
    use_local: bool = True


@app.post("/evaluate")
async def run_evaluation(request: EvaluationRequest):
    """Run RAGAS evaluation on test set and return metrics."""
    logger.info("Starting RAGAS evaluation with test set: %s", request.test_set_path)
    
    try:
        # Lazy import to avoid loading RAGAS on every API startup
        from scripts.evaluate_ragas import RAGASEvaluator
        
        evaluator = RAGASEvaluator(test_set_path=request.test_set_path)
        results = evaluator.evaluate_test_set(use_local=request.use_local)
        
        if request.save_results:
            output_path = evaluator.save_results(results)
            logger.info("Evaluation results saved to: %s", output_path)
        
        return {
            "status": "success",
            "metadata": results['metadata'],
            "scores": {
                "faithfulness": float(results['ragas_scores']['faithfulness']),
                "answer_relevancy": float(results['ragas_scores']['answer_relevancy']),
                "context_precision": float(results['ragas_scores']['context_precision']),
                "context_recall": float(results['ragas_scores']['context_recall'])
            },
            "saved_to": output_path if request.save_results else None
        }
    except FileNotFoundError as e:
        logger.error("Test set not found: %s", e)
        return {
            "status": "error",
            "message": f"Test set not found: {str(e)}"
        }
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        return {
            "status": "error",
            "message": f"Evaluation failed: {str(e)}"
        }

