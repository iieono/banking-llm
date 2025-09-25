"""FastAPI web service for Bank AI LLM system."""

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from .config import settings
from .database import db_manager
from .excel_export import excel_exporter
from .llm_service import llm_service


# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    export_format: str = "excel"


class QueryResponse(BaseModel):
    success: bool
    sql_query: Optional[str] = None
    results: Optional[List[Dict]] = None
    export_path: Optional[str] = None
    error: Optional[str] = None
    explanation: Optional[str] = None


class DatabaseStats(BaseModel):
    clients: int
    accounts: int
    transactions: int
    regions: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="Bank AI LLM Data Analyst",
    description="Transform natural language queries into SQL and get professional Excel reports",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add request timeout and size limits
from contextlib import asynccontextmanager
from fastapi import Request
import asyncio

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    """Add request timeout middleware."""
    try:
        return await asyncio.wait_for(
            call_next(request),
            timeout=settings.request_timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Bank AI LLM Data Analyst API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Execute natural language query",
            "GET /stats": "Get database statistics",
            "GET /samples": "Get sample queries",
            "POST /setup": "Initialize database",
            "GET /download/{filename}": "Download generated files"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """Execute a natural language query and return results with Excel export."""
    try:
        logger.info(f"Received query: {request.query}")

        # Generate SQL from natural language
        llm_result = llm_service.generate_sql(request.query)

        if not llm_result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to generate SQL: {llm_result.get('error', 'Unknown error')}"
            )

        sql_query = llm_result['sql_query']

        # Execute SQL query
        results = db_manager.execute_query(sql_query)

        # Export to Excel if requested
        export_path = None
        if request.export_format.lower() == "excel" and results:
            export_path = excel_exporter.export_query_results(results, llm_result)

        # Get query explanation
        explanation = llm_service.get_query_explanation(sql_query)

        return QueryResponse(
            success=True,
            sql_query=sql_query,
            results=results[:100],  # Limit results in API response
            export_path=export_path,
            explanation=explanation
        )

    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get database statistics."""
    try:
        stats = db_manager.get_database_stats()
        return DatabaseStats(
            clients=stats['clients'],
            accounts=stats['accounts'],
            transactions=stats['transactions'],
            regions=stats['regions']
        )
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/samples")
async def get_sample_queries():
    """Get sample queries that users can try."""
    return {
        "samples": llm_service.suggest_sample_queries()
    }


@app.post("/setup")
async def setup_database():
    """Initialize database with mock data."""
    try:
        logger.info("Starting database setup...")
        db_manager.create_tables()
        db_manager.generate_mock_data()
        stats = db_manager.get_database_stats()

        return {
            "success": True,
            "message": "Database initialized successfully",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated Excel files."""
    try:
        file_path = settings.excel_output_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        stats = db_manager.get_database_stats()

        # Test LLM service (simple test)
        test_result = llm_service.generate_sql("test query")

        return {
            "status": "healthy",
            "database": "connected",
            "llm_service": "connected" if test_result else "error",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )