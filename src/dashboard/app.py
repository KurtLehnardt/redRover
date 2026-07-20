"""FastAPI dashboard for redRover — local web UI."""

import asyncio
from pathlib import Path
from datetime import datetime

import httpx

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import load_config
from ..database import Database
from ..sensors.simulator import generate_sample
from ..sensors.vibration import FaultType, extract_features
from ..ai.analyzer import VibrationAnalyzer
from ..telemetry import init_telemetry

app = FastAPI(title="redRover Dashboard", version="0.1.0")

# Auto-instrument FastAPI with OpenTelemetry
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)
except ImportError:
    pass

TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

config = load_config()
db = Database(config.database.path)


@app.on_event("startup")
async def startup():
    await db.init()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    patrols = await db.get_recent_patrols(10)
    faults = await db.get_active_faults()
    return templates.TemplateResponse(request, "index.html", context={
        "patrols": patrols,
        "faults": faults,
        "now": datetime.now().isoformat(),
    })


@app.get("/station/{station_id}", response_class=HTMLResponse)
async def station_detail(request: Request, station_id: str):
    history = await db.get_station_history(station_id, 50)
    return templates.TemplateResponse(request, "station.html", context={
        "station_id": station_id,
        "history": history,
    })


@app.post("/api/demo-patrol")
async def trigger_demo_patrol():
    """Trigger a simulated patrol for demo purposes."""
    from ..main import run_patrol
    results = await run_patrol(simulate=True, skip_ai=False)
    return {
        "status": "complete",
        "stations_visited": 4,
        "faults_detected": len([r for r in results if r.fault_type != FaultType.NORMAL]),
    }


@app.get("/api/faults")
async def get_faults():
    """Get current active faults (HTMX endpoint)."""
    faults = await db.get_active_faults()
    return faults


@app.get("/api/alerts")
async def get_alerts():
    """Get recent alerts (from in-memory alert manager)."""
    from ..alerting import AlertManager
    mgr = AlertManager()
    return mgr.recent_alerts


@app.get("/api/station/{station_id}/history")
async def get_station_history(station_id: str):
    history = await db.get_station_history(station_id)
    return history


async def _check_ai_status() -> dict:
    """Check Ollama connectivity and model availability."""
    ollama_host = config.ai.ollama_host
    model = config.ai.model
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            response.raise_for_status()
            data = response.json()
            available_models = [
                m.get("name", "") for m in data.get("models", [])
            ]
            # Ollama model names may include a tag suffix (e.g. "gemma3:latest")
            model_found = any(
                m == model or m.startswith(f"{model}:") for m in available_models
            )
            if model_found:
                return {
                    "ai_status": "online",
                    "ai_model": model,
                    "inference_mode": "llm",
                    "message": f"Ollama is running and model '{model}' is available",
                }
            else:
                return {
                    "ai_status": "model_missing",
                    "ai_model": model,
                    "inference_mode": "rule_based",
                    "message": f"Ollama is running but model '{model}' is not found. Available: {available_models}",
                }
    except Exception:
        return {
            "ai_status": "offline",
            "ai_model": model,
            "inference_mode": "rule_based",
            "message": f"Cannot reach Ollama at {ollama_host}. Fusion analysis will use rule-based fallback.",
        }


@app.get("/api/system-status")
async def system_status():
    return await _check_ai_status()


@app.get("/api/health")
async def health():
    ai_status = await _check_ai_status()
    return {"status": "ok", "version": "0.1.0", "ai": ai_status}


def start():
    """Entry point for running the dashboard."""
    import uvicorn
    uvicorn.run(
        "src.dashboard.app:app",
        host=config.dashboard.host,
        port=config.dashboard.port,
        reload=True,
    )


if __name__ == "__main__":
    start()
