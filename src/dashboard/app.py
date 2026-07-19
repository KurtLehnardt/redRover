"""FastAPI dashboard for redRover — local web UI."""

import asyncio
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import load_config
from ..database import Database
from ..sensors.simulator import generate_sample
from ..sensors.vibration import FaultType, extract_features
from ..ai.analyzer import VibrationAnalyzer

app = FastAPI(title="redRover Dashboard", version="0.1.0")

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
    return templates.TemplateResponse("index.html", {
        "request": request,
        "patrols": patrols,
        "faults": faults,
        "now": datetime.now().isoformat(),
    })


@app.get("/station/{station_id}", response_class=HTMLResponse)
async def station_detail(request: Request, station_id: str):
    history = await db.get_station_history(station_id, 50)
    return templates.TemplateResponse("station.html", {
        "request": request,
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


@app.get("/api/station/{station_id}/history")
async def get_station_history(station_id: str):
    history = await db.get_station_history(station_id)
    return history


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


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
