"""FastAPI dashboard for redRover — local web UI."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime

import httpx

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import load_config
from ..database import Database
from ..sensors.simulator import generate_sample
from ..sensors.vibration import FaultType, extract_features
from ..ai.analyzer import VibrationAnalyzer
from ..telemetry import init_telemetry

logger = logging.getLogger(__name__)

app = FastAPI(title="redRover Dashboard", version="0.1.0")

# Room mapping state
_map_task: asyncio.Task | None = None
_map_status: str = "idle"  # idle, mapping, complete, error
_map_last_error: str = ""

# Enable CORS for Grafana dashboard panels (localhost:3000 -> localhost:8080)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # Load existing map stats into Prometheus gauges
    stats = _load_map_stats()
    _update_prometheus_map_metrics(stats)


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
    from ..ai.fusion import OverallHealth
    results = await run_patrol(simulate=True, skip_ai=False)
    return {
        "status": "complete",
        "stations_visited": len(results),
        "faults_detected": len([r for r in results if r.overall_health != OverallHealth.HEALTHY]),
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


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint for Grafana scraping."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from starlette.responses import Response
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return {"error": "prometheus_client not installed"}


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


# ---------------------------------------------------------------------------
# Room mapping API
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MAP_IMAGE_PATH = _PROJECT_ROOT / "data" / "room_map.png"
_MAP_STATS_PATH = _PROJECT_ROOT / "data" / "room_map_stats.json"


def _load_map_stats() -> dict:
    """Load the latest map stats from disk."""
    if _MAP_STATS_PATH.exists():
        try:
            return json.loads(_MAP_STATS_PATH.read_text())
        except Exception:
            pass
    return {
        "free": 0, "wall": 0, "unknown": 0,
        "coverage_pct": 0.0, "area_free_m2": 0.0,
        "timestamp": None, "status": "no_map",
    }


# Prometheus gauges for map stats
try:
    from prometheus_client import Gauge
    _g_free = Gauge('redrover_map_free_cells', 'Free cells in room map')
    _g_wall = Gauge('redrover_map_wall_cells', 'Wall cells in room map')
    _g_coverage = Gauge('redrover_map_coverage_pct', 'Map coverage percentage')
    _g_area = Gauge('redrover_map_area_explored_m2', 'Explored area in square metres')
    _prometheus_map_gauges = True
except ImportError:
    _prometheus_map_gauges = False


def _update_prometheus_map_metrics(stats: dict):
    """Push map stats into Prometheus gauges."""
    if not _prometheus_map_gauges:
        return
    _g_free.set(stats.get('free', 0))
    _g_wall.set(stats.get('wall', 0))
    _g_coverage.set(stats.get('coverage_pct', 0.0))
    _g_area.set(stats.get('area_free_m2', 0.0))


@app.get("/api/map/image")
async def get_map_image():
    """Serve the latest room map PNG."""
    if _MAP_IMAGE_PATH.exists():
        return FileResponse(
            str(_MAP_IMAGE_PATH),
            media_type="image/png",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return JSONResponse({"error": "No map available. Run a mapping session first."}, status_code=404)


@app.get("/api/map/stats")
async def get_map_stats():
    """Return JSON stats from the latest mapping run."""
    stats = _load_map_stats()
    stats["mapping_status"] = _map_status
    if _map_status == "error":
        stats["last_error"] = _map_last_error
    return stats


@app.post("/api/remap")
async def trigger_remap(
    speed: int = 50,
    duration: float = 60.0,
    room_bounds: float = 3.0,
    simulate: bool = False,
):
    """Trigger a new room mapping run in the background."""
    global _map_task, _map_status, _map_last_error

    if _map_status == "mapping" and _map_task and not _map_task.done():
        return JSONResponse(
            {"error": "Mapping already in progress", "status": "mapping"},
            status_code=409,
        )

    _map_status = "mapping"
    _map_last_error = ""
    _map_task = asyncio.create_task(
        _run_mapping(speed=speed, duration=duration, room_bounds=room_bounds, simulate=simulate)
    )
    return {"status": "started", "speed": speed, "duration": duration, "room_bounds": room_bounds}


@app.post("/api/remap/stop")
async def stop_remap():
    """Stop a mapping run in progress."""
    global _map_status
    if _map_status != "mapping":
        return {"status": _map_status, "message": "No mapping in progress"}
    # The explorer checks _running flag — we signal via the task
    if _map_task and not _map_task.done():
        _map_task.cancel()
    _map_status = "idle"
    return {"status": "stopped"}


async def _run_mapping(speed: int, duration: float, room_bounds: float, simulate: bool):
    """Background task: run the room explorer and save results."""
    global _map_status, _map_last_error
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))

    try:
        from scripts.explore_map import OccupancyGrid, RoomExplorer
        from src.rover.controller import RoverController

        grid = OccupancyGrid(
            width_m=room_bounds * 2,
            height_m=room_bounds * 2,
            cell_cm=5,
        )
        rover = RoverController(connection='ble', simulate=simulate)
        explorer = RoomExplorer(
            rover=rover, grid=grid,
            speed=speed, duration=duration,
            room_bounds_m=room_bounds, simulate=simulate,
        )

        await rover.connect()
        logger.info("Remap: rover connected, starting exploration")
        await explorer.run()

        # Stop streaming & disconnect
        try:
            if rover._streaming:
                await rover.stop_sensor_streaming()
        except Exception:
            pass
        try:
            await rover.disconnect()
        except Exception:
            pass

        # Save map image
        s = grid.stats()
        title = (
            f"Room Map - {s['free']} free, {s['wall']} wall cells "
            f"({s['coverage_pct']:.1f}% coverage, {s['area_free_m2']:.2f} m² explored)"
        )
        os.makedirs(_MAP_IMAGE_PATH.parent, exist_ok=True)
        grid.save_png(str(_MAP_IMAGE_PATH), title=title)

        # Save stats JSON
        s["timestamp"] = datetime.now().isoformat()
        s["status"] = "complete"
        s["duration_requested"] = duration
        s["speed"] = speed
        s["room_bounds"] = room_bounds
        s["path_points"] = len(grid.path)
        _MAP_STATS_PATH.write_text(json.dumps(s, indent=2))

        _update_prometheus_map_metrics(s)
        _map_status = "complete"
        logger.info("Remap complete: %d free, %d wall, %.1f%% coverage",
                     s['free'], s['wall'], s['coverage_pct'])

    except asyncio.CancelledError:
        _map_status = "idle"
        logger.info("Remap cancelled by user")
    except Exception as e:
        _map_status = "error"
        _map_last_error = str(e)
        logger.error("Remap failed: %s", e, exc_info=True)


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
