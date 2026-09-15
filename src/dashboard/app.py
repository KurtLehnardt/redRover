"""FastAPI dashboard for redRover — local web UI.

Security note
-------------
This API can physically move the robot (``/api/remap``, ``/api/demo-patrol``).
Every state-changing endpoint therefore requires the shared secret from
``[dashboard].auth_token`` (or ``REDROVER_DASHBOARD__AUTH_TOKEN``), sent as
``X-RedRover-Token`` or ``Authorization: Bearer <token>``.  With no token
configured those endpoints return 503 rather than running unauthenticated, and
CORS is restricted to the origins listed in config — never ``*``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..ai.ollama import OllamaClient
from ..alerting import get_alert_manager
from ..config import load_config
from ..database import Database
from ..mapping import OccupancyGrid, RoomExplorer
from ..rover.controller import RoverController
from ..telemetry import init_telemetry, shutdown_telemetry

logger = logging.getLogger(__name__)

config = load_config()
db = Database(config.database.path)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MAP_IMAGE_PATH = _PROJECT_ROOT / "data" / "room_map.png"
_MAP_STATS_PATH = _PROJECT_ROOT / "data" / "room_map_stats.json"

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


class MappingSession:
    """Owns the one mapping run that may be in flight.

    Stopping asks the explorer to finish its current iteration and awaits the
    task, so the rover is commanded to stop by code that is still running.
    Cancelling the task outright can interrupt that stop mid-await and leave
    the robot driving.
    """

    def __init__(self):
        self.status = "idle"  # idle | mapping | complete | error | stopped
        self.last_error = ""
        self._task: asyncio.Task | None = None
        self._explorer: RoomExplorer | None = None
        self._rover: RoverController | None = None
        self._lock = asyncio.Lock()

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, **kwargs) -> bool:
        async with self._lock:
            if self.busy:
                return False
            self.status = "mapping"
            self.last_error = ""
            self._task = asyncio.create_task(self._run(**kwargs))
            return True

    async def stop(self, timeout: float = 15.0) -> str:
        async with self._lock:
            if not self.busy:
                return self.status
            if self._explorer is not None:
                self._explorer.request_stop()
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
            except TimeoutError:
                logger.error("mapping did not stop within %.0fs; cancelling", timeout)
                self._task.cancel()
            self.status = "stopped"
            return self.status

    async def emergency_stop(self) -> None:
        """Latch the rover's e-stop immediately, then unwind the session."""
        if self._rover is not None:
            await self._rover.emergency_stop()
        if self._explorer is not None:
            self._explorer.request_stop()

    async def _run(self, speed: int, duration: float, room_bounds: float, simulate: bool):
        grid = OccupancyGrid(
            width_m=room_bounds * 2, height_m=room_bounds * 2, cell_cm=5,
        )
        rover = RoverController(
            connection=config.rover.connection,
            simulate=simulate,
            max_speed_mps=config.rover.max_speed_mps,
            max_drive_seconds=config.rover.max_drive_seconds,
        )
        explorer = RoomExplorer(
            rover=rover, grid=grid, speed=speed, duration=duration,
            room_bounds_m=room_bounds, simulate=simulate,
            robot_radius_m=config.rover.robot_radius_m,
        )
        self._rover, self._explorer = rover, explorer

        try:
            await rover.connect()
            logger.info("Remap: rover connected, starting exploration")
            await explorer.run()
            stats = _persist_map(grid, speed, duration, room_bounds)
            self.status = "complete"
            logger.info(
                "Remap complete: %d free, %d wall, %.1f%% coverage",
                stats["free"], stats["wall"], stats["coverage_pct"],
            )
        except asyncio.CancelledError:
            self.status = "stopped"
            raise
        except Exception as e:
            self.status = "error"
            self.last_error = str(e)
            logger.error("Remap failed: %s", e, exc_info=True)
        finally:
            try:
                if rover.streaming:
                    await rover.stop_sensor_streaming()
            except Exception as exc:
                logger.debug("remap: stop streaming: %s", exc)
            try:
                await rover.disconnect()
            except Exception as exc:
                logger.debug("remap: disconnect: %s", exc)
            self._rover = self._explorer = None


mapping_session = MappingSession()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_telemetry(
        service_name=config.telemetry.service_name,
        endpoint=config.telemetry.endpoint or None,
        enabled=config.telemetry.enabled,
        export_interval_ms=config.telemetry.export_interval_ms,
        console_export=config.telemetry.console_export,
        environment=config.telemetry.environment,
    )
    await db.init()
    _update_prometheus_map_metrics(_load_map_stats())
    try:
        yield
    finally:
        await mapping_session.stop()
        await db.close()
        shutdown_telemetry()


app = FastAPI(title="redRover Dashboard", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.dashboard.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-RedRover-Token"],
)

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)
except ImportError:
    pass

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


async def require_token(request: Request) -> None:
    """Guard every endpoint that can change state or move the robot."""
    expected = config.dashboard.auth_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Control endpoints are disabled: set [dashboard].auth_token or "
                "REDROVER_DASHBOARD__AUTH_TOKEN to enable them."
            ),
        )

    supplied = request.headers.get("X-RedRover-Token", "")
    if not supplied:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:]

    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token",
        )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    patrols = await db.get_recent_patrols(10)
    faults = await db.get_active_faults()
    return templates.TemplateResponse(request, "index.html", context={
        "patrols": patrols,
        "faults": faults,
        "now": datetime.now(UTC).isoformat(),
    })


@app.get("/station/{station_id}", response_class=HTMLResponse)
async def station_detail(request: Request, station_id: str):
    history = await db.get_station_history(station_id, 50)
    return templates.TemplateResponse(request, "station.html", context={
        "station_id": station_id,
        "history": history,
    })


# ---------------------------------------------------------------------------
# Read-only API
# ---------------------------------------------------------------------------


@app.get("/api/faults")
async def get_faults():
    """Current active faults (HTMX endpoint)."""
    return await db.get_active_faults()


@app.get("/api/alerts")
async def get_alerts():
    """Recent alerts from the process-wide alert manager."""
    return get_alert_manager(config).recent_alerts


@app.get("/api/stations")
async def get_stations():
    return await db.get_stations()


@app.get("/api/station/{station_id}/history")
async def get_station_history(station_id: str):
    return await db.get_station_history(station_id)


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint for Grafana scraping."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        from starlette.responses import Response
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return JSONResponse(
            {"error": "prometheus_client not installed"}, status_code=501,
        )


async def _check_ai_status() -> dict:
    """Check Ollama connectivity and model availability."""
    client = OllamaClient(host=config.ai.ollama_host, model=config.ai.model)
    try:
        present, available = await client.available()
    finally:
        await client.aclose()

    if present:
        return {
            "ai_status": "online",
            "ai_model": config.ai.model,
            "inference_mode": "llm",
            "message": f"Ollama is running and model '{config.ai.model}' is available",
        }
    if available:
        return {
            "ai_status": "model_missing",
            "ai_model": config.ai.model,
            "inference_mode": "rule_based",
            "message": (
                f"Ollama is running but model '{config.ai.model}' is not found. "
                f"Available: {available}"
            ),
        }
    return {
        "ai_status": "offline",
        "ai_model": config.ai.model,
        "inference_mode": "rule_based",
        "message": (
            f"Cannot reach Ollama at {config.ai.ollama_host}. "
            "Fusion analysis will use the rule-based fallback."
        ),
    }


@app.get("/api/system-status")
async def system_status():
    return await _check_ai_status()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "mapping": mapping_session.status,
        "control_enabled": bool(config.dashboard.auth_token),
        "ai": await _check_ai_status(),
    }


# ---------------------------------------------------------------------------
# Room mapping
# ---------------------------------------------------------------------------


def _load_map_stats() -> dict:
    """Load the latest map stats from disk."""
    if _MAP_STATS_PATH.exists():
        try:
            return json.loads(_MAP_STATS_PATH.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("map stats unreadable: %s", exc)
    return {
        "free": 0, "wall": 0, "unknown": 0,
        "coverage_pct": 0.0, "area_free_m2": 0.0,
        "timestamp": None, "status": "no_map",
    }


try:
    from prometheus_client import Gauge
    _g_free = Gauge('redrover_map_free_cells', 'Free cells in room map')
    _g_wall = Gauge('redrover_map_wall_cells', 'Wall cells in room map')
    _g_coverage = Gauge('redrover_map_coverage_pct', 'Map coverage percentage')
    _g_area = Gauge('redrover_map_area_explored_m2', 'Explored area in square metres')
    _prometheus_map_gauges = True
except ImportError:
    _prometheus_map_gauges = False


def _update_prometheus_map_metrics(stats: dict) -> None:
    if not _prometheus_map_gauges:
        return
    _g_free.set(stats.get('free', 0))
    _g_wall.set(stats.get('wall', 0))
    _g_coverage.set(stats.get('coverage_pct', 0.0))
    _g_area.set(stats.get('area_free_m2', 0.0))


def _persist_map(grid: OccupancyGrid, speed: int, duration: float,
                 room_bounds: float) -> dict:
    stats = grid.stats()
    title = (
        f"Room Map - {stats['free']} free, {stats['wall']} wall cells "
        f"({stats['coverage_pct']:.1f}% coverage, {stats['area_free_m2']:.2f} m2 explored)"
    )
    _MAP_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save_png(str(_MAP_IMAGE_PATH), title=title)

    stats["timestamp"] = datetime.now(UTC).isoformat()
    stats["status"] = "complete"
    stats["duration_requested"] = duration
    stats["speed"] = speed
    stats["room_bounds"] = room_bounds
    stats["path_points"] = len(grid.path)
    _MAP_STATS_PATH.write_text(json.dumps(stats, indent=2))
    _update_prometheus_map_metrics(stats)
    return stats


@app.get("/api/map/image")
async def get_map_image():
    """Serve the latest room map PNG."""
    if _MAP_IMAGE_PATH.exists():
        return FileResponse(
            str(_MAP_IMAGE_PATH),
            media_type="image/png",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return JSONResponse(
        {"error": "No map available. Run a mapping session first."}, status_code=404,
    )


@app.get("/api/map/stats")
async def get_map_stats():
    """Return JSON stats from the latest mapping run."""
    stats = _load_map_stats()
    stats["mapping_status"] = mapping_session.status
    if mapping_session.status == "error":
        stats["last_error"] = mapping_session.last_error
    return stats


# ---------------------------------------------------------------------------
# Control endpoints (authenticated — these move the robot)
# ---------------------------------------------------------------------------


@app.post("/api/remap", dependencies=[Depends(require_token)])
async def trigger_remap(
    speed: int = 50,
    duration: float = 60.0,
    room_bounds: float = 3.0,
    simulate: bool = False,
):
    """Trigger a new room mapping run in the background."""
    speed = max(0, min(255, speed))
    duration = max(1.0, min(1800.0, duration))
    room_bounds = max(0.5, min(50.0, room_bounds))

    started = await mapping_session.start(
        speed=speed, duration=duration, room_bounds=room_bounds, simulate=simulate,
    )
    if not started:
        return JSONResponse(
            {"error": "Mapping already in progress", "status": mapping_session.status},
            status_code=409,
        )
    return {
        "status": "started", "speed": speed,
        "duration": duration, "room_bounds": room_bounds,
    }


@app.post("/api/remap/stop", dependencies=[Depends(require_token)])
async def stop_remap():
    """Ask a mapping run to finish and wait for the rover to be stopped."""
    if not mapping_session.busy:
        return {"status": mapping_session.status, "message": "No mapping in progress"}
    return {"status": await mapping_session.stop()}


@app.post("/api/estop", dependencies=[Depends(require_token)])
async def emergency_stop():
    """Latch an emergency stop on whatever the robot is currently doing."""
    await mapping_session.emergency_stop()
    logger.warning("EMERGENCY STOP requested via API")
    return {"status": "estop_engaged"}


@app.post("/api/demo-patrol", dependencies=[Depends(require_token)])
async def trigger_demo_patrol():
    """Trigger a simulated patrol for demo purposes."""
    from ..ai.fusion import OverallHealth
    from ..main import run_patrol

    results = await run_patrol(simulate=True, skip_ai=False, config=config)
    return {
        "status": "complete",
        "stations_visited": len(results),
        "faults_detected": len(
            [r for r in results if r.overall_health is not OverallHealth.HEALTHY]
        ),
    }


def start():
    """Entry point for running the dashboard."""
    import uvicorn
    uvicorn.run(
        "src.dashboard.app:app",
        host=config.dashboard.host,
        port=config.dashboard.port,
        reload=config.dashboard.reload,
    )


if __name__ == "__main__":
    start()
