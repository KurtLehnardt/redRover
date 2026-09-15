"""SQLite storage for patrol logs, measurements, diagnoses, and stations."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite

from .telemetry import get_meter, get_tracer

logger = logging.getLogger(__name__)

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS patrols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    route_name TEXT NOT NULL,
    stations_visited INTEGER DEFAULT 0,
    faults_detected INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patrol_id INTEGER NOT NULL,
    station_id TEXT NOT NULL,
    measured_at TEXT NOT NULL,
    rms REAL,
    peak REAL,
    crest_factor REAL,
    kurtosis REAL,
    dominant_freq_hz REAL,
    energy_0_100 REAL,
    energy_100_500 REAL,
    energy_500_1000 REAL,
    energy_1000_2000 REAL,
    FOREIGN KEY (patrol_id) REFERENCES patrols(id)
);

CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id INTEGER NOT NULL,
    station_id TEXT NOT NULL,
    fault_type TEXT NOT NULL,
    confidence REAL,
    severity TEXT,
    recommendation TEXT,
    reasoning TEXT,
    diagnosed_at TEXT NOT NULL,
    FOREIGN KEY (measurement_id) REFERENCES measurements(id)
);

CREATE TABLE IF NOT EXISTS stations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    heading REAL DEFAULT 0,
    machine_type TEXT,
    rpm REAL DEFAULT 1800,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_measurements_station
    ON measurements(station_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_measurements_patrol
    ON measurements(patrol_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_station
    ON diagnoses(station_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_diagnoses_measurement
    ON diagnoses(measurement_id);
"""

# Columns added after the first release.  Applied idempotently on init() so an
# existing data/redRover.db upgrades in place rather than failing to open.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("measurements", "sample_rate_hz", "REAL"),
    ("measurements", "bearing_analysis_available", "INTEGER"),
    ("measurements", "source_name", "TEXT"),
    ("measurements", "simulated", "INTEGER DEFAULT 0"),
    ("diagnoses", "health", "TEXT"),
    ("diagnoses", "priority", "INTEGER"),
    ("diagnoses", "inference_mode", "TEXT"),
    ("diagnoses", "correlation_tags", "TEXT"),
    ("diagnoses", "simulated", "INTEGER DEFAULT 0"),
)


@dataclass
class DiagnosisRecord:
    """What gets persisted for one station assessment.

    Built from a :class:`~src.ai.fusion.FusedDiagnosis` via
    :meth:`from_fused`, replacing the ad-hoc objects that scripts used to
    fabricate at the call site.
    """

    fault_type: str
    confidence: float
    severity: str
    recommendation: str
    reasoning: str
    health: str = ""
    priority: int = 4
    inference_mode: str = "rule_based"
    correlation_tags: list[str] = field(default_factory=list)
    simulated: bool = False

    @classmethod
    def from_fused(cls, diagnosis, simulated: bool = False) -> DiagnosisRecord:
        return cls(
            fault_type=diagnosis.primary_fault,
            confidence=diagnosis.overall_confidence,
            severity=_worst_severity(diagnosis),
            recommendation=diagnosis.recommendation,
            reasoning=diagnosis.reasoning,
            health=diagnosis.overall_health.value,
            priority=diagnosis.priority,
            inference_mode=diagnosis.inference_mode,
            correlation_tags=list(diagnosis.correlation_tags),
            simulated=simulated,
        )


_SEVERITY_ORDER = {"none": 0, "incipient": 1, "moderate": 2, "severe": 3, "critical": 4}


def _worst_severity(diagnosis) -> str:
    severities = [
        mr.severity for mr in diagnosis.modality_results if mr.fault_detected
    ]
    if not severities:
        return "none"
    return max(severities, key=lambda s: _SEVERITY_ORDER.get(s, 0))


class Database:
    """Async SQLite access over a single shared connection.

    One connection is held for the object's lifetime instead of opening a new
    one per statement: WAL plus a shared handle removes the per-call open/close
    cost and lets the indexes above stay hot.
    """

    def __init__(self, path: str = "data/redRover.db"):
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._tracer = get_tracer("redrover.db")
        self._meter = get_meter("redrover.db")
        self._query_duration = self._meter.create_histogram(
            "redrover.db.query_seconds",
            description="Database query duration",
            unit="s",
        )
        self._query_counter = self._meter.create_counter(
            "redrover.db.query_count",
            description="Total database queries executed",
        )

    # -- lifecycle ----------------------------------------------------------

    async def init(self) -> None:
        """Create the schema, apply migrations, and open the shared connection."""
        start = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = await self._connect()
        await conn.executescript(DB_SCHEMA)
        await self._migrate(conn)
        await conn.commit()
        self._record_query("init", start)

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None

    async def __aenter__(self) -> Database:
        await self.init()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def _connect(self) -> aiosqlite.Connection:
        async with self._lock:
            if self._conn is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = await aiosqlite.connect(self.path)
                self._conn.row_factory = aiosqlite.Row
                await self._conn.execute("PRAGMA journal_mode=WAL")
                await self._conn.execute("PRAGMA foreign_keys=ON")
            return self._conn

    async def _migrate(self, conn: aiosqlite.Connection) -> None:
        for table, column, decl in _MIGRATIONS:
            cursor = await conn.execute(f"PRAGMA table_info({table})")
            existing = {row["name"] for row in await cursor.fetchall()}
            if column not in existing:
                await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                logger.info("db: added column %s.%s", table, column)

    def _record_query(self, operation: str, start: float) -> None:
        duration = time.time() - start
        self._query_duration.record(duration, {"db.operation": operation})
        self._query_counter.add(1, {"db.operation": operation})

    # -- patrols ------------------------------------------------------------

    async def start_patrol(self, route_name: str, started_at: str) -> int:
        start = time.time()
        conn = await self._connect()
        cursor = await conn.execute(
            "INSERT INTO patrols (started_at, route_name) VALUES (?, ?)",
            (started_at, route_name),
        )
        await conn.commit()
        self._record_query("start_patrol", start)
        return cursor.lastrowid

    async def complete_patrol(
        self, patrol_id: int, completed_at: str, stations: int, faults: int,
    ) -> None:
        start = time.time()
        conn = await self._connect()
        await conn.execute(
            "UPDATE patrols SET completed_at=?, stations_visited=?, faults_detected=? "
            "WHERE id=?",
            (completed_at, stations, faults, patrol_id),
        )
        await conn.commit()
        self._record_query("complete_patrol", start)

    async def get_recent_patrols(self, limit: int = 20) -> list[dict]:
        conn = await self._connect()
        cursor = await conn.execute(
            "SELECT * FROM patrols ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in await cursor.fetchall()]

    # -- measurements -------------------------------------------------------

    async def log_measurement(
        self,
        patrol_id: int,
        station_id: str,
        measured_at: str,
        features: dict,
        source_name: str = "",
        simulated: bool = False,
    ) -> int:
        start = time.time()
        conn = await self._connect()
        cursor = await conn.execute(
            """INSERT INTO measurements
               (patrol_id, station_id, measured_at, rms, peak, crest_factor, kurtosis,
                dominant_freq_hz, energy_0_100, energy_100_500, energy_500_1000,
                energy_1000_2000, sample_rate_hz, bearing_analysis_available,
                source_name, simulated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                patrol_id, station_id, measured_at,
                features.get("rms"), features.get("peak"),
                features.get("crest_factor"), features.get("kurtosis"),
                features.get("dominant_frequency_hz"),
                features.get("energy_0_100hz"), features.get("energy_100_500hz"),
                features.get("energy_500_1000hz"), features.get("energy_1000_2000hz"),
                features.get("sample_rate_hz"),
                int(bool(features.get("bearing_analysis_available"))),
                source_name,
                int(bool(simulated)),
            ),
        )
        await conn.commit()
        self._record_query("log_measurement", start)
        return cursor.lastrowid

    # -- diagnoses ----------------------------------------------------------

    async def log_diagnosis(
        self,
        measurement_id: int,
        station_id: str,
        diagnosis: DiagnosisRecord,
        diagnosed_at: str,
    ) -> int:
        """Persist a diagnosis.

        ``diagnosis`` must be a :class:`DiagnosisRecord`; use
        ``DiagnosisRecord.from_fused()`` to build one from a fused diagnosis.
        """
        if not isinstance(diagnosis, DiagnosisRecord):
            raise TypeError(
                "log_diagnosis expects a DiagnosisRecord; "
                "use DiagnosisRecord.from_fused(diagnosis)"
            )
        start = time.time()
        conn = await self._connect()
        cursor = await conn.execute(
            """INSERT INTO diagnoses
               (measurement_id, station_id, fault_type, confidence, severity,
                recommendation, reasoning, diagnosed_at, health, priority,
                inference_mode, correlation_tags, simulated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                measurement_id, station_id, diagnosis.fault_type,
                diagnosis.confidence, diagnosis.severity,
                diagnosis.recommendation, diagnosis.reasoning, diagnosed_at,
                diagnosis.health, diagnosis.priority, diagnosis.inference_mode,
                ",".join(diagnosis.correlation_tags),
                int(bool(diagnosis.simulated)),
            ),
        )
        await conn.commit()
        self._record_query("log_diagnosis", start)
        return cursor.lastrowid

    async def get_station_history(self, station_id: str, limit: int = 50) -> list[dict]:
        conn = await self._connect()
        cursor = await conn.execute(
            """SELECT m.*, d.fault_type, d.confidence, d.severity, d.recommendation,
                      d.health, d.inference_mode
               FROM measurements m
               LEFT JOIN diagnoses d ON d.measurement_id = m.id
               WHERE m.station_id = ?
               ORDER BY m.id DESC LIMIT ?""",
            (station_id, limit),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def get_station_trend(self, station_id: str, limit: int = 5) -> list[dict]:
        """Recent measurement + diagnosis history for a station, oldest first.

        Feeds trend context to the fusion engine; ``fault_type`` values are
        canonical FaultCode strings so they compare cleanly against a fresh
        diagnosis.
        """
        start = time.time()
        conn = await self._connect()
        cursor = await conn.execute(
            """SELECT m.measured_at, m.rms, m.peak, m.kurtosis, m.crest_factor,
                      d.fault_type, d.severity, d.confidence, d.health
               FROM measurements m
               LEFT JOIN diagnoses d ON d.measurement_id = m.id
               WHERE m.station_id = ?
               ORDER BY m.id DESC LIMIT ?""",
            (station_id, limit),
        )
        rows = await cursor.fetchall()
        self._record_query("get_station_trend", start)
        return [dict(r) for r in reversed(rows)]

    async def get_active_faults(self) -> list[dict]:
        """Most recent diagnosis for each station that currently has a fault."""
        conn = await self._connect()
        cursor = await conn.execute(
            """SELECT d.* FROM diagnoses d
               INNER JOIN (
                   SELECT station_id, MAX(id) as max_id
                   FROM diagnoses GROUP BY station_id
               ) latest ON d.id = latest.max_id
               WHERE d.fault_type != 'normal'
               ORDER BY d.confidence DESC"""
        )
        return [dict(r) for r in await cursor.fetchall()]

    # -- stations -----------------------------------------------------------

    async def upsert_station(
        self,
        station_id: str,
        name: str,
        x: float,
        y: float,
        heading: float = 0.0,
        machine_type: str | None = None,
        rpm: float = 1800.0,
        notes: str | None = None,
    ) -> None:
        conn = await self._connect()
        await conn.execute(
            """INSERT INTO stations (id, name, x, y, heading, machine_type, rpm, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   name=excluded.name, x=excluded.x, y=excluded.y,
                   heading=excluded.heading, machine_type=excluded.machine_type,
                   rpm=excluded.rpm, notes=excluded.notes""",
            (station_id, name, x, y, heading, machine_type, rpm, notes),
        )
        await conn.commit()

    async def get_stations(self) -> list[dict]:
        conn = await self._connect()
        cursor = await conn.execute("SELECT * FROM stations ORDER BY id")
        return [dict(r) for r in await cursor.fetchall()]
