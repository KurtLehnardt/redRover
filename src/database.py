"""SQLite database for patrol logs and vibration history."""

import aiosqlite
from pathlib import Path

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
"""


class Database:
    def __init__(self, path: str = "data/redRover.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self):
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(DB_SCHEMA)
            await db.commit()

    async def start_patrol(self, route_name: str, started_at: str) -> int:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "INSERT INTO patrols (started_at, route_name) VALUES (?, ?)",
                (started_at, route_name),
            )
            await db.commit()
            return cursor.lastrowid

    async def complete_patrol(self, patrol_id: int, completed_at: str, stations: int, faults: int):
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "UPDATE patrols SET completed_at=?, stations_visited=?, faults_detected=? WHERE id=?",
                (completed_at, stations, faults, patrol_id),
            )
            await db.commit()

    async def log_measurement(self, patrol_id: int, station_id: str, measured_at: str, features: dict) -> int:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                """INSERT INTO measurements
                   (patrol_id, station_id, measured_at, rms, peak, crest_factor, kurtosis,
                    dominant_freq_hz, energy_0_100, energy_100_500, energy_500_1000, energy_1000_2000)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    patrol_id, station_id, measured_at,
                    features["rms"], features["peak"], features["crest_factor"],
                    features["kurtosis"], features["dominant_frequency_hz"],
                    features["energy_0_100hz"], features["energy_100_500hz"],
                    features["energy_500_1000hz"], features["energy_1000_2000hz"],
                ),
            )
            await db.commit()
            return cursor.lastrowid

    async def log_diagnosis(self, measurement_id: int, station_id: str, diagnosis, diagnosed_at: str):
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """INSERT INTO diagnoses
                   (measurement_id, station_id, fault_type, confidence, severity, recommendation, reasoning, diagnosed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    measurement_id, station_id, diagnosis.fault_type.value,
                    diagnosis.confidence, diagnosis.severity,
                    diagnosis.recommendation, diagnosis.reasoning, diagnosed_at,
                ),
            )
            await db.commit()

    async def get_recent_patrols(self, limit: int = 20) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM patrols ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_station_history(self, station_id: str, limit: int = 50) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT m.*, d.fault_type, d.confidence, d.severity, d.recommendation
                   FROM measurements m
                   LEFT JOIN diagnoses d ON d.measurement_id = m.id
                   WHERE m.station_id = ?
                   ORDER BY m.id DESC LIMIT ?""",
                (station_id, limit),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_station_trend(self, station_id: str, limit: int = 5) -> list[dict]:
        """Get recent measurement + diagnosis history for a station, oldest first.
        Used to provide trend context to AI analysis."""
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT m.measured_at, m.rms, m.peak, m.kurtosis, m.crest_factor,
                          d.fault_type, d.severity, d.confidence
                   FROM measurements m
                   LEFT JOIN diagnoses d ON d.measurement_id = m.id
                   WHERE m.station_id = ?
                   ORDER BY m.id DESC LIMIT ?""",
                (station_id, limit),
            )
            rows = await cursor.fetchall()
            # Reverse to oldest-first order
            return [dict(r) for r in reversed(rows)]

    async def get_active_faults(self) -> list[dict]:
        """Get most recent diagnosis for each station that has a fault."""
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT d.* FROM diagnoses d
                   INNER JOIN (
                       SELECT station_id, MAX(id) as max_id
                       FROM diagnoses GROUP BY station_id
                   ) latest ON d.id = latest.max_id
                   WHERE d.fault_type != 'normal'
                   ORDER BY d.confidence DESC"""
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
