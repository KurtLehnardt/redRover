"""Frontier-based autonomous room exploration."""

from __future__ import annotations

import asyncio
import logging
import math
import time

import numpy as np

from ..rover.backends.base import RoverBackend
from ..rover.controller import heading_difference, heading_to_vector
from .occupancy import OccupancyGrid

logger = logging.getLogger(__name__)

# Gravity reads ~1 g on the down axis; an impact shows up as total magnitude
# well above that.  Requiring consecutive samples rejects single-frame spikes
# from carpet transitions and BLE jitter.
BUMP_THRESHOLD_G = 2.0
BUMP_CONSECUTIVE_SAMPLES = 2
STALL_VELOCITY_MPS = 0.02
STALL_HOLD_SECONDS = 0.8
# Ignore stalls while the rover is still accelerating from rest.
DRIVE_GRACE_SECONDS = 3.0


class SimulatedEnvironment:
    """A simple rectangular room for testing without hardware."""

    def __init__(self, width_m: float = 3.0, height_m: float = 4.0, seed: int = 42):
        self.half_w = width_m / 2.0
        self.half_h = height_m / 2.0
        self.x = 0.0
        self.y = 0.0
        self.heading_deg = 0.0  # RVR+ convention: 0 = +Y
        self.speed_mps = 0.0
        self._rng = np.random.default_rng(seed)

    def set_drive(self, speed_byte: int, heading_deg: float) -> None:
        self.heading_deg = heading_deg % 360
        self.speed_mps = speed_byte / 255.0 * 2.0  # 0-255 -> ~0-2 m/s

    def stop(self) -> None:
        self.speed_mps = 0.0

    def tick(self, dt: float) -> dict:
        """Advance the simulation by ``dt`` seconds; returns a sensor-like dict."""
        ux, uy = heading_to_vector(self.heading_deg)
        new_x = self.x + self.speed_mps * ux * dt
        new_y = self.y + self.speed_mps * uy * dt

        hit_wall = False
        if abs(new_x) >= self.half_w:
            new_x = math.copysign(self.half_w - 0.01, new_x)
            hit_wall = True
        if abs(new_y) >= self.half_h:
            new_y = math.copysign(self.half_h - 0.01, new_y)
            hit_wall = True

        self.x, self.y = new_x, new_y

        accel_mag = 3.0 + self._rng.normal(0, 0.3) if hit_wall else 1.0
        vx = 0.0 if hit_wall else self.speed_mps * ux
        vy = 0.0 if hit_wall else self.speed_mps * uy

        return {
            'locator': (self.x + self._rng.normal(0, 0.002),
                        self.y + self._rng.normal(0, 0.002)),
            'velocity': (vx, vy),
            'accelerometer': (
                accel_mag * ux + self._rng.normal(0, 0.05),
                accel_mag * uy + self._rng.normal(0, 0.05),
                -1.0 + self._rng.normal(0, 0.02),
            ),
            'gyroscope': tuple(self._rng.normal(0, 5.0, 3)),
            'hit_wall': hit_wall,
        }


class RoomExplorer:
    """Frontier-based autonomous exploration controller."""

    def __init__(
        self,
        rover: RoverBackend,
        grid: OccupancyGrid,
        speed: int = 60,
        duration: float = 120.0,
        room_bounds_m: float = 5.0,
        simulate: bool = False,
        robot_radius_m: float = 0.12,
        seed: int | None = None,
    ):
        self.rover = rover
        self.grid = grid
        self.speed = speed
        self.duration = duration
        self.room_bounds = room_bounds_m
        self.simulate = simulate
        self.robot_radius = robot_radius_m
        self._rng = np.random.default_rng(seed)
        self._sim_env: SimulatedEnvironment | None = None
        self._running = False
        self._heading = 0.0
        self._stall_start: float | None = None
        self._bump_samples = 0
        self._last_status_time = 0.0

    # -- sensor callback (real hardware) ------------------------------------

    async def _on_sensor_data(self, data: dict) -> None:
        """Called by the rover's sensor streaming on each update."""
        x, y = data.get('locator', (0.0, 0.0))
        self.grid.record_pose(x, y)

    # -- direction choosing (frontier-based) --------------------------------

    def choose_heading(self, x: float, y: float) -> float:
        """Heading (degrees) toward the direction with the most unknown cells."""
        best_heading = self._heading
        best_unknown = -1
        for candidate in range(0, 360, 30):
            unknown = self.grid.count_unknown_along(x, y, float(candidate), max_range_m=1.0)
            if unknown > best_unknown:
                best_unknown = unknown
                best_heading = float(candidate)
        return best_heading

    def random_turn(self) -> float:
        """A heading 90-180 degrees away from the current one."""
        offset = float(self._rng.integers(90, 181))
        if self._rng.random() < 0.5:
            offset = -offset
        return (self._heading + offset) % 360

    def heading_home(self, x: float, y: float) -> float:
        """Heading that points back toward the origin."""
        from ..rover.controller import vector_to_heading
        return vector_to_heading(-x, -y)

    # -- shared step logic ---------------------------------------------------

    def _mark_obstacle(self, x: float, y: float) -> None:
        """Place a wall just outside the chassis, along the current heading."""
        standoff = max(self.robot_radius, self.grid.cell_cm / 100.0)
        wx, wy = self.grid.mark_obstacle_ahead(x, y, self._heading, standoff)
        logger.debug("wall marked at (%.2f, %.2f)", wx, wy)

    def _update_heading(self, x: float, y: float) -> None:
        """Pick the next heading: bounds first, then frontier."""
        if abs(x) > self.room_bounds or abs(y) > self.room_bounds:
            self._heading = self.heading_home(x, y)
            return
        if self._stall_start is not None:
            return
        candidate = self.choose_heading(x, y)
        diff = heading_difference(candidate, self._heading)
        if abs(diff) > 20:
            self._heading = (self._heading + diff * 0.3) % 360

    def _print_status(self, start_time: float, x: float, y: float) -> None:
        now = time.monotonic()
        if now - self._last_status_time < 5.0:
            return
        s = self.grid.stats()
        logger.info(
            "[%6.1fs] pos=(%.2f,%.2f) heading=%.0f free=%d wall=%d coverage=%.1f%%",
            now - start_time, x, y, self._heading, s['free'], s['wall'], s['coverage_pct'],
        )
        self._last_status_time = now

    # -- main exploration loop -----------------------------------------------

    async def run(self) -> None:
        self._running = True
        start_time = time.monotonic()
        self._last_status_time = start_time
        if self.simulate:
            await self._run_simulated(start_time)
        else:
            await self._run_real(start_time)

    async def _run_real(self, start_time: float) -> None:
        """Exploration loop using real BLE hardware."""
        self.rover.add_sensor_callback(self._on_sensor_data)

        await self.rover.reset_locator()
        await asyncio.sleep(0.5)
        await self.rover.start_sensor_streaming(period_ms=100)
        await asyncio.sleep(0.5)
        await self.rover.reset_yaw()
        await asyncio.sleep(0.5)

        self._heading = 0.0
        self._stall_start = None
        drive_start = time.monotonic()

        try:
            while self._running and (time.monotonic() - start_time) < self.duration:
                sensor = self.rover.sensor_data
                x, y = sensor.get('locator', (0.0, 0.0))
                vx, vy = sensor.get('velocity', (0.0, 0.0))
                ax, ay, az = sensor.get('accelerometer', (0.0, 0.0, 0.0))
                vel_mag = math.hypot(vx, vy)
                accel_mag = math.sqrt(ax * ax + ay * ay + az * az)

                elapsed_since_drive = time.monotonic() - drive_start
                obstacle = False

                if accel_mag > BUMP_THRESHOLD_G:
                    self._bump_samples += 1
                    if self._bump_samples >= BUMP_CONSECUTIVE_SAMPLES:
                        logger.info(
                            "BUMP detected (accel=%.2fg) at (%.2f, %.2f)", accel_mag, x, y
                        )
                        obstacle = True
                else:
                    self._bump_samples = 0

                if not obstacle and elapsed_since_drive > DRIVE_GRACE_SECONDS \
                        and self.speed > 0 and vel_mag < STALL_VELOCITY_MPS:
                    if self._stall_start is None:
                        self._stall_start = time.monotonic()
                    elif time.monotonic() - self._stall_start > STALL_HOLD_SECONDS:
                        logger.info("STALL detected at (%.2f, %.2f)", x, y)
                        obstacle = True
                elif not obstacle:
                    self._stall_start = None

                if obstacle:
                    self._mark_obstacle(x, y)
                    try:
                        await self.rover.stop()
                    except Exception as e:
                        logger.warning("stop() failed: %s", e)
                    await asyncio.sleep(0.3)
                    self._heading = self.random_turn()
                    self._stall_start = None
                    self._bump_samples = 0
                    drive_start = time.monotonic()
                else:
                    self._update_heading(x, y)

                try:
                    await self.rover.drive_with_heading(self.speed, int(self._heading))
                except Exception as e:
                    logger.warning("drive command failed: %s", e)

                self._print_status(start_time, x, y)
                await asyncio.sleep(0.1)  # match streaming period

        finally:
            self.rover.remove_sensor_callback(self._on_sensor_data)
            await self._safe_stop()

    async def _run_simulated(self, start_time: float) -> None:
        """Exploration loop using the simulated environment."""
        self._sim_env = SimulatedEnvironment(width_m=3.0, height_m=4.0)
        self._heading = 0.0
        self._stall_start = None
        dt = 0.1

        try:
            while self._running and (time.monotonic() - start_time) < self.duration:
                self._sim_env.set_drive(self.speed, self._heading)
                sensor = self._sim_env.tick(dt)

                x, y = sensor['locator']
                vx, vy = sensor['velocity']
                ax, ay, az = sensor['accelerometer']
                vel_mag = math.hypot(vx, vy)
                accel_mag = math.sqrt(ax * ax + ay * ay + az * az)

                # Feed the controller through its public API rather than
                # writing to its private state.
                self.rover.inject_sensor_data(
                    locator=(x, y), velocity=(vx, vy),
                    accelerometer=(ax, ay, az), gyroscope=sensor['gyroscope'],
                )
                self.grid.record_pose(x, y)

                obstacle = accel_mag > BUMP_THRESHOLD_G or sensor['hit_wall']
                if not obstacle and self.speed > 0 and vel_mag < STALL_VELOCITY_MPS:
                    if self._stall_start is None:
                        self._stall_start = time.monotonic()
                    elif time.monotonic() - self._stall_start > 0.5:
                        obstacle = True
                elif not obstacle:
                    self._stall_start = None

                if obstacle:
                    self._mark_obstacle(x, y)
                    self._sim_env.stop()
                    self._heading = self.random_turn()
                    self._stall_start = None
                else:
                    self._update_heading(x, y)

                self._print_status(start_time, x, y)
                await asyncio.sleep(dt)

        finally:
            if self._sim_env:
                self._sim_env.stop()

    async def _safe_stop(self) -> None:
        """Stop the motors even if the surrounding task was cancelled."""
        try:
            await asyncio.shield(self.rover.stop())
        except asyncio.CancelledError:
            logger.warning("explorer: cancelled during stop; command still in flight")
            raise
        except Exception as e:
            logger.warning("explorer: stop failed: %s", e)

    def request_stop(self) -> None:
        """Ask the loop to finish at the next iteration."""
        self._running = False

    @property
    def running(self) -> bool:
        return self._running
