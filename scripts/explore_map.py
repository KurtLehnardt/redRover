#!/usr/bin/env python3
"""Autonomous room exploration and mapping for RVR+.

Drives the rover around a room, detects walls via accelerometer bumps and
velocity stalls, builds an occupancy grid, and saves a PNG map at the end.

Usage:
    python -m scripts.explore_map --speed 60 --duration 120 --output data/room_map.png
    python -m scripts.explore_map --simulate  # no hardware needed
"""

import argparse
import asyncio
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rover.controller import RoverController, Waypoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------

CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_WALL = 2


class OccupancyGrid:
    """2-D occupancy grid centred on the origin.

    Coordinates are in **centimetres** internally; the public API accepts
    metres and converts.
    """

    def __init__(self, width_m: float = 10.0, height_m: float = 10.0, cell_cm: int = 5):
        self.cell_cm = cell_cm
        # Grid dimensions in cells
        self.cols = int(width_m * 100 / cell_cm)
        self.rows = int(height_m * 100 / cell_cm)
        # Origin sits at the centre of the grid
        self.origin_col = self.cols // 2
        self.origin_row = self.rows // 2
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        # Path history (list of (x_m, y_m) tuples)
        self.path: list[tuple[float, float]] = []

    # -- coordinate helpers -------------------------------------------------

    def _world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        col = self.origin_col + int(round(x_m * 100 / self.cell_cm))
        row = self.origin_row - int(round(y_m * 100 / self.cell_cm))  # y-up -> row-down
        return row, col

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    # -- marking ------------------------------------------------------------

    def mark_free(self, x_m: float, y_m: float):
        r, c = self._world_to_cell(x_m, y_m)
        if self._in_bounds(r, c) and self.grid[r, c] != CELL_WALL:
            self.grid[r, c] = CELL_FREE
        self.path.append((x_m, y_m))

    def mark_wall(self, x_m: float, y_m: float):
        r, c = self._world_to_cell(x_m, y_m)
        if self._in_bounds(r, c):
            self.grid[r, c] = CELL_WALL

    # -- stats --------------------------------------------------------------

    def stats(self) -> dict:
        total = self.rows * self.cols
        free = int(np.sum(self.grid == CELL_FREE))
        wall = int(np.sum(self.grid == CELL_WALL))
        unknown = total - free - wall
        area_free_m2 = free * (self.cell_cm / 100.0) ** 2
        return {
            'total_cells': total,
            'free': free,
            'wall': wall,
            'unknown': unknown,
            'coverage_pct': (free + wall) / total * 100 if total else 0.0,
            'area_free_m2': area_free_m2,
        }

    # -- visualisation ------------------------------------------------------

    def save_png(self, path: str, title: str = "Room Map"):
        """Render the occupancy grid to a PNG image using matplotlib."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(figsize=(10, 10))

        # Colour map: 0=gray (unknown), 1=white (free), 2=black (wall)
        cmap = ListedColormap(['#C0C0C0', '#FFFFFF', '#000000'])
        extent = [
            -self.origin_col * self.cell_cm / 100.0,
            (self.cols - self.origin_col) * self.cell_cm / 100.0,
            -self.origin_row * self.cell_cm / 100.0,
            (self.rows - self.origin_row) * self.cell_cm / 100.0,
        ]
        # Flip vertically so y-up matches display
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=2, origin='upper', extent=extent)

        # Draw path as blue line
        if len(self.path) > 1:
            xs = [p[0] for p in self.path]
            ys = [p[1] for p in self.path]
            ax.plot(xs, ys, 'b-', linewidth=0.8, alpha=0.6, label='Path')

        # Mark start position
        ax.plot(0, 0, 'go', markersize=10, label='Start', zorder=5)

        ax.set_xlabel('X (metres)')
        ax.set_ylabel('Y (metres)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Map saved to %s", path)


# ---------------------------------------------------------------------------
# Simulated rover environment
# ---------------------------------------------------------------------------

class SimulatedEnvironment:
    """A simple rectangular room for testing without hardware."""

    def __init__(self, width_m: float = 3.0, height_m: float = 4.0):
        self.half_w = width_m / 2.0
        self.half_h = height_m / 2.0
        self.x = 0.0
        self.y = 0.0
        self.heading_deg = 0.0  # 0 = positive X
        self.speed_mps = 0.0
        self._rng = np.random.default_rng(42)

    def set_drive(self, speed_byte: int, heading_deg: float):
        self.heading_deg = heading_deg % 360
        # speed_byte 0-255 -> ~ 0-2 m/s
        self.speed_mps = speed_byte / 255.0 * 2.0

    def stop(self):
        self.speed_mps = 0.0

    def tick(self, dt: float) -> dict:
        """Advance simulation by ``dt`` seconds.  Returns sensor-like dict."""
        heading_rad = math.radians(self.heading_deg)
        dx = self.speed_mps * math.cos(heading_rad) * dt
        dy = self.speed_mps * math.sin(heading_rad) * dt

        new_x = self.x + dx
        new_y = self.y + dy

        hit_wall = False
        if abs(new_x) >= self.half_w:
            new_x = math.copysign(self.half_w - 0.01, new_x)
            hit_wall = True
        if abs(new_y) >= self.half_h:
            new_y = math.copysign(self.half_h - 0.01, new_y)
            hit_wall = True

        self.x = new_x
        self.y = new_y

        # Add noise
        noise_x = self._rng.normal(0, 0.002)
        noise_y = self._rng.normal(0, 0.002)

        accel_mag = 1.0  # gravity baseline
        if hit_wall:
            accel_mag = 3.0 + self._rng.normal(0, 0.3)  # bump

        vx = self.speed_mps * math.cos(heading_rad) if not hit_wall else 0.0
        vy = self.speed_mps * math.sin(heading_rad) if not hit_wall else 0.0

        return {
            'locator': (self.x + noise_x, self.y + noise_y),
            'velocity': (vx, vy),
            'accelerometer': (
                accel_mag * math.cos(heading_rad) + self._rng.normal(0, 0.05),
                accel_mag * math.sin(heading_rad) + self._rng.normal(0, 0.05),
                -1.0 + self._rng.normal(0, 0.02),
            ),
            'gyroscope': (
                self._rng.normal(0, 5.0),
                self._rng.normal(0, 5.0),
                self._rng.normal(0, 5.0),
            ),
            'hit_wall': hit_wall,
        }


# ---------------------------------------------------------------------------
# Explorer logic
# ---------------------------------------------------------------------------

class RoomExplorer:
    """Frontier-based autonomous exploration controller."""

    def __init__(
        self,
        rover: RoverController,
        grid: OccupancyGrid,
        speed: int = 60,
        duration: float = 120.0,
        room_bounds_m: float = 5.0,
        simulate: bool = False,
    ):
        self.rover = rover
        self.grid = grid
        self.speed = speed
        self.duration = duration
        self.room_bounds = room_bounds_m
        self.simulate = simulate
        self._sim_env: SimulatedEnvironment | None = None
        self._running = False
        self._heading = 0.0
        self._stall_start: float | None = None
        self._last_status_time = 0.0

    # -- sensor callback (real hardware) ------------------------------------

    async def _on_sensor_data(self, data: dict):
        """Called by the rover's sensor streaming on each update."""
        x, y = data.get('locator', (0.0, 0.0))
        self.grid.mark_free(x, y)

    # -- direction choosing (frontier-based) --------------------------------

    def _choose_heading(self, x: float, y: float) -> float:
        """Pick the heading (degrees) toward the direction with the most
        unknown cells in a ~1m radius fan."""
        best_heading = self._heading
        best_unknown = -1

        for candidate_deg in range(0, 360, 30):
            rad = math.radians(candidate_deg)
            unknown_count = 0
            for dist_cm in range(10, 100, 10):
                px = x + (dist_cm / 100.0) * math.cos(rad)
                py = y + (dist_cm / 100.0) * math.sin(rad)
                r, c = self.grid._world_to_cell(px, py)
                if self.grid._in_bounds(r, c) and self.grid.grid[r, c] == CELL_UNKNOWN:
                    unknown_count += 1
            if unknown_count > best_unknown:
                best_unknown = unknown_count
                best_heading = float(candidate_deg)

        return best_heading

    def _random_turn(self) -> float:
        """Return a heading turned 90-180 degrees from current heading."""
        offset = np.random.default_rng().integers(90, 181)
        if np.random.default_rng().random() < 0.5:
            offset = -offset
        return (self._heading + offset) % 360

    # -- main exploration loop -----------------------------------------------

    async def run(self):
        self._running = True
        start_time = time.monotonic()
        self._last_status_time = start_time

        if self.simulate:
            await self._run_simulated(start_time)
        else:
            await self._run_real(start_time)

    async def _run_real(self, start_time: float):
        """Exploration loop using real BLE hardware."""
        # Register sensor callback
        self.rover.add_sensor_callback(self._on_sensor_data)

        # Reset locator and start streaming
        await self.rover.reset_locator()
        await asyncio.sleep(0.5)
        await self.rover.start_sensor_streaming(period_ms=100)
        await asyncio.sleep(0.5)

        # Reset yaw so heading=0 is forward
        await self.rover.reset_yaw()
        await asyncio.sleep(0.5)

        self._heading = 0.0
        self._stall_start = None

        try:
            while self._running and (time.monotonic() - start_time) < self.duration:
                sensor = self.rover.sensor_data
                x, y = sensor.get('locator', (0.0, 0.0))
                vx, vy = sensor.get('velocity', (0.0, 0.0))
                ax, ay, az = sensor.get('accelerometer', (0.0, 0.0, 0.0))
                vel_mag = math.sqrt(vx ** 2 + vy ** 2)
                accel_mag = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)

                # -- Bump detection --
                if accel_mag > 2.0:
                    logger.info("BUMP detected (accel=%.2fg) at (%.2f, %.2f)", accel_mag, x, y)
                    # Mark wall slightly ahead of current position
                    wall_x = x + 0.05 * math.cos(math.radians(self._heading))
                    wall_y = y + 0.05 * math.sin(math.radians(self._heading))
                    self.grid.mark_wall(wall_x, wall_y)
                    await self.rover.stop()
                    await asyncio.sleep(0.3)
                    self._heading = self._random_turn()
                    self._stall_start = None

                # -- Stall detection --
                elif self.speed > 0 and vel_mag < 0.02:
                    if self._stall_start is None:
                        self._stall_start = time.monotonic()
                    elif time.monotonic() - self._stall_start > 0.5:
                        logger.info("STALL detected at (%.2f, %.2f)", x, y)
                        wall_x = x + 0.05 * math.cos(math.radians(self._heading))
                        wall_y = y + 0.05 * math.sin(math.radians(self._heading))
                        self.grid.mark_wall(wall_x, wall_y)
                        await self.rover.stop()
                        await asyncio.sleep(0.3)
                        self._heading = self._random_turn()
                        self._stall_start = None
                else:
                    self._stall_start = None

                # -- Boundary check --
                if abs(x) > self.room_bounds or abs(y) > self.room_bounds:
                    logger.info("Boundary reached at (%.2f, %.2f) — turning back", x, y)
                    self._heading = (math.degrees(math.atan2(-y, -x))) % 360

                # -- Frontier-based direction --
                if self._stall_start is None:  # only re-choose if not stalled
                    candidate = self._choose_heading(x, y)
                    # Blend toward frontier heading to avoid jitter
                    diff = (candidate - self._heading + 180) % 360 - 180
                    if abs(diff) > 20:
                        self._heading = (self._heading + diff * 0.3) % 360

                # -- Drive --
                heading_int = int(self._heading) % 360
                heading_msb = (heading_int >> 8) & 0xFF
                heading_lsb = heading_int & 0xFF
                from src.rover.controller import (
                    _DID_DRIVE, _CID_DRIVE_WITH_HEADING, _TID_ST,
                )
                await self.rover._ble_send_no_response(
                    _DID_DRIVE, _CID_DRIVE_WITH_HEADING, _TID_ST,
                    data=bytes([self.speed & 0xFF, heading_msb, heading_lsb, 0]),
                )

                # -- Status print --
                now = time.monotonic()
                if now - self._last_status_time >= 5.0:
                    elapsed = now - start_time
                    s = self.grid.stats()
                    print(
                        f"[{elapsed:6.1f}s] pos=({x:.2f},{y:.2f}) "
                        f"heading={self._heading:.0f} "
                        f"free={s['free']} wall={s['wall']} "
                        f"coverage={s['coverage_pct']:.1f}%"
                    )
                    self._last_status_time = now

                await asyncio.sleep(0.1)  # match streaming period

        finally:
            await self.rover.stop()

    async def _run_simulated(self, start_time: float):
        """Exploration loop using the simulated environment."""
        self._sim_env = SimulatedEnvironment(width_m=3.0, height_m=4.0)
        self._heading = 0.0
        self._stall_start = None

        dt = 0.1  # simulation tick interval

        try:
            while self._running and (time.monotonic() - start_time) < self.duration:
                # Drive in the simulated env
                self._sim_env.set_drive(self.speed, self._heading)
                sensor = self._sim_env.tick(dt)

                x, y = sensor['locator']
                vx, vy = sensor['velocity']
                ax, ay, az = sensor['accelerometer']
                hit_wall = sensor['hit_wall']
                vel_mag = math.sqrt(vx ** 2 + vy ** 2)
                accel_mag = math.sqrt(ax ** 2 + ay ** 2 + az ** 2)

                # Update rover's internal sensor_data for consistency
                self.rover._sensor_data['locator'] = (x, y)
                self.rover._sensor_data['velocity'] = (vx, vy)
                self.rover._sensor_data['accelerometer'] = (ax, ay, az)
                self.rover._sensor_data['gyroscope'] = sensor['gyroscope']
                self.rover._position = (x, y)

                # Mark current position as free
                self.grid.mark_free(x, y)

                # -- Bump detection --
                if accel_mag > 2.0 or hit_wall:
                    wall_x = x + 0.05 * math.cos(math.radians(self._heading))
                    wall_y = y + 0.05 * math.sin(math.radians(self._heading))
                    self.grid.mark_wall(wall_x, wall_y)
                    self._sim_env.stop()
                    self._heading = self._random_turn()
                    self._stall_start = None

                # -- Stall detection --
                elif self.speed > 0 and vel_mag < 0.02:
                    if self._stall_start is None:
                        self._stall_start = time.monotonic()
                    elif time.monotonic() - self._stall_start > 0.5:
                        wall_x = x + 0.05 * math.cos(math.radians(self._heading))
                        wall_y = y + 0.05 * math.sin(math.radians(self._heading))
                        self.grid.mark_wall(wall_x, wall_y)
                        self._sim_env.stop()
                        self._heading = self._random_turn()
                        self._stall_start = None
                else:
                    self._stall_start = None

                # -- Boundary --
                if abs(x) > self.room_bounds or abs(y) > self.room_bounds:
                    self._heading = (math.degrees(math.atan2(-y, -x))) % 360

                # -- Frontier heading --
                if self._stall_start is None:
                    candidate = self._choose_heading(x, y)
                    diff = (candidate - self._heading + 180) % 360 - 180
                    if abs(diff) > 20:
                        self._heading = (self._heading + diff * 0.3) % 360

                # -- Status --
                now = time.monotonic()
                if now - self._last_status_time >= 5.0:
                    elapsed = now - start_time
                    s = self.grid.stats()
                    print(
                        f"[{elapsed:6.1f}s] pos=({x:.2f},{y:.2f}) "
                        f"heading={self._heading:.0f} "
                        f"free={s['free']} wall={s['wall']} "
                        f"coverage={s['coverage_pct']:.1f}%"
                    )
                    self._last_status_time = now

                await asyncio.sleep(dt)

        finally:
            if self._sim_env:
                self._sim_env.stop()

    def request_stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RVR+ room explorer & mapper")
    p.add_argument('--speed', type=int, default=60, help='Drive speed 0-255 (default 60)')
    p.add_argument('--duration', type=float, default=120, help='Exploration time in seconds (default 120)')
    p.add_argument('--output', type=str, default='data/room_map.png', help='Output map image path')
    p.add_argument('--simulate', action='store_true', help='Run in simulated mode (no hardware)')
    p.add_argument('--cell-size', type=int, default=5, help='Grid cell size in cm (default 5)')
    p.add_argument('--room-bounds', type=float, default=5.0, help='Max distance from origin in metres (default 5)')
    p.add_argument('-v', '--verbose', action='store_true', help='Enable debug logging')
    return p.parse_args()


async def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    )

    grid = OccupancyGrid(
        width_m=args.room_bounds * 2,
        height_m=args.room_bounds * 2,
        cell_cm=args.cell_size,
    )

    rover = RoverController(connection='ble', simulate=args.simulate)
    explorer = RoomExplorer(
        rover=rover,
        grid=grid,
        speed=args.speed,
        duration=args.duration,
        room_bounds_m=args.room_bounds,
        simulate=args.simulate,
    )

    # Handle Ctrl+C gracefully
    loop = asyncio.get_running_loop()

    def _signal_handler():
        print("\nCtrl+C — stopping exploration...")
        explorer.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        # Connect
        print(f"Connecting to RVR+ ({'simulated' if args.simulate else 'BLE'})...")
        await rover.connect()

        # Explore
        print(f"Starting exploration: speed={args.speed}, duration={args.duration}s")
        await explorer.run()

    except KeyboardInterrupt:
        pass
    finally:
        # Stop streaming if running
        if rover._streaming:
            print("Stopping sensor streaming...")
            await rover.stop_sensor_streaming()

        # Attempt return to origin
        try:
            print("Attempting return to origin...")
            if args.simulate:
                # In sim mode just note final position
                x, y = rover.position
                print(f"  Final position: ({x:.2f}, {y:.2f})")
            else:
                await rover.return_home()
        except Exception as e:
            logger.warning("Return home failed: %s", e)

        # Disconnect
        await rover.disconnect()

        # Save map
        s = grid.stats()
        title = (
            f"Room Map - {s['free']} free, {s['wall']} wall cells "
            f"({s['coverage_pct']:.1f}% coverage, {s['area_free_m2']:.2f} m2 explored)"
        )
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(str(_PROJECT_ROOT), output_path)
        grid.save_png(output_path, title=title)

        # Print summary
        print("\n=== Exploration Summary ===")
        print(f"  Free cells:     {s['free']}")
        print(f"  Wall cells:     {s['wall']}")
        print(f"  Unknown cells:  {s['unknown']}")
        print(f"  Coverage:       {s['coverage_pct']:.1f}%")
        print(f"  Area explored:  {s['area_free_m2']:.2f} m2")
        print(f"  Map saved to:   {output_path}")


if __name__ == '__main__':
    asyncio.run(main())
