#!/usr/bin/env python3
"""Autonomous room exploration and mapping for the RVR+ (CLI wrapper).

The mapping logic lives in :mod:`src.mapping`; this script is the command-line
front end so the dashboard and the CLI share one implementation instead of the
dashboard importing a script off ``sys.path``.

Usage:
    python -m scripts.explore_map --speed 60 --duration 120 --output data/room_map.png
    python -m scripts.explore_map --simulate  # no hardware needed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.mapping import OccupancyGrid, RoomExplorer  # noqa: E402
from src.rover.backends import create_rover  # noqa: E402

logger = logging.getLogger("redRover.explore_map")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RVR+ room explorer & mapper")
    p.add_argument('--speed', type=int, default=60, help='Drive speed 0-255 (default 60)')
    p.add_argument('--duration', type=float, default=120,
                   help='Exploration time in seconds (default 120)')
    p.add_argument('--output', type=str, default='data/room_map.png',
                   help='Output map image path')
    p.add_argument('--simulate', action='store_true', help='Run without hardware')
    p.add_argument('--cell-size', type=int, default=5, help='Grid cell size in cm (default 5)')
    p.add_argument('--room-bounds', type=float, default=5.0,
                   help='Max distance from origin in metres (default 5)')
    p.add_argument('--seed', type=int, default=None,
                   help='Seed the exploration RNG for reproducible runs')
    p.add_argument('-v', '--verbose', action='store_true', help='Enable debug logging')
    return p.parse_args()


def save_results(grid: OccupancyGrid, output_path: str, args: argparse.Namespace) -> dict:
    """Write the map PNG and the stats JSON; returns the stats."""
    stats = grid.stats()
    title = (
        f"Room Map - {stats['free']} free, {stats['wall']} wall cells "
        f"({stats['coverage_pct']:.1f}% coverage, {stats['area_free_m2']:.2f} m2 explored)"
    )
    grid.save_png(output_path, title=title)

    stats["timestamp"] = datetime.now(UTC).isoformat()
    stats["status"] = "complete"
    stats["duration_requested"] = args.duration
    stats["speed"] = args.speed
    stats["room_bounds"] = args.room_bounds
    stats["path_points"] = len(grid.path)

    stats_path = os.path.splitext(output_path)[0] + '_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    stats["_stats_path"] = stats_path
    return stats


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    )

    config = load_config()
    grid = OccupancyGrid(
        width_m=args.room_bounds * 2,
        height_m=args.room_bounds * 2,
        cell_cm=args.cell_size,
    )
    # Through the factory, so [rover].connection = "serial" maps a firmware
    # rover instead of handing "serial" to the Sphero controller, which
    # rejects it.
    rover = create_rover(config, simulate=args.simulate)
    explorer = RoomExplorer(
        rover=rover,
        grid=grid,
        speed=args.speed,
        duration=args.duration,
        room_bounds_m=args.room_bounds,
        simulate=args.simulate,
        robot_radius_m=config.rover.robot_radius_m,
        seed=args.seed,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, explorer.request_stop)

    try:
        logger.info("Connecting to RVR+ (%s) ...", 'simulated' if args.simulate else 'BLE')
        await rover.connect()
        logger.info("Starting exploration: speed=%d, duration=%ss", args.speed, args.duration)
        await explorer.run()
    finally:
        try:
            if rover.streaming:
                await rover.stop_sensor_streaming()
        except Exception as e:
            logger.warning("Stop streaming failed: %s", e)

        try:
            if args.simulate:
                x, y = rover.position
                logger.info("Final position: (%.2f, %.2f)", x, y)
            else:
                await rover.return_home()
        except Exception as e:
            logger.warning("Return home failed: %s", e)

        try:
            await rover.disconnect()
        except Exception as e:
            logger.warning("Disconnect failed: %s", e)

        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(str(_PROJECT_ROOT), output_path)
        stats = save_results(grid, output_path, args)

        print("\n=== Exploration Summary ===")
        print(f"  Free cells:     {stats['free']}")
        print(f"  Wall cells:     {stats['wall']}")
        print(f"  Unknown cells:  {stats['unknown']}")
        print(f"  Coverage:       {stats['coverage_pct']:.1f}%")
        print(f"  Area explored:  {stats['area_free_m2']:.2f} m2")
        print(f"  Map saved to:   {output_path}")
        print(f"  Stats saved to: {stats['_stats_path']}")


if __name__ == '__main__':
    asyncio.run(main())
