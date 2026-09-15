"""Occupancy grid and explorer behaviour."""

from __future__ import annotations

import pytest

from src.mapping import CELL_FREE, CELL_WALL, OccupancyGrid, RoomExplorer
from src.rover.controller import RoverController


def test_world_cell_roundtrip():
    grid = OccupancyGrid(width_m=4.0, height_m=4.0, cell_cm=10)
    for x, y in [(0.0, 0.0), (1.0, -1.0), (-1.5, 0.5)]:
        r, c = grid.world_to_cell(x, y)
        back = grid.cell_to_world(r, c)
        assert back == pytest.approx((x, y), abs=0.05)


def test_ray_cast_fills_between_poses():
    """Free space is the line the robot travelled, not a single stamped cell.

    Regression: marking only the cell under the robot produced a path trace
    with unknown cells between consecutive poses, which starved the frontier
    planner.
    """
    grid = OccupancyGrid(width_m=4.0, height_m=4.0, cell_cm=5)
    grid.record_pose(0.0, 0.0)
    grid.record_pose(1.0, 0.0)  # 1 m jump == 20 cells at 5 cm

    # Every cell along the way is free, not just the endpoints.
    for step in range(0, 21):
        assert grid.cell_state(step * 0.05, 0.0) == CELL_FREE
    assert grid.stats()["free"] >= 20


def test_walls_are_never_downgraded_to_free():
    grid = OccupancyGrid(width_m=2.0, height_m=2.0, cell_cm=5)
    grid.mark_wall(0.5, 0.0)
    grid.record_pose(0.0, 0.0)
    grid.record_pose(1.0, 0.0)
    assert grid.cell_state(0.5, 0.0) == CELL_WALL


def test_obstacle_is_marked_outside_the_chassis():
    """A bumped wall sits at the bumper, not under the robot's centre."""
    grid = OccupancyGrid(width_m=4.0, height_m=4.0, cell_cm=5)
    wall_x, wall_y = grid.mark_obstacle_ahead(0.0, 0.0, heading_deg=0.0, standoff_m=0.20)
    assert wall_y == pytest.approx(0.20)
    assert wall_x == pytest.approx(0.0, abs=1e-9)
    assert grid.cell_state(0.0, 0.20) == CELL_WALL
    # The robot's own cell stays traversable.
    grid.record_pose(0.0, 0.0)
    assert grid.cell_state(0.0, 0.0) == CELL_FREE


def test_unknown_count_stops_at_walls():
    grid = OccupancyGrid(width_m=4.0, height_m=4.0, cell_cm=5)
    open_count = grid.count_unknown_along(0.0, 0.0, 0.0, max_range_m=1.0)
    grid.mark_wall(0.0, 0.2)
    blocked_count = grid.count_unknown_along(0.0, 0.0, 0.0, max_range_m=1.0)
    assert blocked_count < open_count


def test_stats_are_consistent():
    grid = OccupancyGrid(width_m=1.0, height_m=1.0, cell_cm=10)
    stats = grid.stats()
    assert stats["unknown"] == stats["total_cells"]
    assert stats["coverage_pct"] == 0.0
    grid.mark_free(0.0, 0.0)
    stats = grid.stats()
    assert stats["free"] == 1
    assert stats["free"] + stats["wall"] + stats["unknown"] == stats["total_cells"]


def test_random_turn_is_seedable():
    grid = OccupancyGrid(width_m=2.0, height_m=2.0, cell_cm=10)
    rover = RoverController(simulate=True)
    a = RoomExplorer(rover=rover, grid=grid, seed=7)
    b = RoomExplorer(rover=rover, grid=grid, seed=7)
    assert [a.random_turn() for _ in range(5)] == [b.random_turn() for _ in range(5)]


@pytest.mark.asyncio
async def test_simulated_exploration_builds_a_map():
    grid = OccupancyGrid(width_m=6.0, height_m=6.0, cell_cm=5)
    rover = RoverController(simulate=True)
    await rover.connect()
    explorer = RoomExplorer(
        rover=rover,
        grid=grid,
        speed=60,
        duration=2.0,
        room_bounds_m=3.0,
        simulate=True,
        seed=1,
    )
    await explorer.run()

    stats = grid.stats()
    assert stats["free"] > 0
    assert stats["unknown"] < stats["total_cells"]


@pytest.mark.asyncio
async def test_request_stop_ends_the_loop():
    grid = OccupancyGrid(width_m=6.0, height_m=6.0, cell_cm=5)
    rover = RoverController(simulate=True)
    await rover.connect()
    explorer = RoomExplorer(
        rover=rover,
        grid=grid,
        speed=60,
        duration=30.0,
        room_bounds_m=3.0,
        simulate=True,
        seed=1,
    )

    import asyncio

    task = asyncio.create_task(explorer.run())
    await asyncio.sleep(0.3)
    explorer.request_stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert not explorer.running
