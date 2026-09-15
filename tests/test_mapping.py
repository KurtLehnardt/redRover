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


# === A mapping run's data must survive a missing renderer ===


def test_stats_are_written_even_when_matplotlib_is_missing(tmp_path, monkeypatch):
    """Regression: rendering ran before the data was persisted.

    A missing matplotlib threw away an entire mapping run — the robot drove the
    full duration and produced nothing at all. The picture is a convenience;
    the measurements are the point.
    """
    import argparse
    import json

    from scripts.explore_map import save_results

    grid = OccupancyGrid(width_m=2.0, height_m=2.0, cell_cm=10)
    grid.record_pose(0.0, 0.0)
    grid.record_pose(0.5, 0.0)
    grid.mark_wall(0.6, 0.0)

    def _no_matplotlib(*_args, **_kwargs):
        raise RuntimeError("matplotlib is required to save a map image")

    monkeypatch.setattr(OccupancyGrid, "save_png", _no_matplotlib)

    output = tmp_path / "room_map.png"
    args = argparse.Namespace(duration=30.0, speed=60, room_bounds=2.0)
    stats = save_results(grid, str(output), args)

    stats_path = tmp_path / "room_map_stats.json"
    assert stats_path.exists(), "the run's data must survive"
    saved = json.loads(stats_path.read_text())
    assert saved["free"] > 0
    assert saved["wall"] == 1
    assert saved["status"] == "complete"
    # And the caller can tell there is no image.
    assert stats["_image_path"] is None


def test_stats_and_image_are_both_written_when_rendering_works(tmp_path):
    import argparse
    import json

    pytest.importorskip("matplotlib")
    from scripts.explore_map import save_results

    grid = OccupancyGrid(width_m=2.0, height_m=2.0, cell_cm=10)
    grid.record_pose(0.0, 0.0)

    output = tmp_path / "room_map.png"
    args = argparse.Namespace(duration=5.0, speed=60, room_bounds=2.0)
    stats = save_results(grid, str(output), args)

    assert output.exists()
    assert stats["_image_path"] == str(output)
    assert json.loads((tmp_path / "room_map_stats.json").read_text())["status"] == "complete"


@pytest.mark.asyncio
async def test_dashboard_map_persist_survives_a_missing_renderer(tmp_path, monkeypatch):
    """A completed mapping session used to report as an error."""
    import json

    from src.dashboard import app as dashboard

    monkeypatch.setattr(dashboard, "_MAP_STATS_PATH", tmp_path / "stats.json")
    monkeypatch.setattr(dashboard, "_MAP_IMAGE_PATH", tmp_path / "map.png")

    def _no_matplotlib(*_args, **_kwargs):
        raise RuntimeError("matplotlib is required to save a map image")

    monkeypatch.setattr(OccupancyGrid, "save_png", _no_matplotlib)

    grid = OccupancyGrid(width_m=2.0, height_m=2.0, cell_cm=10)
    grid.record_pose(0.0, 0.0)

    stats = dashboard._persist_map(grid, speed=50, duration=10.0, room_bounds=1.0)

    assert stats["status"] == "complete"
    assert json.loads((tmp_path / "stats.json").read_text())["free"] > 0
