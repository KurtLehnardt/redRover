"""2-D occupancy grid with ray-cast free-space marking."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_WALL = 2


@dataclass
class Pose:
    x: float
    y: float
    heading: float = 0.0


class OccupancyGrid:
    """2-D occupancy grid centred on the origin.

    Coordinates are stored in **centimetres** internally; the public API
    accepts metres and converts.

    Free space is marked by ray-casting between consecutive poses rather than
    by stamping the single cell under the robot.  Stamping only the current
    cell produces a path trace — a thin line through an otherwise unknown room
    — which is not an occupancy map and cannot be used for frontier planning.
    """

    def __init__(self, width_m: float = 10.0, height_m: float = 10.0, cell_cm: int = 5):
        if cell_cm <= 0:
            raise ValueError("cell_cm must be positive")
        self.cell_cm = cell_cm
        self.cols = max(1, int(width_m * 100 / cell_cm))
        self.rows = max(1, int(height_m * 100 / cell_cm))
        self.origin_col = self.cols // 2
        self.origin_row = self.rows // 2
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.path: list[tuple[float, float]] = []
        self._last_pose: tuple[float, float] | None = None

    # -- coordinate helpers -------------------------------------------------

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Grid (row, col) for a world position in metres."""
        col = self.origin_col + int(round(x_m * 100 / self.cell_cm))
        row = self.origin_row - int(round(y_m * 100 / self.cell_cm))  # y-up -> row-down
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x_m = (col - self.origin_col) * self.cell_cm / 100.0
        y_m = (self.origin_row - row) * self.cell_cm / 100.0
        return x_m, y_m

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def cell_state(self, x_m: float, y_m: float) -> int:
        """Occupancy state at a world position, ``CELL_WALL`` when off-grid."""
        r, c = self.world_to_cell(x_m, y_m)
        if not self.in_bounds(r, c):
            return CELL_WALL
        return int(self.grid[r, c])

    # -- marking ------------------------------------------------------------

    def mark_free(self, x_m: float, y_m: float) -> None:
        """Mark a single cell free (walls are never downgraded)."""
        r, c = self.world_to_cell(x_m, y_m)
        if self.in_bounds(r, c) and self.grid[r, c] != CELL_WALL:
            self.grid[r, c] = CELL_FREE

    def mark_wall(self, x_m: float, y_m: float) -> None:
        r, c = self.world_to_cell(x_m, y_m)
        if self.in_bounds(r, c):
            self.grid[r, c] = CELL_WALL

    def record_pose(self, x_m: float, y_m: float) -> None:
        """Append a pose to the path and ray-cast free space from the last one."""
        if self._last_pose is not None:
            self.mark_ray_free(*self._last_pose, x_m, y_m)
        else:
            self.mark_free(x_m, y_m)
        self._last_pose = (x_m, y_m)
        self.path.append((x_m, y_m))

    def mark_ray_free(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Mark every cell along the segment (x0,y0)->(x1,y1) as free.

        Bresenham over grid cells: the robot demonstrably traversed this line,
        so every cell it passed through is free space.
        """
        r0, c0 = self.world_to_cell(x0, y0)
        r1, c1 = self.world_to_cell(x1, y1)

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        step_r = 1 if r0 < r1 else -1
        step_c = 1 if c0 < c1 else -1
        err = dc - dr

        r, c = r0, c0
        # A traversal longer than the grid diagonal means the pose jumped;
        # bound the loop instead of trusting the endpoints.
        max_steps = self.rows + self.cols
        for _ in range(max_steps):
            if self.in_bounds(r, c) and self.grid[r, c] != CELL_WALL:
                self.grid[r, c] = CELL_FREE
            if r == r1 and c == c1:
                break
            err2 = 2 * err
            if err2 > -dr:
                err -= dr
                c += step_c
            if err2 < dc:
                err += dc
                r += step_r

    def mark_obstacle_ahead(
        self,
        x_m: float,
        y_m: float,
        heading_deg: float,
        standoff_m: float,
    ) -> tuple[float, float]:
        """Mark a wall ``standoff_m`` ahead of the robot along ``heading_deg``.

        ``standoff_m`` should be at least the chassis radius: a wall the robot
        bumped is at its front bumper, not under its centre, and marking it
        under the centre carves the obstacle out of the robot's own footprint.

        Returns the world coordinates that were marked.
        """
        from ..rover.controller import heading_to_vector

        ux, uy = heading_to_vector(heading_deg)
        wall_x = x_m + ux * standoff_m
        wall_y = y_m + uy * standoff_m
        self.mark_wall(wall_x, wall_y)
        return wall_x, wall_y

    def count_unknown_along(
        self,
        x_m: float,
        y_m: float,
        heading_deg: float,
        max_range_m: float = 1.0,
        step_m: float = 0.1,
    ) -> int:
        """Number of unknown cells sampled along a heading — frontier score."""
        from ..rover.controller import heading_to_vector

        ux, uy = heading_to_vector(heading_deg)
        unknown = 0
        distance = step_m
        while distance <= max_range_m:
            px = x_m + ux * distance
            py = y_m + uy * distance
            r, c = self.world_to_cell(px, py)
            if not self.in_bounds(r, c):
                break
            if self.grid[r, c] == CELL_WALL:
                break  # cannot see past a wall
            if self.grid[r, c] == CELL_UNKNOWN:
                unknown += 1
            distance += step_m
        return unknown

    # -- stats --------------------------------------------------------------

    def stats(self) -> dict:
        total = self.rows * self.cols
        free = int(np.sum(self.grid == CELL_FREE))
        wall = int(np.sum(self.grid == CELL_WALL))
        return {
            "total_cells": total,
            "free": free,
            "wall": wall,
            "unknown": total - free - wall,
            "coverage_pct": (free + wall) / total * 100 if total else 0.0,
            "area_free_m2": free * (self.cell_cm / 100.0) ** 2,
        }

    # -- visualisation ------------------------------------------------------

    def save_png(self, path: str, title: str = "Room Map") -> None:
        """Render the occupancy grid to a PNG image using matplotlib."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required to save a map image (pip install matplotlib)"
            ) from exc

        fig, ax = plt.subplots(figsize=(10, 10))

        cmap = ListedColormap(["#C0C0C0", "#FFFFFF", "#000000"])
        extent = [
            -self.origin_col * self.cell_cm / 100.0,
            (self.cols - self.origin_col) * self.cell_cm / 100.0,
            -self.origin_row * self.cell_cm / 100.0,
            (self.rows - self.origin_row) * self.cell_cm / 100.0,
        ]
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=2, origin="upper", extent=extent)

        if len(self.path) > 1:
            xs = [p[0] for p in self.path]
            ys = [p[1] for p in self.path]
            ax.plot(xs, ys, "b-", linewidth=0.8, alpha=0.6, label="Path")

        ax.plot(0, 0, "go", markersize=10, label="Start", zorder=5)
        ax.set_xlabel("X (metres)")
        ax.set_ylabel("Y (metres)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Map saved to %s", path)
