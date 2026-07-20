"""Patrol scheduler — runs patrol cycles at configured intervals."""

import asyncio
import logging
from datetime import datetime, time as dt_time

from ..config import load_config

logger = logging.getLogger("redRover.scheduler")


def _in_quiet_hours(start_str: str, end_str: str) -> bool:
    """Check if current time is within quiet hours window."""
    now = datetime.now().time()
    start = dt_time.fromisoformat(start_str)
    end = dt_time.fromisoformat(end_str)

    if start <= end:
        return start <= now <= end
    else:
        # Overnight window (e.g., 22:00 - 06:00)
        return now >= start or now <= end


async def run_scheduler(
    simulate: bool = True,
    skip_ai: bool = False,
    enable_drone: bool = True,
    quiet_hours_only: bool = False,
):
    """Run patrol cycles on a schedule.

    Args:
        simulate: Use simulated hardware
        skip_ai: Skip AI analysis
        enable_drone: Enable drone deployment
        quiet_hours_only: Only patrol during quiet hours
    """
    from ..main import run_patrol

    config = load_config()
    interval = config.scheduler.patrol_interval * 60  # minutes to seconds
    patrol_count = 0

    logger.info("=" * 70)
    logger.info("redRover Scheduler Started")
    logger.info("  Patrol interval: %d minutes", config.scheduler.patrol_interval)
    if quiet_hours_only:
        logger.info("  Quiet hours: %s - %s", config.scheduler.quiet_hours_start, config.scheduler.quiet_hours_end)
    logger.info("=" * 70)

    while True:
        if quiet_hours_only and not _in_quiet_hours(
            config.scheduler.quiet_hours_start,
            config.scheduler.quiet_hours_end,
        ):
            logger.info("Outside quiet hours — skipping patrol (next check in 5 min)")
            await asyncio.sleep(300)
            continue

        patrol_count += 1
        logger.info("Starting scheduled patrol #%d at %s", patrol_count, datetime.now().isoformat())

        try:
            results = await run_patrol(
                simulate=simulate,
                skip_ai=skip_ai,
                enable_drone=enable_drone,
            )
            n_faults = len([r for r in results if hasattr(r, 'overall_health') and r.overall_health.value != "healthy"])
            logger.info("Patrol #%d complete — %d faults detected", patrol_count, n_faults)
        except Exception as e:
            logger.error("Patrol #%d failed: %s", patrol_count, e)

        logger.info("Next patrol in %d minutes", config.scheduler.patrol_interval)
        await asyncio.sleep(interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="redRover Patrol Scheduler")
    parser.add_argument("--simulate", action="store_true", default=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--skip-ai", action="store_true")
    parser.add_argument("--no-drone", action="store_true")
    parser.add_argument("--quiet-hours-only", action="store_true",
                        help="Only run patrols during configured quiet hours")
    args = parser.parse_args()

    asyncio.run(run_scheduler(
        simulate=not args.real,
        skip_ai=args.skip_ai,
        enable_drone=not args.no_drone,
        quiet_hours_only=args.quiet_hours_only,
    ))


if __name__ == "__main__":
    main()
