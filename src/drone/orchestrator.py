"""Drone-Rover orchestrator — coordinates the ground robot and aerial drone.

The orchestrator manages the full lifecycle:

1. RVR+ drives to a station and performs ground-level sensing
2. If an anomaly needs aerial confirmation -> deploy the drone
3. RVR+ stays still (it is the landing pad) while the drone inspects
4. Drone returns and docks
5. RVR+ continues the patrol

Safety rules:
- RVR+ must be stationary during drone flight
- Drone must have enough battery for the mission *and* the return leg
- If the drone loses contact, the RVR+ stays put as a landing beacon
- Emergency land if battery drops below threshold
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..rover.controller import RoverController, RoverState
from .controller import AerialCapture, DroneController
from .mission import FAULT_TO_MISSION, DroneMission, should_deploy_drone

logger = logging.getLogger(__name__)


@dataclass
class DeploymentResult:
    """Result of a drone deployment."""
    mission_id: str
    station_id: str
    success: bool
    captures: list[AerialCapture]
    flight_duration: float
    battery_used: int
    error: str | None = None


class DroneRoverOrchestrator:
    """Coordinates the ground robot and the piggyback drone."""

    def __init__(
        self,
        rover: RoverController,
        drone: DroneController,
        min_deploy_battery: int = 40,
        max_mission_duration: float = 90.0,
    ):
        self.rover = rover
        self.drone = drone
        self.min_deploy_battery = min_deploy_battery
        self.max_mission_duration = max_mission_duration
        self._deployments: list[DeploymentResult] = []

    async def can_deploy(self) -> tuple[bool, str]:
        """Whether a deployment is possible, and why not when it isn't."""
        # Check the e-stop first: it is the more specific and more important
        # reason, and reporting "must be dwelling" would hide it.
        if getattr(self.rover, "estopped", False):
            return False, "rover is in emergency stop"
        if self.rover.state is not RoverState.DWELLING:
            return False, f"rover is {self.rover.state.value}, must be dwelling"
        if not await self.drone.is_flight_ready():
            return False, f"drone state={self.drone.state.value}"
        battery = await self.drone.get_battery()
        if battery < self.min_deploy_battery:
            return False, f"battery {battery}% < {self.min_deploy_battery}% minimum"
        return True, ""

    async def evaluate_and_deploy(
        self,
        station_id: str,
        fault_type: str,
        station_config: dict | None = None,
    ) -> DeploymentResult | None:
        """Evaluate whether the drone should deploy, and execute if so.

        Returns a :class:`DeploymentResult` if deployed, ``None`` otherwise.
        """
        if not should_deploy_drone(fault_type, station_config):
            logger.info("[ORCH] Fault '%s' does not require aerial inspection", fault_type)
            return None

        ok, reason = await self.can_deploy()
        if not ok:
            logger.warning("[ORCH] Drone not available for deployment: %s", reason)
            return None

        mission = self._generate_mission(station_id, fault_type)
        if mission is None:
            return None

        if mission.estimated_duration > self.max_mission_duration:
            logger.warning(
                "[ORCH] Mission too long (%.0fs > %.0fs limit)",
                mission.estimated_duration, self.max_mission_duration,
            )
            return None

        return await self.deploy(mission)

    async def deploy(self, mission: DroneMission) -> DeploymentResult:
        """Execute a drone mission. The RVR+ remains stationary as the pad."""
        logger.info("")
        logger.info("  +-- DRONE DEPLOYMENT -------------------------------")
        logger.info("  |  Type:    %s", mission.mission_type.value)
        logger.info("  |  Station: %s", mission.station_id)
        logger.info("  |  Targets: %d  (est. %.0fs)",
                    mission.total_targets, mission.estimated_duration)
        logger.info("  |  Reason:  %s", mission.reason)
        logger.info("  +---------------------------------------------------")

        start_time = time.time()
        start_battery = await self.drone.get_battery()
        captures: list[AerialCapture] = []
        error: str | None = None

        try:
            logger.info("  [ORCH] Phase 1: launch from cradle")
            if not await self.drone.launch():
                return self._record(DeploymentResult(
                    mission_id=mission.mission_id,
                    station_id=mission.station_id,
                    success=False,
                    captures=[],
                    flight_duration=time.time() - start_time,
                    battery_used=0,
                    error="Launch failed",
                ))

            logger.info(
                "  [ORCH] Phase 2: executing %d inspection targets", mission.total_targets
            )
            for i, target in enumerate(mission.targets):
                battery = await self.drone.get_battery()
                # Reserve enough charge for the return leg, not just a margin
                # above zero.
                if battery < self.min_deploy_battery // 2:
                    logger.warning(
                        "  [ORCH] Low battery (%d%%), aborting mission", battery
                    )
                    error = "Low battery abort"
                    break

                elapsed = time.time() - start_time
                if elapsed > mission.timeout:
                    logger.warning("  [ORCH] Mission timeout (%.0fs), aborting", elapsed)
                    error = "Timeout abort"
                    break

                await self.drone.fly_to_target(target)
                capture = await self.drone.inspect(target)
                captures.append(capture)

                logger.info(
                    "  [ORCH]   Target %d/%d complete: %s (%d images)",
                    i + 1, mission.total_targets, target.name, len(capture.images),
                )

            logger.info("  [ORCH] Phase 3: return to cradle")
            await self.drone.return_to_cradle()

        except Exception as e:
            logger.error("  [ORCH] Mission error: %s", e)
            error = str(e)
            try:
                await self.drone.emergency_land()
            except Exception as land_exc:
                logger.error("  [ORCH] Emergency land also failed: %s", land_exc)

        flight_duration = time.time() - start_time
        battery_used = max(0, start_battery - await self.drone.get_battery())

        result = self._record(DeploymentResult(
            mission_id=mission.mission_id,
            station_id=mission.station_id,
            success=error is None,
            captures=captures,
            flight_duration=flight_duration,
            battery_used=battery_used,
            error=error,
        ))

        logger.info(
            "  [ORCH] Deployment complete: %s | Duration: %.1fs | Battery used: %d%%",
            "SUCCESS" if result.success else f"FAILED ({error})",
            flight_duration, battery_used,
        )
        return result

    def _record(self, result: DeploymentResult) -> DeploymentResult:
        self._deployments.append(result)
        return result

    def _generate_mission(self, station_id: str, fault_type: str) -> DroneMission | None:
        """Generate a mission based on fault type."""
        entry = FAULT_TO_MISSION.get(fault_type)
        if entry is None:
            logger.warning("[ORCH] No mission template for fault: %s", fault_type)
            return None
        _, generator = entry
        return generator(
            station_id=station_id, reason=f"Ground sensors detected: {fault_type}"
        )

    @property
    def deployments(self) -> list[DeploymentResult]:
        return list(self._deployments)

    @property
    def total_deployments(self) -> int:
        return len(self._deployments)

    @property
    def successful_deployments(self) -> int:
        return sum(1 for d in self._deployments if d.success)
