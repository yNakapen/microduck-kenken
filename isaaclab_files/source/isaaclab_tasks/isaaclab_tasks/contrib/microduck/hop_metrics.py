# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation of MicroDuck single-leg stands and hops on Newton MJWarp with BAM servos.

:func:`read_hop_signals` samples one control step of a live environment, :class:`HopMetrics` accumulates those
samples over an evaluation episode, and :func:`stage_passes` applies the test-plan criteria to the per-environment
summaries. :func:`set_supply_voltage` pins each environment's servo supply, which BAM otherwise draws once at
start-up from ``vin_range``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators.newton import read_group_parameter, write_group_parameter

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

from .flat_env_cfg import MICRODUCK_SOLE_TO_ANKLE_OFFSET
from .hop_env_cfg import MICRODUCK_LIFTED_FOOT, MICRODUCK_SINGLE_LEG_MIN_TRUNK_HEIGHT, MICRODUCK_SUPPORT_FOOT

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

STAND_HOLD_TIME = 10.0
"""Single-support time an environment must hold to pass the stand stage [s]."""
SINGLE_HOP_MIN_AIR_TIME = 0.06
"""Shortest flight of a passing single hop [s]."""
SINGLE_HOP_POST_LANDING_HOLD = 2.0
"""Single-support time after the landing of a passing single hop [s]."""
SINGLE_HOP_PASS_FRACTION = 0.8
"""Fraction of environments that must pass the single-hop stage."""
HOP_MIN_COUNT = 5
"""Hops a standing environment must make to pass the continuous-hop stage."""
HOP_MAX_FALL_FRACTION = 0.2
"""Largest fraction of environments that may fall in the continuous-hop stage."""


@dataclass
class HopSignals:
    """One control step of every environment, each of shape ``(num_envs,)``."""

    support_contact: torch.Tensor
    """Whether the support (right) foot touches the ground."""
    lifted_contact: torch.Tensor
    """Whether the lifted (left) foot touches anything."""
    support_touchdown: torch.Tensor
    """Whether the support foot touched down during the last step."""
    support_last_air_time: torch.Tensor
    """Duration of the support foot's last flight [s], valid on touchdown."""
    com_height: torch.Tensor
    """Whole-body centre-of-mass height above the environment origin [m]."""
    support_sole_height: torch.Tensor
    """Support-foot sole height above the environment origin [m]."""
    fallen: torch.Tensor
    """Whether the environment terminated this step, for any reason but a time-out (a fall or a diverged state)."""
    trunk_low: torch.Tensor
    """Whether the trunk is below the standing height, as in a crouch or a collapse."""
    saturated_fraction: torch.Tensor
    """Fraction of servos delivering their full available torque (duty, back-EMF and current limits)."""
    max_joint_speed: torch.Tensor
    """Fastest servo speed [rad/s]."""
    support_force: torch.Tensor
    """Contact-force magnitude on the support foot [N]."""


def set_supply_voltage(
    env: ManagerBasedRLEnv,
    vin: torch.Tensor,
    sag_gain: float = 0.0,
    asset_name: str = "robot",
    actuator_name: str = "servos",
) -> None:
    """Pin each environment's BAM supply to ``vin`` [V], shape ``(num_envs,)``, with battery sag ``sag_gain``.

    BAM samples the supply once at start-up and keeps it through resets, so one call covers the whole evaluation.
    """
    actuators = env.scene[asset_name].actuators
    current = read_group_parameter(actuators, actuator_name, "drive", "vin")
    values = vin.to(current.device, current.dtype)[:, None].expand_as(current).contiguous()
    write_group_parameter(actuators, actuator_name, "drive", "vin", values)
    write_group_parameter(actuators, actuator_name, "drive", "sag_gain", torch.full_like(current, sag_gain))


def _torque_saturation(robot: Articulation, actuator_name: str, margin: float) -> torch.Tensor:
    """Fraction of each environment's servos whose torque reaches ``margin`` of what BAM can deliver right now.

    The available torque is the full-duty voltage minus back-EMF, ``kt (max_pwm vin - kt w sign(tau)) / R``, capped by
    the current limit ``kt max_current``. All servos share one motor model, so the first servo's parameters apply.
    """
    actuators = robot.actuators

    def param(attr: str) -> torch.Tensor:
        return read_group_parameter(actuators, actuator_name, "drive", attr)[:, :1]

    kt, resistance = param("kt"), param("resistance")
    joint_ids = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)[0]
    torque = actuators.applied_effort.torch[:, joint_ids]
    speed = robot.data.joint_vel.torch[:, joint_ids]
    available = kt / resistance * (param("max_pwm") * param("vin") - kt * speed * torch.sign(torque))
    available = torch.minimum(available.clamp(min=0.0), kt * param("max_current"))
    return (torque.abs() >= margin * available).float().mean(dim=1)


def read_hop_signals(
    env: ManagerBasedRLEnv,
    sensor_name: str = "contact_forces",
    asset_name: str = "robot",
    actuator_name: str = "servos",
    saturation_margin: float = 0.95,
    min_trunk_height: float = MICRODUCK_SINGLE_LEG_MIN_TRUNK_HEIGHT,
) -> HopSignals:
    """Sample the evaluation signals after an environment step.

    A fall is the termination manager's terminal signal; a trunk below ``min_trunk_height`` [m] is reported apart,
    since an upright crouch is not a fall.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    robot: Articulation = env.scene[asset_name]
    (lifted_id, support_id), _ = sensor.find_sensors([MICRODUCK_LIFTED_FOOT, MICRODUCK_SUPPORT_FOOT], True)
    contact_time = sensor.data.current_contact_time.torch
    origin_z = env.scene.env_origins[:, 2]

    mass = robot.data.body_mass.torch
    com = (mass.unsqueeze(-1) * robot.data.body_com_pos_w.torch).sum(dim=1) / mass.sum(dim=1, keepdim=True)
    body_id = robot.find_bodies(MICRODUCK_SUPPORT_FOOT)[0][0]
    sole = robot.data.body_link_pos_w.torch[:, body_id, 2] - origin_z - MICRODUCK_SOLE_TO_ANKLE_OFFSET
    trunk = robot.data.root_link_pos_w.torch[:, 2] - origin_z
    joint_ids = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)[0]

    return HopSignals(
        support_contact=contact_time[:, support_id] > 0.0,
        lifted_contact=contact_time[:, lifted_id] > 0.0,
        support_touchdown=sensor.compute_first_contact(env.step_dt).torch[:, support_id] > 0.5,
        support_last_air_time=sensor.data.last_air_time.torch[:, support_id].clone(),
        com_height=com[:, 2] - origin_z,
        support_sole_height=sole,
        fallen=env.termination_manager.terminated.clone(),
        trunk_low=~(trunk >= min_trunk_height),
        saturated_fraction=_torque_saturation(robot, actuator_name, saturation_margin),
        max_joint_speed=robot.data.joint_vel.torch[:, joint_ids].abs().amax(dim=1),
        support_force=sensor.data.net_forces_w.torch[:, support_id].norm(dim=-1),
    )


class HopMetrics:
    """Accumulate :class:`HopSignals` over an episode, ignoring every step from an environment's first fall on.

    An environment whose trunk ever drops below the standing height is flagged as collapsed but still evaluated.

    A hop is a flight of the support foot that ends on that foot, lasts at least ``min_air_time`` [s], and during
    which the lifted foot never touches anything.
    """

    def __init__(self, num_envs: int, step_dt: float, min_air_time: float = 0.02, device: str | torch.device = "cpu"):
        self.step_dt = step_dt
        self.min_air_time = min_air_time

        def zeros(dtype=torch.float32):
            return torch.zeros(num_envs, dtype=dtype, device=device)

        self._alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._collapsed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._steps = 0
        self._fall_time = zeros()
        self._prev_support = zeros(torch.bool)
        self._prev_lifted = zeros(torch.bool)
        self._prev_com = zeros()
        self._airborne = zeros(torch.bool)
        self._flight_steps = zeros()
        self._lifted_in_air = zeros(torch.bool)
        self._com_takeoff = zeros()
        self._com_peak = zeros()
        self._sole_peak = zeros()
        self._streak = zeros()
        self._post_active = zeros(torch.bool)
        self._post_streak = zeros()
        self._first_step = True
        self.totals = {
            name: zeros()
            for name in (
                "hold_time",
                "post_landing_hold",
                "hops",
                "lifted_touchdowns",
                "air_time_sum",
                "max_air_time",
                "com_rise_sum",
                "max_com_rise",
                "max_sole_clearance",
                "saturation_sum",
                "alive_steps",
                "max_joint_speed",
                "peak_force",
            )
        }

    def update(self, signals: HopSignals) -> None:
        """Fold one control step into the episode statistics."""
        t = self.totals
        dt = self.step_dt
        support, lifted = signals.support_contact, signals.lifted_contact
        com = signals.com_height
        com_valid = torch.isfinite(com)
        sole = torch.nan_to_num(signals.support_sole_height, nan=0.0, posinf=0.0, neginf=0.0)
        self._steps += 1

        falls = self._alive & signals.fallen
        self._fall_time = torch.where(falls, torch.full_like(self._fall_time, self._steps * dt), self._fall_time)
        self._alive &= ~signals.fallen
        live = self._alive
        self._collapsed |= signals.trunk_low & live
        if self._first_step:
            # Only transitions within the episode count; the reset stance is the reference.
            self._prev_support, self._prev_lifted = support.clone(), lifted.clone()
            self._prev_com = torch.where(com_valid, com, torch.zeros_like(com))
            self._first_step = False

        t["lifted_touchdowns"] += (lifted & ~self._prev_lifted & live).float()

        takeoff = self._prev_support & ~support & live
        self._com_takeoff = torch.where(takeoff, self._prev_com, self._com_takeoff)
        self._com_peak = torch.where(takeoff, self._prev_com, self._com_peak)
        self._sole_peak = torch.where(takeoff, torch.zeros_like(sole), self._sole_peak)
        self._lifted_in_air = self._lifted_in_air & ~takeoff
        self._flight_steps = torch.where(takeoff, torch.zeros_like(self._flight_steps), self._flight_steps)
        self._airborne = (self._airborne | takeoff) & live

        flying = self._airborne & ~support
        self._flight_steps += flying.float()
        self._com_peak = torch.where(flying & com_valid, torch.maximum(self._com_peak, com), self._com_peak)
        self._sole_peak = torch.where(flying, torch.maximum(self._sole_peak, sole), self._sole_peak)
        self._lifted_in_air |= flying & lifted

        landing = self._airborne & support
        air_time = torch.where(
            signals.support_touchdown, signals.support_last_air_time, self._flight_steps * dt
        ).nan_to_num(0.0)
        hop = landing & ~self._lifted_in_air & ~lifted & (air_time >= self.min_air_time)
        rise = (self._com_peak - self._com_takeoff).clamp(min=0.0)
        t["hops"] += hop.float()
        t["air_time_sum"] += torch.where(hop, air_time, 0.0)
        t["max_air_time"] = torch.where(hop, torch.maximum(t["max_air_time"], air_time), t["max_air_time"])
        t["com_rise_sum"] += torch.where(hop, rise, 0.0)
        t["max_com_rise"] = torch.where(hop, torch.maximum(t["max_com_rise"], rise), t["max_com_rise"])
        t["max_sole_clearance"] = torch.where(
            hop, torch.maximum(t["max_sole_clearance"], self._sole_peak), t["max_sole_clearance"]
        )
        self._airborne &= ~landing

        single = support & ~lifted & live
        self._streak = torch.where(single, self._streak + dt, torch.zeros_like(self._streak))
        t["hold_time"] = torch.maximum(t["hold_time"], self._streak)
        self._post_active = (self._post_active | hop) & single
        self._post_streak = torch.where(
            self._post_active, torch.where(hop, 0.0, self._post_streak) + dt, torch.zeros_like(self._post_streak)
        )
        t["post_landing_hold"] = torch.maximum(t["post_landing_hold"], self._post_streak)

        t["saturation_sum"] += torch.where(live, signals.saturated_fraction.nan_to_num(0.0), 0.0)
        t["alive_steps"] += live.float()
        speed = signals.max_joint_speed.nan_to_num(0.0, posinf=0.0)
        t["max_joint_speed"] = torch.where(live, torch.maximum(t["max_joint_speed"], speed), t["max_joint_speed"])
        force = signals.support_force.nan_to_num(0.0, posinf=0.0)
        t["peak_force"] = torch.where(live, torch.maximum(t["peak_force"], force), t["peak_force"])

        self._prev_support, self._prev_lifted = support.clone(), lifted.clone()
        self._prev_com = torch.where(com_valid, com, self._prev_com)

    def summary(self) -> dict[str, torch.Tensor]:
        """Per-environment episode statistics; times in [s], heights in [m], speeds in [rad/s], forces in [N]."""
        t = self.totals
        hops = t["hops"].clamp(min=1.0)
        fell = ~self._alive
        elapsed = torch.full_like(self._fall_time, self._steps * self.step_dt)
        return {
            "fell": fell,
            "collapsed": self._collapsed.clone(),
            "fall_time": torch.where(fell, self._fall_time, elapsed),
            "hold_time": t["hold_time"].clone(),
            "post_landing_hold": t["post_landing_hold"].clone(),
            "hops": t["hops"].clone(),
            "lifted_touchdowns": t["lifted_touchdowns"].clone(),
            "mean_air_time": t["air_time_sum"] / hops,
            "max_air_time": t["max_air_time"].clone(),
            "mean_com_rise": t["com_rise_sum"] / hops,
            "max_com_rise": t["max_com_rise"].clone(),
            "max_sole_clearance": t["max_sole_clearance"].clone(),
            "saturation_ratio": t["saturation_sum"] / t["alive_steps"].clamp(min=1.0),
            "max_joint_speed": t["max_joint_speed"].clone(),
            "peak_force": t["peak_force"].clone(),
        }


def stage_passes(stage: str, summary: dict[str, torch.Tensor]) -> bool:
    """Whether the evaluated environments meet the test plan for ``stage``: ``"stand"``, ``"single"`` or ``"hop"``.

    A collapsed environment fails like a fallen one.
    """
    failed = summary["fell"] | summary["collapsed"]
    n = failed.numel()
    if stage == "stand":
        return bool((~failed).all() and (summary["hold_time"] >= STAND_HOLD_TIME).all())
    if stage == "single":
        ok = (
            ~failed
            & (summary["max_air_time"] >= SINGLE_HOP_MIN_AIR_TIME)
            & (summary["post_landing_hold"] >= SINGLE_HOP_POST_LANDING_HOLD)
        )
        return int(ok.sum()) >= math.ceil(SINGLE_HOP_PASS_FRACTION * n)
    if stage == "hop":
        standing_ok = (summary["hops"] >= HOP_MIN_COUNT) & (summary["lifted_touchdowns"] == 0)
        return int(failed.sum()) <= math.floor(HOP_MAX_FALL_FRACTION * n) and bool((failed | standing_ok).all())
    raise ValueError(f"Unknown stage: {stage!r}")
