# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command and rewards of hopping in place on the stance foot of the flamingo pose.

The command occupies the twist slots as ``[0, 0, hop]``: 1 asks for hopping, 0 for a still one-foot stand. An
internal blend ``beta`` follows the hop flag over a short ramp and gates the rewards, so the policy only sees the raw
flag. Penalties are non-negative costs for negative weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .flamingo import (
    foot_in_contact,
    foot_site_height_gaussian,
    joint_vel_gaussian,
    pose_target_match,
    projected_gravity_match,
)
from .observations import zero_non_finite

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


class HopCommand(CommandTerm):
    """``[0, 0, hop]`` alternating a still stand and hopping, starting still after every reset.

    Each phase lasts a time drawn from ``still_time_range`` or ``hop_time_range``. :attr:`beta` slews toward the hop
    flag over ``ramp_s`` seconds.
    """

    cfg: HopCommandCfg

    def __init__(self, cfg: HopCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._beta = torch.zeros(self.num_envs, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._hop_time = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """``[0, 0, hop]``, shape ``(num_envs, 3)``."""
        return self._command

    @property
    def beta(self) -> torch.Tensor:
        """Blend from the still (0) to the hopping (1) rewards, shape ``(num_envs,)``."""
        return self._beta

    @property
    def hop_time(self) -> torch.Tensor:
        """Time [s] since hopping was commanded, 0 while still, shape ``(num_envs,)``."""
        return self._hop_time

    def set_command(self, command: torch.Tensor) -> None:
        """Script all environments: write ``command`` ``(num_envs, 3)`` and stop the alternation."""
        self._command[:] = command
        self.time_left[:] = torch.inf

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self._fresh[ids] = True
        self._beta[ids] = 0.0
        self._hop_time[ids] = 0.0
        return super().reset(env_ids)

    def compute(self, dt: float):
        super().compute(dt)
        step = dt / max(self.cfg.ramp_s, 1e-6)
        self._beta += torch.clamp(self._command[:, 2] - self._beta, -step, step)
        hopping = self._command[:, 2] > 0.5
        self._hop_time = torch.where(hopping, self._hop_time + dt, torch.zeros_like(self._hop_time))

    def _resample_command(self, env_ids: Sequence[int] | slice):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(ids) == 0:
            return
        hop = torch.where(self._fresh[ids], torch.zeros_like(self._beta[ids]), 1.0 - self._command[ids, 2])
        self._command[ids, 2] = hop
        still = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.still_time_range)
        hopping = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.hop_time_range)
        self.time_left[ids] = torch.where(hop > 0.5, hopping, still)
        self._fresh[ids] = False

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class HopCommandCfg(CommandTermCfg):
    """Configuration of :class:`HopCommand`; ``resampling_time_range`` is overridden by the phase ranges."""

    class_type: type = HopCommand
    still_time_range: tuple[float, float] = (1.5, 3.0)
    """Duration [s] of a still phase."""
    hop_time_range: tuple[float, float] = (2.0, 4.0)
    """Duration [s] of a hopping phase."""
    ramp_s: float = 0.3
    """Time [s] for ``beta`` to cross from 0 to 1."""


def _beta(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).beta


def hop_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_air_time: float,
    max_air_time: float,
    swing_sensor_cfg: SceneEntityCfg | None = None,
    swing_lead_time: float = 0.0,
) -> torch.Tensor:
    """Score in [0, 1] of the stance foot's flight ending this step while hopping: 0 at ``min_air_time`` [s] or
    shorter, 1 at ``max_air_time`` [s] or longer. Only flights of the stance foot count, so a tapping swing foot earns
    nothing.

    With ``swing_sensor_cfg``, a landing pays only when the swing foot has been in the air at least the stance flight
    plus ``swing_lead_time`` [s]: it was up before the takeoff, so it did not push off, and stayed up to the landing.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    landed = sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids].any(dim=1).float()
    air = zero_non_finite(sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]).max(dim=1).values
    score = torch.clamp((air - min_air_time) / (max_air_time - min_air_time), 0.0, 1.0)
    if swing_sensor_cfg is not None:
        swing_sensor: ContactSensor = env.scene.sensors[swing_sensor_cfg.name]
        swing_air = zero_non_finite(swing_sensor.data.current_air_time.torch[:, swing_sensor_cfg.body_ids])
        score = score * (swing_air.min(dim=1).values >= air + swing_lead_time - 1e-6).float()
    return _beta(env, command_name) * landed * score


def hop_ground_time_cost(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, max_contact_time: float
) -> torch.Tensor:
    """1 while hopping and the stance foot has stayed down longer than ``max_contact_time`` [s]."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = zero_non_finite(sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]).max(dim=1).values
    return _beta(env, command_name) * (contact_time > max_contact_time).float()


def hop_wait_cost(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """While hopping, the time [s] the stance foot has stood since its last takeoff or the start of hopping, whichever
    is later: it grows the longer the robot waits on one foot and resets on a takeoff.

    It is not capped, so standing through a hop phase eventually costs more than the still-stand rewards pay.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = zero_non_finite(sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]).max(dim=1).values
    term = env.command_manager.get_term(command_name)
    return term.beta * torch.minimum(contact_time, term.hop_time)


def hop_drift_cost(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_weight: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """While hopping, squared horizontal trunk speed [m/s] plus ``yaw_weight`` times the squared yaw rate [rad/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin = zero_non_finite(asset.data.root_lin_vel_w.torch[:, :2])
    yaw = zero_non_finite(asset.data.root_ang_vel_w.torch[:, 2])
    return _beta(env, command_name) * ((lin**2).sum(dim=-1) + yaw_weight * yaw**2)


def still_foot_in_contact(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """:func:`foot_in_contact`, faded out while hopping."""
    return (1.0 - _beta(env, command_name)) * foot_in_contact(env, sensor_cfg)


def still_joint_vel_gaussian(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
    gate_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """:func:`joint_vel_gaussian`, faded out while hopping."""
    return (1.0 - _beta(env, command_name)) * joint_vel_gaussian(env, std, asset_cfg, gate_sensor_cfg)


def still_foot_site_height_gaussian(
    env: ManagerBasedRLEnv,
    command_name: str,
    target: float,
    std: float,
    asset_cfg: SceneEntityCfg,
    gate_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """:func:`foot_site_height_gaussian`, faded out while hopping."""
    reward = foot_site_height_gaussian(env, target, std, asset_cfg, gate_sensor_cfg)
    return (1.0 - _beta(env, command_name)) * reward


def still_pose_target_match(
    env: ManagerBasedRLEnv, command_name: str, target: Sequence[float], std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """:func:`pose_target_match`, faded out while hopping."""
    return (1.0 - _beta(env, command_name)) * pose_target_match(env, target, std, asset_cfg)


def still_projected_gravity_match(
    env: ManagerBasedRLEnv,
    command_name: str,
    target: tuple[float, float, float],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """:func:`projected_gravity_match`, faded out while hopping."""
    return (1.0 - _beta(env, command_name)) * projected_gravity_match(env, target, std, asset_cfg)


def still_action_rate_l2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Squared change of the policy actions, faded out while hopping."""
    rate = env.action_manager.action - env.action_manager.prev_action
    return (1.0 - _beta(env, command_name)) * (rate**2).sum(dim=1)
