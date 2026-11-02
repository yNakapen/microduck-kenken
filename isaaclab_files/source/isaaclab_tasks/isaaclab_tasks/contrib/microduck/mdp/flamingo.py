# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measurements, rewards and the reset of the one-foot flamingo task.

The rewards port the stage-1 terms of microduck_rl (``src/mjlab_microduck/tasks/mdp.py`` on the ``flamingo``
branch, ``4757d6ae``). microduck_rl writes penalties as non-positive values with positive weights; here they are
non-negative costs for negative weights, as in the rest of the MicroDuck task. Foot contact is the contact sensor's
per-body contact time instead of microduck_rl's per-geom ``found`` flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from .observations import zero_non_finite

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensor

MICRODUCK_FOOT_SITE_OFFSET = (0.0, -0.02381, -0.01409)
"""Position [m] of the ``left_foot`` / ``right_foot`` sites in their ankle frames, identical in the MJCF and USD."""


def robot_com_pos_w(asset: Articulation) -> torch.Tensor:
    """Whole-robot center of mass [m] in the world frame, shape ``(num_envs, 3)``.

    The mass-weighted mean of the body CoMs, which matches MuJoCo's root ``subtree_com``. The root link's own CoM
    is centimeters away from it with the heavy head and a lifted leg.
    """
    mass = asset.data.body_mass.torch
    com = asset.data.body_com_pos_w.torch
    return (mass.unsqueeze(-1) * com).sum(dim=1) / mass.sum(dim=1, keepdim=True)


def foot_site_pos_w(
    asset: Articulation, body_ids: Sequence[int], offset: tuple[float, float, float] = MICRODUCK_FOOT_SITE_OFFSET
) -> torch.Tensor:
    """World positions [m] of the foot sites of ankle bodies ``body_ids``, shape ``(num_envs, len(body_ids), 3)``."""
    pos = asset.data.body_link_pos_w.torch[:, body_ids]
    quat = asset.data.body_link_quat_w.torch[:, body_ids]
    local = torch.tensor(offset, device=pos.device, dtype=pos.dtype).expand_as(pos)
    return pos + quat_apply(quat, local)


"""
Rewards.
"""


def foot_in_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 while every selected foot touches the ground, else 0."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    return (zero_non_finite(contact_time) > 0.0).all(dim=1).float()


def com_over_foot(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Gaussian of the horizontal distance between the whole-robot CoM and the foot site of ankle ``asset_cfg``."""
    asset: Articulation = env.scene[asset_cfg.name]
    com_xy = robot_com_pos_w(asset)[:, :2]
    foot_xy = foot_site_pos_w(asset, asset_cfg.body_ids)[:, 0, :2]
    dist2 = torch.nan_to_num(((com_xy - foot_xy) ** 2).sum(dim=-1), nan=1.0)
    return torch.exp(-dist2 / std**2)


def foot_site_height_gaussian(
    env: ManagerBasedRLEnv,
    target: float,
    std: float,
    asset_cfg: SceneEntityCfg,
    gate_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Gaussian of the foot-site height [m] of ankle ``asset_cfg`` above the env origin around ``target``.

    With ``gate_sensor_cfg``, paid only while that (stance) foot is down, so hopping cannot farm it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = foot_site_pos_w(asset, asset_cfg.body_ids)[:, 0, 2] - env.scene.env_origins[:, 2]
    reward = torch.exp(-(((zero_non_finite(z) - target) / std) ** 2))
    if gate_sensor_cfg is not None:
        reward = reward * foot_in_contact(env, gate_sensor_cfg)
    return reward


def projected_gravity_match(
    env: ManagerBasedRLEnv,
    target: tuple[float, float, float],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian of the distance between the trunk-frame gravity direction and ``target``, to hold a trunk lean."""
    asset: Articulation = env.scene[asset_cfg.name]
    gravity = zero_non_finite(asset.data.projected_gravity_b.torch)
    target_t = torch.tensor(target, device=gravity.device, dtype=gravity.dtype)
    return torch.exp(-((gravity - target_t) ** 2).sum(dim=-1) / std**2)


def lateral_tilt_cost(
    env: ManagerBasedRLEnv,
    threshold: float,
    direction: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared excess of ``direction * g_b,y`` over ``threshold``: rolling further toward one side than the pose.

    With ``direction=-1`` this is the stance side of the right-foot pose, the fall direction with no foot to catch it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_y = zero_non_finite(asset.data.projected_gravity_b.torch[:, 1])
    return torch.clamp(direction * gravity_y - threshold, min=0.0) ** 2


def joint_vel_gaussian(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    gate_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Stillness: Gaussian of the mean squared joint speed [rad/s], optionally only while the stance foot is down."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = zero_non_finite(asset.data.joint_vel.torch[:, asset_cfg.joint_ids])
    reward = torch.exp(-(joint_vel**2).mean(dim=-1) / std**2)
    if gate_sensor_cfg is not None:
        reward = reward * foot_in_contact(env, gate_sensor_cfg)
    return reward


def pose_target_match(
    env: ManagerBasedRLEnv, target: Sequence[float], std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Mean over joints ``asset_cfg`` of per-joint Gaussians around the joint positions ``target`` [rad]."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = zero_non_finite(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    target_t = torch.tensor(target, device=joint_pos.device, dtype=joint_pos.dtype)
    return torch.exp(-(((joint_pos - target_t) / std) ** 2)).mean(dim=-1)


def joint_limit_proximity(env: ManagerBasedRLEnv, margin: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Summed depth [rad] of joints ``asset_cfg`` inside a ``margin`` band next to their hard limits.

    Unlike the soft-limit penalty, this bites before a low-gain servo parks a joint on its stop.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = zero_non_finite(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    limits = asset.data.joint_pos_limits.torch[:, asset_cfg.joint_ids]
    below = torch.clamp(limits[..., 0] + margin - joint_pos, min=0.0)
    above = torch.clamp(joint_pos - (limits[..., 1] - margin), min=0.0)
    return (below + above).sum(dim=-1)


class joint_torque_rate_l2(ManagerTermBase):
    """Sum of squared changes of the servo torques [N.m] since the previous evaluation.

    As in microduck_rl, the first evaluation returns zero and the previous torque is kept across resets. The reward
    manager skips zero-weight terms, so the history starts when a curriculum first gives the term a weight.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_torque: torch.Tensor | None = None

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        torque = zero_non_finite(asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]).clone()
        if self._prev_torque is None:
            self._prev_torque = torque
            return torch.zeros(env.num_envs, device=env.device)
        rate = torque - self._prev_torque
        self._prev_torque = torque
        return (rate**2).sum(dim=-1)


"""
Events.
"""


def reset_flamingo_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    joint_pose: Sequence[float],
    base_roll: float,
    base_pitch: float,
    z_range: tuple[float, float],
    xy_range: float = 0.0,
    tilt_noise: float = 0.0,
    joint_noise_std: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Place environments at rest in the one-foot pose, as microduck_rl's ``set_flamingo_state``.

    The trunk gets roll ``base_roll`` and pitch ``base_pitch`` [rad] each with uniform noise of ``tilt_noise`` [rad],
    a uniform yaw, a height drawn from ``z_range`` [m] and a horizontal offset within ``xy_range`` [m] of the env
    origin. Joints ``asset_cfg`` get ``joint_pose`` [rad] plus Gaussian noise of ``joint_noise_std`` [rad], not clipped
    to the joint ranges.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if env_ids is not None else slice(None)]
    n = len(env_ids)
    if n == 0:
        return

    def uniform(low: float, high: float) -> torch.Tensor:
        return torch.empty(n, device=env.device).uniform_(low, high)

    roll = base_roll + uniform(-tilt_noise, tilt_noise)
    pitch = base_pitch + uniform(-tilt_noise, tilt_noise)
    yaw = uniform(-torch.pi, torch.pi)
    root = torch.zeros(n, 7, device=env.device)
    root[:, :3] = env.scene.env_origins[env_ids]
    root[:, 0] += uniform(-xy_range, xy_range)
    root[:, 1] += uniform(-xy_range, xy_range)
    root[:, 2] += uniform(*z_range)
    root[:, 3:] = quat_from_euler_xyz(roll, pitch, yaw)
    asset.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=env_ids)

    joint_pos = asset.data.default_joint_pos.torch[env_ids].clone()
    pose = torch.tensor(joint_pose, device=env.device, dtype=joint_pos.dtype).expand(n, -1)
    joint_pos[:, asset_cfg.joint_ids] = pose + joint_noise_std * torch.randn_like(pose)
    asset.write_joint_state_to_sim_index(position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
