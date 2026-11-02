# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command, reset and rewards of the flamingo cycle: stand, lift one foot, hold, lower, on either side.

Ports ``FlamingoCommand``, ``set_flamingo_cycle_state`` and the ``fl_*`` rewards of microduck_rl
(``src/mjlab_microduck/tasks/mdp.py`` on the ``flamingo`` branch, ``4757d6ae``). The command occupies the twist slots
as ``[flag, side, 0]``: ``flag`` 1 asks for one foot, 0 for two; ``side`` +1 stands on the right foot, -1 on the
left. An internal blend ``alpha`` slews toward the flag and drives every shaping target, so the policy only sees the
raw command. Penalties are non-negative costs for negative weights, as in the rest of the MicroDuck task.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from .flamingo import MICRODUCK_FOOT_SITE_OFFSET, foot_site_pos_w, robot_com_pos_w
from .observations import zero_non_finite

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _ids(env_ids: Sequence[int] | slice | torch.Tensor | None, num_envs: int, device) -> torch.Tensor:
    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)[env_ids]
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


class FlamingoCommand(CommandTerm):
    """``[flag, side, 0]`` resampled every dwell time, with a slewed blend ``alpha`` toward the flag.

    The flag is 1 with probability ``flamingo_prob``. The stance side is redrawn only at an episode start or after
    a two-foot command, so a hold never switches feet; a side requested by the reset event wins. The rewards read the
    latched ±1 :attr:`stance_side`, while the observed side is zeroed with probability ``zero_side_prob`` during
    two-foot commands so that the deployed idle command ``[0, 0, 0]`` is trained.
    """

    cfg: FlamingoCommandCfg

    def __init__(self, cfg: FlamingoCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._alpha = torch.zeros(self.num_envs, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_side = torch.zeros(self.num_envs, device=self.device)
        self._stance_side = torch.ones(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """``[flag, observed side, 0]``, shape ``(num_envs, 3)``."""
        return self._command

    @property
    def alpha(self) -> torch.Tensor:
        """Blend from the two-foot (0) to the one-foot (1) targets, shape ``(num_envs,)``."""
        return self._alpha

    @property
    def flag(self) -> torch.Tensor:
        """Commanded posture: 1 one foot, 0 two feet, shape ``(num_envs,)``."""
        return self._command[:, 0]

    @property
    def stance_side(self) -> torch.Tensor:
        """Latched stance side for the rewards, +1 right or -1 left, shape ``(num_envs,)``."""
        return self._stance_side

    def set_alpha(self, env_ids: torch.Tensor, value: torch.Tensor) -> None:
        """Start the blend of environments ``env_ids`` at ``value``, as the reset pose demands."""
        self._alpha[env_ids] = value

    def request_side(self, env_ids: torch.Tensor, side: torch.Tensor) -> None:
        """Pin the stance side of environments ``env_ids`` for the resample of the coming command reset."""
        self._pending_side[env_ids] = side
        self._stance_side[env_ids] = side
        self._command[env_ids, 1] = side

    def set_command(self, command: torch.Tensor, stance_side: torch.Tensor) -> None:
        """Script all environments: write ``command`` ``(num_envs, 3)`` and the latched ``stance_side`` (±1), and stop
        the random resampling. The blend keeps slewing toward the written flag."""
        self._command[:] = command
        self._stance_side[:] = stance_side
        self.time_left[:] = torch.inf

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
        self._fresh[_ids(env_ids, self.num_envs, self.device)] = True
        return super().reset(env_ids)

    def compute(self, dt: float):
        super().compute(dt)
        step = dt / max(self.cfg.ramp_s, 1e-6)
        self._alpha += torch.clamp(self._command[:, 0] - self._alpha, -step, step)

    def _resample_command(self, env_ids: Sequence[int] | slice):
        ids = _ids(env_ids, self.num_envs, self.device)
        n = len(ids)
        if n == 0:
            return
        prev_flag = self._command[ids, 0]
        new_flag = (torch.rand(n, device=self.device) < self.cfg.flamingo_prob).float()
        new_side = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        redraw = self._fresh[ids] | (prev_flag < 0.5)
        side = torch.where(redraw, new_side, self._stance_side[ids])
        pending = self._pending_side[ids]
        side = torch.where(pending.abs() > 0.5, pending, side)
        self._pending_side[ids] = 0.0
        self._stance_side[ids] = side
        zero = (new_flag < 0.5) & (torch.rand(n, device=self.device) < self.cfg.zero_side_prob)
        self._command[ids] = 0.0
        self._command[ids, 0] = new_flag
        self._command[ids, 1] = torch.where(zero, torch.zeros_like(side), side)
        self._fresh[ids] = False

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class FlamingoCommandCfg(CommandTermCfg):
    """Configuration of :class:`FlamingoCommand`."""

    class_type: type = FlamingoCommand
    flamingo_prob: float = 0.6
    """Probability that a resample commands one foot."""
    ramp_s: float = 1.5
    """Time [s] for ``alpha`` to cross from 0 to 1."""
    zero_side_prob: float = 0.5
    """Probability that the observed side is 0 during a two-foot command."""


"""
Reset.
"""


def reset_flamingo_cycle_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    pose_right: Sequence[float],
    pose_left: Sequence[float],
    base_roll: float,
    base_pitch: float,
    z_range: tuple[float, float],
    standing_z_range: tuple[float, float],
    command_name: str,
    in_pose_prob: float,
    xy_range: float = 0.0,
    tilt_noise: float = 0.0,
    joint_noise_std: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Place environments in the one-foot pose of a random side with ``in_pose_prob``, else standing at HOME.

    As microduck_rl's ``set_flamingo_cycle_state``: the side is pinned into command ``command_name`` and its blend
    starts at 1 in the pose or 0 standing. In the pose the trunk rolls ``side * base_roll`` and pitches ``base_pitch``
    [rad] with uniform ``tilt_noise`` [rad] at a height from ``z_range`` [m]; standing it is level at a height from
    ``standing_z_range`` [m]. Joints ``asset_cfg`` get Gaussian noise of ``joint_noise_std`` [rad] in both buckets.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _ids(env_ids, env.num_envs, env.device)
    n = len(ids)
    if n == 0:
        return
    term: FlamingoCommand = env.command_manager.get_term(command_name)

    def uniform(low: float, high: float) -> torch.Tensor:
        return torch.empty(n, device=env.device).uniform_(low, high)

    side = torch.where(torch.rand(n, device=env.device) < 0.5, -1.0, 1.0)
    term.request_side(ids, side)
    in_pose = torch.rand(n, device=env.device) < in_pose_prob

    roll = torch.where(in_pose, side * base_roll + uniform(-tilt_noise, tilt_noise), torch.zeros(n, device=env.device))
    pitch = torch.where(in_pose, base_pitch + uniform(-tilt_noise, tilt_noise), torch.zeros(n, device=env.device))
    yaw = uniform(-torch.pi, torch.pi)
    root = torch.zeros(n, 7, device=env.device)
    root[:, :3] = env.scene.env_origins[ids]
    root[:, 0] += uniform(-xy_range, xy_range)
    root[:, 1] += uniform(-xy_range, xy_range)
    root[:, 2] += torch.where(in_pose, uniform(*z_range), uniform(*standing_z_range))
    root[:, 3:] = quat_from_euler_xyz(roll, pitch, yaw)
    asset.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=ids)

    joint_pos = asset.data.default_joint_pos.torch[ids].clone()
    home = joint_pos[:, asset_cfg.joint_ids]
    right = torch.tensor(pose_right, device=env.device, dtype=joint_pos.dtype).expand(n, -1)
    left = torch.tensor(pose_left, device=env.device, dtype=joint_pos.dtype).expand(n, -1)
    pose = torch.where(in_pose.unsqueeze(1), torch.where((side > 0).unsqueeze(1), right, left), home)
    joint_pos[:, asset_cfg.joint_ids] = pose + joint_noise_std * torch.randn_like(pose)
    asset.write_joint_state_to_sim_index(position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=ids)
    term.set_alpha(ids, in_pose.float())


"""
Rewards. ``feet_sensor_cfg`` and ``feet_cfg`` select the ankles in the order [left, right].
"""


def _smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = torch.clamp((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _context(env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    term: FlamingoCommand = env.command_manager.get_term(command_name)
    return term.flag, term.stance_side, term.alpha


def _stance_swing_contact(
    env: ManagerBasedRLEnv, feet_sensor_cfg: SceneEntityCfg, side: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
    contact = (zero_non_finite(sensor.data.current_contact_time.torch[:, feet_sensor_cfg.body_ids]) > 0.0).float()
    left, right = contact[:, 0], contact[:, 1]
    right_stance = side > 0
    return torch.where(right_stance, right, left), torch.where(right_stance, left, right)


def _feet_sites(
    env: ManagerBasedRLEnv, feet_cfg: SceneEntityCfg, site_offset: tuple[float, float, float]
) -> tuple[Articulation, torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[feet_cfg.name]
    sites = zero_non_finite(foot_site_pos_w(asset, feet_cfg.body_ids, site_offset))
    return asset, sites[:, 0], sites[:, 1]


def cycle_com_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    feet_cfg: SceneEntityCfg,
    std: float,
    site_offset: tuple[float, float, float] = MICRODUCK_FOOT_SITE_OFFSET,
) -> torch.Tensor:
    """Gaussian of the whole-robot CoM against ``(1 - alpha)`` times the feet midpoint plus ``alpha`` times the stance
    foot, horizontally: the weight shift leads the transition."""
    _, side, alpha = _context(env, command_name)
    asset, left, right = _feet_sites(env, feet_cfg, site_offset)
    stance = torch.where((side > 0).unsqueeze(1), right, left)
    a = alpha.unsqueeze(1)
    target = (1.0 - a) * 0.5 * (left + right)[:, :2] + a * stance[:, :2]
    dist2 = torch.nan_to_num(((robot_com_pos_w(asset)[:, :2] - target) ** 2).sum(dim=-1), nan=1.0)
    return torch.exp(-dist2 / std**2)


def cycle_stance_foot_grounded(
    env: ManagerBasedRLEnv, command_name: str, feet_sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """1 while the stance foot touches the ground, in both postures."""
    _, side, _ = _context(env, command_name)
    return _stance_swing_contact(env, feet_sensor_cfg, side)[0]


def cycle_swing_foot_contact(
    env: ManagerBasedRLEnv, command_name: str, feet_sensor_cfg: SceneEntityCfg, lo: float, hi: float
) -> torch.Tensor:
    """Swing-foot contact: +1 while standing is due (``alpha <= lo``), -1 once the lift is due (``alpha >= hi``)."""
    _, side, alpha = _context(env, command_name)
    swing = _stance_swing_contact(env, feet_sensor_cfg, side)[1]
    return swing * (1.0 - 2.0 * _smoothstep(alpha, lo, hi))


def cycle_swing_foot_clear(
    env: ManagerBasedRLEnv,
    command_name: str,
    feet_cfg: SceneEntityCfg,
    feet_sensor_cfg: SceneEntityCfg,
    target: float,
    std: float,
    lo: float,
    hi: float,
    site_offset: tuple[float, float, float] = MICRODUCK_FOOT_SITE_OFFSET,
) -> torch.Tensor:
    """Gaussian of the swing-foot site height [m] above the env origin around ``target``, while the stance foot is
    down and as the lift becomes due."""
    _, side, alpha = _context(env, command_name)
    _, left, right = _feet_sites(env, feet_cfg, site_offset)
    swing = torch.where((side > 0).unsqueeze(1), left, right)
    z = swing[:, 2] - env.scene.env_origins[:, 2]
    stance = _stance_swing_contact(env, feet_sensor_cfg, side)[0]
    return torch.exp(-(((z - target) / std) ** 2)) * stance * _smoothstep(alpha, lo, hi)


def cycle_pose_track(
    env: ManagerBasedRLEnv,
    command_name: str,
    pose_right: Sequence[float],
    pose_left: Sequence[float],
    std: float,
    mid_attenuation: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mean per-joint Gaussian against ``(1 - alpha)`` HOME plus ``alpha`` times the side's one-foot pose, faded by
    ``mid_attenuation`` mid-transition where the CoM term leads."""
    _, side, alpha = _context(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    home = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    right = torch.tensor(pose_right, device=home.device, dtype=home.dtype)
    left = torch.tensor(pose_left, device=home.device, dtype=home.dtype)
    a = alpha.unsqueeze(1)
    target = (1.0 - a) * home + a * torch.where((side > 0).unsqueeze(1), right, left)
    joint_pos = zero_non_finite(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    match = torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)
    return match * (1.0 - mid_attenuation * 4.0 * alpha * (1.0 - alpha))


def cycle_gravity_track(
    env: ManagerBasedRLEnv,
    command_name: str,
    gravity_right: tuple[float, float, float],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian of the trunk-frame gravity against the normalized blend of upright and the side's lean; the left lean
    mirrors ``gravity_right`` in y."""
    _, side, alpha = _context(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    gravity = zero_non_finite(asset.data.projected_gravity_b.torch)
    lean = torch.tensor(gravity_right, device=gravity.device, dtype=gravity.dtype).expand_as(gravity).clone()
    lean[:, 1] *= side
    up = torch.tensor([0.0, 0.0, -1.0], device=gravity.device, dtype=gravity.dtype)
    a = alpha.unsqueeze(1)
    target = (1.0 - a) * up + a * lean
    target = target / target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.exp(-((gravity - target) ** 2).sum(dim=-1) / std**2)


def cycle_stillness(
    env: ManagerBasedRLEnv,
    command_name: str,
    feet_sensor_cfg: SceneEntityCfg,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Gaussian of the mean squared joint speed, paid once the blend has reached the flag and the stance foot is
    down: quiet rests in both postures, nothing mid-transition."""
    flag, side, alpha = _context(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = zero_non_finite(asset.data.joint_vel.torch[:, asset_cfg.joint_ids])
    reward = torch.exp(-(joint_vel**2).mean(dim=-1) / std**2)
    ramp_done = ((flag - alpha).abs() < 0.02).float()
    return reward * ramp_done * _stance_swing_contact(env, feet_sensor_cfg, side)[0]


def cycle_stance_side_tilt_cost(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Squared roll past ``threshold`` (in ``|g_b,y|``) toward the stance side, mirrored by the side."""
    _, side, _ = _context(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_y = zero_non_finite(asset.data.projected_gravity_b.torch[:, 1])
    return torch.clamp(-side * gravity_y - threshold, min=0.0) ** 2


def cycle_single_support_success(
    env: ManagerBasedRLEnv, command_name: str, feet_sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Logged indicator: 1 when the commanded posture holds, one foot down and the other up or both down."""
    flag, side, _ = _context(env, command_name)
    stance, swing = _stance_swing_contact(env, feet_sensor_cfg, side)
    return torch.where(flag > 0.5, stance * (1.0 - swing), stance * swing)
