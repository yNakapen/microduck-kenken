# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static checks of microduck_rl's flamingo pose on the Newton MJWarp MicroDuck.

:func:`make_check_env` builds the all-collisions robot without gearbox play, randomization or pushes.
:func:`solve_flat_sole_placement` finds the trunk roll, pitch and height that put a stance sole flat on the floor
for a given joint pose, :func:`place_robot` teleports the robot there, and :func:`measure_pose` reports the quantities
microduck_rl's design validated the pose with. Support hulls come from the sole collision meshes of the robot USD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_CFG, MICRODUCK_JOINT_NAMES

from . import mdp
from .hop_metrics import set_supply_voltage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CHECK_TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
"""Registered task whose scene, servos and termination terms the checks reuse."""

SOLE_FACE_TOLERANCE = 0.002
"""Height band [m] above a sole's lowest vertex that counts as its contact face."""

SOLE_UP_AXIS_IN_ANKLE = (0.0, 1.0, 0.0)
"""Foot-site z axis expressed in the ankle frame, the same for both feet in the MJCF and USD."""

_RANDOMIZATION_EVENTS = (
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "mass_inertia",
    "randomize_com",
    "randomize_head_com",
    "randomize_joint_friction",
    "randomize_armature",
)
_RANDOMIZATION_CURRICULA = ("com_range", "head_com_range")


def make_check_env(device: str = "cuda:0", num_envs: int = 1, vin: float = 7.4) -> ManagerBasedRLEnv:
    """Build the all-collisions robot without backlash, randomization or pushes, at a fixed supply ``vin`` [V]."""
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = parse_env_cfg(CHECK_TASK, device=device, num_envs=num_envs)
    cfg.scene.robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # microduck_rl on mjlab uses MuJoCo Warp's own collision detection instead of Newton's pipeline.
    cfg.sim.physics.solver_cfg.use_mujoco_contacts = True
    cfg.sim.physics.collision_cfg = None
    for name in _RANDOMIZATION_EVENTS:
        setattr(cfg.events, name, None)
    for name in _RANDOMIZATION_CURRICULA:
        setattr(cfg.curriculum, name, None)
    sim_utils.create_new_stage()
    env = gym.make(CHECK_TASK, cfg=cfg).unwrapped
    env.reset()
    set_supply_voltage(env, torch.full((env.num_envs,), vin, device=env.device), sag_gain=0.0)
    return env


def servo_joint_ids(env: ManagerBasedRLEnv) -> list[int]:
    """Articulation indices of the 14 servos in the robot's servo order."""
    return env.scene["robot"].find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)[0]


def place_robot(
    env: ManagerBasedRLEnv, pose: list[float], quat: torch.Tensor, z: float, env_ids: torch.Tensor | None = None
) -> None:
    """Teleport environments to joint ``pose`` [rad] (servo order) with root orientation ``quat`` (x, y, z, w) and
    root height ``z`` [m] above the environment origin, at rest."""
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    root = torch.zeros(n, 7, device=env.device)
    root[:, :3] = env.scene.env_origins[env_ids]
    root[:, 2] += z
    root[:, 3:] = quat.to(env.device).expand(n, 4)
    robot.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=env_ids)
    joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    joint_pos[:, servo_joint_ids(env)] = torch.tensor(pose, device=env.device)
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=torch.zeros_like(joint_pos), env_ids=env_ids)


@cache
def _sole_vertices_in_ankle(usd_path: str, foot: str) -> tuple[tuple[float, float, float], ...]:
    """Vertices [m] of the enabled collision mesh under ankle body ``foot``, in the ankle frame."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(retrieve_file_path(usd_path))
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    ankle = next(p for p in prims if p.GetName() == foot and not p.IsA(UsdGeom.Mesh))
    cache_ = UsdGeom.XformCache()
    vertices = []
    for prim in Usd.PrimRange(ankle, Usd.TraverseInstanceProxies()):
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        to_ankle, _ = cache_.ComputeRelativeTransform(prim, ankle)
        vertices += [tuple(to_ankle.Transform(p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
    if not vertices:
        raise ValueError(f"No enabled collision mesh under '{foot}' in {usd_path}.")
    return tuple(vertices)


def _sole_vertices(env: ManagerBasedRLEnv, foot: str) -> torch.Tensor:
    return torch.tensor(_sole_vertices_in_ankle(env.cfg.scene.robot.spawn.usd_path, foot), device=env.device)


def sole_normal_in_ankle(env: ManagerBasedRLEnv, foot: str) -> tuple[torch.Tensor, float]:
    """Outward (downward) normal of a sole's contact face in the ankle frame, and the face's RMS flatness [m].

    The face is the band of vertices furthest along the foot site's downward axis; the normal is its plane fit.
    """
    vertices = _sole_vertices(env, foot)
    down = -torch.tensor(SOLE_UP_AXIS_IN_ANKLE, device=env.device)
    depth = vertices @ down
    face = vertices[depth >= depth.max() - SOLE_FACE_TOLERANCE]
    centered = face - face.mean(dim=0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal if normal @ down > 0 else -normal
    return normal, (centered @ normal).square().mean().sqrt().item()


def sole_vertices_w(env: ManagerBasedRLEnv, foot: str, env_id: int = 0) -> torch.Tensor:
    """Sole collision vertices [m] of one environment in the world frame, shape ``(V, 3)``."""
    robot = env.scene["robot"]
    body = robot.find_bodies(foot)[0][0]
    pos = robot.data.body_link_pos_w.torch[env_id, body]
    quat = robot.data.body_link_quat_w.torch[env_id, body]
    vertices = _sole_vertices(env, foot)
    return pos + quat_apply(quat.expand(len(vertices), 4), vertices)


def sole_support_hull(env: ManagerBasedRLEnv, foot: str, env_id: int = 0) -> torch.Tensor:
    """World vertices [m] within :data:`SOLE_FACE_TOLERANCE` of a sole's lowest point, shape ``(K, 3)``."""
    vertices = sole_vertices_w(env, foot, env_id)
    return vertices[vertices[:, 2] <= vertices[:, 2].min() + SOLE_FACE_TOLERANCE]


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Counter-clockwise convex hull (Andrew's monotone chain), without collinear points."""
    points = sorted(set(points))
    if len(points) < 3:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_signed_distance(point: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Distance [m] from ``point`` (2,) to the convex hull of ``points`` (K, 2): positive inside, negative outside."""
    hull = torch.tensor(_convex_hull([tuple(p) for p in points.tolist()]), dtype=point.dtype, device=point.device)
    start, end = hull, torch.roll(hull, -1, dims=0)
    edge = end - start
    t = (((point - start) * edge).sum(dim=1) / edge.square().sum(dim=1)).clamp(0.0, 1.0)
    distance = (start + t[:, None] * edge - point).norm(dim=1).min()
    rel = point - start
    inside = (edge[:, 0] * rel[:, 1] - edge[:, 1] * rel[:, 0] >= 0.0).all()
    return distance if inside else -distance


@dataclass
class Placement:
    """Root orientation (x, y, z, w) and height [m] that put a stance sole flat on the floor."""

    quat: torch.Tensor
    z: float
    roll_deg: float
    pitch_deg: float


def solve_flat_sole_placement(env: ManagerBasedRLEnv, pose: list[float], stance: str) -> Placement:
    """Trunk roll, pitch and height that put the ``stance`` ("right" or "left") sole flat on the floor in ``pose``.

    Yaw is zero. Leaves environment 0 placed accordingly.
    """
    robot = env.scene["robot"]
    foot = f"ankle_{stance}"
    body = robot.find_bodies(foot)[0][0]
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device)
    lift = 0.5
    place_robot(env, pose, identity, lift)
    normal, _ = sole_normal_in_ankle(env, foot)
    # Sole normal in the trunk frame; the trunk must turn it straight down.
    v = quat_apply(robot.data.body_link_quat_w.torch[0:1, body], normal[None])[0]
    down = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    axis = torch.linalg.cross(v, down)
    angle = torch.atan2(axis.norm(), v @ down)
    axis = axis / axis.norm().clamp(min=1e-12)
    half = 0.5 * angle
    quat = torch.cat([axis * torch.sin(half), torch.cos(half)[None]])[None]
    roll, pitch, _ = euler_xyz_from_quat(quat)
    quat = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
    place_robot(env, pose, quat, lift)
    lowest = (sole_vertices_w(env, foot)[:, 2] - env.scene.env_origins[0, 2]).min().item()
    z = lift - lowest
    place_robot(env, pose, quat, z)
    return Placement(quat=quat, z=z, roll_deg=math.degrees(roll.item()), pitch_deg=math.degrees(pitch.item()))


@dataclass
class PoseMeasurement:
    """Geometry of environment 0 standing on one foot, as validated in microduck_rl."""

    roll_deg: float
    pitch_deg: float
    z: float
    gravity_b: list[float]
    com_to_stance_site: float
    com_hull_margin: float
    lifted_site_z: float
    lifted_sole_min_z: float
    stance_sole_min_z: float
    stance_sole_tilt_deg: float
    min_joint_limit_margin: float
    min_joint_limit_joint: str


def measure_pose(env: ManagerBasedRLEnv, pose: list[float], stance: str) -> PoseMeasurement:
    """Measure environment 0 as currently placed, for joint ``pose`` standing on the ``stance`` foot."""
    robot = env.scene["robot"]
    lifted = "left" if stance == "right" else "right"
    stance_foot, lifted_foot = f"ankle_{stance}", f"ankle_{lifted}"
    origin_z = env.scene.env_origins[0, 2].item()
    bodies = robot.find_bodies([stance_foot, lifted_foot], preserve_order=True)[0]
    sites = mdp.foot_site_pos_w(robot, bodies)[0]
    com_xy = mdp.robot_com_pos_w(robot)[0, :2]
    hull = sole_support_hull(env, stance_foot)

    normal, _ = sole_normal_in_ankle(env, stance_foot)
    normal_w = quat_apply(robot.data.body_link_quat_w.torch[0:1, bodies[0]], normal[None])[0]
    tilt = torch.acos((-normal_w[2]).clamp(-1.0, 1.0))

    joint_ids = servo_joint_ids(env)
    limits = robot.data.joint_pos_limits.torch[0, joint_ids]
    q = torch.tensor(pose, device=env.device)
    margins = torch.minimum(q - limits[:, 0], limits[:, 1] - q)
    worst = int(margins.argmin())

    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w.torch[0:1])
    return PoseMeasurement(
        roll_deg=math.degrees(roll.item()),
        pitch_deg=math.degrees(pitch.item()),
        z=robot.data.root_link_pos_w.torch[0, 2].item() - origin_z,
        gravity_b=robot.data.projected_gravity_b.torch[0].tolist(),
        com_to_stance_site=(com_xy - sites[0, :2]).norm().item(),
        com_hull_margin=polygon_signed_distance(com_xy, hull[:, :2]).item(),
        lifted_site_z=sites[1, 2].item() - origin_z,
        lifted_sole_min_z=sole_vertices_w(env, lifted_foot)[:, 2].min().item() - origin_z,
        stance_sole_min_z=sole_vertices_w(env, stance_foot)[:, 2].min().item() - origin_z,
        stance_sole_tilt_deg=math.degrees(tilt.item()),
        min_joint_limit_margin=margins[worst].item(),
        min_joint_limit_joint=MICRODUCK_JOINT_NAMES[worst],
    )
