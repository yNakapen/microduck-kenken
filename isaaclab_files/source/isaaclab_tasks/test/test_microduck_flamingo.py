# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flamingo reproduction: the one-foot pose, whole-robot CoM, support hulls, and pose placement."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flamingo_checks import (
    make_check_env,
    measure_pose,
    place_robot,
    polygon_signed_distance,
    sole_support_hull,
    solve_flat_sole_placement,
)
from isaaclab_tasks.contrib.microduck.flamingo_pose import FLAMINGO_POSE_LEFT, FLAMINGO_POSE_RIGHT, mirror_pose

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES


def test_mirrored_pose_swaps_the_legs_and_flips_head_yaw_and_roll():
    assert mirror_pose(mirror_pose(FLAMINGO_POSE_RIGHT)) == pytest.approx(FLAMINGO_POSE_RIGHT)
    # microduck_flamingo_cycle_env_cfg.mirror_pose applied to FLAMINGO_POSE by hand.
    expected = [
        *[-0.386, 0.334, -0.258, -0.005, 0.253],
        *[0.349, 0.350, 1.500, 0.000],
        *[0.000, -0.300, -1.200, 0.800, -0.800],
    ]
    assert pytest.approx(expected) == FLAMINGO_POSE_LEFT


def test_polygon_signed_distance_is_positive_inside_and_negative_outside():
    square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    points = torch.tensor([[0.5, 0.5], [0.9, 0.5], [1.5, 0.5], [0.5, 0.0], [2.0, 2.0]])
    expected = torch.tensor([0.5, 0.1, -0.5, 0.0, -math.sqrt(2.0)])
    distance = torch.stack([polygon_signed_distance(p, square) for p in points])
    torch.testing.assert_close(distance, expected)


def test_robot_com_is_the_mass_weighted_mean_of_body_coms():
    mass = torch.tensor([[1.0, 3.0]])
    com = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, 2.0]]])
    asset = SimpleNamespace(
        data=SimpleNamespace(body_mass=SimpleNamespace(torch=mass), body_com_pos_w=SimpleNamespace(torch=com))
    )
    torch.testing.assert_close(mdp.robot_com_pos_w(asset), torch.tensor([[3.0, 0.0, 1.5]]))


@pytest.fixture(scope="module")
def check_env():
    device = test_devices(DeviceScope.CUDA)[0]
    env = make_check_env(device=device)
    yield env
    env.close()
    SimulationContext.clear_instance()


@pytest.mark.integration
@pytest.mark.kitless
def test_check_env_uses_the_all_collisions_robot_without_backlash(check_env):
    robot = check_env.scene["robot"]
    assert check_env.cfg.scene.robot.spawn.usd_path.endswith("microduck_allcollisions.usd")
    assert not any(name.startswith("passive_") for name in robot.joint_names)
    assert check_env.cfg.events.push_robot is None
    # microduck_rl on mjlab runs MuJoCo Warp's own collision detection; so does the check.
    assert check_env.cfg.sim.physics.solver_cfg.use_mujoco_contacts
    assert check_env.cfg.sim.physics.collision_cfg is None


@pytest.mark.integration
@pytest.mark.kitless
def test_placement_writes_the_requested_root_pose_and_joints(check_env):
    robot = check_env.scene["robot"]
    check_env.reset()
    quat = torch.tensor([[0.0, 0.0, math.sin(0.2), math.cos(0.2)]], device=check_env.device)
    place_robot(check_env, FLAMINGO_POSE_RIGHT, quat, 0.3)
    joint_ids = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)[0]
    torch.testing.assert_close(
        robot.data.joint_pos.torch[0, joint_ids], torch.tensor(FLAMINGO_POSE_RIGHT, device=check_env.device)
    )
    torch.testing.assert_close(robot.data.root_link_quat_w.torch, quat)
    assert robot.data.root_link_pos_w.torch[0, 2].item() == pytest.approx(0.3, abs=1e-6)
    # Forward kinematics follow the write: the lifted left foot is well above the right one.
    feet = robot.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)[0]
    left_z, right_z = robot.data.body_link_pos_w.torch[0, feet, 2].tolist()
    assert left_z - right_z > 0.03


@pytest.mark.integration
@pytest.mark.kitless
def test_home_stance_is_level_with_the_com_inside_both_soles(check_env):
    """Sign and frame sanity of the measurements on the default two-foot stance."""
    check_env.reset()
    robot = check_env.scene["robot"]
    home = robot.data.default_joint_pos.torch[0, robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)[0]]
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=check_env.device)
    place_robot(check_env, home.tolist(), identity, 0.2)
    gravity = robot.data.projected_gravity_b.torch[0]
    torch.testing.assert_close(gravity, torch.tensor([0.0, 0.0, -1.0], device=check_env.device), atol=0.1, rtol=0)
    hull = torch.cat([sole_support_hull(check_env, foot) for foot in ("ankle_left", "ankle_right")])
    com_xy = mdp.robot_com_pos_w(robot)[0, :2]
    assert polygon_signed_distance(com_xy, hull[:, :2]) > 0.0


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("stance", ["right", "left"])
def test_flat_sole_placement_puts_the_stance_sole_on_the_ground(check_env, stance):
    pose = FLAMINGO_POSE_RIGHT if stance == "right" else FLAMINGO_POSE_LEFT
    check_env.reset()
    placement = solve_flat_sole_placement(check_env, pose, stance)
    place_robot(check_env, pose, placement.quat, placement.z)
    measured = measure_pose(check_env, pose, stance)
    assert measured.stance_sole_min_z == pytest.approx(0.0, abs=1e-4)
    assert measured.stance_sole_tilt_deg < 0.5
    assert measured.lifted_sole_min_z > 0.03
    # The pose stays 0.05 rad off every hard limit (microduck_rl test plan). microduck_rl checks against ranges
    # rounded to 0.384 rad; the 22 deg hip-roll limit leaves 0.04997 rad, so the comparison allows 1e-4 rad of rounding.
    assert measured.min_joint_limit_margin >= 0.05 - 1e-4
    # Leaning over the stance foot rolls the trunk toward it.
    assert math.copysign(1.0, measured.roll_deg) == (1.0 if stance == "right" else -1.0)
