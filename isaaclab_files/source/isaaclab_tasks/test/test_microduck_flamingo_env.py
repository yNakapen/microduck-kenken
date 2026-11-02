# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flamingo task, microduck_rl's stage 1 (``Mjlab-Flamingo-Flat-MicroDuck``) on Newton MJWarp.

Reward values are compared with the microduck_rl formulas (``src/mjlab_microduck/tasks/mdp.py`` on the ``flamingo``
branch, ``4757d6ae``); the environment tests check the settings copied from ``microduck_flamingo_env_cfg.py``.
"""

import math
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flamingo_pose import (
    FLAMINGO_BASE_PITCH,
    FLAMINGO_BASE_ROLL,
    FLAMINGO_GRAVITY_B,
    FLAMINGO_POSE_RIGHT,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

FLAMINGO_TASK = "IsaacContrib-Flamingo-Flat-MicroDuck"

_ROBOT_CFG = SceneEntityCfg("robot", joint_ids=[0, 1], body_ids=[0])


def _fake_env(**data):
    """An environment whose robot exposes ``data`` fields as ``.torch`` arrays."""
    robot = SimpleNamespace(data=SimpleNamespace(**{k: SimpleNamespace(torch=v) for k, v in data.items()}))
    return SimpleNamespace(scene={"robot": robot})


def test_projected_gravity_match_peaks_at_the_target_lean():
    target = torch.tensor(FLAMINGO_GRAVITY_B)
    gravity = torch.stack([target, torch.tensor([0.0, 0.0, -1.0])])
    env = _fake_env(projected_gravity_b=gravity)
    reward = mdp.projected_gravity_match(env, target=FLAMINGO_GRAVITY_B, std=0.15)
    expected = torch.exp(-((gravity - target) ** 2).sum(-1) / 0.15**2)
    torch.testing.assert_close(reward, expected)
    assert reward[0].item() == pytest.approx(1.0)


def test_lateral_tilt_cost_is_zero_inside_the_threshold_and_quadratic_toward_the_stance_side():
    # g_b,y of the pose, at the threshold, past it toward the stance side, and toward the swing side.
    gravity = torch.tensor([[0.0, -0.38, -0.9], [0.0, -0.45, -0.9], [0.0, -0.65, -0.7], [0.0, 0.65, -0.7]])
    env = _fake_env(projected_gravity_b=gravity)
    cost = mdp.lateral_tilt_cost(env, threshold=0.45, direction=-1.0)
    torch.testing.assert_close(cost, torch.tensor([0.0, 0.0, 0.2**2, 0.0]))


def test_pose_target_match_averages_per_joint_gaussians():
    joint_pos = torch.tensor([[0.3, -0.2], [0.8, -0.2]])
    env = _fake_env(joint_pos=joint_pos)
    reward = mdp.pose_target_match(env, target=[0.3, -0.2], std=0.5, asset_cfg=_ROBOT_CFG)
    torch.testing.assert_close(reward, torch.tensor([1.0, (math.exp(-1.0) + 1.0) / 2.0]))


def test_joint_limit_proximity_ramps_linearly_inside_the_band_of_the_hard_limits():
    limits = torch.tensor([[[-0.4, 0.4], [-1.0, 1.0]]]).expand(3, 2, 2)
    # Interior; 0.05 into the band of joint 0's upper limit; past joint 1's lower stop by 0.1.
    joint_pos = torch.tensor([[0.0, 0.0], [0.35, 0.0], [0.0, -1.1]])
    env = _fake_env(joint_pos=joint_pos, joint_pos_limits=limits)
    cost = mdp.joint_limit_proximity(env, margin=0.1, asset_cfg=_ROBOT_CFG)
    torch.testing.assert_close(cost, torch.tensor([0.0, 0.05, 0.2]))


def test_joint_vel_gaussian_is_gated_on_the_stance_foot():
    joint_vel = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    contact_time = torch.tensor([[0.1], [0.1], [0.0]])
    env = _fake_env(joint_vel=joint_vel)
    sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=SimpleNamespace(torch=contact_time)))
    env.scene = type("Scene", (dict,), {"sensors": {"contact_forces": sensor}})(env.scene)
    gate = SceneEntityCfg("contact_forces", body_ids=[0])
    reward = mdp.joint_vel_gaussian(env, std=2.0, asset_cfg=_ROBOT_CFG, gate_sensor_cfg=gate)
    torch.testing.assert_close(reward, torch.tensor([1.0, math.exp(-0.5), 0.0]))


def test_reset_orientation_matches_microduck_rl_zyx_euler():
    """``set_flamingo_state`` composes yaw, pitch and roll in this order; Isaac Lab's helper must agree."""
    roll, pitch, yaw = torch.tensor([0.39]), torch.tensor([-0.16]), torch.tensor([2.1])
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    wxyz = torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=-1,
    )
    torch.testing.assert_close(quat_from_euler_xyz(roll, pitch, yaw), wxyz[:, [1, 2, 3, 0]])


@pytest.mark.parametrize("play", [False, True])
def test_every_event_schedule_targets_an_enabled_event(play):
    """Play mode drops the pushes; a schedule still pointing at them fails on the first reset."""
    cfg = parse_env_cfg(FLAMINGO_TASK, device="cpu", num_envs=1)
    if play:
        cfg.play_mode()
    for name, term in cfg.curriculum.to_dict().items():
        if term is None:
            continue
        address = term["params"]["address"].split(".")
        if address[0] == "events":
            assert getattr(cfg.events, address[1]) is not None, f"curriculum {name} targets disabled {address[1]}"


"""
Environment.
"""


@pytest.fixture(scope="module")
def flamingo_env():
    device = test_devices(DeviceScope.CUDA)[0]
    cfg = parse_env_cfg(FLAMINGO_TASK, device=device, num_envs=64)
    sim_utils.create_new_stage()
    env = gym.make(FLAMINGO_TASK, cfg=cfg).unwrapped
    env.reset()
    yield env
    env.close()
    SimulationContext.clear_instance()


@pytest.mark.integration
@pytest.mark.kitless
def test_flamingo_keeps_the_deployed_interface(flamingo_env):
    env = flamingo_env
    assert env.observation_manager.group_obs_dim["policy"] == (61,)
    assert env.action_manager.total_action_dim == 14
    # microduck_rl zero-pads the body-pose slots and drops the critic's foot heights.
    torch.testing.assert_close(
        env.command_manager.get_command("body_pose"), torch.zeros(env.num_envs, 6, device=env.device)
    )
    assert "foot_height" not in env.observation_manager.active_terms["critic"]
    assert env.cfg.commands.base_velocity.resampling_time_range == (6.0, 12.0)
    assert env.cfg.commands.base_velocity.rel_standing_envs == 0.0
    assert env.max_episode_length_s == pytest.approx(6.0)


@pytest.mark.integration
@pytest.mark.kitless
def test_flamingo_runs_the_all_collisions_robot_like_microduck_rl(flamingo_env):
    env = flamingo_env
    robot = env.scene["robot"]
    assert env.cfg.scene.robot.spawn.usd_path.endswith("microduck_allcollisions.usd")
    assert not any(name.startswith("passive_") for name in robot.joint_names)
    assert env.cfg.sim.physics.solver_cfg.use_mujoco_contacts
    assert env.cfg.sim.physics.collision_cfg is None
    # mjlab's velocity-env buffers, which microduck_rl's flamingo keeps: the walking task's njmax=96 overflows with
    # whole-body contacts.
    assert env.cfg.sim.physics.solver_cfg.njmax == 1500
    assert env.cfg.sim.physics.solver_cfg.nconmax == 35
    # microduck_rl comments out the BAM current limit; zero disables it.
    assert env.cfg.scene.robot.actuators["servos"].motor.max_current == 0.0
    # Self-collisions are watched over the whole body, as microduck_rl's trunk_base subtree sensor.
    sensor = env.scene["self_collision"]
    assert sorted(sensor.body_names) == sorted(robot.body_names)
    assert sensor.data.force_matrix_w.torch.shape[1:3] == (robot.num_bodies, robot.num_bodies)


@pytest.mark.integration
@pytest.mark.kitless
def test_flamingo_rewards_match_microduck_rl_stage_one(flamingo_env):
    """Weights of microduck_rl's terms, with its self-negating terms flipped to Isaac Lab's cost convention."""
    expected = {
        "com_over_stance_foot": 3.0,
        "stance_foot_grounded": 1.0,
        "swing_foot_clear": 1.5,
        "swing_foot_touch": -0.5,
        "pose_flamingo": 1.5,
        "gravity_flamingo": 2.0,
        "stillness": 1.0,
        "stance_side_tilt": -4.0,
        "joint_limit_proximity": -1.0,
        "action_rate_l2": -0.1,
        "joint_torque_rate_l2": 0.0,
        "body_ang_vel": -0.05,
        "angular_momentum": -0.02,
        "self_collisions": -1.0,
    }
    manager = flamingo_env.reward_manager
    weights = {name: manager.get_term_cfg(name).weight for name in manager.active_terms}
    assert weights == pytest.approx(expected)
    assert manager.get_term_cfg("com_over_stance_foot").params["std"] == 0.03
    # microduck_rl's Cycle fix (c008a7f3): 0.05 +- 0.03 contradicted the pose's 0.17 m foot.
    assert manager.get_term_cfg("swing_foot_clear").params["target"] == 0.10
    assert manager.get_term_cfg("swing_foot_clear").params["std"] == 0.06


@pytest.mark.integration
@pytest.mark.kitless
def test_flamingo_spawns_in_the_noised_one_foot_pose(flamingo_env):
    env = flamingo_env
    env.reset()
    robot = env.scene["robot"]
    joint_ids = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)[0]
    error = robot.data.joint_pos.torch[:, joint_ids] - torch.tensor(FLAMINGO_POSE_RIGHT, device=env.device)
    # Gaussian joint noise of 0.05 rad around the pose.
    assert error.abs().mean().item() == pytest.approx(0.05 * math.sqrt(2.0 / math.pi), rel=0.25)
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w.torch)
    tilt = math.radians(3.0) + 1e-4
    assert ((roll - FLAMINGO_BASE_ROLL).abs() <= tilt).all()
    assert ((pitch - FLAMINGO_BASE_PITCH).abs() <= tilt).all()
    offset = robot.data.root_link_pos_w.torch - env.scene.env_origins
    assert ((offset[:, 2] >= 0.120 - 1e-4) & (offset[:, 2] <= 0.130 + 1e-4)).all()
    assert (offset[:, :2].abs() <= 0.5 + 1e-4).all()
    torch.testing.assert_close(robot.data.root_link_vel_w.torch, torch.zeros(env.num_envs, 6, device=env.device))


@pytest.mark.integration
@pytest.mark.kitless
def test_flamingo_steps_with_finite_rewards_and_lands_on_the_right_foot(flamingo_env):
    env = flamingo_env
    env.reset()
    actions = torch.zeros(env.num_envs, 14, device=env.device)
    for _ in range(10):
        _, reward, *_ = env.step(actions)
        assert torch.isfinite(reward).all()
    stance_cfg = SceneEntityCfg("contact_forces", body_names=["ankle_right"])
    stance_cfg.resolve(env.scene)
    stance = mdp.foot_in_contact(env, stance_cfg)
    assert stance.mean().item() > 0.5


@pytest.mark.integration
@pytest.mark.kitless
def test_flamingo_curriculum_reaches_the_final_microduck_rl_stages(flamingo_env):
    env = flamingo_env
    saved = env.common_step_counter
    try:
        env.common_step_counter = 1500 * 24
        env.curriculum_manager.compute(env_ids=torch.arange(env.num_envs, device=env.device))
        # microduck_rl's Cycle fix (e4a3ca5f): the 0.25 m/s push on top of the -1.0 action-rate tax collapsed stage 1.
        assert env.event_manager.get_term_cfg("push_robot").params["velocity_range"] == {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
        }
        assert env.reward_manager.get_term_cfg("action_rate_l2").weight == -0.5
        assert env.reward_manager.get_term_cfg("joint_torque_rate_l2").weight == -1e-3
        assert env.reward_manager.get_term_cfg("com_over_stance_foot").params["std"] == 0.02
        assert env.event_manager.get_term_cfg("randomize_com").params["com_range"]["x"] == (-0.015, 0.015)
        assert env.event_manager.get_term_cfg("randomize_head_com").params["com_range"]["x"] == (-0.01, 0.01)
    finally:
        env.common_step_counter = saved
