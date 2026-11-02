# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck single-leg tasks: support-foot rewards, reward signs, and the deployed policy interface."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

STAND_TASK = "IsaacContrib-Hop-Stand-Flat-MicroDuck"

POLICY_TERMS = [
    "imu",
    "joint_pos",
    "joint_vel",
    "actions",
    "velocity_commands",
    "head_pose_commands",
    "body_pose_commands",
]
"""Actor input order of the deployed ONNX policy, shared with the walking task."""

# Contact-time columns: [lifted (left), support (right)].
_LIFTED_CFG = SceneEntityCfg("contact_forces", body_ids=[0])
_SUPPORT_CFG = SceneEntityCfg("contact_forces", body_ids=[1])


def _fake_env(contact_time, trunk_z, foot_z, origin_z=0.0):
    """Two envs-worth of contact times [s], trunk heights [m] and lifted-foot heights [m]."""
    contact_time = torch.tensor(contact_time, dtype=torch.float32)
    n = contact_time.shape[0]
    sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=SimpleNamespace(torch=contact_time)))
    root = torch.zeros(n, 3)
    root[:, 2] = torch.tensor(trunk_z) + origin_z
    bodies = torch.zeros(n, 1, 3)
    bodies[:, 0, 2] = torch.tensor(foot_z) + origin_z
    robot = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=SimpleNamespace(torch=root), body_link_pos_w=SimpleNamespace(torch=bodies))
    )

    class Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, origin_z]]).expand(n, 3)
        sensors = {"contact_forces": sensor}

    return SimpleNamespace(scene=Scene(robot=robot))


def test_single_support_requires_only_the_support_foot_and_height():
    """Reward 1 only for: support foot down, lifted foot up, trunk above ``min_height``."""
    env = _fake_env(
        # [lifted, support] contact times; rows: single support, double support, wrong foot, flight, too low
        contact_time=[[0.0, 0.2], [0.1, 0.2], [0.2, 0.0], [0.0, 0.0], [0.0, 0.2]],
        trunk_z=[0.12, 0.12, 0.12, 0.12, 0.05],
        foot_z=[0.05] * 5,
        origin_z=3.0,
    )
    reward = mdp.single_support(env, lifted_sensor_cfg=_LIFTED_CFG, support_sensor_cfg=_SUPPORT_CFG, min_height=0.08)
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_lifted_foot_height_is_gaussian_and_gated_by_support():
    """Peak 1 at the target height above the env origin; zero without the support foot or when fallen."""
    target, std = 0.06, 0.02
    env = _fake_env(
        contact_time=[[0.0, 0.2], [0.0, 0.2], [0.0, 0.0], [0.0, 0.2]],
        trunk_z=[0.12, 0.12, 0.12, 0.05],
        foot_z=[target, target + std, target, target],
        origin_z=-1.0,
    )
    reward = mdp.lifted_foot_height(
        env,
        target_height=target,
        std=std,
        asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
        support_sensor_cfg=_SUPPORT_CFG,
        min_height=0.08,
    )
    expected = torch.tensor([1.0, torch.exp(torch.tensor(-1.0)).item(), 0.0, 0.0])
    torch.testing.assert_close(reward, expected)


def test_lifted_foot_contact_is_a_nonnegative_cost():
    env = _fake_env(contact_time=[[0.0, 0.2], [0.3, 0.2]], trunk_z=[0.12, 0.12], foot_z=[0.05, 0.0])
    cost = mdp.lifted_foot_contact(env, sensor_cfg=_LIFTED_CFG)
    torch.testing.assert_close(cost, torch.tensor([0.0, 1.0]))


def test_single_leg_rewards_stay_finite_on_nan_state():
    """Rewards precede resets, so a diverged state must not leak NaN into the return."""
    nan = float("nan")
    env = _fake_env(contact_time=[[nan, 0.2]], trunk_z=[nan], foot_z=[nan])
    params = {"lifted_sensor_cfg": _LIFTED_CFG, "support_sensor_cfg": _SUPPORT_CFG, "min_height": 0.08}
    assert torch.isfinite(mdp.single_support(env, **params)).all()
    height = mdp.lifted_foot_height(
        env, 0.06, 0.02, SceneEntityCfg("robot", body_ids=[0]), support_sensor_cfg=_SUPPORT_CFG, min_height=0.08
    )
    assert torch.isfinite(height).all()
    assert torch.isfinite(mdp.lifted_foot_contact(env, sensor_cfg=_LIFTED_CFG)).all()


def test_stand_reward_signs_follow_the_cost_convention():
    """Costs return values >= 0 and carry negative weights; the single-leg objectives carry positive weights."""
    cfg = parse_env_cfg(STAND_TASK, device="cpu", num_envs=1)
    rewards = cfg.rewards
    for name in ("single_support", "lifted_foot_height", "upright", "track_lin_vel", "track_ang_vel"):
        assert getattr(rewards, name).weight > 0.0, name
    for name in ("lifted_foot_contact", "body_ang_vel", "angular_momentum", "action_rate_l2", "self_collisions"):
        assert getattr(rewards, name).weight < 0.0, name
    # Walking-gait terms do not apply to a single-leg stand.
    for name in ("air_time", "foot_clearance", "foot_swing_height", "pose"):
        assert getattr(rewards, name, None) is None, name
    # The right foot supports, the left foot lifts.
    assert rewards.single_support.params["support_sensor_cfg"].body_names == ["ankle_right"]
    assert rewards.single_support.params["lifted_sensor_cfg"].body_names == ["ankle_left"]
    assert rewards.lifted_foot_height.params["asset_cfg"].body_names == ["ankle_left"]


@pytest.mark.parametrize("presets", [(), ("presets=backlash",)])
def test_stand_trains_on_newton_mjwarp_with_bam_servos(presets):
    """Training, playback and evaluation share the Newton MJWarp solver with native BAM servos, without Kit."""
    cfg = parse_env_cfg(STAND_TASK, device="cpu", num_envs=1, overrides=presets)
    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert cfg.sim.use_newton_actuators


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_stand_keeps_the_deployed_interface_and_starts_in_double_support(device):
    """The 61-wide actor input and 14 actions are unchanged; the double-support start earns no single-leg reward."""
    cfg = parse_env_cfg(STAND_TASK, device=device, num_envs=2)
    sim_utils.create_new_stage()
    env = gym.make(STAND_TASK, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 61)
        assert env.observation_manager.active_terms["policy"] == POLICY_TERMS
        assert env.action_manager.total_action_dim == 14
        with torch.inference_mode():
            for _ in range(10):
                obs, reward, *_ = env.step(torch.zeros(2, 14, device=env.device))
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
        term = env.reward_manager.get_term_cfg("single_support")
        torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2, device=env.device))
    finally:
        env.close()
        SimulationContext.clear_instance()
