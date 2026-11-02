# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flamingo cycle, microduck_rl's stage 2 (``Mjlab-FlamingoCycle-Flat-MicroDuck``) on Newton MJWarp.

The command and reward behaviour follows ``FlamingoCommand`` and the ``fl_*`` terms of microduck_rl's
``src/mjlab_microduck/tasks/mdp.py`` (``flamingo`` branch, ``4757d6ae``); the environment tests check the settings of
``microduck_flamingo_cycle_env_cfg.py``.
"""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flamingo_pose import FLAMINGO_GRAVITY_B, FLAMINGO_POSE_LEFT, FLAMINGO_POSE_RIGHT
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

CYCLE_TASK = "IsaacContrib-FlamingoCycle-Flat-MicroDuck"


def _command(num_envs=4, flamingo_prob=0.6, ramp_s=1.5, zero_side_prob=0.5):
    cfg = mdp.FlamingoCommandCfg(
        resampling_time_range=(2.5, 5.0), flamingo_prob=flamingo_prob, ramp_s=ramp_s, zero_side_prob=zero_side_prob
    )
    return mdp.FlamingoCommand(cfg, SimpleNamespace(num_envs=num_envs, device="cpu"))


def test_alpha_slews_toward_the_flag_over_the_ramp_time():
    term = _command(num_envs=1, flamingo_prob=1.0)
    term.reset()
    assert term.flag.item() == 1.0
    term.time_left[:] = 100.0
    # 0.6 s of a 1.5 s ramp.
    for _ in range(30):
        term.compute(0.02)
    assert term.alpha.item() == pytest.approx(0.4, abs=1e-5)
    for _ in range(100):
        term.compute(0.02)
    assert term.alpha.item() == pytest.approx(1.0)


def test_the_stance_side_is_latched_during_a_hold_and_redrawn_only_from_stand():
    torch.manual_seed(0)
    term = _command(num_envs=256, flamingo_prob=1.0)
    term.reset()
    first = term.stance_side.clone()
    # Every resample after a one-foot command keeps the side: no side flip mid-hold.
    for _ in range(5):
        term._resample(slice(None))
        torch.testing.assert_close(term.stance_side, first)
    assert set(first.unique().tolist()) == {-1.0, 1.0}


def test_the_observed_side_is_zeroed_only_while_standing():
    torch.manual_seed(0)
    term = _command(num_envs=2048, flamingo_prob=0.5, zero_side_prob=0.5)
    term.reset()
    flag, side = term.command[:, 0], term.command[:, 1]
    assert (side[flag > 0.5].abs() == 1.0).all()
    zeroed = (side[flag < 0.5] == 0.0).float().mean().item()
    assert zeroed == pytest.approx(0.5, abs=0.06)
    assert (term.stance_side.abs() == 1.0).all()
    assert (term.command[:, 2] == 0.0).all()


def test_a_side_requested_by_the_spawn_survives_the_reset_resample():
    term = _command(num_envs=4)
    ids = torch.arange(4)
    term.request_side(ids, torch.tensor([1.0, -1.0, 1.0, -1.0]))
    term.reset(ids)
    torch.testing.assert_close(term.stance_side, torch.tensor([1.0, -1.0, 1.0, -1.0]))


def test_a_scripted_command_overrides_the_sampler_and_keeps_the_latched_side():
    term = _command(num_envs=2)
    term.reset()
    term.set_command(torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), stance_side=torch.tensor([1.0, -1.0]))
    for _ in range(500):
        term.compute(0.02)
    torch.testing.assert_close(term.command, torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(term.stance_side, torch.tensor([1.0, -1.0]))
    assert term.alpha.tolist() == pytest.approx([1.0, 0.0])


"""
Rewards on a fake environment: two envs, columns [left, right] for feet.
"""


def _fake_env(alpha, side, flag=None, contact=None, feet_z=None, joint_vel=None, gravity=None):
    n = len(alpha)
    term = SimpleNamespace(
        alpha=torch.tensor(alpha),
        stance_side=torch.tensor(side),
        flag=torch.tensor(flag if flag is not None else alpha),
    )
    command_manager = SimpleNamespace(get_term=lambda name: term)
    contact_time = torch.tensor(contact if contact is not None else [[0.1, 0.1]] * n)
    sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=SimpleNamespace(torch=contact_time)))
    pos = torch.zeros(n, 2, 3)
    if feet_z is not None:
        pos[:, :, 2] = torch.tensor(feet_z)
    quat = torch.zeros(n, 2, 4)
    quat[..., 3] = 1.0
    data = {
        "body_link_pos_w": pos,
        "body_link_quat_w": quat,
        "joint_vel": torch.tensor(joint_vel if joint_vel is not None else [[0.0, 0.0]] * n),
        "projected_gravity_b": torch.tensor(gravity if gravity is not None else [[0.0, 0.0, -1.0]] * n),
    }
    robot = SimpleNamespace(data=SimpleNamespace(**{k: SimpleNamespace(torch=v) for k, v in data.items()}))

    class Scene(dict):
        sensors = {"contact_forces": sensor}
        env_origins = torch.zeros(n, 3)

    return SimpleNamespace(scene=Scene(robot=robot), command_manager=command_manager, device="cpu", num_envs=n)


_FEET_SENSOR = SceneEntityCfg("contact_forces", body_ids=[0, 1])
_FEET_BODIES = SceneEntityCfg("robot", body_ids=[0, 1], joint_ids=[0, 1])


def test_swing_foot_contact_is_paid_while_standing_and_taxed_on_one_foot():
    # Both feet down; alpha 0 (stand), 0.65 (mid), 1 (one foot); right stance.
    env = _fake_env(alpha=[0.0, 0.65, 1.0], side=[1.0, 1.0, 1.0], contact=[[0.1, 0.1]] * 3)
    reward = mdp.cycle_swing_foot_contact(env, "base_velocity", _FEET_SENSOR, lo=0.4, hi=0.9)
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0, -1.0]))


def test_swing_foot_clear_follows_the_side_and_waits_for_the_lift():
    # Left foot at 0.10 m for right stance, right foot at 0.10 m for left stance; the lift is due only at alpha 1.
    env = _fake_env(alpha=[1.0, 1.0, 0.3], side=[1.0, -1.0, 1.0], feet_z=[[0.10, 0.0], [0.0, 0.10], [0.10, 0.0]])
    offset = (0.0, 0.0, 0.0)
    reward = mdp.cycle_swing_foot_clear(
        env, "base_velocity", _FEET_BODIES, _FEET_SENSOR, target=0.10, std=0.06, lo=0.6, hi=1.0, site_offset=offset
    )
    torch.testing.assert_close(reward, torch.tensor([1.0, 1.0, 0.0]))


def test_stillness_waits_for_the_ramp_to_finish():
    env = _fake_env(alpha=[1.0, 0.5], side=[1.0, 1.0], flag=[1.0, 1.0])
    reward = mdp.cycle_stillness(env, "base_velocity", _FEET_SENSOR, std=2.0, asset_cfg=_FEET_BODIES)
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0]))


def test_gravity_target_blends_upright_into_the_side_lean():
    g_right = torch.tensor(FLAMINGO_GRAVITY_B)
    g_left = g_right * torch.tensor([1.0, -1.0, 1.0])
    env = _fake_env(
        alpha=[0.0, 1.0, 1.0], side=[1.0, 1.0, -1.0], gravity=[[0.0, 0.0, -1.0], g_right.tolist(), g_left.tolist()]
    )
    reward = mdp.cycle_gravity_track(env, "base_velocity", FLAMINGO_GRAVITY_B, std=0.15)
    torch.testing.assert_close(reward, torch.ones(3))


def test_stance_side_tilt_is_mirrored_by_the_side():
    # Rolled 0.2 past the threshold toward the stance side for each side, and toward the swing side.
    env = _fake_env(
        alpha=[1.0, 1.0, 1.0], side=[1.0, -1.0, 1.0], gravity=[[0, -0.65, -0.7], [0, 0.65, -0.7], [0, 0.65, -0.7]]
    )
    cost = mdp.cycle_stance_side_tilt_cost(env, "base_velocity", threshold=0.45)
    torch.testing.assert_close(cost, torch.tensor([0.04, 0.04, 0.0]))


"""
Environment.
"""


@pytest.fixture(scope="module")
def cycle_env():
    device = test_devices(DeviceScope.CUDA)[0]
    cfg = parse_env_cfg(CYCLE_TASK, device=device, num_envs=64)
    sim_utils.create_new_stage()
    env = gym.make(CYCLE_TASK, cfg=cfg).unwrapped
    env.reset()
    yield env
    env.close()
    SimulationContext.clear_instance()


@pytest.mark.integration
@pytest.mark.kitless
def test_cycle_keeps_the_deployed_interface(cycle_env):
    env = cycle_env
    assert env.observation_manager.group_obs_dim["policy"] == (61,)
    assert env.action_manager.total_action_dim == 14
    assert isinstance(env.command_manager.get_term("base_velocity"), mdp.FlamingoCommand)
    assert env.max_episode_length_s == pytest.approx(10.0)


@pytest.mark.integration
@pytest.mark.kitless
def test_cycle_rewards_match_microduck_rl(cycle_env):
    expected = {
        "com_target": 3.0,
        "stance_foot_grounded": 1.0,
        "swing_foot_contact": 0.5,
        "swing_foot_clear": 1.5,
        "pose_track": 1.5,
        "gravity_track": 2.0,
        "stillness": 1.0,
        "stance_side_tilt": -4.0,
        "commanded_support": 1e-3,
        "joint_limit_proximity": -1.0,
        "action_rate_l2": -0.1,
        "joint_torque_rate_l2": 0.0,
        "body_ang_vel": -0.05,
        "angular_momentum": -0.02,
        "self_collisions": -1.0,
    }
    manager = cycle_env.reward_manager
    weights = {name: manager.get_term_cfg(name).weight for name in manager.active_terms}
    assert weights == pytest.approx(expected)


@pytest.mark.integration
@pytest.mark.kitless
def test_cycle_spawn_matches_the_command_side_and_alpha(cycle_env):
    env = cycle_env
    env.reset()
    term = env.command_manager.get_term("base_velocity")
    robot = env.scene["robot"]
    joint_ids = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)[0]
    q = robot.data.joint_pos.torch[:, joint_ids]
    home = robot.data.default_joint_pos.torch[:, joint_ids]
    poses = {
        1.0: torch.tensor(FLAMINGO_POSE_RIGHT, device=env.device),
        -1.0: torch.tensor(FLAMINGO_POSE_LEFT, device=env.device),
    }
    for i in range(env.num_envs):
        target = poses[term.stance_side[i].item()] if term.alpha[i] > 0.5 else home[i]
        assert (q[i] - target).abs().mean().item() < 0.15
    # In-pose spawns lean over the stance foot.
    gy = robot.data.projected_gravity_b.torch[:, 1]
    in_pose = term.alpha > 0.5
    assert (torch.sign(gy[in_pose]) == -term.stance_side[in_pose]).all()
    assert set(term.alpha.unique().tolist()) <= {0.0, 1.0}


@pytest.mark.integration
@pytest.mark.kitless
def test_cycle_steps_with_finite_rewards(cycle_env):
    env = cycle_env
    env.reset()
    actions = torch.zeros(env.num_envs, 14, device=env.device)
    for _ in range(10):
        _, reward, *_ = env.step(actions)
        assert torch.isfinite(reward).all()


@pytest.mark.integration
@pytest.mark.kitless
def test_cycle_curriculum_reaches_the_final_microduck_rl_stages(cycle_env):
    env = cycle_env
    saved = env.common_step_counter
    try:
        env.common_step_counter = 1500 * 24
        env.curriculum_manager.compute(env_ids=torch.arange(env.num_envs, device=env.device))
        push = env.event_manager.get_term_cfg("push_robot").params["velocity_range"]
        assert push == {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
        assert env.reward_manager.get_term_cfg("action_rate_l2").weight == -0.5
        assert env.reward_manager.get_term_cfg("joint_torque_rate_l2").weight == -1e-3
        assert env.reward_manager.get_term_cfg("com_target").params["std"] == 0.02
        assert env.event_manager.get_term_cfg("reset_flamingo_cycle_pose").params["in_pose_prob"] == 0.3
    finally:
        env.common_step_counter = saved


@pytest.mark.parametrize("play", [False, True])
def test_cycle_schedules_target_enabled_events(play):
    cfg = parse_env_cfg(CYCLE_TASK, device="cpu", num_envs=1)
    if play:
        cfg.play_mode()
    for name, term in cfg.curriculum.to_dict().items():
        if term is None:
            continue
        address = term["params"]["address"].split(".")
        if address[0] == "events":
            assert getattr(cfg.events, address[1]) is not None, f"curriculum {name} targets disabled {address[1]}"


def test_cycle_pose_target_blends_home_into_the_side_pose_and_fades_mid_transition():
    # Two joints; HOME (0, 0); right pose (1, -1), left pose (-1, 1).
    env = _fake_env(alpha=[0.0, 1.0, 1.0, 0.5], side=[1.0, 1.0, -1.0, 1.0])
    joint_pos = torch.tensor([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0], [0.5, -0.5]])
    env.scene["robot"].data.joint_pos = SimpleNamespace(torch=joint_pos)
    env.scene["robot"].data.default_joint_pos = SimpleNamespace(torch=torch.zeros(4, 2))
    reward = mdp.cycle_pose_track(
        env, "base_velocity", [1.0, -1.0], [-1.0, 1.0], std=0.5, mid_attenuation=0.75, asset_cfg=_FEET_BODIES
    )
    # On target everywhere; mid-transition the match is scaled by 1 - 0.75 * 4 * 0.5 * 0.5 = 0.25.
    torch.testing.assert_close(reward, torch.tensor([1.0, 1.0, 1.0, 0.25]))
