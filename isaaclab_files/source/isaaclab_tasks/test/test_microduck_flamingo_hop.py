# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck one-foot hopping on the right foot: the hop command, the hop rewards and the task contract."""

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
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

HOP_TASK = "IsaacContrib-FlamingoHop-Flat-MicroDuck"


def _command(num_envs=2):
    cfg = mdp.HopCommandCfg(
        resampling_time_range=(1.0, 1.0), still_time_range=(1.5, 1.5), hop_time_range=(2.0, 2.0), ramp_s=0.3
    )
    return mdp.HopCommand(cfg, SimpleNamespace(num_envs=num_envs, device="cpu"))


def test_hop_command_starts_still_then_alternates_with_the_dwell_of_each_phase():
    term = _command()
    term.reset()
    assert term.command.tolist() == [[0.0, 0.0, 0.0]] * 2
    assert term.time_left.tolist() == pytest.approx([1.5, 1.5])
    term._resample(slice(None))
    assert term.command[:, 2].tolist() == [1.0, 1.0]
    assert term.time_left.tolist() == pytest.approx([2.0, 2.0])
    term._resample(slice(None))
    assert term.command[:, 2].tolist() == [0.0, 0.0]


def test_hop_blend_ramps_over_the_ramp_time_and_a_script_freezes_the_command():
    term = _command(num_envs=1)
    term.reset()
    term.set_command(torch.tensor([[0.0, 0.0, 1.0]]))
    for _ in range(9):  # 0.18 s of a 0.3 s ramp
        term.compute(0.02)
    assert term.beta.item() == pytest.approx(0.6, abs=1e-5)
    for _ in range(200):
        term.compute(0.02)
    assert term.beta.item() == pytest.approx(1.0)
    assert term.command[0, 2].item() == 1.0


def test_hop_time_counts_the_current_hop_phase_and_resets_when_still():
    term = _command(num_envs=1)
    term.reset()
    term.set_command(torch.tensor([[0.0, 0.0, 1.0]]))
    for _ in range(10):
        term.compute(0.02)
    assert term.hop_time.item() == pytest.approx(0.2)
    term.set_command(torch.tensor([[0.0, 0.0, 0.0]]))
    term.compute(0.02)
    assert term.hop_time.item() == 0.0


def _fake_env(
    beta,
    contact_time=None,
    first_contact=None,
    last_air_time=None,
    lin_vel=None,
    ang_vel=None,
    current_air_time=None,
    hop_time=None,
):
    n = len(beta)
    term = SimpleNamespace(
        beta=torch.tensor(beta), hop_time=torch.tensor(hop_time if hop_time is not None else [9.0] * n)
    )
    contact_time = torch.tensor(contact_time if contact_time is not None else [[0.1]] * n)
    first = torch.tensor(first_contact if first_contact is not None else [[False]] * n)
    air = torch.tensor(last_air_time if last_air_time is not None else [[0.0]] * n)
    current_air = torch.tensor(current_air_time if current_air_time is not None else [[0.0]] * n)
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_contact_time=SimpleNamespace(torch=contact_time),
            last_air_time=SimpleNamespace(torch=air),
            current_air_time=SimpleNamespace(torch=current_air),
        ),
        compute_first_contact=lambda dt: SimpleNamespace(torch=first),
    )
    data = {
        "root_lin_vel_w": torch.tensor(lin_vel if lin_vel is not None else [[0.0, 0.0, 0.0]] * n),
        "root_ang_vel_w": torch.tensor(ang_vel if ang_vel is not None else [[0.0, 0.0, 0.0]] * n),
    }
    robot = SimpleNamespace(data=SimpleNamespace(**{k: SimpleNamespace(torch=v) for k, v in data.items()}))

    class Scene(dict):
        sensors = {"contact_forces": sensor}

    return SimpleNamespace(
        scene=Scene(robot=robot),
        command_manager=SimpleNamespace(get_term=lambda name: term),
        step_dt=0.02,
        num_envs=n,
        device="cpu",
    )


_STANCE = SceneEntityCfg("contact_forces", body_ids=[0])
_SWING = SceneEntityCfg("contact_forces", body_ids=[1])


def test_hop_air_time_refuses_takeoffs_pushed_with_the_swing_foot():
    # 80 ms stance flights; the swing foot left the floor with the stance foot (both feet pushed off), 50 ms before
    # the takeoff, and 0.1 s or more before it.
    env = _fake_env(
        beta=[1.0, 1.0, 1.0, 1.0],
        first_contact=[[True, False]] * 4,
        last_air_time=[[0.08, 0.0]] * 4,
        current_air_time=[[0.0, 0.08], [0.0, 0.13], [0.0, 0.18], [0.0, 1.0]],
    )
    reward = mdp.hop_air_time(
        env,
        "base_velocity",
        _STANCE,
        min_air_time=0.04,
        max_air_time=0.12,
        swing_sensor_cfg=_SWING,
        swing_lead_time=0.1,
    )
    torch.testing.assert_close(reward, torch.tensor([0.0, 0.0, 0.5, 0.5]))


def test_hop_air_time_pays_only_landings_with_the_swing_foot_up_through_the_flight():
    # Columns [stance, swing]; every row lands after an 80 ms stance flight while hopping. The swing foot has been up
    # for 1 s, down now, up for only 40 ms (it touched during the flight), and up for exactly the flight.
    env = _fake_env(
        beta=[1.0, 1.0, 1.0, 1.0],
        first_contact=[[True, False]] * 4,
        last_air_time=[[0.08, 0.0]] * 4,
        current_air_time=[[0.0, 1.0], [0.0, 0.0], [0.0, 0.04], [0.0, 0.08]],
    )
    reward = mdp.hop_air_time(
        env, "base_velocity", _STANCE, min_air_time=0.04, max_air_time=0.12, swing_sensor_cfg=_SWING
    )
    torch.testing.assert_close(reward, torch.tensor([0.5, 0.0, 0.0, 0.5]))


def test_hop_air_time_pays_landings_of_real_flights_while_hopping():
    # Landing after 80 ms while hopping, after 20 ms (too short), after 300 ms (capped), while still, and no landing.
    env = _fake_env(
        beta=[1.0, 1.0, 1.0, 0.0, 1.0],
        first_contact=[[True], [True], [True], [True], [False]],
        last_air_time=[[0.08], [0.02], [0.30], [0.08], [0.08]],
    )
    reward = mdp.hop_air_time(env, "base_velocity", _STANCE, min_air_time=0.04, max_air_time=0.12)
    torch.testing.assert_close(reward, torch.tensor([0.5, 0.0, 1.0, 0.0, 0.0]))


def test_hop_ground_time_cost_bites_only_long_stance_contacts_while_hopping():
    env = _fake_env(beta=[1.0, 1.0, 0.0], contact_time=[[0.5], [0.2], [0.9]])
    cost = mdp.hop_ground_time_cost(env, "base_velocity", _STANCE, max_contact_time=0.4)
    torch.testing.assert_close(cost, torch.tensor([1.0, 0.0, 0.0]))


def test_hop_wait_cost_grows_with_the_stance_time_since_the_last_takeoff_or_the_hop_start():
    # Stance down 0.3 s and 1.2 s into a long hop phase; down 1.2 s but the hop phase started 0.5 s ago; 5 s without a
    # takeoff keeps growing; just took off (contact time 0); still phase.
    env = _fake_env(
        beta=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        contact_time=[[0.3], [1.2], [1.2], [5.0], [0.0], [1.2]],
        hop_time=[3.0, 3.0, 0.5, 9.0, 3.0, 0.0],
    )
    cost = mdp.hop_wait_cost(env, "base_velocity", _STANCE)
    torch.testing.assert_close(cost, torch.tensor([0.3, 1.2, 0.5, 5.0, 0.0, 0.0]))


def test_hop_drift_cost_penalizes_horizontal_and_yaw_motion_while_hopping():
    env = _fake_env(
        beta=[1.0, 0.0], lin_vel=[[0.3, 0.4, 1.0], [0.3, 0.4, 0.0]], ang_vel=[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
    )
    torch.testing.assert_close(mdp.hop_drift_cost(env, "base_velocity", yaw_weight=0.1), torch.tensor([0.65, 0.0]))


def test_still_scaled_terms_fade_out_while_hopping():
    env = _fake_env(beta=[0.0, 0.5, 1.0], contact_time=[[0.1], [0.1], [0.1]])
    reward = mdp.still_foot_in_contact(env, "base_velocity", _STANCE)
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.5, 0.0]))


def test_still_action_rate_fades_out_while_hopping():
    env = _fake_env(beta=[0.0, 0.5, 1.0])
    env.action_manager = SimpleNamespace(action=torch.ones(3, 2), prev_action=torch.zeros(3, 2))
    torch.testing.assert_close(mdp.still_action_rate_l2(env, "base_velocity"), torch.tensor([2.0, 1.0, 0.0]))


def test_still_gravity_and_pose_matches_fade_out_while_hopping():
    env = _fake_env(beta=[0.0, 1.0])
    robot = env.scene["robot"]
    robot.data.projected_gravity_b = SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -1.0]] * 2))
    robot.data.joint_pos = SimpleNamespace(torch=torch.zeros(2, 2))
    gravity = mdp.still_projected_gravity_match(env, "base_velocity", target=(0.0, 0.0, -1.0), std=0.15)
    torch.testing.assert_close(gravity, torch.tensor([1.0, 0.0]))
    pose = mdp.still_pose_target_match(
        env, "base_velocity", target=[0.0, 0.0], std=0.5, asset_cfg=SceneEntityCfg("robot", joint_ids=[0, 1])
    )
    torch.testing.assert_close(pose, torch.tensor([1.0, 0.0]))


"""
Environment.
"""


@pytest.fixture(scope="module")
def hop_env():
    device = test_devices(DeviceScope.CUDA)[0]
    cfg = parse_env_cfg(HOP_TASK, device=device, num_envs=32)
    sim_utils.create_new_stage()
    env = gym.make(HOP_TASK, cfg=cfg).unwrapped
    env.reset()
    yield env
    env.close()
    SimulationContext.clear_instance()


@pytest.mark.integration
@pytest.mark.kitless
def test_hop_keeps_the_deployed_interface_and_the_flamingo_physics(hop_env):
    env = hop_env
    assert env.observation_manager.group_obs_dim["policy"] == (61,)
    assert env.action_manager.total_action_dim == 14
    assert isinstance(env.command_manager.get_term("base_velocity"), mdp.HopCommand)
    assert env.cfg.sim.physics.solver_cfg.njmax == 1500
    assert env.max_episode_length_s == pytest.approx(10.0)
    # Pushes from the start at the stage-1 final magnitude, with no schedules.
    assert env.cfg.events.push_robot.params["velocity_range"] == {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
    assert all(term is None for term in env.cfg.curriculum.to_dict().values())


@pytest.mark.integration
@pytest.mark.kitless
def test_hop_rewards(hop_env):
    expected = {
        "com_over_stance_foot": 3.0,
        "stance_foot_grounded": 1.0,
        "swing_foot_clear": 1.5,
        "swing_foot_touch": -1.5,
        "pose_flamingo": 1.5,
        "gravity_flamingo": 2.0,
        "stillness": 1.0,
        "stance_side_tilt": -4.0,
        "joint_limit_proximity": -1.0,
        "action_rate_l2": -0.5,
        "joint_torque_rate_l2": -1e-3,
        "body_ang_vel": -0.05,
        "angular_momentum": -0.02,
        "self_collisions": -1.0,
        "hop_air_time": 30.0,
        "hop_ground_time": -2.0,
        "hop_wait": -2.0,
        "hop_drift": -4.0,
    }
    manager = hop_env.reward_manager
    weights = {name: manager.get_term_cfg(name).weight for name in manager.active_terms}
    assert weights == pytest.approx(expected)
    assert manager.get_term_cfg("com_over_stance_foot").params["std"] == 0.02
    # Hops landing on both feet earn nothing.
    assert manager.get_term_cfg("hop_air_time").params["swing_sensor_cfg"].body_names == ["ankle_left"]
    # The swing foot must be up 0.1 s before the takeoff: the first policy pushed off with it 93 % of the time.
    assert manager.get_term_cfg("hop_air_time").params["swing_lead_time"] == 0.1
    # Terms that pay for keeping still are off while hopping.
    still_funcs = {
        "stance_foot_grounded": mdp.still_foot_in_contact,
        "stillness": mdp.still_joint_vel_gaussian,
        "swing_foot_clear": mdp.still_foot_site_height_gaussian,
        "pose_flamingo": mdp.still_pose_target_match,
        "gravity_flamingo": mdp.still_projected_gravity_match,
        "action_rate_l2": mdp.still_action_rate_l2,
    }
    for name, func in still_funcs.items():
        assert manager.get_term_cfg(name).func is func, name


@pytest.mark.integration
@pytest.mark.kitless
def test_hop_steps_with_finite_rewards(hop_env):
    env = hop_env
    env.reset()
    actions = torch.zeros(env.num_envs, 14, device=env.device)
    for _ in range(10):
        _, reward, *_ = env.step(actions)
        assert torch.isfinite(reward).all()
