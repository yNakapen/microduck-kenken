# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck hop evaluation: episode metrics from contact timelines, stage verdicts, and fixed supply voltage."""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.hop_metrics import (
    HopMetrics,
    HopSignals,
    read_hop_signals,
    set_supply_voltage,
    stage_passes,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

STAND_TASK = "IsaacContrib-Hop-Stand-Flat-MicroDuck"
STEP_DT = 0.02


def _signals(support, lifted, *, touchdown=False, air_time=0.0, com=0.10, sole=0.0, fallen=False, low=False, force=0.0):
    """One control step of a single environment."""

    def vec(value, dtype=torch.float32):
        return torch.tensor([value], dtype=dtype)

    return HopSignals(
        support_contact=vec(support, torch.bool),
        lifted_contact=vec(lifted, torch.bool),
        support_touchdown=vec(touchdown, torch.bool),
        support_last_air_time=vec(air_time),
        com_height=vec(com),
        support_sole_height=vec(sole),
        fallen=vec(fallen, torch.bool),
        trunk_low=vec(low, torch.bool),
        saturated_fraction=vec(0.0),
        max_joint_speed=vec(1.0),
        support_force=vec(force),
    )


def _run(steps):
    metrics = HopMetrics(num_envs=1, step_dt=STEP_DT)
    for signals in steps:
        metrics.update(signals)
    return {name: value.item() for name, value in metrics.summary().items()}


def _single_hop_timeline(air_steps=4, hold_steps=110):
    """Double support, lift the left foot, one flight on the right foot, then hold single support."""
    steps = [_signals(True, True) for _ in range(10)]
    steps += [_signals(True, False) for _ in range(50)]
    for i in range(air_steps):
        steps.append(_signals(False, False, com=0.10 + 0.0025 * (i + 1), sole=0.002 * (i + 1)))
    steps.append(_signals(True, False, touchdown=True, air_time=air_steps * STEP_DT, force=30.0))
    steps += [_signals(True, False) for _ in range(hold_steps)]
    return steps


def test_single_hop_is_counted_with_air_time_com_rise_and_post_landing_hold():
    summary = _run(_single_hop_timeline())
    assert summary["hops"] == 1
    assert summary["max_air_time"] == pytest.approx(0.08)
    assert summary["max_com_rise"] == pytest.approx(0.01)
    assert summary["max_sole_clearance"] == pytest.approx(0.008)
    assert summary["post_landing_hold"] == pytest.approx(111 * STEP_DT)
    assert summary["hold_time"] == pytest.approx(111 * STEP_DT)
    assert summary["lifted_touchdowns"] == 0
    assert summary["peak_force"] == pytest.approx(30.0)
    assert not summary["fell"]
    assert not summary["collapsed"]


def test_a_crouch_below_the_trunk_limit_is_a_collapse_not_a_fall():
    """An upright robot that sinks below the standing height keeps being evaluated, but is flagged."""
    steps = _single_hop_timeline(hold_steps=10)
    steps[20] = _signals(True, False, low=True)
    summary = _run(steps)
    assert summary["collapsed"]
    assert not summary["fell"]
    assert summary["hops"] == 1


def test_a_step_onto_the_lifted_foot_is_a_touchdown_not_a_hop():
    steps = [_signals(True, True) for _ in range(5)] + [_signals(True, False) for _ in range(20)]
    steps += [_signals(False, True), _signals(False, True)]
    steps += [_signals(True, True, touchdown=True, air_time=0.04)]
    summary = _run(steps)
    assert summary["hops"] == 0
    assert summary["lifted_touchdowns"] == 1


def test_nothing_accumulates_after_a_fall():
    steps = _single_hop_timeline(hold_steps=10) + [_signals(True, False, fallen=True)]
    # The automatic reset puts both feet back down; it must not count as a touchdown or a hop.
    steps += [_signals(True, True) for _ in range(5)]
    steps += _single_hop_timeline()
    summary = _run(steps)
    assert summary["fell"]
    assert summary["fall_time"] == pytest.approx((len(_single_hop_timeline(hold_steps=10)) + 1) * STEP_DT)
    assert summary["hops"] == 1
    assert summary["lifted_touchdowns"] == 0


def test_non_finite_signals_do_not_leak_into_the_summary():
    nan = float("nan")
    steps = _single_hop_timeline(hold_steps=3)
    steps[62] = _signals(False, False, com=nan, sole=nan, force=nan)
    summary = _run(steps)
    assert all(torch.isfinite(torch.tensor(value, dtype=torch.float32)) for value in summary.values())


def test_stage_verdicts_follow_the_test_plan():
    n = 10
    zeros, ones = torch.zeros(n), torch.ones(n)

    def flags():
        return {"fell": torch.zeros(n, dtype=torch.bool), "collapsed": torch.zeros(n, dtype=torch.bool)}

    stand = {**flags(), "hold_time": 10.5 * ones}
    assert stage_passes("stand", stand)
    stand["collapsed"][0] = True
    assert not stage_passes("stand", stand)
    stand = {**flags(), "hold_time": 10.5 * ones}
    stand["fell"][0] = True
    assert not stage_passes("stand", stand)

    single = {
        **flags(),
        "max_air_time": 0.07 * ones,
        "post_landing_hold": 2.5 * ones,
    }
    single["max_air_time"][:2] = 0.05
    assert stage_passes("single", single)  # 8/10
    single["post_landing_hold"][2] = 1.0
    assert not stage_passes("single", single)  # 7/10

    hop = {**flags(), "hops": 6 * ones, "lifted_touchdowns": zeros.clone()}
    hop["fell"][0] = True
    hop["collapsed"][1] = True
    assert stage_passes("hop", hop)  # 2/10 failures
    hop["collapsed"][2] = True
    assert not stage_passes("hop", hop)  # 3/10 failures
    hop["collapsed"][2] = False
    hop["lifted_touchdowns"][5] = 1
    assert not stage_passes("hop", hop)


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_supply_voltage_stays_fixed_per_environment(device):
    """Each environment keeps its evaluation voltage through steps and resets, without battery sag."""
    cfg = parse_env_cfg(STAND_TASK, device=device, num_envs=3, overrides=("presets=backlash",))
    sim_utils.create_new_stage()
    env = gym.make(STAND_TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        vins = torch.tensor([6.5, 7.4, 8.2], device=env.device)
        set_supply_voltage(env, vins, sag_gain=0.0)
        robot = env.scene["robot"]
        drive = robot.actuators["servos"].drive
        with torch.inference_mode():
            for _ in range(5):
                env.step(torch.zeros(3, 14, device=env.device))
            env.reset()
            # The robot is reset slightly above the ground and settles within a few steps.
            for _ in range(10):
                env.step(torch.zeros(3, 14, device=env.device))
        stored = read_group_parameter(robot.actuators, "servos", "drive", "vin")
        torch.testing.assert_close(stored, vins[:, None].expand_as(stored))
        effective = torch.as_tensor(drive.effective_vin.numpy(), device=env.device).reshape(3, -1)
        torch.testing.assert_close(effective, vins[:, None].expand_as(effective))
        signals = read_hop_signals(env)
        for name, value in vars(signals).items():
            assert value.shape == (3,), name
            assert torch.isfinite(value.float()).all(), name
        assert ((signals.saturated_fraction >= 0.0) & (signals.saturated_fraction <= 1.0)).all()
        # A settled robot stands on both feet.
        assert signals.support_contact.all() and signals.lifted_contact.all()
        assert (signals.saturated_fraction == 0.0).all(), "holding the default pose needs little torque"
        # A position target far from the current pose drives servos to their full torque once each
        # environment's random 3-6 physics-step command delay has passed.
        peak = torch.zeros(3, device=env.device)
        with torch.inference_mode():
            for _ in range(2):
                env.step(torch.full((3, 14), 1.5, device=env.device))
                peak = torch.maximum(peak, read_hop_signals(env).saturated_fraction)
        assert (peak > 0.0).all()
    finally:
        env.close()
        SimulationContext.clear_instance()
