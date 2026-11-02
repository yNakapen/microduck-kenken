# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode metrics of the MicroDuck flamingo evaluation: microduck_rl's stage-1 success metric and fall sides."""

import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.flamingo_metrics import (
    CycleEpisodeMetrics,
    FlamingoEpisodeMetrics,
    HopEpisodeMetrics,
    cycle_protocol_command,
    fall_side,
    hop_protocol_command,
    set_eval_conditions,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

DT = 0.5


def _run(steps):
    """Feed ``(stance, swing, done, timeout, gravity)`` rows for two environments."""
    metrics = FlamingoEpisodeMetrics(num_envs=2, step_dt=DT, success_hold=2.0)
    for stance, swing, done, timeout, gravity in steps:
        metrics.update(
            stance_contact=torch.tensor(stance),
            swing_contact=torch.tensor(swing),
            done=torch.tensor(done),
            time_out=torch.tensor(timeout),
            gravity_b=torch.tensor(gravity),
        )
    return metrics.summary()


_POSE_G = [-0.154, -0.379, -0.912]


def test_success_needs_a_long_enough_single_support_streak_and_no_fall():
    # env 0: 4 steps (2 s) of single support then a time-out; env 1: the swing foot lands in the middle.
    rows = [
        ([True, True], [False, False], [False, False], [False, False], [_POSE_G, _POSE_G]),
        ([True, True], [False, True], [False, False], [False, False], [_POSE_G, _POSE_G]),
        ([True, True], [False, False], [False, False], [False, False], [_POSE_G, _POSE_G]),
        ([True, True], [False, False], [True, True], [True, True], [_POSE_G, _POSE_G]),
    ]
    summary = _run(rows)
    torch.testing.assert_close(summary["longest_single_support"], torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(summary["success"], torch.tensor([True, False]))
    torch.testing.assert_close(summary["fell"], torch.tensor([False, False]))
    torch.testing.assert_close(summary["swing_touchdowns"], torch.tensor([0, 1]))


def test_a_fall_fails_and_freezes_the_episode():
    stance_fall = [0.0, -0.95, -0.3]
    # The terminal step's gravity is read after the automatic reset, so the fall shows the step before.
    rows = [
        ([True, True], [False, False], [False, False], [False, False], [stance_fall, _POSE_G]),
        ([True, True], [False, False], [True, False], [False, False], [_POSE_G, _POSE_G]),
        # After the reset env 0 must not count any more.
        ([True, True], [False, False], [False, False], [False, False], [_POSE_G, _POSE_G]),
        ([True, True], [False, False], [False, True], [False, True], [_POSE_G, _POSE_G]),
    ]
    summary = _run(rows)
    torch.testing.assert_close(summary["fell"], torch.tensor([True, False]))
    torch.testing.assert_close(summary["longest_single_support"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(summary["success"], torch.tensor([False, True]))
    assert summary["fall_side"] == ["stance", ""]


def test_fall_side_is_the_largest_departure_from_the_pose_lean():
    gravity = torch.tensor([[0.0, -0.95, -0.3], [0.0, 0.5, -0.8], [0.9, -0.38, -0.2], [-0.9, -0.38, -0.2]])
    assert fall_side(gravity, torch.tensor(_POSE_G)) == ["stance", "swing", "backward", "forward"]


def test_eval_conditions_freeze_the_final_randomization_with_the_tested_push():
    cfg = parse_env_cfg("IsaacContrib-Flamingo-Flat-MicroDuck", device="cpu", num_envs=1)
    set_eval_conditions(cfg, 0.2)
    assert all(term is None for term in cfg.curriculum.to_dict().values())
    assert cfg.events.push_robot.params["velocity_range"] == {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
    assert cfg.events.randomize_com.params["com_range"]["z"] == (-0.015, 0.015)
    assert cfg.events.randomize_head_com.params["com_range"]["z"] == (-0.01, 0.01)


def test_cycle_protocol_stands_lifts_holds_and_lowers():
    side = torch.tensor([1.0, -1.0])
    assert cycle_protocol_command(1.0, side).tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert cycle_protocol_command(3.0, side).tolist() == [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]
    assert cycle_protocol_command(8.0, side).tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_cycle_metrics_score_the_hold_and_the_final_stand_windows():
    # Two envs at 1 s steps over a 10 s protocol; env 1 drops its swing foot during the hold.
    metrics = CycleEpisodeMetrics(num_envs=2, step_dt=1.0)
    for t in range(10):
        hold = 4.0 <= t + 1 <= 7.0
        swing = torch.tensor([not hold, not hold or t + 1 == 5.0])
        metrics.update(
            time=t + 1.0,
            stance_contact=torch.tensor([True, True]),
            swing_contact=swing,
            swing_height=torch.tensor([0.1 if hold else 0.0, 0.1 if hold else 0.0]),
            done=torch.tensor([False, False]),
            time_out=torch.tensor([False, False]),
        )
    summary = metrics.summary()
    torch.testing.assert_close(summary["hold_single_support"], torch.tensor([1.0, 0.75]))
    torch.testing.assert_close(summary["stand_two_feet"], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(summary["success"], torch.tensor([True, False]))


def test_hop_protocol_stills_hops_and_stills_again():
    assert hop_protocol_command(1.0, 2).tolist() == [[0.0, 0.0, 0.0]] * 2
    assert hop_protocol_command(3.0, 2).tolist() == [[0.0, 0.0, 1.0]] * 2
    assert hop_protocol_command(6.0, 2).tolist() == [[0.0, 0.0, 0.0]] * 2


def test_hop_metrics_count_stance_hops_swing_touchdowns_and_the_final_still_stand():
    # 0.1 s steps over 8 s. env 0: lands 6 times during the hop window, then stands still on one foot.
    # env 1: lands twice and touches the floor with its swing foot once.
    dt = 0.1
    metrics = HopEpisodeMetrics(num_envs=2, step_dt=dt)
    for k in range(80):
        t = round((k + 1) * dt, 6)
        hopping = 2.2 <= t <= 5.2
        landing0 = hopping and round(t * 10) % 5 == 0
        landing1 = hopping and t in (3.0, 4.0)
        metrics.update(
            time=t,
            stance_contact=torch.tensor([not hopping or landing0, not hopping or landing1]),
            stance_landing=torch.tensor([landing0, landing1]),
            stance_last_air_time=torch.tensor([0.4, 0.9]),
            swing_contact=torch.tensor([False, t == 4.5]),
            done=torch.tensor([False, False]),
            time_out=torch.tensor([False, False]),
        )
    summary = metrics.summary()
    assert summary["hops"].tolist() == [6, 2]
    assert summary["swing_touchdowns"].tolist() == [0, 1]
    # Still window 5.5-8.0 s: 26 steps of 0.1 s.
    torch.testing.assert_close(summary["still_single_support"], torch.tensor([2.6, 2.6]))
    assert summary["success"].tolist() == [True, False]
