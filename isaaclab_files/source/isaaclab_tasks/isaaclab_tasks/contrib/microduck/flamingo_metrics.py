# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode metrics of the MicroDuck flamingo evaluation.

microduck_rl's stage-1 success metric (``docs/superpowers/specs/2026-08-28-flamingo-design.md`` on the ``flamingo``
branch) counts an episode as a success when the robot holds single support, stance foot down and swing foot clear,
for at least 5 s without a hard fall. Here the hold is the longest uninterrupted single-support streak, and a fall is
any termination other than a time-out.
"""

from __future__ import annotations

import torch

FLAMINGO_SUCCESS_HOLD = 5.0
"""Single-support hold [s] microduck_rl's stage-1 success metric asks for."""


def set_eval_conditions(cfg, push: float) -> None:
    """Freeze a flamingo task config ``cfg`` at its final randomization with pushes of ``push`` [m/s].

    The schedules are removed, so the CoM randomization takes its last curriculum range and the interval pushes
    sample x and y velocities within ``push``.
    """
    for name in list(cfg.curriculum.to_dict()):
        setattr(cfg.curriculum, name, None)
    cfg.events.push_robot.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
    cfg.events.randomize_com.params["com_range"] = {axis: (-0.015, 0.015) for axis in "xyz"}
    cfg.events.randomize_head_com.params["com_range"] = {axis: (-0.01, 0.01) for axis in "xyz"}


def fall_side(gravity_b: torch.Tensor, pose_gravity_b: torch.Tensor) -> list[str]:
    """Label each trunk-frame gravity direction by its largest horizontal departure from the pose's.

    For the right-foot pose, ``g_b,y`` falling below the pose value rolls over the stance foot (``"stance"``) and
    rising above it rolls toward the lifted foot (``"swing"``); ``g_b,x`` above or below the pose value is a fall
    backward or forward.
    """
    delta = gravity_b[:, :2] - pose_gravity_b[:2].to(gravity_b)
    labels = []
    for dx, dy in delta.tolist():
        if abs(dy) >= abs(dx):
            labels.append("stance" if dy < 0.0 else "swing")
        else:
            labels.append("backward" if dx > 0.0 else "forward")
    return labels


class FlamingoEpisodeMetrics:
    """Accumulate the first episode of each environment; later episodes after an automatic reset are ignored."""

    def __init__(
        self,
        num_envs: int,
        step_dt: float,
        success_hold: float = FLAMINGO_SUCCESS_HOLD,
        pose_gravity_b: tuple[float, float, float] = (-0.154, -0.379, -0.912),
        device: str = "cpu",
    ):
        self.step_dt = step_dt
        self.success_hold = success_hold
        self._pose_gravity_b = torch.tensor(pose_gravity_b, device=device)
        self._active = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._streak = torch.zeros(num_envs, device=device)
        self._longest = torch.zeros(num_envs, device=device)
        self._duration = torch.zeros(num_envs, device=device)
        self._swing_time = torch.zeros(num_envs, device=device)
        self._touchdowns = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._swing_was_down = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._last_gravity = torch.zeros(num_envs, 3, device=device)
        self._fall_side = [""] * num_envs

    def update(
        self,
        stance_contact: torch.Tensor,
        swing_contact: torch.Tensor,
        done: torch.Tensor,
        time_out: torch.Tensor,
        gravity_b: torch.Tensor,
    ) -> None:
        """Record one step: foot contacts, the step's terminal flags and the trunk-frame gravity after the step.

        The gravity of a terminal step is read after the automatic reset, so falls are labelled from the gravity of
        the step before.
        """
        active = self._active
        single = stance_contact & ~swing_contact
        self._streak = torch.where(active & single, self._streak + self.step_dt, torch.zeros_like(self._streak))
        self._longest = torch.where(active, torch.maximum(self._longest, self._streak), self._longest)
        self._duration += active.float() * self.step_dt
        self._swing_time += (active & swing_contact).float() * self.step_dt
        self._touchdowns += (active & swing_contact & ~self._swing_was_down).long()
        self._swing_was_down = swing_contact

        fell = active & done & ~time_out
        if fell.any():
            for i, label in zip(
                fell.nonzero().flatten().tolist(), fall_side(self._last_gravity[fell], self._pose_gravity_b)
            ):
                self._fall_side[i] = label
        self._fell |= fell
        self._last_gravity = torch.where(active.unsqueeze(-1), gravity_b, self._last_gravity)
        self._active = active & ~done

    def summary(self) -> dict:
        """Per-environment results of the first episode."""
        return {
            "fell": self._fell.clone(),
            "fall_side": list(self._fall_side),
            "duration": self._duration.clone(),
            "longest_single_support": self._longest.clone(),
            "swing_contact_time": self._swing_time.clone(),
            "swing_touchdowns": self._touchdowns.clone(),
            "success": ~self._fell & (self._longest >= self.success_hold - 1e-6),
        }


"""
Flamingo cycle.
"""

CYCLE_LIFT_TIME = 2.0
"""Protocol time [s] at which one foot is commanded, after standing on both."""
CYCLE_LOWER_TIME = 7.0
"""Protocol time [s] at which both feet are commanded again."""
CYCLE_HOLD_WINDOW = (4.0, 7.0)
"""Protocol window [s] scored as the one-foot hold: the 1.5 s blend has finished and 0.5 s passed."""
CYCLE_STAND_WINDOW = (9.0, 10.0)
"""Protocol window [s] scored as the final two-foot stand, after the blend back and 0.5 s."""
CYCLE_MIN_FRACTION = 0.9
"""Fraction of each scored window the commanded posture must hold for a success."""


def cycle_protocol_command(time: float, side: torch.Tensor) -> torch.Tensor:
    """Twist-slot command ``[flag, side, 0]`` of the cycle protocol at ``time`` [s] for stance sides ``side`` (±1).

    The robot idles on both feet with the deployed all-zero command, lifts the foot opposite ``side`` at
    :data:`CYCLE_LIFT_TIME` and lowers it at :data:`CYCLE_LOWER_TIME`.
    """
    command = torch.zeros(len(side), 3, device=side.device)
    if CYCLE_LIFT_TIME <= time < CYCLE_LOWER_TIME:
        command[:, 0] = 1.0
        command[:, 1] = side
    return command


class CycleEpisodeMetrics:
    """Score one cycle-protocol episode per environment; steps after a termination are ignored.

    A success holds single support (stance foot down, swing foot up) for :data:`CYCLE_MIN_FRACTION` of the hold
    window, both feet down for that fraction of the final stand window, and does not fall.
    """

    def __init__(self, num_envs: int, step_dt: float, device: str = "cpu"):
        self.step_dt = step_dt
        zeros = lambda dtype=torch.float32: torch.zeros(num_envs, dtype=dtype, device=device)  # noqa: E731
        self._active = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._fell = zeros(torch.bool)
        self._hold_steps, self._hold_single, self._hold_clear = zeros(), zeros(), zeros()
        self._stand_steps, self._stand_two = zeros(), zeros()

    def update(
        self,
        time: float,
        stance_contact: torch.Tensor,
        swing_contact: torch.Tensor,
        swing_height: torch.Tensor,
        done: torch.Tensor,
        time_out: torch.Tensor,
        clear_height: float = 0.03,
    ) -> None:
        """Record the step ending at protocol ``time`` [s]; ``swing_height`` is the swing-foot site height [m]."""
        active = self._active
        if CYCLE_HOLD_WINDOW[0] <= time <= CYCLE_HOLD_WINDOW[1]:
            self._hold_steps += active.float()
            self._hold_single += (active & stance_contact & ~swing_contact).float()
            self._hold_clear += (active & (swing_height > clear_height)).float()
        if CYCLE_STAND_WINDOW[0] <= time <= CYCLE_STAND_WINDOW[1]:
            self._stand_steps += active.float()
            self._stand_two += (active & stance_contact & swing_contact).float()
        self._fell |= active & done & ~time_out
        self._active = active & ~done

    def summary(self) -> dict[str, torch.Tensor]:
        """Per-environment fractions of the scored windows, falls and successes."""
        hold = self._hold_single / self._hold_steps.clamp(min=1.0)
        stand = self._stand_two / self._stand_steps.clamp(min=1.0)
        return {
            "fell": self._fell.clone(),
            "hold_single_support": hold,
            "hold_swing_clear_frac": self._hold_clear / self._hold_steps.clamp(min=1.0),
            "stand_two_feet": stand,
            "success": ~self._fell
            & (hold >= CYCLE_MIN_FRACTION - 1e-6)
            & (stand >= CYCLE_MIN_FRACTION - 1e-6)
            & (self._hold_steps > 0)
            & (self._stand_steps > 0),
        }


"""
Flamingo hop.
"""

HOP_START_TIME = 2.0
"""Protocol time [s] at which hopping is commanded, after a still one-foot stand."""
HOP_STOP_TIME = 5.0
"""Protocol time [s] at which a still one-foot stand is commanded again."""
HOP_COUNT_WINDOW = (2.0, 5.3)
"""Protocol window [s] whose stance-foot landings count as hops; it covers the 0.3 s blend back."""
HOP_STILL_WINDOW = (5.5, 8.0)
"""Protocol window [s] scored as the final still stand."""
HOP_MIN_AIR_TIME = 0.02
"""Shortest stance-foot flight [s] that counts as a hop, as in :mod:`~isaaclab_tasks.contrib.microduck.hop_metrics`."""
HOP_MIN_HOPS = 5
"""Hops a non-fallen episode must make."""
HOP_MIN_STILL = 2.0
"""Uninterrupted single support [s] a non-fallen episode must hold in the still window."""
HOP_MAX_FALL_FRACTION = 0.2
"""Largest fraction of episodes that may fall."""


def hop_protocol_command(time: float, num_envs: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Twist-slot command ``[0, 0, hop]`` of the hop protocol at ``time`` [s]: still, hopping, still."""
    command = torch.zeros(num_envs, 3, device=device)
    if HOP_START_TIME <= time < HOP_STOP_TIME:
        command[:, 2] = 1.0
    return command


class HopEpisodeMetrics:
    """Score one hop-protocol episode per environment; steps after a termination are ignored.

    A hop is a landing of the stance foot after a flight of at least :data:`HOP_MIN_AIR_TIME` within
    :data:`HOP_COUNT_WINDOW`; a swing touchdown is any new swing-foot contact. A non-fallen episode succeeds with at
    least :data:`HOP_MIN_HOPS` hops, no swing touchdown and :data:`HOP_MIN_STILL` of uninterrupted single support in
    :data:`HOP_STILL_WINDOW`.
    """

    def __init__(self, num_envs: int, step_dt: float, device: str = "cpu"):
        self.step_dt = step_dt
        self._active = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._hops = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._touchdowns = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._swing_was_down = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._streak = torch.zeros(num_envs, device=device)
        self._still = torch.zeros(num_envs, device=device)

    def update(
        self,
        time: float,
        stance_contact: torch.Tensor,
        stance_landing: torch.Tensor,
        stance_last_air_time: torch.Tensor,
        swing_contact: torch.Tensor,
        done: torch.Tensor,
        time_out: torch.Tensor,
    ) -> None:
        """Record the step ending at protocol ``time`` [s]; ``stance_landing`` flags a stance-foot touchdown this step
        and ``stance_last_air_time`` [s] its preceding flight."""
        active = self._active
        if HOP_COUNT_WINDOW[0] <= time <= HOP_COUNT_WINDOW[1]:
            self._hops += (active & stance_landing & (stance_last_air_time >= HOP_MIN_AIR_TIME)).long()
        self._touchdowns += (active & swing_contact & ~self._swing_was_down).long()
        self._swing_was_down = swing_contact.clone()
        if HOP_STILL_WINDOW[0] <= time <= HOP_STILL_WINDOW[1] + 1e-6:
            single = active & stance_contact & ~swing_contact
            self._streak = torch.where(single, self._streak + self.step_dt, torch.zeros_like(self._streak))
            self._still = torch.maximum(self._still, self._streak)
        self._fell |= active & done & ~time_out
        self._active = active & ~done

    def summary(self) -> dict[str, torch.Tensor]:
        """Per-environment hops, swing touchdowns, final still support [s], falls and successes."""
        return {
            "fell": self._fell.clone(),
            "hops": self._hops.clone(),
            "swing_touchdowns": self._touchdowns.clone(),
            "still_single_support": self._still.clone(),
            "success": ~self._fell
            & (self._hops >= HOP_MIN_HOPS)
            & (self._touchdowns == 0)
            & (self._still >= HOP_MIN_STILL - 1e-6),
        }
