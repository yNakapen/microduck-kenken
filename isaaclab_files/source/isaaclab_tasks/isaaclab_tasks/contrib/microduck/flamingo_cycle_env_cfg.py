# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flamingo cycle: stand, lift either foot on command, hold, lower, with one policy; microduck_rl's stage 2.

A port of ``Mjlab-FlamingoCycle-Flat-MicroDuck`` (``src/mjlab_microduck/tasks/microduck_flamingo_cycle_env_cfg.py``
on the microduck_rl ``flamingo`` branch, ``4757d6ae``). The twist slots carry ``[flag, side, 0]`` from
:class:`~isaaclab_tasks.contrib.microduck.mdp.FlamingoCommand`; the rewards follow its slewed blend ``alpha`` from
the two-foot to the one-foot targets. Everything else is the flamingo stage-1 task's.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

from . import mdp
from .flamingo_env_cfg import (
    FLAMINGO_MAX_PUSH,
    FLAMINGO_STANCE_SIDE_TILT,
    FLAMINGO_SWING_FOOT_STD,
    FLAMINGO_SWING_FOOT_TARGET_Z,
    FlamingoCommandsCfg,
    FlamingoCurriculumCfg,
    FlamingoEventsCfg,
    MicroDuckFlamingoFlatEnvCfg,
)
from .flamingo_pose import (
    FLAMINGO_BASE_PITCH,
    FLAMINGO_BASE_ROLL,
    FLAMINGO_GRAVITY_B,
    FLAMINGO_POSE_LEFT,
    FLAMINGO_POSE_RIGHT,
    FLAMINGO_Z,
)
from .flat_env_cfg import _schedule

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=["trunk_base"])
_FEET_CFG = SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"], preserve_order=True)
_FEET_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=["ankle_left", "ankle_right"], preserve_order=True)
_COMMAND = "base_velocity"

FLAMINGO_CYCLE_DWELL_S = (2.5, 5.0)
"""Time [s] in each commanded posture before the flag may flip; longer than the 1.5 s ramp."""


@configclass
class FlamingoCycleCommandsCfg(FlamingoCommandsCfg):
    """``[flag, side, 0]`` in the twist slots; the head and body commands of the stage-1 task."""

    base_velocity = mdp.FlamingoCommandCfg(
        resampling_time_range=FLAMINGO_CYCLE_DWELL_S, flamingo_prob=0.6, ramp_s=1.5, zero_side_prob=0.5
    )


@configclass
class FlamingoCycleRewardsCfg:
    """microduck_rl's cycle rewards; its non-positive terms are costs with negative weights here."""

    # The weight shift leads: the CoM target slides from the feet midpoint to the stance foot.
    com_target = RewTerm(
        func=mdp.cycle_com_target, weight=3.0, params={"command_name": _COMMAND, "feet_cfg": _FEET_CFG, "std": 0.03}
    )
    stance_foot_grounded = RewTerm(
        func=mdp.cycle_stance_foot_grounded,
        weight=1.0,
        params={"command_name": _COMMAND, "feet_sensor_cfg": _FEET_SENSOR_CFG},
    )
    # +1 while standing is due, -1 once the lift is due: mixed sign by design.
    swing_foot_contact = RewTerm(
        func=mdp.cycle_swing_foot_contact,
        weight=0.5,
        params={"command_name": _COMMAND, "feet_sensor_cfg": _FEET_SENSOR_CFG, "lo": 0.4, "hi": 0.9},
    )
    swing_foot_clear = RewTerm(
        func=mdp.cycle_swing_foot_clear,
        weight=1.5,
        params={
            "command_name": _COMMAND,
            "feet_cfg": _FEET_CFG,
            "feet_sensor_cfg": _FEET_SENSOR_CFG,
            "target": FLAMINGO_SWING_FOOT_TARGET_Z,
            "std": FLAMINGO_SWING_FOOT_STD,
            "lo": 0.6,
            "hi": 1.0,
        },
    )
    pose_track = RewTerm(
        func=mdp.cycle_pose_track,
        weight=1.5,
        params={
            "command_name": _COMMAND,
            "pose_right": FLAMINGO_POSE_RIGHT,
            "pose_left": FLAMINGO_POSE_LEFT,
            "std": 0.5,
            "mid_attenuation": 0.75,
            "asset_cfg": _SERVO_JOINT_CFG,
        },
    )
    gravity_track = RewTerm(
        func=mdp.cycle_gravity_track,
        weight=2.0,
        params={"command_name": _COMMAND, "gravity_right": FLAMINGO_GRAVITY_B, "std": 0.15},
    )
    stillness = RewTerm(
        func=mdp.cycle_stillness,
        weight=1.0,
        params={
            "command_name": _COMMAND,
            "feet_sensor_cfg": _FEET_SENSOR_CFG,
            "std": 2.0,
            "asset_cfg": _SERVO_JOINT_CFG,
        },
    )
    stance_side_tilt = RewTerm(
        func=mdp.cycle_stance_side_tilt_cost,
        weight=-4.0,
        params={"command_name": _COMMAND, "threshold": FLAMINGO_STANCE_SIDE_TILT},
    )
    # Logged success indicator; a zero weight would skip the term entirely.
    commanded_support = RewTerm(
        func=mdp.cycle_single_support_success,
        weight=1e-3,
        params={"command_name": _COMMAND, "feet_sensor_cfg": _FEET_SENSOR_CFG},
    )
    joint_limit_proximity = RewTerm(
        func=mdp.joint_limit_proximity, weight=-1.0, params={"margin": 0.1, "asset_cfg": _SERVO_JOINT_CFG}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_torque_rate_l2 = RewTerm(func=mdp.joint_torque_rate_l2, weight=0.0, params={"asset_cfg": _SERVO_JOINT_CFG})
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    self_collisions = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("self_collision")}
    )


@configclass
class FlamingoCycleEventsCfg(FlamingoEventsCfg):
    """Spawn in either side's one-foot pose or standing at HOME, with the side pinned into the command."""

    reset_flamingo_pose = None
    reset_flamingo_cycle_pose = EventTerm(
        func=mdp.reset_flamingo_cycle_pose,
        mode="reset",
        params={
            "pose_right": FLAMINGO_POSE_RIGHT,
            "pose_left": FLAMINGO_POSE_LEFT,
            "base_roll": FLAMINGO_BASE_ROLL,
            "base_pitch": FLAMINGO_BASE_PITCH,
            "z_range": (FLAMINGO_Z, FLAMINGO_Z + 0.01),
            "standing_z_range": (0.11, 0.12),
            "command_name": _COMMAND,
            "in_pose_prob": 0.6,
            "xy_range": 0.5,
            "tilt_noise": math.radians(3.0),
            "joint_noise_std": 0.05,
            "asset_cfg": _SERVO_JOINT_CFG,
        },
    )


@configclass
class FlamingoCycleCurriculumCfg(FlamingoCurriculumCfg):
    """microduck_rl's cycle schedules: transitions are harder than the hold, so the ramps come later."""

    push_magnitude = _schedule(
        "events.push_robot.params.velocity_range",
        (500, {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}),
        (1000, {"x": (-FLAMINGO_MAX_PUSH, FLAMINGO_MAX_PUSH), "y": (-FLAMINGO_MAX_PUSH, FLAMINGO_MAX_PUSH)}),
    )
    action_rate_weight = _schedule("rewards.action_rate_l2.weight", (600, -0.5))
    torque_rate_weight = _schedule("rewards.joint_torque_rate_l2.weight", (800, -1e-3))
    com_std = _schedule("rewards.com_target.params.std", (700, 0.02))
    in_pose_prob = _schedule("events.reset_flamingo_cycle_pose.params.in_pose_prob", (600, 0.4), (1200, 0.3))


@configclass
class MicroDuckFlamingoCycleFlatEnvCfg(MicroDuckFlamingoFlatEnvCfg):
    """MicroDuck stand and one-foot cycle on either foot with Newton MJWarp and BAM servos."""

    commands: FlamingoCycleCommandsCfg = FlamingoCycleCommandsCfg()
    rewards: FlamingoCycleRewardsCfg = FlamingoCycleRewardsCfg()
    events: FlamingoCycleEventsCfg = FlamingoCycleEventsCfg()
    curriculum: FlamingoCycleCurriculumCfg = FlamingoCycleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
