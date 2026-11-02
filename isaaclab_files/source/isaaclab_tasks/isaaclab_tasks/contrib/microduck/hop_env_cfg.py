# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-leg stand on the right foot, the first stage towards in-place hopping.

The task keeps the walking task's scene, servos, randomization and 61-wide actor input, so a stand policy can be
swapped in on the robot. The twist, head and body command slots stay narrow but non-zero for the same reason.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .flat_env_cfg import (
    MICRODUCK_SOLE_TO_ANKLE_OFFSET,
    CommandsCfg,
    CurriculumCfg,
    MicroDuckVelocityFlatEnvCfg,
    RewardsCfg,
)

MICRODUCK_SUPPORT_FOOT = "ankle_right"
MICRODUCK_LIFTED_FOOT = "ankle_left"
MICRODUCK_LIFTED_FOOT_CLEARANCE = 0.04
"""Target sole clearance of the lifted foot [m]."""
MICRODUCK_SINGLE_LEG_MIN_TRUNK_HEIGHT = 0.08
"""Lowest trunk height [m] that still counts as standing; the default stance is about 0.125 m."""

_SUPPORT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=[MICRODUCK_SUPPORT_FOOT])
_LIFTED_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=[MICRODUCK_LIFTED_FOOT])
_LIFTED_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_LIFTED_FOOT])


@configclass
class HopStandCommandsCfg(CommandsCfg):
    """Near-zero twist with a zero-command bucket; head and body commands as in walking's first stage."""

    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.1,
        debug_vis=False,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.01, 0.01), lin_vel_y=(-0.01, 0.01), ang_vel_z=(-0.05, 0.05)
        ),
    )


@configclass
class HopStandRewardsCfg(RewardsCfg):
    """Stand on the right foot with the left foot held up, level and in place."""

    single_support = RewTerm(
        func=mdp.single_support,
        weight=2.0,
        params={
            "lifted_sensor_cfg": _LIFTED_SENSOR_CFG,
            "support_sensor_cfg": _SUPPORT_SENSOR_CFG,
            "min_height": MICRODUCK_SINGLE_LEG_MIN_TRUNK_HEIGHT,
        },
    )
    lifted_foot_height = RewTerm(
        func=mdp.lifted_foot_height,
        weight=2.0,
        params={
            "target_height": MICRODUCK_LIFTED_FOOT_CLEARANCE + MICRODUCK_SOLE_TO_ANKLE_OFFSET,
            "std": 0.02,
            "asset_cfg": _LIFTED_BODY_CFG,
            "support_sensor_cfg": _SUPPORT_SENSOR_CFG,
            "min_height": MICRODUCK_SINGLE_LEG_MIN_TRUNK_HEIGHT,
        },
    )
    lifted_foot_contact = RewTerm(func=mdp.lifted_foot_contact, weight=-1.0, params={"sensor_cfg": _LIFTED_SENSOR_CFG})

    # Walking-gait terms do not apply; the default-pose prior would pull the left foot down.
    pose = None
    air_time = None
    foot_clearance = None
    foot_swing_height = None

    def __post_init__(self):
        # Hold the place and heading at any command, including the near-zero ones.
        self.track_lin_vel.weight = 2.0
        self.track_ang_vel.weight = 1.0
        self.foot_slip.params["command_threshold"] = -1.0


@configclass
class HopStandCurriculumCfg(CurriculumCfg):
    """Keep the CoM randomization and action-rate schedules; head and twist commands stay narrow."""

    head_pose_bias_weight = None
    standing_envs = None
    head_pose_range = None


@configclass
class MicroDuckHopStandFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """MicroDuck single-leg stand on the right foot with Newton MJWarp and BAM servos."""

    commands: HopStandCommandsCfg = HopStandCommandsCfg()
    rewards: HopStandRewardsCfg = HopStandRewardsCfg()
    curriculum: HopStandCurriculumCfg = HopStandCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Single support tolerates smaller shoves than walking.
        self.events.push_robot.params["velocity_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
