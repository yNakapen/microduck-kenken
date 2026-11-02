# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hopping in place on the right foot of the flamingo pose, then holding still, under pushes.

Fine-tunes the flamingo stage-1 policy: the robot spawns in the one-foot pose, and the twist slots carry
``[0, 0, hop]`` from :class:`~isaaclab_tasks.contrib.microduck.mdp.HopCommand`, alternating a still stand and hopping.
While hopping the stance foot must leave the floor and land again with the lifted foot up from before the takeoff
(40 to 120 ms flights are paid), may not stay down longer than 0.4 s, and the trunk must stay in place; the
still-stand terms fade out over the 0.3 s blend and back.
The stage-1 schedules start at their final values, with 0.15 m/s pushes throughout.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .flamingo_env_cfg import (
    FLAMINGO_MAX_PUSH,
    FlamingoCommandsCfg,
    FlamingoEventsCfg,
    FlamingoRewardsCfg,
    MicroDuckFlamingoFlatEnvCfg,
)

_COMMAND = "base_velocity"
_STANCE_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=["ankle_right"])
_SWING_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=["ankle_left"])

FLAMINGO_HOP_MIN_AIR_TIME = 0.04
"""Flight [s] below which a stance-foot landing earns nothing; contact flicker stays under it."""
FLAMINGO_HOP_MAX_AIR_TIME = 0.12
"""Flight [s] earning the full hop reward, about the 7.4 V jump estimate of the planar hop model (105 ms)."""


@configclass
class FlamingoHopCommandsCfg(FlamingoCommandsCfg):
    """``[0, 0, hop]`` in the twist slots; the head and body commands of the stage-1 task."""

    base_velocity = mdp.HopCommandCfg(
        resampling_time_range=(1.5, 3.0), still_time_range=(1.5, 3.0), hop_time_range=(2.0, 4.0), ramp_s=0.3
    )


@configclass
class FlamingoHopRewardsCfg(FlamingoRewardsCfg):
    """Stage-1 rewards at their final schedule values, the still-stand terms faded by the hop blend, and hop terms."""

    hop_air_time = RewTerm(
        func=mdp.hop_air_time,
        weight=30.0,
        params={
            "command_name": _COMMAND,
            "sensor_cfg": _STANCE_SENSOR_CFG,
            "min_air_time": FLAMINGO_HOP_MIN_AIR_TIME,
            "max_air_time": FLAMINGO_HOP_MAX_AIR_TIME,
            # The swing foot must be up from 0.1 s before the takeoff to the landing: the first policy pushed off with
            # it within 60 ms before 93 % of its takeoffs.
            "swing_sensor_cfg": _SWING_SENSOR_CFG,
            "swing_lead_time": 0.1,
        },
    )
    hop_ground_time = RewTerm(
        func=mdp.hop_ground_time_cost,
        weight=-2.0,
        params={"command_name": _COMMAND, "sensor_cfg": _STANCE_SENSOR_CFG, "max_contact_time": 0.4},
    )
    # Grows with the time spent standing on one foot before the next takeoff and resets when the foot leaves the floor.
    hop_wait = RewTerm(
        func=mdp.hop_wait_cost,
        weight=-2.0,
        params={"command_name": _COMMAND, "sensor_cfg": _STANCE_SENSOR_CFG},
    )
    hop_drift = RewTerm(func=mdp.hop_drift_cost, weight=-4.0, params={"command_name": _COMMAND, "yaw_weight": 0.1})

    def __post_init__(self):
        self.stance_foot_grounded.func = mdp.still_foot_in_contact
        self.stance_foot_grounded.params = {"command_name": _COMMAND, **self.stance_foot_grounded.params}
        self.stillness.func = mdp.still_joint_vel_gaussian
        self.stillness.params = {"command_name": _COMMAND, **self.stillness.params}
        # Terms that pay for keeping still are off while hopping, too.
        self.swing_foot_clear.func = mdp.still_foot_site_height_gaussian
        self.swing_foot_clear.params = {"command_name": _COMMAND, **self.swing_foot_clear.params}
        self.pose_flamingo.func = mdp.still_pose_target_match
        self.pose_flamingo.params = {"command_name": _COMMAND, **self.pose_flamingo.params}
        self.gravity_flamingo.func = mdp.still_projected_gravity_match
        self.gravity_flamingo.params = {"command_name": _COMMAND, **self.gravity_flamingo.params}
        self.action_rate_l2.func = mdp.still_action_rate_l2
        self.action_rate_l2.params = {"command_name": _COMMAND}
        # microduck_rl's Ballet v2 triples the lifted-foot tax against a tapping or parked foot.
        self.swing_foot_touch.weight = -1.5
        # The stage-1 schedules at their final values.
        self.action_rate_l2.weight = -0.5
        self.joint_torque_rate_l2.weight = -1e-3
        self.com_over_stance_foot.params["std"] = 0.02


@configclass
class FlamingoHopEventsCfg(FlamingoEventsCfg):
    """Stage-1 spawn and randomization at the final CoM ranges, with full pushes from the start."""

    def __post_init__(self):
        push = FLAMINGO_MAX_PUSH
        self.push_robot.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        self.randomize_com.params["com_range"] = {axis: (-0.015, 0.015) for axis in "xyz"}
        self.randomize_head_com.params["com_range"] = {axis: (-0.01, 0.01) for axis in "xyz"}


@configclass
class FlamingoHopCurriculumCfg:
    """No schedules: the fine-tuning starts from the stage-1 final values."""


@configclass
class MicroDuckFlamingoHopFlatEnvCfg(MicroDuckFlamingoFlatEnvCfg):
    """MicroDuck hopping in place on the right foot and holding still, with Newton MJWarp and BAM servos."""

    commands: FlamingoHopCommandsCfg = FlamingoHopCommandsCfg()
    rewards: FlamingoHopRewardsCfg = FlamingoHopRewardsCfg()
    events: FlamingoHopEventsCfg = FlamingoHopEventsCfg()
    curriculum: FlamingoHopCurriculumCfg = FlamingoHopCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
