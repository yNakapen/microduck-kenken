# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    # actions
    "BiasedJointPositionAction",
    "BiasedJointPositionActionCfg",
    # commands
    "MicroDuckRoughVelocityCommand",
    "MicroDuckVelocityCommand",
    "MicroDuckVelocityCommandCfg",
    "UniformPoseDeltaCommand",
    "UniformPoseDeltaCommandCfg",
    # curriculums
    "microduck_terrain_levels",
    "staged_value",
    # events
    "randomize_bam_friction",
    "randomize_encoder_bias",
    # flamingo
    "MICRODUCK_FOOT_SITE_OFFSET",
    "com_over_foot",
    "foot_in_contact",
    "foot_site_height_gaussian",
    "foot_site_pos_w",
    "joint_limit_proximity",
    "joint_torque_rate_l2",
    "joint_vel_gaussian",
    "lateral_tilt_cost",
    "pose_target_match",
    "projected_gravity_match",
    "reset_flamingo_pose",
    "robot_com_pos_w",
    # flamingo cycle
    "FlamingoCommand",
    "FlamingoCommandCfg",
    "cycle_com_target",
    "cycle_gravity_track",
    "cycle_pose_track",
    "cycle_single_support_success",
    "cycle_stance_foot_grounded",
    "cycle_stance_side_tilt_cost",
    "cycle_stillness",
    "cycle_swing_foot_clear",
    "cycle_swing_foot_contact",
    "reset_flamingo_cycle_pose",
    # flamingo hop
    "HopCommand",
    "HopCommandCfg",
    "hop_air_time",
    "hop_drift_cost",
    "hop_ground_time_cost",
    "hop_wait_cost",
    "still_action_rate_l2",
    "still_foot_in_contact",
    "still_foot_site_height_gaussian",
    "still_joint_vel_gaussian",
    "still_pose_target_match",
    "still_projected_gravity_match",
    # observations
    "ImuNoiseCfg",
    "foot_air_time_safe",
    "foot_contact",
    "foot_contact_forces_safe",
    "foot_height_safe",
    "joint_pos_rel_biased",
    "joint_vel_rel_backlash",
    "misaligned_imu",
    # rewards
    "angular_momentum_l2",
    "body_ang_vel_xy_l2",
    "feet_air_time_windowed",
    "foot_clearance",
    "foot_slip",
    "foot_swing_height",
    "head_pose_bias_penalty",
    "head_pose_tracking",
    "joint_deviation_exp",
    "joint_pos_limits_finite",
    "lifted_foot_contact",
    "lifted_foot_height",
    "self_collision_cost",
    "single_support",
    "track_angular_velocity",
    "track_angular_velocity_world",
    "track_linear_velocity",
    "track_linear_velocity_heading",
    "upright",
    # terminations
    "robot_state_is_nan",
]

from .actions import BiasedJointPositionAction, BiasedJointPositionActionCfg
from .commands import (
    MicroDuckRoughVelocityCommand,
    MicroDuckVelocityCommand,
    MicroDuckVelocityCommandCfg,
    UniformPoseDeltaCommand,
    UniformPoseDeltaCommandCfg,
)
from .curriculums import microduck_terrain_levels, staged_value
from .events import randomize_bam_friction, randomize_encoder_bias
from .flamingo import (
    MICRODUCK_FOOT_SITE_OFFSET,
    com_over_foot,
    foot_in_contact,
    foot_site_height_gaussian,
    foot_site_pos_w,
    joint_limit_proximity,
    joint_torque_rate_l2,
    joint_vel_gaussian,
    lateral_tilt_cost,
    pose_target_match,
    projected_gravity_match,
    reset_flamingo_pose,
    robot_com_pos_w,
)
from .flamingo_cycle import (
    FlamingoCommand,
    FlamingoCommandCfg,
    cycle_com_target,
    cycle_gravity_track,
    cycle_pose_track,
    cycle_single_support_success,
    cycle_stance_foot_grounded,
    cycle_stance_side_tilt_cost,
    cycle_stillness,
    cycle_swing_foot_clear,
    cycle_swing_foot_contact,
    reset_flamingo_cycle_pose,
)
from .flamingo_hop import (
    HopCommand,
    HopCommandCfg,
    hop_air_time,
    hop_drift_cost,
    hop_ground_time_cost,
    hop_wait_cost,
    still_action_rate_l2,
    still_foot_in_contact,
    still_foot_site_height_gaussian,
    still_joint_vel_gaussian,
    still_pose_target_match,
    still_projected_gravity_match,
)
from .observations import (
    ImuNoiseCfg,
    foot_air_time_safe,
    foot_contact,
    foot_contact_forces_safe,
    foot_height_safe,
    joint_pos_rel_biased,
    joint_vel_rel_backlash,
    misaligned_imu,
)
from .rewards import (
    angular_momentum_l2,
    body_ang_vel_xy_l2,
    feet_air_time_windowed,
    foot_clearance,
    foot_slip,
    foot_swing_height,
    head_pose_bias_penalty,
    head_pose_tracking,
    joint_deviation_exp,
    joint_pos_limits_finite,
    lifted_foot_contact,
    lifted_foot_height,
    self_collision_cost,
    single_support,
    track_angular_velocity,
    track_angular_velocity_world,
    track_linear_velocity,
    track_linear_velocity_heading,
    upright,
)
from .terminations import robot_state_is_nan

from isaaclab.envs.mdp import *
