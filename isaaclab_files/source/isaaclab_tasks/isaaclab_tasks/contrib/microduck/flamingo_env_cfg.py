# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-foot "flamingo" balance on the right foot, microduck_rl's stage 1 on Newton MJWarp.

A port of ``Mjlab-Flamingo-Flat-MicroDuck`` (``src/mjlab_microduck/tasks/microduck_flamingo_env_cfg.py`` on the
microduck_rl ``flamingo`` branch, ``4757d6ae``). The robot spawns in the noised one-foot pose with a near-zero
command and must hold it under growing pushes. Like microduck_rl it runs the all-collisions robot without gearbox
play, MuJoCo Warp's own contacts and no BAM current limit; it keeps the walking task's 61-wide actor input,
randomization and terminations.

It carries the stage-1 fixes microduck_rl made in its Cycle task: the swing-foot clearance target 0.10 +- 0.06 m
instead of 0.05 +- 0.03 m (``c008a7f3``), and pushes capped at 0.15 m/s with the action-rate weight capped at -0.5
(``e4a3ca5f``), because the 0.25 m/s stage on top of the -1.0 weight halved the stage-1 reward for good.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_CFG, MICRODUCK_JOINT_NAMES

from . import mdp
from .flamingo_pose import FLAMINGO_BASE_PITCH, FLAMINGO_BASE_ROLL, FLAMINGO_GRAVITY_B, FLAMINGO_POSE_RIGHT, FLAMINGO_Z
from .flat_env_cfg import (
    CommandsCfg,
    CurriculumCfg,
    EventsCfg,
    MicroDuckSceneCfg,
    MicroDuckVelocityFlatEnvCfg,
    ObservationsCfg,
    _schedule,
)

_SERVOS = MICRODUCK_ALLCOLLISIONS_CFG.actuators["servos"]
MICRODUCK_FLAMINGO_ROBOT_CFG = MICRODUCK_ALLCOLLISIONS_CFG.replace(
    actuators={"servos": _SERVOS.replace(motor=_SERVOS.motor.replace(max_current=0.0))}
)
"""The all-collisions MicroDuck without the BAM current limit, which microduck_rl leaves commented out."""

_ROBOT_GEOMETRY = "{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*)?"
_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=["trunk_base"])
_STANCE_FOOT_CFG = SceneEntityCfg("robot", body_names=["ankle_right"])
_SWING_FOOT_CFG = SceneEntityCfg("robot", body_names=["ankle_left"])
_STANCE_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=["ankle_right"])
_SWING_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=["ankle_left"])

FLAMINGO_STANCE_SIDE_TILT = 0.45
"""Trunk-frame gravity ``-g_b,y`` beyond which the trunk rolls further toward the stance side than the pose (0.38)."""
FLAMINGO_SWING_FOOT_TARGET_Z = 0.10
"""Swing-foot site height target [m] of microduck_rl's Cycle task; its stage-1 0.05 m contradicted the pose's 0.17 m."""
FLAMINGO_SWING_FOOT_STD = 0.06
"""Swing-foot height tolerance [m]: pays 0.5 at 5 cm, 1.0 at 10 cm and 0.26 at the pose's 17 cm."""
FLAMINGO_MAX_PUSH = 0.15
"""Final push magnitude [m/s]; microduck_rl's stage-1 run collapsed at its 0.25 m/s stage."""


@configclass
class FlamingoSceneCfg(MicroDuckSceneCfg):
    """The all-collisions robot with self-contacts sensed between any two bodies."""

    robot = MICRODUCK_FLAMINGO_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    self_collision = ContactSensorCfg(prim_path=_ROBOT_GEOMETRY, filter_prim_paths_expr=[_ROBOT_GEOMETRY])


@configclass
class FlamingoObservationsCfg(ObservationsCfg):
    """The walking observations without the critic's foot heights, which microduck_rl drops."""

    def __post_init__(self):
        self.critic.foot_height = None


@configclass
class FlamingoCommandsCfg(CommandsCfg):
    """microduck_rl's near-zero twist resampled every 6 to 12 s, the walking head command and a zero body pose."""

    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 12.0),
        rel_standing_envs=0.0,
        debug_vis=False,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.01, 0.01), lin_vel_y=(-0.01, 0.01), ang_vel_z=(-0.05, 0.05)
        ),
    )
    body_pose = mdp.UniformPoseDeltaCommandCfg(resampling_time_range=(2.0, 5.0), ranges=((0.0, 0.0),) * 6)


@configclass
class FlamingoRewardsCfg:
    """microduck_rl's stage-1 rewards; its non-positive terms are costs with negative weights here."""

    # Balance: the whole-robot CoM over the stance foot site; std tightened to 0.02 by the curriculum.
    com_over_stance_foot = RewTerm(
        func=mdp.com_over_foot, weight=3.0, params={"std": 0.03, "asset_cfg": _STANCE_FOOT_CFG}
    )
    stance_foot_grounded = RewTerm(func=mdp.foot_in_contact, weight=1.0, params={"sensor_cfg": _STANCE_SENSOR_CFG})
    swing_foot_clear = RewTerm(
        func=mdp.foot_site_height_gaussian,
        weight=1.5,
        params={
            "target": FLAMINGO_SWING_FOOT_TARGET_Z,
            "std": FLAMINGO_SWING_FOOT_STD,
            "asset_cfg": _SWING_FOOT_CFG,
            "gate_sensor_cfg": _STANCE_SENSOR_CFG,
        },
    )
    # Soft failure: the free foot lands; cheap and never terminal.
    swing_foot_touch = RewTerm(func=mdp.foot_in_contact, weight=-0.5, params={"sensor_cfg": _SWING_SENSOR_CFG})
    pose_flamingo = RewTerm(
        func=mdp.pose_target_match,
        weight=1.5,
        params={"target": FLAMINGO_POSE_RIGHT, "std": 0.5, "asset_cfg": _SERVO_JOINT_CFG},
    )
    # The trunk lean is part of the pose, so its gravity direction is rewarded instead of uprightness.
    gravity_flamingo = RewTerm(
        func=mdp.projected_gravity_match, weight=2.0, params={"target": FLAMINGO_GRAVITY_B, "std": 0.15}
    )
    stillness = RewTerm(
        func=mdp.joint_vel_gaussian,
        weight=1.0,
        params={"std": 2.0, "asset_cfg": _SERVO_JOINT_CFG, "gate_sensor_cfg": _STANCE_SENSOR_CFG},
    )
    # Hard failure: rolling over the stance foot has no catch.
    stance_side_tilt = RewTerm(
        func=mdp.lateral_tilt_cost, weight=-4.0, params={"threshold": FLAMINGO_STANCE_SIDE_TILT, "direction": -1.0}
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
class FlamingoEventsCfg(EventsCfg):
    """Spawn in the one-foot pose instead of the walking stance; pushes start at zero."""

    reset_base = None
    reset_robot_joints = None
    reset_flamingo_pose = EventTerm(
        func=mdp.reset_flamingo_pose,
        mode="reset",
        params={
            "joint_pose": FLAMINGO_POSE_RIGHT,
            "base_roll": FLAMINGO_BASE_ROLL,
            "base_pitch": FLAMINGO_BASE_PITCH,
            "z_range": (FLAMINGO_Z, FLAMINGO_Z + 0.01),
            # microduck_rl's generic reset_base offsets x and y before set_flamingo_state overrides the rest.
            "xy_range": 0.5,
            "tilt_noise": math.radians(3.0),
            "joint_noise_std": 0.05,
            "asset_cfg": _SERVO_JOINT_CFG,
        },
    )

    def __post_init__(self):
        self.push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}


@configclass
class FlamingoCurriculumCfg(CurriculumCfg):
    """microduck_rl's stage-1 schedules in its iterations, with pushes and action-rate capped as in its Cycle task."""

    head_pose_bias_weight = None
    standing_envs = None
    head_pose_range = None
    action_rate_weight = _schedule("rewards.action_rate_l2.weight", (400, -0.5))
    push_magnitude = _schedule(
        "events.push_robot.params.velocity_range",
        (300, {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}),
        (600, {"x": (-FLAMINGO_MAX_PUSH, FLAMINGO_MAX_PUSH), "y": (-FLAMINGO_MAX_PUSH, FLAMINGO_MAX_PUSH)}),
    )
    torque_rate_weight = _schedule("rewards.joint_torque_rate_l2.weight", (600, -1e-3))
    com_std = _schedule("rewards.com_over_stance_foot.params.std", (500, 0.02))


@configclass
class MicroDuckFlamingoFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """MicroDuck one-foot balance on the right foot with Newton MJWarp and BAM servos."""

    scene: FlamingoSceneCfg = FlamingoSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: FlamingoObservationsCfg = FlamingoObservationsCfg()
    commands: FlamingoCommandsCfg = FlamingoCommandsCfg()
    rewards: FlamingoRewardsCfg = FlamingoRewardsCfg()
    events: FlamingoEventsCfg = FlamingoEventsCfg()
    curriculum: FlamingoCurriculumCfg = FlamingoCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 6.0
        # microduck_rl runs MuJoCo Warp's own collision detection instead of Newton's pipeline.
        self.sim.physics.solver_cfg.use_mujoco_contacts = True
        self.sim.physics.collision_cfg = None
        # mjlab's velocity-env defaults, which microduck_rl's flamingo keeps. The walking task's njmax=96 is sized for
        # sole-only collisions and overflows the constraint rows of whole-body contacts.
        self.sim.physics.solver_cfg.njmax = 1500
        self.sim.physics.solver_cfg.nconmax = 35

    def play_mode(self):
        """Disable interval pushes together with their magnitude schedule."""
        super().play_mode()
        self.curriculum.push_magnitude = None
