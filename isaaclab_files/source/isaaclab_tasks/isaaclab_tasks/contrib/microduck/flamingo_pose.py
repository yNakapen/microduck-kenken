# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The one-foot "flamingo" pose of pollen-robotics/microduck_rl, reproduced on Newton MJWarp.

Values are copied from ``src/mjlab_microduck/tasks/microduck_flamingo_env_cfg.py`` and
``docs/superpowers/specs/2026-08-28-flamingo-design.md`` on the microduck_rl ``flamingo`` branch (``4757d6ae``).
Joint values follow :data:`~isaaclab_assets.robots.microduck.MICRODUCK_JOINT_NAMES`: left leg, neck and head,
right leg.
"""

import math

FLAMINGO_POSE_RIGHT: list[float] = [
    *[0.000, 0.300, 1.200, -0.800, 0.800],  # left leg (lifted): swung forward
    *[0.349, 0.350, -1.500, 0.000],  # neck and head: beak turned toward the stance side
    *[0.386, -0.334, 0.258, 0.005, -0.253],  # right leg (stance): yawed, rolled, slight forward lean
]
"""Joint positions [rad] of the pose standing on the right foot, every joint at least 3 deg inside its range."""

FLAMINGO_BASE_ROLL = math.radians(22.6)
"""microduck_rl's trunk roll [rad] with the right sole flat on the floor."""

FLAMINGO_BASE_PITCH = math.radians(-8.9)
"""microduck_rl's trunk pitch [rad] with the right sole flat on the floor."""

FLAMINGO_Z = 0.120
"""microduck_rl's trunk height [m] with the right sole flat on the floor."""

FLAMINGO_GRAVITY_B = (-0.154, -0.379, -0.912)
"""microduck_rl's gravity direction in the trunk frame at the right-stance pose."""

FLAMINGO_COM_MARGIN = 0.0121
"""microduck_rl's distance [m] of the whole-robot CoM inside the stance sole's contact hull."""


def mirror_pose(pose: list[float]) -> list[float]:
    """Return the pose standing on the other foot.

    The right-leg joint axes are the negatives of the left-leg ones, so the legs swap with every leg joint negated;
    the head keeps its pitches and negates yaw and roll.
    """
    left_leg = [-v for v in pose[9:14]]
    right_leg = [-v for v in pose[0:5]]
    head = [pose[5], pose[6], -pose[7], -pose[8]]
    return [v + 0.0 for v in left_leg + head + right_leg]


FLAMINGO_POSE_LEFT: list[float] = mirror_pose(FLAMINGO_POSE_RIGHT)
"""Joint positions [rad] of the pose standing on the left foot."""
