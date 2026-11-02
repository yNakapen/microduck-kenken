# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check microduck_rl's flamingo one-foot pose on the Newton MJWarp MicroDuck before training on it.

C1 (geometry, no physics step): for each stance side, place the robot with the stance sole flat and measure the
trunk lean, CoM margin inside the stance sole, lifted-foot clearance and joint-limit margins, next to microduck_rl's
values and to microduck_rl's trunk placement taken as is.
C2 (hold): from that placement, hold the pose's joint targets for ``--duration`` seconds and record when and which
way the robot falls; the default two-foot stance is run the same way for reference.

Results go to ``logs/flamingo_static/<time>/report.md`` and ``timeseries.csv``.

.. code-block:: bash

    uv run python scripts/reinforcement_learning/microduck/check_flamingo_pose.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import asdict
from datetime import datetime

import torch

from isaaclab.sim import SimulationContext
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flamingo_checks import (
    make_check_env,
    measure_pose,
    place_robot,
    polygon_signed_distance,
    servo_joint_ids,
    sole_support_hull,
    sole_vertices_w,
    solve_flat_sole_placement,
)
from isaaclab_tasks.contrib.microduck.flamingo_pose import (
    FLAMINGO_BASE_PITCH,
    FLAMINGO_BASE_ROLL,
    FLAMINGO_COM_MARGIN,
    FLAMINGO_GRAVITY_B,
    FLAMINGO_POSE_LEFT,
    FLAMINGO_POSE_RIGHT,
    FLAMINGO_Z,
)

LIFTED_CLEARANCE_MIN = 0.05
"""Lowest lifted-sole height [m] that counts as clearly lifted (microduck_rl: swing foot at least 5 cm up)."""

JOINT_MARGIN_MIN = 0.05 - 1e-4
"""Joint-limit margin [rad] the pose must keep; 1e-4 rad covers microduck_rl's rounded hip-roll range."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=3.0, help="Hold time per case [s].")
    parser.add_argument("--vin", type=float, default=7.4, help="Servo supply voltage [V].")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default=None, help="Defaults to logs/flamingo_static/<time>.")
    return parser.parse_args()


def _geometry(env, pose: list[float], stance: str) -> dict:
    """C1 for one stance side: the flat-sole placement and microduck_rl's placement taken as is."""
    sign = 1.0 if stance == "right" else -1.0
    env.reset()
    placement = solve_flat_sole_placement(env, pose, stance)
    flat = measure_pose(env, pose, stance)
    roll = torch.tensor([sign * FLAMINGO_BASE_ROLL], device=env.device)
    microduck_rl_quat = quat_from_euler_xyz(roll, torch.tensor([FLAMINGO_BASE_PITCH], device=env.device), roll * 0.0)
    place_robot(env, pose, microduck_rl_quat, FLAMINGO_Z)
    microduck_rl = measure_pose(env, pose, stance)
    checks = {
        "stance sole flat (tilt < 0.5 deg, lowest point at 0 +- 0.1 mm)": flat.stance_sole_tilt_deg < 0.5
        and abs(flat.stance_sole_min_z) < 1e-4,
        "CoM inside the stance sole (margin > 0)": flat.com_hull_margin > 0.0,
        f"lifted sole >= {LIFTED_CLEARANCE_MIN * 100:.0f} cm up": flat.lifted_sole_min_z >= LIFTED_CLEARANCE_MIN,
        "every joint >= 0.05 rad off its limit (1e-4 rad rounding)": flat.min_joint_limit_margin >= JOINT_MARGIN_MIN,
    }
    return {"stance": stance, "placement": placement, "flat": flat, "microduck_rl": microduck_rl, "checks": checks}


def _hold(env, pose: list[float], stance: str, duration: float, label: str, rows: list[dict]) -> dict:
    """C2: hold the joint targets of ``pose`` from the flat-sole placement and record the fall."""
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    lifted = "left" if stance == "right" else "right"
    (stance_id, lifted_id), _ = sensor.find_sensors([f"ankle_{stance}", f"ankle_{lifted}"], True)
    weight = robot.data.body_mass.torch[0].sum().item() * 9.81
    sign = 1.0 if stance == "right" else -1.0

    # Resetting clears the servo command delay, so the first delayed command is already the pose target.
    env.reset()
    placement = solve_flat_sole_placement(env, pose, stance)
    pose_t = torch.tensor(pose, device=env.device)
    actions = (pose_t - robot.data.default_joint_pos.torch[0, servo_joint_ids(env)])[None]
    roll0 = placement.roll_deg

    fall_time, first_lifted_contact, stance_forces, stance_sole_z = None, None, [], []
    roll = pitch = roll0
    for step in range(round(duration / env.step_dt)):
        env.step(actions)
        t = (step + 1) * env.step_dt
        if env.termination_manager.terminated[0]:
            fall_time = t
            break
        forces = sensor.data.net_forces_w.torch[0].norm(dim=-1)
        r, p, _ = euler_xyz_from_quat(robot.data.root_link_quat_w.torch[0:1])
        roll, pitch = math.degrees(r.item()), math.degrees(p.item())
        com_xy = mdp.robot_com_pos_w(robot)[0, :2]
        margin = polygon_signed_distance(com_xy, sole_support_hull(env, f"ankle_{stance}")[:, :2]).item()
        sole_z = sole_vertices_w(env, f"ankle_{stance}")[:, 2].min().item() - env.scene.env_origins[0, 2].item()
        if first_lifted_contact is None and forces[lifted_id] > 0.1:
            first_lifted_contact = t
        stance_forces.append(forces[stance_id].item())
        stance_sole_z.append(sole_z)
        rows.append(
            {
                "case": label,
                "t": round(t, 3),
                "roll_deg": roll,
                "pitch_deg": pitch,
                "trunk_z": robot.data.root_link_pos_w.torch[0, 2].item(),
                "com_margin": margin,
                "stance_force": forces[stance_id].item(),
                "lifted_force": forces[lifted_id].item(),
                "stance_sole_min_z": sole_z,
            }
        )
    # Rolling further over the stance foot is the hard failure (no foot to catch the fall).
    toward = "stance side" if sign * (roll - roll0) > 0 else "lifted side"
    return {
        "case": label,
        "start roll/pitch [deg]": f"{roll0:.1f} / {placement.pitch_deg:.1f}",
        "fall time [s]": f"{fall_time:.2f}" if fall_time is not None else f"none in {duration:.1f}",
        "last roll/pitch [deg]": f"{roll:.1f} / {pitch:.1f}",
        "rolled toward": f"{toward} ({roll - roll0:+.1f} deg)",
        "first lifted-foot contact [s]": f"{first_lifted_contact:.2f}" if first_lifted_contact is not None else "none",
        "stance force / weight": f"{sum(stance_forces) / max(len(stance_forces), 1) / weight:.2f}",
        "stance sole min z [mm]": f"{min(stance_sole_z) * 1e3:.2f} .. {max(stance_sole_z) * 1e3:.2f}"
        if stance_sole_z
        else "-",
    }


def _report(args, geometry: list[dict], holds: list[dict]) -> list[str]:
    lines = [
        "# Flamingo pose static check (Newton MJWarp, MuJoCo Warp contacts)",
        "",
        f"robot: all-collisions USD, no backlash, no randomization, supply {args.vin} V without sag",
        "",
        "## C1 geometry",
        "",
    ]
    for g in geometry:
        flat, up = g["flat"], g["microduck_rl"]
        sign = 1.0 if g["stance"] == "right" else -1.0
        gravity = (FLAMINGO_GRAVITY_B[0], sign * FLAMINGO_GRAVITY_B[1], FLAMINGO_GRAVITY_B[2])
        # (quantity, microduck_rl value, value under microduck_rl's placement, value under the flat-sole placement)
        rows = [
            ("trunk roll [deg]", f"{sign * math.degrees(FLAMINGO_BASE_ROLL):.1f}", f"{up.roll_deg:.1f}",
             f"{flat.roll_deg:.2f}"),
            ("trunk pitch [deg]", f"{math.degrees(FLAMINGO_BASE_PITCH):.1f}", f"{up.pitch_deg:.1f}",
             f"{flat.pitch_deg:.2f}"),
            ("trunk z [m]", f"{FLAMINGO_Z:.3f}", f"{up.z:.3f}", f"{flat.z:.4f}"),
            ("gravity in trunk", str(list(gravity)), str([round(v, 3) for v in up.gravity_b]),
             str([round(v, 3) for v in flat.gravity_b])),
            ("stance sole tilt [deg]", "0", f"{up.stance_sole_tilt_deg:.2f}", f"{flat.stance_sole_tilt_deg:.2f}"),
            ("stance sole lowest z [mm]", "0", f"{up.stance_sole_min_z * 1e3:.2f}",
             f"{flat.stance_sole_min_z * 1e3:.2f}"),
            ("CoM margin in stance sole [cm]", f"{FLAMINGO_COM_MARGIN * 100:.2f}", f"{up.com_hull_margin * 100:.2f}",
             f"{flat.com_hull_margin * 100:.2f}"),
            ("CoM to stance site [cm]", "-", f"{up.com_to_stance_site * 100:.2f}",
             f"{flat.com_to_stance_site * 100:.2f}"),
            ("lifted site z [m]", "0.136 (design) / ~0.17 (cfg)", f"{up.lifted_site_z:.3f}",
             f"{flat.lifted_site_z:.3f}"),
            ("lifted sole lowest z [m]", "-", f"{up.lifted_sole_min_z:.3f}", f"{flat.lifted_sole_min_z:.3f}"),
            ("min joint-limit margin [rad]", "0.05", f"{up.min_joint_limit_margin:.5f}",
             f"{flat.min_joint_limit_margin:.5f} ({flat.min_joint_limit_joint})"),
        ]  # fmt: skip
        lines += [
            f"### {g['stance']} stance",
            "",
            "| quantity | microduck_rl value | microduck_rl placement here | flat-sole placement here |",
            "|---|---|---|---|",
            *["| " + " | ".join(row) + " |" for row in rows],
            "",
            "| criterion (flat-sole placement) | pass |",
            "|---|---|",
            *[f"| {name} | {'yes' if ok else 'no'} |" for name, ok in g["checks"].items()],
            "",
        ]
    lines += ["## C2 hold with fixed joint targets (recorded, not judged)", ""]
    keys = list(holds[0].keys())
    lines += ["| " + " | ".join(keys) + " |", "|---" * len(keys) + "|"]
    lines += ["| " + " | ".join(str(h[k]) for k in keys) + " |" for h in holds]
    return lines


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir or os.path.join("logs", "flamingo_static", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    env = make_check_env(device=args.device, vin=args.vin)
    try:
        robot = env.scene["robot"]
        home = robot.data.default_joint_pos.torch[0, servo_joint_ids(env)].tolist()
        geometry = [_geometry(env, FLAMINGO_POSE_RIGHT, "right"), _geometry(env, FLAMINGO_POSE_LEFT, "left")]
        rows: list[dict] = []
        with torch.inference_mode():
            holds = [
                _hold(env, FLAMINGO_POSE_RIGHT, "right", args.duration, "flamingo right", rows),
                _hold(env, FLAMINGO_POSE_LEFT, "left", args.duration, "flamingo left", rows),
                _hold(env, home, "right", args.duration, "home two feet", rows),
            ]
    finally:
        env.close()
        SimulationContext.clear_instance()

    with open(os.path.join(out_dir, "timeseries.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = _report(args, geometry, holds)
    lines += ["", "Placements: " + "; ".join(f"{g['stance']}: {asdict(g['flat'])}" for g in geometry)]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[check_flamingo_pose] wrote {path}")


if __name__ == "__main__":
    main()
