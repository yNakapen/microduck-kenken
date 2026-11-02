# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an exported MicroDuck flamingo-cycle policy on Newton MJWarp with a scripted stand, lift, hold, lower.

Half of the environments stand on the right foot, half on the left. Each starts standing at HOME with the deployed
idle command ``[0, 0, 0]``, is commanded ``[1, side, 0]`` from 2 s to 7 s and ``[0, 0, 0]`` again until 10 s, under
the training randomization at its final curriculum stage with pushes of the given magnitude every 3 to 6 s. A success
holds single support for 90 % of the 4 to 7 s window and both feet down for 90 % of the 9 to 10 s window without a
fall; the target is 70 % successes on each side at 0.15 m/s pushes. Results go to ``episodes.csv`` and
``summary.md``.

.. code-block:: bash

    uv run python scripts/reinforcement_learning/microduck/eval_flamingo_cycle.py \\
        --policy logs/rsl_rl/microduck_flamingo_cycle_flat/<run>/exported/policy.pt
"""

from __future__ import annotations

import argparse
import csv
import os

import torch

from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.flamingo_eval import make_cycle_env, run_cycle_protocol

TARGET_PUSH = 0.15
"""Push magnitude [m/s] of the success target."""
TARGET_SUCCESS = 0.70
"""Success fraction each side must reach at :data:`TARGET_PUSH`."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="IsaacContrib-FlamingoCycle-Flat-MicroDuck")
    parser.add_argument("--policy", required=True, help="Exported TorchScript policy.pt.")
    parser.add_argument("--pushes", type=float, nargs="+", default=[0.0, TARGET_PUSH], help="Push [m/s].")
    parser.add_argument("--num_envs", type=int, default=512, help="Even; half per stance side.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default=None, help="Defaults to <policy dir>/eval_flamingo_cycle.")
    return parser.parse_args()


def _write_results(args: argparse.Namespace, results: dict[float, dict]) -> str:
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.policy)), "eval_flamingo_cycle")
    os.makedirs(out_dir, exist_ok=True)
    names = ["side", "fell", "hold_single_support", "hold_swing_clear_frac", "stand_two_feet", "success"]
    with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["push", "env", *names])
        for push, summary in results.items():
            for i in range(args.num_envs):
                writer.writerow([push, i, *(float(summary[name][i]) for name in names)])

    lines = [
        f"# {args.task}: flamingo cycle evaluation",
        "",
        f"policy: `{args.policy}`, envs per push: {args.num_envs} (half per side), seed: {args.seed}. Protocol: stand "
        "with [0, 0, 0], [1, side, 0] from 2 s to 7 s, [0, 0, 0] until 10 s; spawned standing; randomization at its "
        "final curriculum stage; pushes every 3-6 s.",
        "",
        "success: single support for 90 % of the 4-7 s hold window, both feet down for 90 % of the 9-10 s window, no "
        f"fall. Target: >= {TARGET_SUCCESS:.0%} on each side at {TARGET_PUSH} m/s.",
        "",
        "| push [m/s] | side | success | falls | hold single support mean | hold swing clear mean | "
        "stand two feet mean |",
        "|---|---|---|---|---|---|---|",
    ]
    verdict = True
    for push, s in results.items():
        for label, value in (("right", 1.0), ("left", -1.0)):
            m = s["side"] == value
            rate = s["success"][m].float().mean().item()
            if push == TARGET_PUSH:
                verdict &= rate >= TARGET_SUCCESS
            lines.append(
                f"| {push:.2f} | {label} | {rate:.1%} | {s['fell'][m].float().mean():.1%} | "
                f"{s['hold_single_support'][m].mean():.2f} | {s['hold_swing_clear_frac'][m].mean():.2f} | "
                f"{s['stand_two_feet'][m].mean():.2f} |"
            )
    if TARGET_PUSH in results:
        lines += ["", f"verdict at {TARGET_PUSH} m/s: **{'pass' if verdict else 'fail'}**"]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = _parse_args()
    policy = torch.jit.load(args.policy, map_location=args.device).eval()
    results = {}
    for push in args.pushes:
        env = make_cycle_env(args.task, args.device, args.num_envs, args.seed, push)
        try:
            results[push] = {
                k: v.cpu() if torch.is_tensor(v) else v for k, v in run_cycle_protocol(env, policy).items()
            }
        finally:
            env.close()
            SimulationContext.clear_instance()
        print(f"[eval_flamingo_cycle] push {push}: success {results[push]['success'].float().mean():.1%}")
    path = _write_results(args, results)
    with open(path) as file:
        print(file.read())


if __name__ == "__main__":
    main()
