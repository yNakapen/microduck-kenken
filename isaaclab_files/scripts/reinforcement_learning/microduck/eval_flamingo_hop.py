# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an exported MicroDuck flamingo-hop policy on Newton MJWarp with a scripted still, hop, still protocol.

Every environment spawns in the right-foot pose and is commanded ``[0, 0, 0]`` until 2 s, ``[0, 0, 1]`` (hop) until
5 s and ``[0, 0, 0]`` until 8 s, under the training randomization with pushes of the given magnitude every 3 to 6 s.
The test plan passes when at most 20 % of the episodes fall and every other episode makes at least 5 stance-foot hops,
never touches down with the lifted foot, and then holds 2 s of uninterrupted single support. Results go to
``episodes.csv`` and ``summary.md``.

.. code-block:: bash

    uv run python scripts/reinforcement_learning/microduck/eval_flamingo_hop.py \\
        --policy logs/rsl_rl/microduck_flamingo_hop_flat/<run>/exported/policy.pt
"""

from __future__ import annotations

import argparse
import csv
import os

import torch

from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.flamingo_eval import make_hop_env, run_hop_protocol
from isaaclab_tasks.contrib.microduck.flamingo_metrics import (
    HOP_MAX_FALL_FRACTION,
    HOP_MIN_HOPS,
    HOP_MIN_STILL,
)

NAMES = ("fell", "hops", "swing_touchdowns", "still_single_support", "success")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="IsaacContrib-FlamingoHop-Flat-MicroDuck")
    parser.add_argument("--policy", required=True, help="Exported TorchScript policy.pt.")
    parser.add_argument("--pushes", type=float, nargs="+", default=[0.0, 0.15], help="Push [m/s].")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default=None, help="Defaults to <policy dir>/eval_flamingo_hop.")
    return parser.parse_args()


def _passes(summary: dict) -> bool:
    fell = summary["fell"]
    return fell.float().mean().item() <= HOP_MAX_FALL_FRACTION and bool((fell | summary["success"]).all())


def _write_results(args: argparse.Namespace, results: dict[float, dict]) -> str:
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.policy)), "eval_flamingo_hop")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["push", "env", *NAMES])
        for push, summary in results.items():
            for i in range(args.num_envs):
                writer.writerow([push, i, *(float(summary[name][i]) for name in NAMES)])

    lines = [
        f"# {args.task}: flamingo hop evaluation",
        "",
        f"policy: `{args.policy}`, envs per push: {args.num_envs}, seed: {args.seed}. Protocol on the right foot: "
        "still until 2 s, hop until 5 s, still until 8 s; randomization at its final stage; pushes every 3-6 s.",
        "",
        f"test plan: falls <= {HOP_MAX_FALL_FRACTION:.0%}, and every other episode makes >= {HOP_MIN_HOPS} stance-foot "
        f"hops (flights >= 20 ms landing on the stance foot, 2-5.3 s), no lifted-foot touchdown, and >= "
        f"{HOP_MIN_STILL:.0f} s of uninterrupted single support in 5.5-8 s.",
        "",
        "| push [m/s] | pass | success | falls | hops mean | hops min (not fallen) | episodes with swing touchdowns | "
        "still single support [s] mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for push, s in results.items():
        standing = ~s["fell"]
        hops_min = int(s["hops"][standing].min()) if standing.any() else 0
        lines.append(
            f"| {push:.2f} | {'yes' if _passes(s) else 'no'} | {s['success'].float().mean():.1%} | "
            f"{s['fell'].float().mean():.1%} | {s['hops'].float().mean():.2f} | {hops_min} | "
            f"{(s['swing_touchdowns'] > 0).float().mean():.1%} | {s['still_single_support'].mean():.2f} |"
        )
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = _parse_args()
    policy = torch.jit.load(args.policy, map_location=args.device).eval()
    results = {}
    for push in args.pushes:
        env = make_hop_env(args.task, args.device, args.num_envs, args.seed, push)
        try:
            results[push] = {k: v.cpu() for k, v in run_hop_protocol(env, policy).items()}
        finally:
            env.close()
            SimulationContext.clear_instance()
        print(f"[eval_flamingo_hop] push {push}: success {results[push]['success'].float().mean():.1%}")
    path = _write_results(args, results)
    with open(path) as file:
        print(file.read())


if __name__ == "__main__":
    main()
