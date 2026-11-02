# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an exported MicroDuck flamingo policy on Newton MJWarp with microduck_rl's stage-1 success metric.

For each push magnitude, one batch of environments runs a single episode from the noised one-foot spawn with the
training randomization at its final curriculum stage and pushes of that magnitude every 3 to 6 s. An episode is a
success when it holds single support (stance foot down, swing foot clear) for at least 5 s without a fall; microduck_rl
targets more than 70 % success at 0.15 m/s pushes. Results are written as ``episodes.csv`` and ``summary.md``.

.. code-block:: bash

    uv run python scripts/reinforcement_learning/microduck/eval_flamingo.py \\
        --policy logs/rsl_rl/microduck_flamingo_flat/<run>/exported/policy.pt
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flamingo_metrics import (
    FLAMINGO_SUCCESS_HOLD,
    FlamingoEpisodeMetrics,
    set_eval_conditions,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TARGET_PUSH = 0.15
"""Push magnitude [m/s] of microduck_rl's stage-1 success target."""
TARGET_SUCCESS = 0.70
"""Success fraction microduck_rl's stage-1 target must exceed."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="IsaacContrib-Flamingo-Flat-MicroDuck")
    parser.add_argument("--policy", required=True, help="Exported TorchScript policy.pt.")
    parser.add_argument("--pushes", type=float, nargs="+", default=[0.0, TARGET_PUSH, 0.25], help="Push [m/s].")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action_noise_std",
        type=float,
        default=0.0,
        help="Gaussian noise [rad] added to the actions, e.g. the trained policy's action std to mimic training.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default=None, help="Defaults to <policy dir>/eval_flamingo.")
    return parser.parse_args()


def _make_env(args: argparse.Namespace, push: float):
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    set_eval_conditions(cfg, push)
    sim_utils.create_new_stage()
    return gym.make(args.task, cfg=cfg).unwrapped


def _evaluate(args: argparse.Namespace, push: float, policy) -> dict:
    env = _make_env(args, push)
    try:
        stance_cfg = SceneEntityCfg("contact_forces", body_names=["ankle_right"])
        swing_cfg = SceneEntityCfg("contact_forces", body_names=["ankle_left"])
        stance_cfg.resolve(env.scene)
        swing_cfg.resolve(env.scene)
        robot = env.scene["robot"]
        metrics = FlamingoEpisodeMetrics(env.num_envs, env.step_dt, device=env.device)
        obs, _ = env.reset()
        with torch.inference_mode():
            for _ in range(env.max_episode_length):
                actions = policy(obs["policy"])
                actions = actions + args.action_noise_std * torch.randn_like(actions)
                obs, _, terminated, truncated, _ = env.step(actions)
                metrics.update(
                    stance_contact=mdp.foot_in_contact(env, stance_cfg).bool(),
                    swing_contact=mdp.foot_in_contact(env, swing_cfg).bool(),
                    done=terminated | truncated,
                    time_out=truncated & ~terminated,
                    gravity_b=robot.data.projected_gravity_b.torch,
                )
        return metrics.summary()
    finally:
        env.close()
        SimulationContext.clear_instance()


def _write_results(args: argparse.Namespace, results: dict[float, dict]) -> str:
    default_name = "eval_flamingo" if args.action_noise_std == 0.0 else f"eval_flamingo_noise{args.action_noise_std:g}"
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.policy)), default_name)
    os.makedirs(out_dir, exist_ok=True)
    tensor_names = [name for name in next(iter(results.values())) if name != "fall_side"]
    with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["push", "env", *tensor_names, "fall_side"])
        for push, summary in results.items():
            for i in range(args.num_envs):
                row = [float(summary[name][i]) for name in tensor_names]
                writer.writerow([push, i, *row, summary["fall_side"][i]])

    lines = [
        f"# {args.task}: flamingo evaluation",
        "",
        f"policy: `{args.policy}`, envs per push: {args.num_envs}, seed: {args.seed}, "
        f"action noise std: {args.action_noise_std} rad, one episode each "
        f"(spawned in the noised pose, randomization at its final curriculum stage, pushes every 3-6 s).",
        "",
        f"success: single support (right foot down, left foot clear) for at least {FLAMINGO_SUCCESS_HOLD:.0f} s "
        "without a fall (any termination but the time-out). microduck_rl's stage-1 target: "
        f"> {TARGET_SUCCESS:.0%} at {TARGET_PUSH} m/s.",
        "",
        "| push [m/s] | success | falls | fall side (stance/swing/forward/backward) | single support [s] mean | "
        "swing touchdowns mean | swing contact [s] mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for push, s in results.items():
        sides = Counter(label for label in s["fall_side"] if label)
        side_text = "/".join(str(sides.get(k, 0)) for k in ("stance", "swing", "forward", "backward"))
        lines.append(
            f"| {push:.2f} | {s['success'].float().mean():.1%} | {s['fell'].float().mean():.1%} | {side_text} | "
            f"{s['longest_single_support'].mean():.2f} | {s['swing_touchdowns'].float().mean():.2f} | "
            f"{s['swing_contact_time'].mean():.2f} |"
        )
    if TARGET_PUSH in results:
        rate = results[TARGET_PUSH]["success"].float().mean().item()
        verdict = "pass" if rate > TARGET_SUCCESS else "fail"
        lines += ["", f"verdict at {TARGET_PUSH} m/s: **{verdict}** ({rate:.1%} vs > {TARGET_SUCCESS:.0%})"]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = _parse_args()
    policy = torch.jit.load(args.policy, map_location=args.device).eval()
    results = {}
    for push in args.pushes:
        results[push] = _evaluate(args, push, policy)
        print(f"[eval_flamingo] push {push}: success {results[push]['success'].float().mean():.1%}")
    path = _write_results(args, results)
    with open(path) as file:
        print(file.read())


if __name__ == "__main__":
    main()
