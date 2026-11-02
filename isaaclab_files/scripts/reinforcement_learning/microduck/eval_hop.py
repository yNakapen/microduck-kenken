# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an exported MicroDuck single-leg policy on Newton MJWarp at fixed servo supply voltages.

Each seed builds one environment per voltage with the task's play-mode configuration (backlash robot by default),
lets the robot settle, then rolls the policy for ``--duration`` seconds and records stand and hop metrics.
Results are written as ``episodes.csv`` (one row per seed and voltage) and ``summary.md`` (one row per voltage,
with the test-plan verdict for ``--stage``).

.. code-block:: bash

    uv run --with onnxruntime python scripts/reinforcement_learning/microduck/eval_hop.py \\
        --task IsaacContrib-Hop-Stand-Flat-MicroDuck --stage stand \\
        --policy logs/rsl_rl/microduck_hop_stand_flat_backlash/<run>/exported/policy.onnx
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.hop_metrics import (
    HopMetrics,
    read_hop_signals,
    set_supply_voltage,
    stage_passes,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="IsaacContrib-Hop-Stand-Flat-MicroDuck")
    parser.add_argument("--stage", choices=("stand", "single", "hop"), default="stand")
    parser.add_argument("--policy", required=True, help="Exported policy: policy.onnx, or a TorchScript policy.pt.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--vin", type=float, nargs="+", default=[6.5, 7.4, 8.2], help="Supply voltages [V].")
    parser.add_argument("--sag_gain", type=float, default=0.0, help="Battery sag gain during evaluation.")
    parser.add_argument("--duration", type=float, default=12.0, help="Evaluated time after settling [s].")
    parser.add_argument("--settle", type=float, default=0.5, help="Time to land after reset, not evaluated [s].")
    parser.add_argument("--no_backlash", action="store_true", help="Evaluate the robot without gearbox play.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default=None, help="Defaults to <policy dir>/eval_<stage>.")
    return parser.parse_args()


def _load_policy(path: str, device: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a batched policy; the exported ONNX graph takes one observation at a time."""
    if path.endswith(".onnx"):
        import onnxruntime as ort

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        def onnx_policy(obs: torch.Tensor) -> torch.Tensor:
            batch = obs.detach().cpu().numpy().astype(np.float32)
            actions = [session.run(None, {input_name: row[None]})[0][0] for row in batch]
            return torch.as_tensor(np.stack(actions), device=obs.device)

        return onnx_policy
    jit = torch.jit.load(path, map_location=device).eval()
    return lambda obs: jit(obs)


def _evaluate_seed(args: argparse.Namespace, seed: int, policy, reference) -> tuple[dict, float]:
    overrides = () if args.no_backlash else ("presets=backlash",)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(args.vin), overrides=overrides)
    cfg.play_mode()
    cfg.seed = seed
    sim_utils.create_new_stage()
    env = gym.make(args.task, cfg=cfg).unwrapped
    max_export_gap = 0.0
    try:
        obs, _ = env.reset()
        set_supply_voltage(env, torch.tensor(args.vin, device=env.device), sag_gain=args.sag_gain)
        fell_while_settling = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        metrics = HopMetrics(env.num_envs, env.step_dt, device=env.device)
        settle_steps = round(args.settle / env.step_dt)
        with torch.inference_mode():
            for step in range(settle_steps + round(args.duration / env.step_dt)):
                actions = policy(obs["policy"])
                if reference is not None:
                    gap = (reference(obs["policy"]) - actions).abs().max().item()
                    max_export_gap = max(max_export_gap, gap)
                obs, *_ = env.step(actions)
                signals = read_hop_signals(env)
                if step < settle_steps:
                    fell_while_settling |= signals.fallen
                    continue
                signals.fallen |= fell_while_settling
                metrics.update(signals)
        return metrics.summary(), max_export_gap
    finally:
        env.close()
        SimulationContext.clear_instance()


def _write_results(args: argparse.Namespace, per_seed: list[dict], max_export_gap: float) -> str:
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.policy)), f"eval_{args.stage}")
    os.makedirs(out_dir, exist_ok=True)
    names = list(per_seed[0].keys())
    with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "vin", *names])
        for seed, summary in zip(args.seeds, per_seed):
            for i, vin in enumerate(args.vin):
                writer.writerow([seed, vin, *(float(summary[name][i]) for name in names)])

    columns = [
        ("falls", lambda s: f"{int(s['fell'].sum())}/{s['fell'].numel()}"),
        ("collapses", lambda s: f"{int(s['collapsed'].sum())}/{s['collapsed'].numel()}"),
        ("hold [s] min", lambda s: f"{s['hold_time'].min():.2f}"),
        ("hops mean", lambda s: f"{s['hops'].mean():.1f}"),
        ("air [ms] max", lambda s: f"{s['max_air_time'].max() * 1e3:.0f}"),
        ("CoM rise [mm] max", lambda s: f"{s['max_com_rise'].max() * 1e3:.1f}"),
        ("sole [mm] max", lambda s: f"{s['max_sole_clearance'].max() * 1e3:.1f}"),
        ("post-land hold [s] min", lambda s: f"{s['post_landing_hold'].min():.2f}"),
        ("lifted touchdowns", lambda s: f"{int(s['lifted_touchdowns'].sum())}"),
        ("saturation [%]", lambda s: f"{s['saturation_ratio'].mean() * 100:.1f}"),
        ("joint speed [rad/s] max", lambda s: f"{s['max_joint_speed'].max():.1f}"),
        ("contact force [N] max", lambda s: f"{s['peak_force'].max():.1f}"),
    ]
    lines = [
        f"# {args.task}: {args.stage} evaluation",
        "",
        f"policy: `{args.policy}`, seeds: {args.seeds}, backlash: {not args.no_backlash}, "
        f"duration: {args.duration} s after {args.settle} s settling, sag gain: {args.sag_gain}",
        "",
        "falls: terminated (tilt beyond 70 deg or diverged state); collapses: trunk ever below the standing height.",
        "",
        "| Vin [V] | pass | " + " | ".join(name for name, _ in columns) + " |",
        "|---" * (len(columns) + 2) + "|",
    ]
    for i, vin in enumerate(args.vin):
        summary = {name: torch.stack([s[name][i] for s in per_seed]).cpu() for name in names}
        verdict = "yes" if stage_passes(args.stage, summary) else "no"
        lines.append(f"| {vin:.1f} | {verdict} | " + " | ".join(fn(summary) for _, fn in columns) + " |")
    if max_export_gap >= 0.0:
        lines += ["", f"max |ONNX - TorchScript| action gap: {max_export_gap:.2e}"]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = _parse_args()
    policy = _load_policy(args.policy, args.device)
    # Cross-check the ONNX graph against the TorchScript export written next to it by `isaaclab play`.
    sibling = os.path.join(os.path.dirname(args.policy), "policy.pt")
    has_sibling = args.policy.endswith(".onnx") and os.path.exists(sibling)
    reference = _load_policy(sibling, args.device) if has_sibling else None
    per_seed, gaps = [], []
    for seed in args.seeds:
        summary, gap = _evaluate_seed(args, seed, policy, reference)
        per_seed.append(summary)
        gaps.append(gap)
        print(f"[eval_hop] seed {seed}: falls {summary['fell'].tolist()}, hops {summary['hops'].tolist()}")
    path = _write_results(args, per_seed, max(gaps) if reference is not None else -1.0)
    with open(path) as file:
        print(file.read())


if __name__ == "__main__":
    main()
