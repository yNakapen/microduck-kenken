# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record MP4 videos of an exported MicroDuck flamingo policy on Newton MJWarp under the evaluation conditions.

A batch of environments runs one episode each, as in ``eval_flamingo.py``, while a camera per environment follows
the trunk. Frames come from Newton's Warp ray-tracing renderer, so no OpenGL library is needed. Every environment's
first episode is streamed to its own MP4; the episodes matching ``--keep`` are kept and the others deleted. The
script prints each kept video's path, outcome and file size.

Writing MP4 needs ffmpeg, which the ``imageio-ffmpeg`` wheel bundles:

.. code-block:: bash

    uv run --with imageio --with imageio-ffmpeg python scripts/reinforcement_learning/microduck/record_flamingo.py \\
        --policy logs/rsl_rl/microduck_flamingo_flat/<run>/exported/policy.pt --push 0.0 --keep success
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import imageio.v2 as imageio
import torch
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flamingo_eval import (
    make_cycle_env,
    make_hop_env,
    run_cycle_protocol,
    run_hop_protocol,
)
from isaaclab_tasks.contrib.microduck.flamingo_metrics import FlamingoEpisodeMetrics, set_eval_conditions
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

CAMERA_OFFSET = (0.45, -0.45, 0.12)
"""Camera eye relative to the trunk [m], in the world frame."""
CAMERA_TARGET_OFFSET = (0.0, 0.0, -0.04)
"""Look-at point relative to the trunk [m], in the world frame."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="IsaacContrib-Flamingo-Flat-MicroDuck")
    parser.add_argument("--policy", required=True, help="Exported TorchScript policy.pt.")
    parser.add_argument("--push", type=float, default=0.0, help="Push magnitude [m/s].")
    parser.add_argument(
        "--cycle",
        action="store_true",
        help="Run the flamingo-cycle protocol of eval_flamingo_cycle.py (stand, lift, hold, lower) instead of one "
        "stage-1 episode; the first half of the environments stands on the right foot.",
    )
    parser.add_argument(
        "--hop",
        action="store_true",
        help="Run the hop protocol of eval_flamingo_hop.py (still, hop, still) on the right foot.",
    )
    parser.add_argument("--num_envs", type=int, default=8, help="Episodes recorded in parallel.")
    parser.add_argument("--keep", choices=("success", "failure", "all"), default="success")
    parser.add_argument("--max_videos", type=int, default=1, help="Kept videos at most, in environment order.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default=None, help="Defaults to <policy dir>/videos_push<push>.")
    return parser.parse_args()


def _camera_cfg(args: argparse.Namespace) -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/RecordCamera",
        update_period=0.0,
        width=args.width,
        height=args.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 20.0)),
        renderer_cfg=NewtonWarpRendererCfg(),
    )


def _make_env(args: argparse.Namespace):
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    set_eval_conditions(cfg, args.push)
    cfg.scene.record_camera = _camera_cfg(args)
    sim_utils.create_new_stage()
    return gym.make(args.task, cfg=cfg).unwrapped


def _aim_cameras(env) -> None:
    """Point every environment's camera at its trunk from a fixed world-frame offset."""
    trunk = env.scene["robot"].data.root_link_pos_w.torch
    offset = torch.tensor(CAMERA_OFFSET, device=trunk.device)
    target = torch.tensor(CAMERA_TARGET_OFFSET, device=trunk.device)
    env.scene["record_camera"].set_world_poses_from_view(trunk + offset, trunk + target)


def _write_frames(env, writers, recording: torch.Tensor) -> None:
    frames = env.scene["record_camera"].data.output["rgb"].torch[..., :3].cpu().numpy()
    for i in recording.cpu().nonzero().flatten().tolist():
        writers[i].append_data(frames[i])


def _record_stage_one(args: argparse.Namespace, policy, writers) -> tuple[dict, str]:
    env = _make_env(args)
    try:
        stance_cfg = SceneEntityCfg("contact_forces", body_names=["ankle_right"])
        swing_cfg = SceneEntityCfg("contact_forces", body_names=["ankle_left"])
        stance_cfg.resolve(env.scene)
        swing_cfg.resolve(env.scene)
        robot = env.scene["robot"]
        metrics = FlamingoEpisodeMetrics(env.num_envs, env.step_dt, device=env.device)
        recording = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        obs, _ = env.reset()
        with torch.inference_mode():
            for _ in range(env.max_episode_length):
                _aim_cameras(env)
                obs, _, terminated, truncated, _ = env.step(policy(obs["policy"]))
                metrics.update(
                    stance_contact=mdp.foot_in_contact(env, stance_cfg).bool(),
                    swing_contact=mdp.foot_in_contact(env, swing_cfg).bool(),
                    done=terminated | truncated,
                    time_out=truncated & ~terminated,
                    gravity_b=robot.data.projected_gravity_b.torch,
                )
                # A terminal step's frame already shows the reset robot, so the episode ends one frame earlier.
                recording &= ~(terminated | truncated)
                _write_frames(env, writers, recording)
    finally:
        env.close()
        SimulationContext.clear_instance()
    return metrics.summary(), "single support {longest:.2f} s, duration {duration:.2f} s"


def _record_cycle(args: argparse.Namespace, policy, writers) -> tuple[dict, str]:
    env = make_cycle_env(
        args.task, args.device, args.num_envs, args.seed, args.push, scene_extras={"record_camera": _camera_cfg(args)}
    )

    def on_step(env, alive):
        _write_frames(env, writers, alive)
        _aim_cameras(env)

    try:
        _aim_cameras(env)
        summary = run_cycle_protocol(env, policy, on_step=on_step)
    finally:
        env.close()
        SimulationContext.clear_instance()
    return summary, "side {side:+.0f}, hold single support {hold:.2f}, stand two feet {stand:.2f}"


def _record_hop(args: argparse.Namespace, policy, writers) -> tuple[dict, str]:
    env = make_hop_env(
        args.task, args.device, args.num_envs, args.seed, args.push, scene_extras={"record_camera": _camera_cfg(args)}
    )

    def on_step(env, alive):
        _write_frames(env, writers, alive)
        _aim_cameras(env)

    try:
        _aim_cameras(env)
        summary = run_hop_protocol(env, policy, on_step=on_step)
    finally:
        env.close()
        SimulationContext.clear_instance()
    return summary, "hops {hops:.0f}, swing touchdowns {touchdowns:.0f}, still single support {still:.2f} s"


def main() -> None:
    args = _parse_args()
    policy = torch.jit.load(args.policy, map_location=args.device).eval()
    mode = "_cycle" if args.cycle else "_hop" if args.hop else ""
    default_dir = f"videos{mode}_push{args.push:g}"
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.policy)), default_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"seed{args.seed}_env{i:03d}.mp4") for i in range(args.num_envs)]
    # Every task steps at 50 Hz (dt 0.005 s, decimation 4).
    writers = [imageio.get_writer(p, fps=50, codec="libx264", quality=7, macro_block_size=8) for p in paths]
    try:
        record = _record_cycle if args.cycle else _record_hop if args.hop else _record_stage_one
        summary, detail = record(args, policy, writers)
    finally:
        for writer in writers:
            writer.close()

    success = summary["success"].cpu()
    wanted = {"success": success, "failure": ~success, "all": torch.ones_like(success)}
    keep = wanted[args.keep].nonzero().flatten().tolist()[: args.max_videos]
    for i, path in enumerate(paths):
        if i not in keep:
            os.remove(path)
    if not keep:
        print(f"[record_flamingo] no {args.keep} episode among {args.num_envs}; try more environments or seeds")
    for i in keep:
        size = os.path.getsize(paths[i])
        values = {
            "longest": float(summary.get("longest_single_support", torch.zeros(args.num_envs))[i]),
            "duration": float(summary.get("duration", torch.zeros(args.num_envs))[i]),
            "side": float(summary.get("side", torch.zeros(args.num_envs))[i]),
            "hold": float(summary.get("hold_single_support", torch.zeros(args.num_envs))[i]),
            "stand": float(summary.get("stand_two_feet", torch.zeros(args.num_envs))[i]),
            "hops": float(summary.get("hops", torch.zeros(args.num_envs))[i]),
            "touchdowns": float(summary.get("swing_touchdowns", torch.zeros(args.num_envs))[i]),
            "still": float(summary.get("still_single_support", torch.zeros(args.num_envs))[i]),
        }
        print(
            f"[record_flamingo] {paths[i]}: success={bool(success[i])}, fell={bool(summary['fell'][i])}, "
            f"{detail.format(**values)}, size {size} bytes ({size / 2**20:.2f} MiB)"
        )


if __name__ == "__main__":
    main()
