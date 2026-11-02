# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted rollouts of the MicroDuck flamingo cycle and hop tasks, shared by their evaluation and recording scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from . import mdp
from .flamingo_metrics import (
    HOP_STILL_WINDOW,
    CycleEpisodeMetrics,
    HopEpisodeMetrics,
    cycle_protocol_command,
    hop_protocol_command,
    set_eval_conditions,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def make_cycle_env(
    task: str, device: str, num_envs: int, seed: int, push: float, scene_extras: dict | None = None
) -> ManagerBasedRLEnv:
    """Build the cycle task under the evaluation conditions, every environment spawning standing at HOME."""
    cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    cfg.seed = seed
    set_eval_conditions(cfg, push)
    cfg.events.reset_flamingo_cycle_pose.params["in_pose_prob"] = 0.0
    for name, value in (scene_extras or {}).items():
        setattr(cfg.scene, name, value)
    sim_utils.create_new_stage()
    return gym.make(task, cfg=cfg).unwrapped


def run_cycle_protocol(env, policy, on_step=None) -> dict:
    """Roll the scripted protocol once; ``on_step(env, alive)`` runs after each step. Returns the cycle summary."""
    term = env.command_manager.get_term("base_velocity")
    side = torch.where(torch.arange(env.num_envs, device=env.device) < env.num_envs // 2, 1.0, -1.0)
    feet_sensor = SceneEntityCfg("contact_forces", body_names=["ankle_left", "ankle_right"], preserve_order=True)
    feet = SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"], preserve_order=True)
    feet_sensor.resolve(env.scene)
    feet.resolve(env.scene)
    robot = env.scene["robot"]
    metrics = CycleEpisodeMetrics(env.num_envs, env.step_dt, device=env.device)
    obs, _ = env.reset()
    term.set_alpha(torch.arange(env.num_envs, device=env.device), torch.zeros(env.num_envs, device=env.device))
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    # The time-out step's state already shows the reset robot, so the protocol stops one step short of it.
    with torch.inference_mode():
        for step in range(env.max_episode_length - 1):
            term.set_command(cycle_protocol_command(step * env.step_dt, side), side)
            obs, _, terminated, truncated, _ = env.step(policy(obs["policy"]))
            sensor = env.scene.sensors["contact_forces"]
            down = sensor.data.current_contact_time.torch[:, feet_sensor.body_ids] > 0.0
            right = side > 0
            stance = torch.where(right, down[:, 1], down[:, 0])
            swing = torch.where(right, down[:, 0], down[:, 1])
            sites = mdp.foot_site_pos_w(robot, feet.body_ids)
            swing_z = torch.where(right, sites[:, 0, 2], sites[:, 1, 2]) - env.scene.env_origins[:, 2]
            metrics.update(
                time=(step + 1) * env.step_dt,
                stance_contact=stance,
                swing_contact=swing,
                swing_height=swing_z,
                done=terminated | truncated,
                time_out=truncated & ~terminated,
            )
            alive &= ~(terminated | truncated)
            if on_step is not None:
                on_step(env, alive)
    summary = metrics.summary()
    summary["side"] = side
    return summary


def make_hop_env(
    task: str, device: str, num_envs: int, seed: int, push: float, scene_extras: dict | None = None
) -> ManagerBasedRLEnv:
    """Build the hop task under the evaluation conditions; every environment spawns in the one-foot pose."""
    cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    cfg.seed = seed
    set_eval_conditions(cfg, push)
    for name, value in (scene_extras or {}).items():
        setattr(cfg.scene, name, value)
    sim_utils.create_new_stage()
    return gym.make(task, cfg=cfg).unwrapped


def run_hop_protocol(env: ManagerBasedRLEnv, policy, on_step=None) -> dict:
    """Roll the scripted still, hop, still protocol once on the right foot; ``on_step(env, alive)`` runs after each
    step. Returns the hop summary."""
    term = env.command_manager.get_term("base_velocity")
    stance_cfg = SceneEntityCfg("contact_forces", body_names=["ankle_right"])
    swing_cfg = SceneEntityCfg("contact_forces", body_names=["ankle_left"])
    stance_cfg.resolve(env.scene)
    swing_cfg.resolve(env.scene)
    sensor = env.scene.sensors["contact_forces"]
    metrics = HopEpisodeMetrics(env.num_envs, env.step_dt, device=env.device)
    obs, _ = env.reset()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    steps = round(HOP_STILL_WINDOW[1] / env.step_dt)
    with torch.inference_mode():
        for step in range(steps):
            term.set_command(hop_protocol_command(step * env.step_dt, env.num_envs, env.device))
            obs, _, terminated, truncated, _ = env.step(policy(obs["policy"]))
            stance = stance_cfg.body_ids[0]
            metrics.update(
                time=(step + 1) * env.step_dt,
                stance_contact=mdp.foot_in_contact(env, stance_cfg).bool(),
                stance_landing=sensor.compute_first_contact(env.step_dt).torch[:, stance].bool(),
                stance_last_air_time=sensor.data.last_air_time.torch[:, stance],
                swing_contact=mdp.foot_in_contact(env, swing_cfg).bool(),
                done=terminated | truncated,
                time_out=truncated & ~terminated,
            )
            alive &= ~(terminated | truncated)
            if on_step is not None:
                on_step(env, alive)
    return metrics.summary()
