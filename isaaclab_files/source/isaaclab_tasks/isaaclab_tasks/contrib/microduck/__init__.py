# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat and rough velocity walking, the single-leg stand towards hopping, one-foot flamingo balance and hopping.

Select the backlash robot with ``presets=backlash``.
"""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Velocity-Flat-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MicroDuckVelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Hop-Stand-Flat-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hop_env_cfg:MicroDuckHopStandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckHopStandPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Flamingo-Flat-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flamingo_env_cfg:MicroDuckFlamingoFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckFlamingoPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-FlamingoHop-Flat-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flamingo_hop_env_cfg:MicroDuckFlamingoHopFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckFlamingoHopPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-FlamingoCycle-Flat-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flamingo_cycle_env_cfg:MicroDuckFlamingoCycleFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckFlamingoCyclePPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MicroDuckVelocityRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckRoughPPORunnerCfg",
    },
)
