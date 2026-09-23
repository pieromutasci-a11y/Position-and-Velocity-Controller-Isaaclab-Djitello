# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG

from ..drone_physics import DroneInertialCfg, TELLO_INERTIAL_CFG

class MyDroneVelEnvWindow(BaseEnvWindow):

    def __init__(self, env, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)

@configclass
class MyDroneVelEnvCfg(DirectRLEnvCfg):

    terminate_su_tilt_eccessivo: bool = True

    episode_length_s = 10.0
    decimation = 2

    action_space = 4

    observation_space = 17
    state_space = 0

    debug_vis = True
    ui_window_class_type = MyDroneVelEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    drone_inertial: DroneInertialCfg = TELLO_INERTIAL_CFG.replace()

    thrust_to_weight = 1.8
    moment_scale = (0.021, 0.021, 0.0085)

    max_lin_vel_xy = 1.0
    max_lin_vel_z = 1.0
    max_ang_vel_z = 1.5

    max_tilt_deg = 65.0

    tilt_free_deg = 35.0

    vel_error_reward_scale = -7.0
    action_rate_reward_scale_start = -0.01
    action_rate_reward_scale_end = -0.1
    action_rate_curriculum_steps = 40000

    unwanted_ang_vel_reward_scale = -0.05
    tilt_reward_scale = -2.0
    died_penalty = -10.0

    hover_frac = 0.15

    curriculum_livelli = (
        {"pesi_modalita": (0.0, 0.0, 0.95, 0.0, 0.05), "hold_min_s": 3.5, "hold_max_s": 5.0, "mode3_min_frac": 0.3},
        {"pesi_modalita": (0.475, 0.0, 0.475, 0.0, 0.05), "hold_min_s": 3.0, "hold_max_s": 4.0, "mode3_min_frac": 0.3},
        {"pesi_modalita": (0.40, 0.15, 0.30, 0.10, 0.05), "hold_min_s": 2.0, "hold_max_s": 3.0, "mode3_min_frac": 0.4},
        {"pesi_modalita": (0.35, 0.15, 0.20, 0.25, 0.05), "hold_min_s": 1.5, "hold_max_s": 2.5, "mode3_min_frac": 0.5},
        {"pesi_modalita": (0.35, 0.15, 0.15, 0.30, 0.05), "hold_min_s": 1.0, "hold_max_s": 2.5, "mode3_min_frac": 0.6},
    )

    curriculum_soglia_successo = 0.02
    curriculum_successi_richiesti = 5
    curriculum_transiente_skip_s = 0.3

    debug_print_every_n_resets = 50
