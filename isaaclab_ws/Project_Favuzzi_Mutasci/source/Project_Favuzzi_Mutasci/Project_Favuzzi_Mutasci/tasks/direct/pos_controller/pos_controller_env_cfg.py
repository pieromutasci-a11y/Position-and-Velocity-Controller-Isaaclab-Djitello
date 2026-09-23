# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

from ..drone_physics import DroneInertialCfg, TELLO_INERTIAL_CFG

@configclass
class MyDronePosEnvCfg(DirectRLEnvCfg):
    decimation = 4
    low_level_decimation = 2
    episode_length_s = 30.0

    n_waypoints = 4
    wp_preview_horizon = 4

    action_space = 4
    observation_space = 52
    state_space = 0
    debug_vis = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    drone_inertial: DroneInertialCfg = TELLO_INERTIAL_CFG.replace()

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    target_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/target_marker",
        markers={
            "target": sim_utils.ConeCfg(
                radius=0.1,
                height=0.2,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    room_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/room_marker",
        markers={
            "room": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.6, 1.0),
                    opacity=0.12,
                ),
            ),
        },
    )

    low_level_policy_path = (
        "/workspace/isaaclab/logs/skrl/vel_controller/2026-07-30_08-29-15_ppo_torch/checkpoints/best_agent.pt"
    )
    thrust_to_weight = 1.8
    moment_scale = (0.021, 0.021, 0.0085)
    target_lin_vel_xy_scale = 1.0
    target_lin_vel_z_scale = 1.0
    target_yaw_vel_scale = 1.5

    room_x_pos_range = (1.0, 4.0)
    room_x_neg_range = (1.0, 4.0)
    room_y_pos_range = (1.0, 4.0)
    room_y_neg_range = (1.0, 4.0)
    room_z_max_range = (1.0, 4.0)
    min_z_pos = 0.1
    target_room_margin = 0.8

    spawn_room_margin = 0.8
    spawn_low_prob = 0.3
    spawn_low_z_range = (0.12, 0.25)

    prob_variabile = 0.8
    hover_frac_in_singolo = 0.30

    prob_canonical_variabile = 0.35
    canonical_room_margin = 0.8
    canonical_min_dist = 0.5

    dof_mask_set = (
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
    )
    dof_mask_probs = (0.5, 0.5)

    target_reach_threshold = 0.15
    target_reach_yaw_threshold = 0.20
    target_hold_time_s = 1.2

    terminate_su_tilt_eccessivo = True
    max_tilt_deg = 65.0

    integral_tau_s = 5.0
    integral_clamp = 1.0
    integral_obs_scale = 0.5

    rew_scale_alive = 0.2

    rew_scale_position_approach = 25.0
    reward_exp_beta = 0.5
    rew_scale_position_prec = 25.0
    reward_exp_beta_prec = 3.0

    rew_scale_yaw_error = -8.0

    rew_scale_yaw_prec = 5.0
    reward_exp_beta_yaw_prec = 4.0

    rew_scale_action_smoothness = -12.0

    rew_scale_reg_ang_vel_xy = -0.00

    rew_scale_reg_ang_vel_wz = -0.03

    rew_scale_aggressive_cmd = -8.0

    rew_scale_reverse_vx = -70.0

    yaw_gate_dist_uniciclo = 0.18

    rew_scale_vy_penalty_uniciclo = -300.0

    rew_scale_vy_real_penalty_uniciclo = -150.0

    approach_brake_dist = 0.5
    rew_scale_approach_brake = -15.0

    uniciclo_vy_hard_mask = False

    obj_w_err = 1.0

    obj_w_smooth = 1.0

    obj_w_rev = 2.0

    obj_w_vy_ref = 2.0

    obj_w_vy_real = 2.0

    rew_scale_target_reached = 80.0
    oob_reward = -50.0
    tilt_death_reward = -50.0

    debug_print_every_n_resets = 50
