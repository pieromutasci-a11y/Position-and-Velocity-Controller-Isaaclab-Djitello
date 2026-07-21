# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Config del CONTROLLORE DI POSIZIONE del drone (Crazyflie), skrl.
# Livello esterno (25Hz); avvolge il controllore di velocita' gia' addestrato
# (interno, 50Hz, CONGELATO) e lo richiama in ZOH.
#
# CAMBIAMENTI RISPETTO ALLA VERSIONE PRECEDENTE:
#   - observation_space: 48 -> 52 (aggiunto integrale errore pos+yaw, 4 valori)
#   - Nuovi parametri integrale: integral_tau_s, integral_clamp, integral_obs_scale
#   - reg_ang_vel: unico parametro -> separato in reg_ang_vel_xy e reg_ang_vel_wz
#   - Nuovo parametro: rew_scale_aggressive_cmd
#   - Nuovo parametro: yaw_gate_dist_uniciclo (gate yaw per uniciclo, vedi sotto)
#   INCOMPATIBILE con checkpoint addestrati a 48 input.
#
# FIX UNICICLO:
#   - yaw_gate_dist_uniciclo: 0.6 -> 0.10 (il drone naviga con yaw libero e
#     allinea solo nell'ultima fase, appena sotto target_reach_threshold)
#   - Il gate e' ora applicato ANCHE nel criterio di avanzamento waypoint
#     (_update_waypoint), non solo nella reward. Senza questo fix il criterio
#     di convergenza pretendeva yaw allineato anche lontano dal target,
#     contraddicendo la reward gated.

from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg


@configclass
class MyDronePosEnvCfg(DirectRLEnvCfg):
    # ===================================================================
    # ENV / TIMING
    # ===================================================================
    decimation = 4
    low_level_decimation = 2
    episode_length_s = 30.0

    # ===================================================================
    # CODA DI WAYPOINT + FINESTRA DI PREVIEW
    # ===================================================================
    n_waypoints = 4
    wp_preview_horizon = 4

    # ===================================================================
    # SPACES
    # ===================================================================
    action_space = 4
    observation_space = 52
    #  3  pos drone            (frame MONDO/env, m)
    #  2  sin(yaw), cos(yaw)   (frame MONDO)
    #  3  lin_vel_b            (frame CORPO, m/s)
    #  3  ang_vel_b            (frame CORPO, rad/s)
    #  3  proj_grav_b          (frame CORPO)
    #  4  prev_hl_actions      ([-1,1])
    #  5  FEEDBACK   : (w0 - pos)   + sin/cos(yaw_w0 - yaw)
    #  5  FEEDFORWARD1: (w1 - w0)   + sin/cos(yaw_w1 - yaw_w0)
    #  5  FEEDFORWARD2: (w2 - w1)   + sin/cos(yaw_w2 - yaw_w1)
    #  5  FEEDFORWARD3: (w3 - w2)   + sin/cos(yaw_w3 - yaw_w2)
    #  6  CLEARANCE   : [x_max-x, y_max-y, z_max-z, x-x_min, y-y_min, z-z_min]
    #  4  DOF_MASK    : [vx, vy, vz, wz] in {0,1}
    #  4  INTEGRALE   : [int_ex, int_ey, int_ez, int_eyaw] / integral_obs_scale
    # = 52
    state_space = 0
    debug_vis = True

    # ===================================================================
    # SIMULAZIONE
    # ===================================================================
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

    # ===================================================================
    # SCENA / ROBOT
    # ===================================================================
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # ===================================================================
    # VISUALIZZAZIONE
    # ===================================================================
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

    # ===================================================================
    # LOW-LEVEL POLICY
    # ===================================================================
    low_level_policy_path = (
        "/workspace/isaaclab/logs/skrl/vel_controller/2026-07-09_16-01-03_ppo_torch/checkpoints/best_agent.pt"
    )
    thrust_to_weight = 1.9
    moment_scale = 0.01
    target_lin_vel_xy_scale = 1.0
    target_lin_vel_z_scale = 1.0
    target_yaw_vel_scale = 1.5

    # ===================================================================
    # STANZA ASIMMETRICA
    # ===================================================================
    room_x_pos_range = (1.0, 4.0)
    room_x_neg_range = (1.0, 4.0)
    room_y_pos_range = (1.0, 4.0)
    room_y_neg_range = (1.0, 4.0)
    room_z_max_range = (1.0, 4.0)
    min_z_pos = 0.1
    target_room_margin = 0.8

    # ===================================================================
    # SPAWN + TAKE-OFF DA TERRA
    # ===================================================================
    spawn_room_margin = 0.8
    spawn_low_prob = 0.3
    spawn_low_z_range = (0.12, 0.25)

    # ===================================================================
    # MODALITA' DI RIFERIMENTO
    # ===================================================================
    prob_variabile = 0.8
    hover_frac_in_singolo = 0.30

    # ===================================================================
    # MASCHERA GRADI DI LIBERTA' (DoF mask)
    # ===================================================================
    dof_mask_set = (
        (1.0, 1.0, 1.0, 1.0),  # full
        (1.0, 0.0, 1.0, 1.0),  # uniciclo         (vx + vz + wz, no vy)
        (1.0, 1.0, 1.0, 0.0),  # planare_olonomo  (vx + vy + vz, no wz)
    )
    dof_mask_probs = (0.50, 0.25, 0.25)

    # ===================================================================
    # AVANZAMENTO WAYPOINT
    # ===================================================================
    target_reach_threshold = 0.15
    target_reach_yaw_threshold = 0.20
    target_hold_time_s = 1.2

    # ===================================================================
    # TERMINAZIONE
    # ===================================================================
    terminate_su_tilt_eccessivo = True
    max_tilt_deg = 65.0

    # ===================================================================
    # LEAKY INTEGRATOR ERRORE POS+YAW
    # ===================================================================
    integral_tau_s = 5.0
    integral_clamp = 1.0
    integral_obs_scale = 0.5

    # ===================================================================
    # REWARD
    # ===================================================================
    rew_scale_alive = 0.2

    rew_scale_position_approach = 15.0
    reward_exp_beta = 0.5
    rew_scale_position_prec = 15.0
    reward_exp_beta_prec = 3.0

    rew_scale_yaw_error = -8.0
    rew_scale_yaw_prec = 5.0
    reward_exp_beta_yaw_prec = 4.0

    # Distanza [m] sotto la quale l'errore di yaw inizia a contare nella reward
    # E nel criterio di avanzamento waypoint, SOLO in modalita' uniciclo
    # (dof_mask con vy=0).
    #
    # LOGICA: in uniciclo il drone PUO' traslare solo avanzando lungo body-x,
    # quindi lontano dal target lo yaw deve essere LIBERO di puntare verso il
    # target. Solo quando il drone e' quasi arrivato (dist < yaw_gate_dist)
    # ha senso richiedere l'allineamento allo yaw del waypoint.
    #
    # Valore scelto appena sotto target_reach_threshold (0.15) cosi' il gate
    # si attiva solo nell'ultima fase di avvicinamento.
    # Non tocca le altre modalita' (full, planare_olonomo): li' il peso resta 1.
    yaw_gate_dist_uniciclo = 0.15

    rew_scale_action_smoothness = -1.5

    # Separazione roll/pitch vs yaw: wz penalizzato meno per non ostacolare uniciclo
    rew_scale_reg_ang_vel_xy = -0.05
    rew_scale_reg_ang_vel_wz = -0.01

    # Penalita' velocita' reali assi mascherati (effetto)
    rew_scale_vel_mask_penalty = -6.0

    # Penalita' aggressivita' comandi con mask non-full (causa).
    # Zero per mask full. Incentiva comandi moderati quando i DoF sono ridotti.
    rew_scale_aggressive_cmd = -0.5


    # Reward bearing: premia l'allineamento body-x verso il target (solo uniciclo,
    # solo lontano dal target). Insegna esplicitamente "punta verso il target
    # prima di avanzare con vx". Scala negativa = penalita' per disallineamento.
    rew_scale_bearing = -4.0

    
    # -- eventi impulsivi (SENZA step_dt) --
    rew_scale_target_reached = 80.0
    oob_reward = -50.0
    tilt_death_reward = -50.0

    # ===================================================================
    # DEBUG / LOGGING
    # ===================================================================
    debug_print_every_n_resets = 50