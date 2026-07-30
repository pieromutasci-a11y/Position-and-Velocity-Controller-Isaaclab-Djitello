# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Config del CONTROLLORE DI POSIZIONE del drone (Crazyflie), skrl.
# Livello esterno (25Hz); avvolge il controllore di velocita' gia' addestrato
# (interno, 50Hz, CONGELATO) e lo richiama in ZOH.
#
# ===================================================================================
# VERSIONE: OBIETTIVO GLOBALE (QUESTA VERSIONE)
#
# Rispetto alla versione precedente (obiettivo asimmetrico full/uniciclo con
# guardia a cerniera su full):
#
#   - ELIMINATO ogni concetto di "proteggere full" / "obiettivo primario
#     uniciclo": l'ERRORE e la SMOOTHNESS sono ora GLOBALI, calcolati come
#     media su TUTTI gli env del batch di reset, senza distinzione di
#     modalita' (full/uniciclo/variabile/canonica/hover). Non esistono piu'
#     obj_full_baseline, obj_w_full_guard, e lo split err_full/err_uniciclo.
#     La smoothness e' un termine SEPARATO nell'obiettivo (prima era fusa
#     dentro l'errore con peso 0.3, ora no: ognuna delle 5 componenti e'
#     indipendente e pesata a se'.
#
#   - RETROMARCIA e USO DI VY (riferimento generato E velocita' reale)
#     restano invece MASCHERATI SOLO SU UNICICLO: sono comportamenti che ha
#     senso penalizzare solo quando l'asse vy e' vincolato dal compito. Full
#     puo' legittimamente strafare lateralmente, quindi penalizzarlo anche
#     li' sarebbe un errore di design, non una generalizzazione. Vedi env,
#     funzione _reset_idx.
#
#   - Ogni componente dell'obiettivo e' loggata SEPARATAMENTE su
#     wandb/tensorboard: il valore grezzo (Objective/*_mean o Diag/*,
#     confrontabile tra trial con pesi diversi) E il contributo GIA' pesato
#     (Objective/*_contrib, nella stessa unita' del composite_target).
#
#   - CSV: l'env accumula in memoria lo storico di ciascuna componente
#     (una entry per chiamata di _reset_idx) ed espone
#     get_objective_history(). train_manual.py lo trasforma in 6 file CSV
#     (uno per componente + uno per il composite) caricati come wandb
#     Artifact a fine training: NESSUN file persistito sull'host, tutto
#     vive nel run di wandb.
#
# NUOVA FUNZIONE OBIETTIVO (Objective/composite_target, da minimizzare):
#
#   err_mean              = mean(final_dist + 0.2*final_yaw_err)   SU TUTTI GLI ENV
#   smooth_mean           = mean(action_oscillation_raw)           SU TUTTI GLI ENV
#   reverse_uniciclo_mean = mean(reverse_amount  | is_uniciclo)    SOLO UNICICLO
#   vy_ref_uniciclo_mean  = mean(|vy_ref|        | is_uniciclo)    SOLO UNICICLO
#   vy_real_uniciclo_mean = mean(|vy_real|       | is_uniciclo)    SOLO UNICICLO
#
#   composite_target = obj_w_err     * err_mean
#                     + obj_w_smooth  * smooth_mean
#                     + obj_w_rev     * reverse_uniciclo_mean
#                     + obj_w_vy_ref  * vy_ref_uniciclo_mean
#                     + obj_w_vy_real * vy_real_uniciclo_mean
#
#   Tutte le metriche sono GREZZE (non scalate da rew_scale_*): il loro
#   significato non cambia quando lo sweep varia i pesi della reward, quindi
#   i trial restano confrontabili tra loro.
#
# I PESI obj_* SONO FISSI, NON SWEEPPATI di proposito: se lo sweep potesse
# sceglierli, potrebbe azzerarli per "vincere" piu' facilmente, rendendo
# l'obiettivo diverso da un trial all'altro e i trial non confrontabili.
# ===================================================================================

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

    # -- PROPRIETA' INERZIALI: geometria Crazyflie, massa/inerzia del DJI Tello -- #
    # Deve coincidere con quella usata per addestrare la policy di velocita' indicata
    # in low_level_policy_path: quella policy e' CONGELATA e vale solo per la dinamica
    # con cui e' stata addestrata. Vedi ../drone_physics.py, nota 3.
    drone_inertial: DroneInertialCfg = TELLO_INERTIAL_CFG.replace()

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
        "/workspace/isaaclab/logs/skrl/vel_controller/2026-07-30_08-29-15_ppo_torch/checkpoints/best_agent.pt"
    )
    # -- ATTUATORI: tarati sul Tello, non piu' sul Crazyflie -- #
    # DEVE coincidere con i valori usati per addestrare la policy di velocita' in
    # low_level_policy_path qui sopra. Derivazione in ../drone_physics.py.
    thrust_to_weight = 1.8
    moment_scale = (0.021, 0.021, 0.0085)  # (roll, pitch, yaw)
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
    # SEQUENZA DI RIFERIMENTI CANONICA (solo dentro la modalita' variabile)
    # ===================================================================
    prob_canonical_variabile = 0.35
    canonical_room_margin = 0.8
    canonical_min_dist = 0.5

    # ===================================================================
    # MASCHERA GRADI DI LIBERTA' (DoF mask) — SOLO DUE MODALITA'
    # ===================================================================
    dof_mask_set = (
        (1.0, 1.0, 1.0, 1.0),  # full
        (1.0, 0.0, 1.0, 1.0),  # uniciclo (vx + vz + wz, no vy)
    )
    dof_mask_probs = (0.5, 0.5)

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
    # REWARD (BASE, condivisa)
    # ===================================================================
    rew_scale_alive = 0.2

    rew_scale_position_approach = 25.0
    reward_exp_beta = 0.5
    rew_scale_position_prec = 25.0
    reward_exp_beta_prec = 3.0

    rew_scale_yaw_error = -8.0

    rew_scale_yaw_prec = 5.0
    reward_exp_beta_yaw_prec = 4.0

    # ANTI-OSCILLAZIONE #1: penalizza i salti step-to-step del comando.
    rew_scale_action_smoothness = -12.0

    rew_scale_reg_ang_vel_xy = -0.00

    # ANTI-OSCILLAZIONE #2: smorza il chattering di wz.
    rew_scale_reg_ang_vel_wz = -0.03

    # ANTI-OSCILLAZIONE #3: penalizza comandi ampi. Attivo per full E
    # uniciclo; il canale wz resta escluso solo per l'uniciclo, che ne ha
    # bisogno per virare.
    rew_scale_aggressive_cmd = -8.0

    # ===================================================================
    # PENALITA' RETROMARCIA (nella reward e' moltiplicata per is_uniciclo,
    # quindi di fatto attiva solo li'; nell'obiettivo dello sweep e'
    # mascherata allo stesso modo, vedi sotto e l'env)
    # ===================================================================
    rew_scale_reverse_vx = -70.0

    # ===================================================================
    # UNICICLO — SOLO VX E WZ (attivi SOLO con is_uniciclo=True)
    # ===================================================================

    # U.3 — GATE YAW MORBIDO: rampa lineare 0->1 con la distanza.
    yaw_gate_dist_uniciclo = 0.18

    # U.1 — PENALITA' VY sul riferimento GENERATO dalla rete.
    rew_scale_vy_penalty_uniciclo = -300.0

    # U.1bis — PENALITA' VY REALE (velocita' body-frame effettiva,
    # root_lin_vel_b[:,1]), non il riferimento generato.
    rew_scale_vy_real_penalty_uniciclo = -150.0

    # U.5 — ANTI-OVERSHOOT: penalita' sulla vx REALE vicino al target.
    approach_brake_dist = 0.5
    rew_scale_approach_brake = -15.0

    # AUTORITA' VY: False = la rete puo' generare vy (penalizzata da U.1).
    uniciclo_vy_hard_mask = False

    # ===================================================================
    # FUNZIONE OBIETTIVO DELLO SWEEP (pesi FISSI, NON sweeppati)
    # Vedi spiegazione estesa in testa al file.
    # ===================================================================

    # Peso dell'errore GLOBALE (posizione + yaw), su TUTTI gli env.
    obj_w_err = 1.0

    # Peso della smoothness GLOBALE (oscillazione grezza dei comandi),
    # su TUTTI gli env. Termine SEPARATO, non piu' fuso dentro l'errore.
    obj_w_smooth = 1.0

    # Peso della retromarcia, SOLO UNICICLO (metrica grezza: quanto vx_ref
    # e' negativa in media).
    obj_w_rev = 2.0

    # Peso dell'uso di vy RIFERIMENTO, SOLO UNICICLO (metrica grezza
    # |vy_ref| media).
    obj_w_vy_ref = 2.0

    # Peso dell'uso di vy REALE, SOLO UNICICLO (metrica grezza |vy_reale|
    # media, velocita' body-frame effettivamente realizzata).
    obj_w_vy_real = 2.0

    # -- eventi impulsivi (SENZA step_dt) --
    rew_scale_target_reached = 80.0
    oob_reward = -50.0
    tilt_death_reward = -50.0

    # ===================================================================
    # DEBUG / LOGGING
    # ===================================================================
    debug_print_every_n_resets = 50