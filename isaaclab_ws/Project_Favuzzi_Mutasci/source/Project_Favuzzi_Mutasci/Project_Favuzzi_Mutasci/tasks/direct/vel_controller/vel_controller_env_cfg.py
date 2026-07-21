# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Config del CONTROLLORE DI VELOCITA' (livello piu' interno/veloce della gerarchia).
#
# FIX rispetto alla versione precedente (skrl, non funzionante):
#   - la reward di vel_error NON viene piu' divisa per n_assi_attivi (vedi
#     vel_controller_env.py, _get_rewards). Nella versione precedente questa divisione
#     diluiva il segnale di tracking proprio nel caso piu' difficile (modalita' 0/3,
#     tutti e 4 gli assi attivi), l'opposto di quello che serve. La normalizzazione per
#     asse attivo resta SOLO nella valutazione del curriculum (dove il vecchio env
#     rsl_rl la usava gia' correttamente), non nella reward vera e propria.
#   - piano di riferimento visivo ora con griglia (asset standard Isaac Lab) invece del
#     MeshCuboidCfg bianco pieno, e con il collider rimosso via USD dopo lo spawn (vedi
#     _setup_scene): drone e griglia visibili, ma ancora nessuna interazione fisica col
#     suolo, come richiesto per questo controllore (che vola in spazio libero).

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG


# =======================================================================
# FINESTRA PER DEBUG
# =======================================================================
class MyDroneVelEnvWindow(BaseEnvWindow):
    """Finestra UI di debug per l'ambiente del controllore di velocita'.
    Per ora vuota, pronta per aggiungere elementi in futuro (slider target, ecc.)."""

    def __init__(self, env, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)


# =======================================================================
# CONFIGURAZIONE AMBIENTE
# =======================================================================
@configclass
class MyDroneVelEnvCfg(DirectRLEnvCfg):

    terminate_su_tilt_eccessivo: bool = True
    """Se False, il drone non viene mai resettato per tilt eccessivo (utile in evaluation
    per vedere se il controllore riesce a riassestarsi da solo dopo un tilt forte). Questo
    controllore vola in spazio libero, senza pavimento fisico: l'unica condizione di
    morte e' il tilt eccessivo."""

    episode_length_s = 10.0
    decimation = 2

    # azioni: thrust totale (1) + 3 momenti (3) = 4 valori, in [-1,1]
    action_space = 4

    # osservazioni, 17 valori totali (senza mask):
    # 3 vel lineare attuale + 3 vel angolare attuale + 3 projected_gravity_b + 4 azione
    # corrente + 4 velocita' target
    observation_space = 17
    state_space = 0

    debug_vis = True
    ui_window_class_type = MyDroneVelEnvWindow

    # -- CONFIGURAZIONE DELLA SIMULAZIONE FISICA -- #
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,  # 100 Hz
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # NOTA: nessun campo "terrain"/collisione col pavimento: questo controllore vola in
    # spazio libero. Il piano visibile in _setup_scene e' SOLO un riferimento grafico
    # (griglia), senza collider fisico - vedi commento in _setup_scene.

    # -- CONFIGURAZIONE DELLA SCENA -- #
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # -- CONFIGURAZIONE DEL DRONE -- #
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # -- RANGE DELLE VELOCITA' TARGET CHE IL DRONE DEVE IMPARARE A RAGGIUNGERE -- #
    max_lin_vel_xy = 1.0    # m/s, per vx e vy
    max_lin_vel_z = 1.0     # m/s, per vz
    max_ang_vel_z = 1.5     # rad/s, per wz

    # -- SOGLIA DI INCLINAZIONE MASSIMA PRIMA DI CONSIDERARE IL DRONE "FUORI CONTROLLO" -- #
    max_tilt_deg = 65.0  # gradi, oltre questa inclinazione l'episodio termina con penalita'

    # -- SOGLIA DI INCLINAZIONE "LIBERA" (nessuna penalita' sotto questo angolo) -- #
    tilt_free_deg = 35.0

    # -- PESI DELLA REWARD FUNCTION -- #
    vel_error_reward_scale = -7.0
    # curriculum sulla penalita' di action_rate: parte bassa e sale progressivamente.
    action_rate_reward_scale_start = -0.01
    action_rate_reward_scale_end = -0.1
    # in quanti "step di controllo" (50Hz) la penalita' passa da start a end.
    action_rate_curriculum_steps = 40000

    unwanted_ang_vel_reward_scale = -0.05
    tilt_reward_scale = -2.0
    died_penalty = -10.0

    # -- MODALITA' 4 (HOVER / PICCOLA CORREZIONE) -- #
    hover_frac = 0.15

    # -- CURRICULUM ADATTIVO PER-AMBIENTE -- #
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

    # -- DEBUG: ogni quante chiamate a _reset_idx stampare un riepilogo del curriculum
    # su console (bypassa il problema di skrl che non mostra self.extras["log"] in
    # TensorBoard). 0 = mai (usa solo TensorBoard/extras).
    debug_print_every_n_resets = 50