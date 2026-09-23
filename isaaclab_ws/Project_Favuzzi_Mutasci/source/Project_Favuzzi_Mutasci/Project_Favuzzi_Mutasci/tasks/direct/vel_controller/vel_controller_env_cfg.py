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

# finestra UI di debug dell'ambiente (nessuna personalizzazione oltre la base)
class MyDroneVelEnvWindow(BaseEnvWindow):

    def __init__(self, env, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)

# configurazione del controllore di velocita' (livello interno, 50Hz)
@configclass
class MyDroneVelEnvCfg(DirectRLEnvCfg):

    # terminazione anticipata per tilt eccessivo
    terminate_su_tilt_eccessivo: bool = True

    # timing: durata episodio e decimazione fisica/rendering
    episode_length_s = 10.0
    decimation = 2

    action_space = 4

    observation_space = 17
    state_space = 0

    debug_vis = True
    ui_window_class_type = MyDroneVelEnvWindow

    # parametri fisici della simulazione
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

    # scena vettorizzata (numero di env in parallelo)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # asset del drone (Crazyflie come base) con override inerziale verso il Tello
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    drone_inertial: DroneInertialCfg = TELLO_INERTIAL_CFG.replace()

    # conversione azioni [-1,1] -> spinta/momento fisico
    thrust_to_weight = 1.8
    moment_scale = (0.021, 0.021, 0.0085)

    # scala massima dei riferimenti di velocita' (lineare xy/z, angolare z)
    max_lin_vel_xy = 1.0
    max_lin_vel_z = 1.0
    max_ang_vel_z = 1.5

    # soglia di tilt per la terminazione anticipata
    max_tilt_deg = 65.0

    # tilt libero (senza penalita') prima che scatti il termine di reward sul tilt
    tilt_free_deg = 35.0

    # pesi dei termini di reward: errore di velocita', action rate (con curriculum), ang. vel indesiderata, tilt, morte
    vel_error_reward_scale = -7.0
    action_rate_reward_scale_start = -0.01
    action_rate_reward_scale_end = -0.1
    action_rate_curriculum_steps = 40000

    unwanted_ang_vel_reward_scale = -0.05
    tilt_reward_scale = -2.0
    died_penalty = -10.0

    # frazione di ampiezza per la correzione casuale in modalita' hover (modalita 4)
    hover_frac = 0.15

    # livelli del curriculum: pesi delle 5 modalita' di riferimento, durata hold, frazione minima modalita 3
    curriculum_livelli = (
        {"pesi_modalita": (0.0, 0.0, 0.95, 0.0, 0.05), "hold_min_s": 3.5, "hold_max_s": 5.0, "mode3_min_frac": 0.3},
        {"pesi_modalita": (0.475, 0.0, 0.475, 0.0, 0.05), "hold_min_s": 3.0, "hold_max_s": 4.0, "mode3_min_frac": 0.3},
        {"pesi_modalita": (0.40, 0.15, 0.30, 0.10, 0.05), "hold_min_s": 2.0, "hold_max_s": 3.0, "mode3_min_frac": 0.4},
        {"pesi_modalita": (0.35, 0.15, 0.20, 0.25, 0.05), "hold_min_s": 1.5, "hold_max_s": 2.5, "mode3_min_frac": 0.5},
        {"pesi_modalita": (0.35, 0.15, 0.15, 0.30, 0.05), "hold_min_s": 1.0, "hold_max_s": 2.5, "mode3_min_frac": 0.6},
    )

    # soglia di errore e numero di successi consecutivi richiesti per salire di livello
    curriculum_soglia_successo = 0.02
    curriculum_successi_richiesti = 5
    curriculum_transiente_skip_s = 0.3

    # frequenza (in numero di reset) della stampa di debug a schermo
    debug_print_every_n_resets = 50
