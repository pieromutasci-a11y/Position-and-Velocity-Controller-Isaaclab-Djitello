# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# CONTROLLORE DI POSIZIONE del drone (Crazyflie), skrl.
# Livello esterno (25Hz) che avvolge il controllore di velocita' gia' addestrato
# (interno, 50Hz, CONGELATO) richiamandolo in ZOH.
#
# ===================================================================================
# FIX STORICI (mantenuti tutti, NON toccare):
#   FIX 1 [CRITICO] ordine feature obs low-level
#   FIX 2 [CRITICO] spawn ASSOLUTO e clampato dentro la stanza
#   FIX 3  yaw in obs come sin/cos (niente discontinuita')
#   FIX 4  terminazione su tilt > 65 deg
#   FIX 5  clamp azioni HL a [-1,1] PRIMA di scalarle
#   FIX 6  scale riferimenti allineate ai range del low-level
#   FIX 7  _reset_idx vettorizzato
#   FIX 8  termini CONTINUI * step_dt; eventi IMPULSIVI SENZA step_dt
#   FIX 9  nessuna penalita' libera su vx/vy/vz
#   FIX 10 extras come torch.Tensor per skrl
#
# ===================================================================================
# VERSIONE: OBIETTIVO GLOBALE (QUESTA VERSIONE)
#
# Rispetto alla versione precedente (obiettivo asimmetrico full/uniciclo con
# guardia a cerniera su full — max(0, err_full - baseline)):
#
#   - ELIMINATI: mask_full/mask_uniciclo per l'ERRORE, err_full, err_uniciclo,
#     err_max (Objective/combined_error_maxmode), full_regression,
#     obj_full_baseline, obj_w_full_guard. Non esiste piu' alcun trattamento
#     differenziato tra full e uniciclo per errore e smoothness.
#
#   - err_mean e smooth_mean sono ora GLOBALI: media su TUTTI gli env del
#     batch di reset. La smoothness e' un termine SEPARATO (prima era fusa
#     dentro l'errore con peso 0.3 in "err_per_env = combined_error +
#     0.3*smooth_penalty"; ora combined_error e smooth_mean sono
#     indipendenti, ciascuno col proprio peso obj_*).
#
#   - RETROMARCIA e VY (rif. generato E reale) restano MASCHERATI SOLO SU
#     UNICICLO (is_uniciclo = dof_mask[:,1] < 0.5): sono comportamenti
#     penalizzabili solo quando l'asse vy e' vincolato dal compito; full
#     puo' legittimamente strafare, quindi non va incluso in queste medie.
#
#   - COMPONENTI SEPARATE SU WANDB/TENSORBOARD: ciascun termine e' loggato
#     sia come valore grezzo (Objective/error_mean, Objective/smoothness_mean,
#     Diag/uniciclo_reverse_ref_mean, Diag/uniciclo_vy_ref_abs_mean,
#     Diag/uniciclo_vy_real_abs_mean) sia come contributo GIA' pesato
#     (Objective/error_contrib, Objective/smoothness_contrib,
#     Objective/reverse_contrib, Objective/vy_ref_contrib,
#     Objective/vy_real_contrib), oltre a Objective/composite_target.
#
#   - STORICO IN MEMORIA + CSV SU WANDB: self._objective_history accumula,
#     ad ogni chiamata di _reset_idx, un valore per ciascuna delle 6
#     componenti (5 + composite). get_objective_history() lo espone;
#     train_manual.py lo scrive come CSV dentro un wandb Artifact a fine
#     training (nessun file persistito sull'host).
#
# TUTTO IL RESTO (fix storici, reward per-step, sequenza canonica, leaky
# integrator, spawn/stanza, avanzamento waypoint, osservazioni, DoF mask,
# terminazione, debug vis) e' INVARIATO rispetto alla versione precedente.
# ===================================================================================

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, sample_uniform

from ..drone_physics import apply_drone_inertial_props, log_inertial_props
from .pos_controller_env_cfg import MyDronePosEnvCfg

MODE_VARIABILE = 0
MODE_SINGOLO = 1
MODE_NAMES = {MODE_VARIABILE: "variabile", MODE_SINGOLO: "singolo"}

# Sequenza canonica per la modalita' variabile: (segno_dx, segno_dy).
# Ordine fissato: AVANTI, INDIETRO, DESTRA, SINISTRA (mappato 1:1 sui
# 4 slot di n_waypoints). "destra" = -y, "sinistra" = +y (convenzione
# arbitraria ma coerente in tutto il file).
_CANONICAL_DIRECTIONS = (
    (1.0, 0.0),   # 0: avanti    (+x)
    (-1.0, 0.0),  # 1: indietro  (-x)
    (0.0, -1.0),  # 2: destra    (-y)
    (0.0, 1.0),   # 3: sinistra  (+y)
)

# Nomi delle componenti dell'obiettivo tracciate nello storico in memoria
# (per il CSV su wandb) — un valore per ciascuna ad ogni chiamata di
# _reset_idx.
_OBJECTIVE_HISTORY_KEYS = (
    "error_mean",
    "smoothness_mean",
    "reverse_uniciclo_mean",
    "vy_ref_uniciclo_mean",
    "vy_real_uniciclo_mean",
    "composite_target",
)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class MyDronePosEnv(DirectRLEnv):
    cfg: MyDronePosEnvCfg

    def __init__(self, cfg: MyDronePosEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        dev = self.device
        n_wp = self.cfg.n_waypoints

        self._actions = torch.zeros(N, 4, device=dev)
        self._thrust = torch.zeros(N, 1, 3, device=dev)
        self._moment = torch.zeros(N, 1, 3, device=dev)

        self._high_level_actions = torch.zeros(N, 4, device=dev)
        self._prev_high_level_actions = torch.zeros(N, 4, device=dev)

        self._vel_ref_scale = torch.tensor(
            [
                self.cfg.target_lin_vel_xy_scale,
                self.cfg.target_lin_vel_xy_scale,
                self.cfg.target_lin_vel_z_scale,
                self.cfg.target_yaw_vel_scale,
            ],
            device=dev,
        )

        self._wp_pos_queue = torch.zeros(N, n_wp, 3, device=dev)
        self._wp_yaw_queue = torch.zeros(N, n_wp, device=dev)
        self._wp_idx = torch.zeros(N, dtype=torch.long, device=dev)
        self._hold_timer = torch.zeros(N, device=dev)
        self._advanced_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self._env_arange = torch.arange(N, device=dev)

        self._room_min = torch.zeros(N, 3, device=dev)
        self._room_max = torch.zeros(N, 3, device=dev)

        self._reference_mode = torch.zeros(N, dtype=torch.long, device=dev)
        self._is_hover = torch.zeros(N, dtype=torch.bool, device=dev)
        # NUOVO: flag per la sequenza di riferimenti CANONICA (solo un
        # sottoinsieme della modalita' variabile, vedi _sample_reference_mode).
        self._is_canonical = torch.zeros(N, dtype=torch.bool, device=dev)

        self._dof_mask = torch.ones(N, 4, device=dev)
        self._dof_mask_set = torch.tensor(self.cfg.dof_mask_set, device=dev, dtype=torch.float)
        self._dof_mask_probs = torch.tensor(self.cfg.dof_mask_probs, device=dev, dtype=torch.float)

        # -- LEAKY INTEGRATOR errore posizione+yaw (N,4) = [int_ex, int_ey, int_ez, int_eyaw] --
        self._err_integral = torch.zeros(N, 4, device=dev)
        self._integral_alpha = 1.0  # placeholder, aggiornato sotto

        self._physics_step_counter = 0

        # -- massa e inerzia del Tello al posto di quelle del Crazyflie --
        # Deve avvenire PRIMA della lettura di _robot_mass qui sotto (usata per scalare
        # la spinta) e coincidere con la dinamica su cui e' stata addestrata la policy
        # di velocita' congelata caricata piu' avanti.
        applied = apply_drone_inertial_props(self.robot, self.cfg.drone_inertial)
        log_inertial_props("pos_controller_env", applied, self.cfg.drone_inertial)

        self._body_id = self.robot.find_bodies("body")[0]
        # moment_scale e' (roll, pitch, yaw): yaw ha meta' autorita' di roll/pitch,
        # vedi ../drone_physics.py. Tensore (1,3) per broadcasting su (N,3).
        self._moment_scale = torch.tensor(self.cfg.moment_scale, device=dev).unsqueeze(0)
        self._robot_mass = float(self.robot.root_physx_view.get_masses()[0].sum())
        self._gravity_magnitude = float(torch.tensor(self.cfg.sim.gravity, device=dev).norm().item())
        self._robot_weight = self._robot_mass * self._gravity_magnitude
        self._max_tilt_rad = math.radians(self.cfg.max_tilt_deg)

        self._reward_keys = [
            "position_approach",
            "position_prec",
            "yaw_error",
            "yaw_error_prec",
            "reg_ang_vel_xy",
            "reg_ang_vel_wz",
            "aggressive_cmd",
            "alive",
            "action_smoothness",
            "reverse_vx_penalty",
            "uniciclo_vy_penalty",
            "uniciclo_vy_real_penalty",
            "approach_brake_penalty",
            "target_reached",
            "out_of_bounds",
            "tilt_death",
        ]
        self._metric_keys = [
            "position_error_abs", "yaw_error_abs", "n_targets_reached", "min_clearance",
            "action_oscillation_raw", "uniciclo_vy_ref_abs", "uniciclo_vy_real_abs",
            "uniciclo_reverse_ref",
        ]
        self._episode_sums = {
            k: torch.zeros(N, dtype=torch.float, device=dev)
            for k in self._reward_keys + self._metric_keys
        }
        self._episode_min_max = {
            "position_error_abs_min": torch.full((N,), float("inf"), device=dev),
            "position_error_abs_max": torch.full((N,), float("-inf"), device=dev),
            "yaw_error_abs_min": torch.full((N,), float("inf"), device=dev),
            "yaw_error_abs_max": torch.full((N,), float("-inf"), device=dev),
        }

        # -- STORICO obiettivo in memoria (per il CSV su wandb, vedi
        # get_objective_history() e train_manual.py). Una entry per ogni
        # chiamata di _reset_idx, indipendentemente da quanti env
        # contenesse quel batch. --
        self._objective_history: dict[str, list[float]] = {k: [] for k in _OBJECTIVE_HISTORY_KEYS}

        self._reset_call_counter = 0

        self.set_debug_vis(self.cfg.debug_vis)
        self._load_low_level_policy()

        # Calcola alpha del leaky integrator ora che step_dt e' disponibile
        self._integral_alpha = math.exp(-self.step_dt / self.cfg.integral_tau_s)
        print(
            f"[pos_env] Leaky integrator: tau={self.cfg.integral_tau_s}s, "
            f"alpha={self._integral_alpha:.4f} per step "
            f"(decadimento a 1/e in {self.cfg.integral_tau_s}s, "
            f"clamp={self.cfg.integral_clamp}, "
            f"obs_scale={self.cfg.integral_obs_scale})"
        )

        all_ids = torch.arange(N, device=dev)
        self._sample_room(all_ids)
        self._sample_reference_mode(all_ids)
        self._sample_dof_mask(all_ids)
        spawn_pos0 = self.robot.data.root_pos_w[all_ids] - self._env_origins[all_ids]
        _, _, spawn_yaw0 = euler_xyz_from_quat(self.robot.data.root_quat_w[all_ids])
        self._build_queue(all_ids, spawn_pos=spawn_pos0, spawn_yaw=spawn_yaw0)

        H = self.cfg.wp_preview_horizon
        print(
            f"[pos_env] step_dt={self.step_dt:.4f}s ({1/self.step_dt:.1f}Hz HL) | "
            f"low-level a {1/(self.cfg.sim.dt*self.cfg.low_level_decimation):.1f}Hz | "
            f"episodio={self.max_episode_length} step ({self.cfg.episode_length_s:.0f}s)\n"
            f"[pos_env] coda: n_waypoints={n_wp}, preview H={H} "
            f"(1 feedback + {H-1} feedforward) | obs={self.cfg.observation_space}\n"
            f"[pos_env] DoF mask: {len(self.cfg.dof_mask_set)} maschere (SOLO full+uniciclo), "
            f"probs={self.cfg.dof_mask_probs} | spawn margin={self.cfg.spawn_room_margin}, "
            f"take-off low_prob={self.cfg.spawn_low_prob} z={self.cfg.spawn_low_z_range}\n"
            f"[pos_env] UNICICLO: yaw_gate={self.cfg.yaw_gate_dist_uniciclo}m "
            f"vy_pen(ref)={self.cfg.rew_scale_vy_penalty_uniciclo} "
            f"vy_pen(reale)={self.cfg.rew_scale_vy_real_penalty_uniciclo} "
            f"brake(d={self.cfg.approach_brake_dist}m)={self.cfg.rew_scale_approach_brake} "
            f"hard_mask={self.cfg.uniciclo_vy_hard_mask} "
            f"yaw_prec_esponenziale=SOLO_FULL\n"
            f"[pos_env] reverse_vx_penalty={self.cfg.rew_scale_reverse_vx} "
            f"(CONDIVISA: attiva per full e uniciclo)\n"
            f"[pos_env] aggressive_cmd={self.cfg.rew_scale_aggressive_cmd} "
            f"(CONDIVISA: attiva per full e uniciclo, wz escluso SOLO per uniciclo)\n"
            f"[pos_env] sequenza CANONICA (variabile): prob={self.cfg.prob_canonical_variabile} "
            f"margin={self.cfg.canonical_room_margin} min_dist={self.cfg.canonical_min_dist}m\n"
            f"[pos_env] OBIETTIVO SWEEP: GLOBALE (errore+smoothness su tutti gli env), "
            f"vy/retromarcia mascherati solo su uniciclo. Pesi: "
            f"obj_w_err={self.cfg.obj_w_err} obj_w_smooth={self.cfg.obj_w_smooth} "
            f"obj_w_rev={self.cfg.obj_w_rev} obj_w_vy_ref={self.cfg.obj_w_vy_ref} "
            f"obj_w_vy_real={self.cfg.obj_w_vy_real}"
        )

    # ===================================================================
    # LOW-LEVEL POLICY (congelato)
    # ===================================================================
    def _load_low_level_policy(self):
        try:
            ckpt = torch.load(self.cfg.low_level_policy_path, map_location=self.device)

            if "state_preprocessor" in ckpt:
                self._running_mean = ckpt["state_preprocessor"]["running_mean"].to(self.device).float()
                self._running_var = ckpt["state_preprocessor"]["running_variance"].to(self.device).float()
            else:
                print("[pos_env] ATTENZIONE: nessun state_preprocessor, uso identita'.")
                self._running_mean = torch.zeros(17, device=self.device)
                self._running_var = torch.ones(17, device=self.device)

            if "policy" not in ckpt:
                raise ValueError("Nessuna chiave 'policy' nel checkpoint del low-level.")
            sd = ckpt["policy"]

            trunk_keys = sorted(
                (k for k in sd if k.startswith("net_container.") and k.endswith(".weight")),
                key=lambda k: int(k.split(".")[1]),
            )
            if not trunk_keys:
                raise ValueError(f"Nessun layer 'net_container.*'. Chiavi: {list(sd)}")
            if "policy_layer.weight" not in sd:
                raise ValueError(f"Nessuna 'policy_layer.weight'. Chiavi: {list(sd)}")

            trunk_dims = [tuple(sd[k].shape[::-1]) for k in trunk_keys]
            head_out, head_in = sd["policy_layer.weight"].shape
            layer_dims = trunk_dims + [(head_in, head_out)]

            in_dim, out_dim = layer_dims[0][0], layer_dims[-1][1]
            hidden = [o for _, o in layer_dims[:-1]]
            print(f"[pos_env] low-level: input={in_dim}, hidden={hidden}, output={out_dim}")

            if in_dim != 17:
                raise ValueError(f"low-level con input_dim={in_dim}, atteso 17.")
            if out_dim != 4:
                raise ValueError(f"low-level con output_dim={out_dim}, atteso 4.")
            if self._running_mean.numel() != 17 or self._running_var.numel() != 17:
                raise ValueError(f"state_preprocessor con {self._running_mean.numel()} elementi, attesi 17.")

            class LowLevelPolicy(torch.nn.Module):
                def __init__(self, dims, act=torch.nn.ELU):
                    super().__init__()
                    layers = []
                    for i, (a, b) in enumerate(dims):
                        layers.append(torch.nn.Linear(a, b))
                        if i < len(dims) - 1:
                            layers.append(act())
                    self.net = torch.nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x)

            self._low_level_policy = LowLevelPolicy(layer_dims).to(self.device)

            new_sd = {}
            lin_idx = [i for i, m in enumerate(self._low_level_policy.net) if isinstance(m, torch.nn.Linear)]
            for orig, dest in zip(trunk_keys, lin_idx[:-1]):
                base = orig.rsplit(".", 1)[0]
                new_sd[f"net.{dest}.weight"] = sd[f"{base}.weight"]
                new_sd[f"net.{dest}.bias"] = sd[f"{base}.bias"]
            new_sd[f"net.{lin_idx[-1]}.weight"] = sd["policy_layer.weight"]
            new_sd[f"net.{lin_idx[-1]}.bias"] = sd["policy_layer.bias"]

            self._low_level_policy.load_state_dict(new_sd, strict=True)
            self._low_level_policy.eval()
            for p in self._low_level_policy.parameters():
                p.requires_grad_(False)

            print(f"[pos_env] low-level CONGELATO, caricato da {self.cfg.low_level_policy_path}")

        except Exception as e:
            print(f"[pos_env] ERRORE caricamento low-level: {e}")
            raise

    # ===================================================================
    # SCENA
    # ===================================================================
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self._target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)
        self._room_marker = VisualizationMarkers(self.cfg.room_marker_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self._env_origins = self.scene.env_origins

        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        grid = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
            scale=(1.0, 1.0, 1.0),
        )
        grid.func("/World/GridPlane", grid, translation=(0.0, 0.0, 0.0))
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ===================================================================
    # STANZA ASIMMETRICA + CLEARANCE
    # ===================================================================
    def _sample_room(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        u = lambda r: sample_uniform(r[0], r[1], (n,), device=self.device)  # noqa: E731
        self._room_min[env_ids, 0] = -u(self.cfg.room_x_neg_range)
        self._room_max[env_ids, 0] = u(self.cfg.room_x_pos_range)
        self._room_min[env_ids, 1] = -u(self.cfg.room_y_neg_range)
        self._room_max[env_ids, 1] = u(self.cfg.room_y_pos_range)
        self._room_min[env_ids, 2] = self.cfg.min_z_pos
        self._room_max[env_ids, 2] = u(self.cfg.room_z_max_range)

    def _clearance(self, pos_env: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._room_max - pos_env, pos_env - self._room_min], dim=-1)

    def _is_out_of_bounds(self, pos_env: torch.Tensor) -> torch.Tensor:
        return torch.any(self._clearance(pos_env) < 0.0, dim=-1)

    def _sample_in_room(self, env_ids: torch.Tensor, margin: float) -> torch.Tensor:
        n = env_ids.numel()
        rmin = self._room_min[env_ids]
        rmax = self._room_max[env_ids]
        center = 0.5 * (rmin + rmax)
        half = 0.5 * (rmax - rmin) * margin
        lo = center - half
        hi = center + half
        lo[:, 2] = torch.clamp(lo[:, 2], min=self.cfg.min_z_pos)
        hi[:, 2] = torch.maximum(hi[:, 2], lo[:, 2] + 1e-3)
        r = sample_uniform(0.0, 1.0, (n, 3), device=self.device)
        return lo + r * (hi - lo)

    # ===================================================================
    # MASCHERA GRADI DI LIBERTA'
    # ===================================================================
    def _sample_dof_mask(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        idx = torch.multinomial(self._dof_mask_probs, n, replacement=True)
        self._dof_mask[env_ids] = self._dof_mask_set[idx]

    # ===================================================================
    # SEQUENZA DI RIFERIMENTI CANONICA (variabile) — NUOVO
    # ===================================================================
    def _sample_canonical_queue(self, env_ids: torch.Tensor, spawn_pos: torch.Tensor) -> None:
        """Sovrascrive i primi len(_CANONICAL_DIRECTIONS) slot della coda
        con una sequenza canonica: AVANTI, INDIETRO, DESTRA, SINISTRA.

        Ogni target e' spostato rispetto allo spawn lungo UN SOLO asse
        (l'altro resta fisso, come richiesto): avanti/indietro muovono x
        (y fissa = y di spawn), destra/sinistra muovono y (x fissa = x di
        spawn). Il target yaw e' il bearing (atan2) dallo spawn al target,
        cosi' che avvicinarsi in linea retta richieda SEMPRE vx>0 in body
        frame una volta ruotato lo yaw — mai una vera retromarcia. Questo
        vale sia per full che per uniciclo: l'uniciclo non puo' strafare
        in y, quindi per raggiungere un target laterale (destra/sinistra)
        deve comunque prima allinearsi in yaw.

        Per "indietro" in particolare, il target e' su -x rispetto allo
        spawn con yaw target ~pi: la rete impara a girarsi di 180 gradi e
        poi volare in avanti nella nuova direzione, invece di dover
        generare una retromarcia vera (che ne' full ne' uniciclo possono
        fare) verso un ostacolo/target posto dietro di se'.
        """
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        rmin = self._room_min[env_ids]
        rmax = self._room_max[env_ids]
        margin = self.cfg.canonical_room_margin
        min_dist = self.cfg.canonical_min_dist

        n_slots = min(len(_CANONICAL_DIRECTIONS), self.cfg.n_waypoints)
        for k in range(n_slots):
            dxs, dys = _CANONICAL_DIRECTIONS[k]

            if dxs != 0.0:
                raw_avail = (rmax[:, 0] - spawn_pos[:, 0]) if dxs > 0 else (spawn_pos[:, 0] - rmin[:, 0])
            else:
                raw_avail = (rmax[:, 1] - spawn_pos[:, 1]) if dys > 0 else (spawn_pos[:, 1] - rmin[:, 1])
            avail = torch.clamp(raw_avail * margin, min=min_dist)

            frac = sample_uniform(0.0, 1.0, (n,), device=self.device)
            mag = min_dist + frac * (avail - min_dist)

            pos = spawn_pos.clone()
            pos[:, 0] = pos[:, 0] + dxs * mag
            pos[:, 1] = pos[:, 1] + dys * mag
            # z: riusa il campionamento random gia' clampato dentro la stanza.
            pos[:, 2] = self._sample_in_room(env_ids, margin)[:, 2]

            # Clamp finale di sicurezza dentro la stanza (stile FIX 2).
            pos[:, 0] = torch.clamp(pos[:, 0], rmin[:, 0] + 1e-3, rmax[:, 0] - 1e-3)
            pos[:, 1] = torch.clamp(pos[:, 1], rmin[:, 1] + 1e-3, rmax[:, 1] - 1e-3)

            yaw = torch.atan2(
                torch.full((n,), dys, device=self.device),
                torch.full((n,), dxs, device=self.device),
            )

            self._wp_pos_queue[env_ids, k] = pos
            self._wp_yaw_queue[env_ids, k] = yaw

    # ===================================================================
    # CODA DI WAYPOINT
    # ===================================================================
    def _build_queue(
        self,
        env_ids: torch.Tensor,
        spawn_pos: torch.Tensor,
        spawn_yaw: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        n_wp = self.cfg.n_waypoints

        for k in range(n_wp):
            self._wp_pos_queue[env_ids, k] = self._sample_in_room(env_ids, self.cfg.target_room_margin)
            self._wp_yaw_queue[env_ids, k] = sample_uniform(-math.pi, math.pi, (n,), device=self.device)

        # NUOVO: sequenza CANONICA per un sottoinsieme della modalita'
        # variabile (SOVRASCRIVE quanto generato random sopra, solo per
        # gli env selezionati). Tutto il resto della generazione random
        # resta invariato.
        canonical = self._is_canonical[env_ids]
        if torch.any(canonical):
            c_ids = env_ids[canonical]
            self._sample_canonical_queue(c_ids, spawn_pos[canonical])

        singolo = self._reference_mode[env_ids] == MODE_SINGOLO
        if torch.any(singolo):
            s_ids = env_ids[singolo]
            p0 = self._wp_pos_queue[s_ids, 0].clone()
            y0 = self._wp_yaw_queue[s_ids, 0].clone()
            for k in range(n_wp):
                self._wp_pos_queue[s_ids, k] = p0
                self._wp_yaw_queue[s_ids, k] = y0

        hover = self._is_hover[env_ids]
        if torch.any(hover):
            h_local = hover.nonzero(as_tuple=False).squeeze(-1)
            h_ids = env_ids[h_local]
            for k in range(n_wp):
                self._wp_pos_queue[h_ids, k] = spawn_pos[h_local]
                self._wp_yaw_queue[h_ids, k] = spawn_yaw[h_local]

        freeze_yaw = self._dof_mask[env_ids, 3] < 0.5
        if torch.any(freeze_yaw):
            fy_ids = env_ids[freeze_yaw]
            yaw_spawn = spawn_yaw[freeze_yaw]
            for k in range(n_wp):
                self._wp_yaw_queue[fy_ids, k] = yaw_spawn

        self._wp_idx[env_ids] = 0
        self._hold_timer[env_ids] = 0.0
        self._err_integral[env_ids] = 0.0

    def _current_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._wp_idx.clamp(max=self.cfg.n_waypoints - 1)
        return (
            self._wp_pos_queue[self._env_arange, idx],
            self._wp_yaw_queue[self._env_arange, idx],
        )

    def _compute_preview(self, pos_env: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        n_wp = self.cfg.n_waypoints
        H = self.cfg.wp_preview_horizon
        ar = self._env_arange
        blocks = []

        i0 = self._wp_idx.clamp(max=n_wp - 1)
        w0_pos = self._wp_pos_queue[ar, i0]
        w0_yaw = self._wp_yaw_queue[ar, i0]
        e0 = wrap_to_pi(w0_yaw - yaw)
        blocks += [w0_pos - pos_env, torch.sin(e0).unsqueeze(-1), torch.cos(e0).unsqueeze(-1)]

        for k in range(1, H):
            i_c = (self._wp_idx + k).clamp(max=n_wp - 1)
            i_p = (self._wp_idx + k - 1).clamp(max=n_wp - 1)
            p_c = self._wp_pos_queue[ar, i_c]
            p_p = self._wp_pos_queue[ar, i_p]
            y_c = self._wp_yaw_queue[ar, i_c]
            y_p = self._wp_yaw_queue[ar, i_p]
            dyaw = wrap_to_pi(y_c - y_p)
            blocks += [p_c - p_p, torch.sin(dyaw).unsqueeze(-1), torch.cos(dyaw).unsqueeze(-1)]

        return torch.cat(blocks, dim=-1)

    def _update_waypoint(self) -> None:
        """Aggiorna il waypoint e l'integrale dell'errore."""
        pos_env = self.robot.data.root_pos_w - self._env_origins
        w0_pos, w0_yaw = self._current_target()
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)

        # -- errori correnti --
        pos_err = pos_env - w0_pos                      # (N,3) frame MONDO
        yaw_err_signed = wrap_to_pi(yaw - w0_yaw)       # (N,) con segno, wrap-to-pi
        dist = torch.norm(pos_err, dim=1)
        yaw_err_abs = torch.abs(yaw_err_signed)

        # -- aggiorna leaky integrator PRIMA di eventuale avanzamento --
        alpha = self._integral_alpha
        clamp = self.cfg.integral_clamp
        self._err_integral[:, :3] = torch.clamp(
            alpha * self._err_integral[:, :3] + pos_err * self.step_dt,
            -clamp, clamp,
        )
        self._err_integral[:, 3] = torch.clamp(
            alpha * self._err_integral[:, 3] + yaw_err_signed * self.step_dt,
            -clamp, clamp,
        )

        # -- criterio di avanzamento waypoint --
        yaw_controllable = self._dof_mask[:, 3] > 0.5
        yaw_err_eff = torch.where(
            yaw_controllable,
            yaw_err_abs,
            torch.zeros_like(yaw_err_abs),
        )

        converged = (dist < self.cfg.target_reach_threshold) & (
            yaw_err_eff < self.cfg.target_reach_yaw_threshold
        )
        self._hold_timer = torch.where(
            converged, self._hold_timer + self.step_dt, torch.zeros_like(self._hold_timer)
        )

        can_advance = self._wp_idx < (self.cfg.n_waypoints - 1)
        self._advanced_now = (self._hold_timer >= self.cfg.target_hold_time_s) & can_advance

        adv = self._advanced_now.nonzero(as_tuple=False).squeeze(-1)
        if adv.numel() > 0:
            self._wp_idx[adv] += 1
            self._hold_timer[adv] = 0.0
            self._episode_sums["n_targets_reached"][adv] += 1.0
            self._err_integral[adv] = 0.0

    # ===================================================================
    # AZIONI
    # ===================================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        self._update_waypoint()
        self._prev_high_level_actions = self._high_level_actions.clone()

        if self.cfg.uniciclo_vy_hard_mask:
            action_mask = self._dof_mask
        else:
            is_uniciclo = self._dof_mask[:, 1] < 0.5
            action_mask = self._dof_mask.clone()
            action_mask[is_uniciclo, 1] = 1.0

        self._high_level_actions = (actions * action_mask).clone()

    def _apply_action(self):
        if self._physics_step_counter % self.cfg.low_level_decimation == 0:
            lin_vel_b = self.robot.data.root_lin_vel_b
            ang_vel_b = self.robot.data.root_ang_vel_b
            proj_grav_b = self.robot.data.projected_gravity_b

            hl = self._high_level_actions.clamp(-1.0, 1.0)
            target_vel_ref = hl * self._vel_ref_scale

            low_obs = torch.cat(
                [lin_vel_b, ang_vel_b, proj_grav_b, self._actions, target_vel_ref], dim=-1
            )
            low_obs_n = (low_obs - self._running_mean) / torch.sqrt(self._running_var + 1e-8)

            with torch.no_grad():
                a_ll = self._low_level_policy(low_obs_n)

            self._actions = a_ll.clamp(-1.0, 1.0)
            self._thrust[:, 0, 2] = (
                self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
            )
            self._moment[:, 0, :] = self._moment_scale * self._actions[:, 1:4]
            self._physics_step_counter = 0

        self.robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        self._physics_step_counter += 1

    # ===================================================================
    # OSSERVAZIONI (52 = 48 + 4 integrale)
    # ===================================================================
    def _get_observations(self) -> dict:
        pos_env = self.robot.data.root_pos_w - self._env_origins
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)

        preview = self._compute_preview(pos_env, yaw)
        clearance = self._clearance(pos_env)

        integral_norm = self._err_integral / self.cfg.integral_obs_scale  # (N,4)

        obs = torch.cat(
            [
                pos_env,                                    # 3  MONDO
                torch.sin(yaw).unsqueeze(-1),               # 1
                torch.cos(yaw).unsqueeze(-1),               # 1
                self.robot.data.root_lin_vel_b,             # 3  CORPO
                self.robot.data.root_ang_vel_b,             # 3  CORPO
                self.robot.data.projected_gravity_b,        # 3  CORPO
                self._prev_high_level_actions,              # 4
                preview,                                    # 20 (feedback + 3 FF)
                clearance,                                  # 6  metri
                self._dof_mask,                             # 4  DoF mask
                integral_norm,                               # 4  leaky integrator errore pos+yaw
            ],
            dim=-1,
        )  # 52
        return {"policy": obs}

    # ===================================================================
    # TERMINAZIONE
    # ===================================================================
    def _is_tilt_excessive(self) -> torch.Tensor:
        if not self.cfg.terminate_su_tilt_eccessivo:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        gz = torch.clamp(-self.robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
        return torch.acos(gz) > self._max_tilt_rad

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos_env = self.robot.data.root_pos_w - self._env_origins
        died = self._is_out_of_bounds(pos_env) | self._is_tilt_excessive()
        return died, time_out

    # ===================================================================
    # REWARD
    # ===================================================================
    def _get_rewards(self) -> torch.Tensor:
        pos_env = self.robot.data.root_pos_w - self._env_origins
        w0_pos, w0_yaw = self._current_target()
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)

        dist = torch.norm(pos_env - w0_pos, dim=1)

        yaw_err_abs = torch.abs(wrap_to_pi(yaw - w0_yaw))
        yaw_controllable = self._dof_mask[:, 3] > 0.5
        yaw_err = torch.where(yaw_controllable, yaw_err_abs, torch.zeros_like(yaw_err_abs))

        position_approach = (
            torch.exp(-self.cfg.reward_exp_beta * dist)
            * self.cfg.rew_scale_position_approach * self.step_dt
        )
        position_prec = (
            torch.exp(-self.cfg.reward_exp_beta_prec * dist)
            * self.cfg.rew_scale_position_prec * self.step_dt
        )

        is_uniciclo = self._dof_mask[:, 1] < 0.5
        uni_off = (~is_uniciclo).float()  # 1.0 per full, 0.0 per uniciclo

        yaw_gate = torch.clamp(
            1.0 - dist / self.cfg.yaw_gate_dist_uniciclo, min=0.0, max=1.0
        )
        yaw_weight = torch.where(is_uniciclo, yaw_gate, torch.ones_like(yaw_gate))

        yaw_error = yaw_err * yaw_weight * self.cfg.rew_scale_yaw_error * self.step_dt

        yaw_error_prec_raw = torch.exp(-self.cfg.reward_exp_beta_yaw_prec * yaw_err)
        yaw_error_prec = (
            torch.where(yaw_controllable, yaw_error_prec_raw, torch.zeros_like(yaw_error_prec_raw))
            * yaw_weight * uni_off * self.cfg.rew_scale_yaw_prec * self.step_dt
        )

        ang_vel_xy_sq = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        reg_ang_vel_xy = ang_vel_xy_sq * self.cfg.rew_scale_reg_ang_vel_xy * self.step_dt

        wz_sq = torch.square(self.robot.data.root_ang_vel_b[:, 2])
        wz_proximity_weight_base = 1.0 + 9.0 * torch.exp(-2.0 * dist)
        wz_proximity_weight = torch.where(
            is_uniciclo, torch.ones_like(dist), wz_proximity_weight_base
        )
        reg_ang_vel_wz = wz_sq * wz_proximity_weight * self.cfg.rew_scale_reg_ang_vel_wz * self.step_dt

        # NUOVO: aggressive_cmd ora e' attiva SEMPRE (anche per full), non
        # piu' gated da is_not_full. Il canale wz resta escluso SOLO per
        # l'uniciclo (che ne ha bisogno per virare); per full il wz e'
        # incluso nella penalita' quadratica come gli altri assi.
        cmd_weight_aggressive = self._dof_mask.clone()
        cmd_weight_aggressive[is_uniciclo, 3] = 0.0
        cmd_attivi = self._high_level_actions * cmd_weight_aggressive
        cmd_sq = torch.sum(torch.square(cmd_attivi), dim=1)
        aggressive_cmd = cmd_sq * self.cfg.rew_scale_aggressive_cmd * self.step_dt

        alive = torch.full_like(dist, self.cfg.rew_scale_alive * self.step_dt)

        smooth_weight = torch.ones(self.num_envs, 4, device=self.device)
        smooth_weight[is_uniciclo, 3] = 1.0

        d_act = self._high_level_actions - self._prev_high_level_actions
        action_smoothness = (
            torch.sum(torch.square(d_act) * smooth_weight, dim=1)
            * self.cfg.rew_scale_action_smoothness * self.step_dt
        )

        vx_ref_generated = self._high_level_actions[:, 0]
        reverse_amount = torch.clamp(-vx_ref_generated, min=0.0)  # >0 solo se vx_ref < 0
        reverse_vx_penalty = (
            is_uniciclo.float() * torch.square(reverse_amount) * self.cfg.rew_scale_reverse_vx * self.step_dt
        )

        # U.1 — penalita' su vy RIFERIMENTO generato dalla rete.
        vy_ref_generated = self._high_level_actions[:, 1]
        uniciclo_vy_penalty = (
            is_uniciclo.float() * torch.square(vy_ref_generated)
            * self.cfg.rew_scale_vy_penalty_uniciclo * self.step_dt
        )

        # U.1bis — NUOVA penalita' su vy REALE (velocita' body-frame
        # effettivamente realizzata dal drone). Distinta da U.1: quella
        # agisce sul comando (vy_ref), questa sul comportamento fisico
        # (root_lin_vel_b[:,1]). Attiva SOLO in uniciclo, quadratica,
        # SENZA gate sulla distanza dal target: la deriva laterale va
        # scoraggiata in ogni fase del volo, non solo in avvicinamento.
        vy_real = self.robot.data.root_lin_vel_b[:, 1]
        uniciclo_vy_real_penalty = (
            is_uniciclo.float() * torch.square(vy_real)
            * self.cfg.rew_scale_vy_real_penalty_uniciclo * self.step_dt
        )

        vx_real = self.robot.data.root_lin_vel_b[:, 0]
        vz_real = self.robot.data.root_lin_vel_b[:,2]
        brake_weight = torch.clamp(
            1.0 - dist / self.cfg.approach_brake_dist, min=0.0, max=1.0
        )
        approach_brake_penalty = (
            is_uniciclo.float() * brake_weight * (torch.square(vx_real)+torch.square(vz_real))
            * self.cfg.rew_scale_approach_brake * self.step_dt
        )

        is_oob = self._is_out_of_bounds(pos_env)
        oob_penalty = is_oob.float() * self.cfg.oob_reward
        is_tilt = self._is_tilt_excessive()
        tilt_penalty = is_tilt.float() * self.cfg.tilt_death_reward
        target_reached = self._advanced_now.float() * self.cfg.rew_scale_target_reached

        rewards = {
            "position_approach": position_approach,
            "position_prec": position_prec,
            "yaw_error": yaw_error,
            "yaw_error_prec": yaw_error_prec,
            "reg_ang_vel_xy": reg_ang_vel_xy,
            "reg_ang_vel_wz": reg_ang_vel_wz,
            "aggressive_cmd": aggressive_cmd,
            "alive": alive,
            "action_smoothness": action_smoothness,
            "reverse_vx_penalty": reverse_vx_penalty,
            "uniciclo_vy_penalty": uniciclo_vy_penalty,
            "uniciclo_vy_real_penalty": uniciclo_vy_real_penalty,
            "approach_brake_penalty": approach_brake_penalty,
            "target_reached": target_reached,
            "out_of_bounds": oob_penalty,
            "tilt_death": tilt_penalty,
        }

        self._episode_sums["position_error_abs"] += dist
        self._episode_sums["yaw_error_abs"] += yaw_err
        self._episode_sums["min_clearance"] += self._clearance(pos_env).min(dim=-1).values

        self._episode_sums["action_oscillation_raw"] += torch.sum(torch.square(d_act), dim=1)
        self._episode_sums["uniciclo_vy_ref_abs"] += is_uniciclo.float() * torch.abs(vy_ref_generated)
        # Metrica GREZZA della vy reale (solo uniciclo), per confrontabilita'
        # tra trial con pesi diversi (stesso pattern di uniciclo_vy_ref_abs).
        self._episode_sums["uniciclo_vy_real_abs"] += is_uniciclo.float() * torch.abs(vy_real)
        # Metrica GREZZA della retromarcia (solo uniciclo): quanto vx_ref e'
        # negativa, in valore assoluto. reverse_amount vale 0 quando vx_ref>=0.
        self._episode_sums["uniciclo_reverse_ref"] += is_uniciclo.float() * reverse_amount

        mm = self._episode_min_max
        mm["position_error_abs_min"] = torch.minimum(mm["position_error_abs_min"], dist)
        mm["position_error_abs_max"] = torch.maximum(mm["position_error_abs_max"], dist)
        mm["yaw_error_abs_min"] = torch.minimum(mm["yaw_error_abs_min"], yaw_err)
        mm["yaw_error_abs_max"] = torch.maximum(mm["yaw_error_abs_max"], yaw_err)
        for k, v in rewards.items():
            self._episode_sums[k] += v

        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    # ===================================================================
    # MODALITA'
    # ===================================================================
    def _sample_reference_mode(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        is_var = torch.rand(n, device=self.device) < self.cfg.prob_variabile
        self._reference_mode[env_ids] = torch.where(
            is_var,
            torch.full((n,), MODE_VARIABILE, dtype=torch.long, device=self.device),
            torch.full((n,), MODE_SINGOLO, dtype=torch.long, device=self.device),
        )
        self._is_hover[env_ids] = (~is_var) & (
            torch.rand(n, device=self.device) < self.cfg.hover_frac_in_singolo
        )
        # NUOVO: sottoinsieme della modalita' variabile che usa la
        # sequenza CANONICA (avanti/indietro/destra/sinistra) invece dei
        # waypoint completamente random. Non tocca la modalita' singolo.
        self._is_canonical[env_ids] = is_var & (
            torch.rand(n, device=self.device) < self.cfg.prob_canonical_variabile
        )

    # ===================================================================
    # STORICO OBIETTIVO (per il CSV su wandb)
    # ===================================================================
    def get_objective_history(self) -> dict[str, list[float]]:
        """Ritorna una COPIA dello storico delle componenti dell'obiettivo.

        Una entry per ogni chiamata di _reset_idx (valore aggregato sul
        batch di env resettati in quella chiamata). Usata da
        train_manual.py per costruire i CSV caricati su wandb come
        Artifact a fine training: nessun file viene persistito sull'host,
        i CSV vengono scritti direttamente nello staging interno
        dell'Artifact wandb (wandb.Artifact.new_file(...)).
        """
        return {k: list(v) for k, v in self._objective_history.items()}

    # ===================================================================
    # RESET
    # ===================================================================
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = ids.numel()

        extras = dict()

        room_diag = torch.norm(self._room_max[ids] - self._room_min[ids], dim=-1)
        died_mask = self.reset_terminated[ids]
        steps_left = (self.max_episode_length - self.episode_length_buf[ids]).clamp(min=0).float()
        fill = died_mask.float() * steps_left
        self._episode_sums["position_error_abs"][ids] += room_diag * fill
        self._episode_sums["yaw_error_abs"][ids] += math.pi * fill

        # FIX BUG PREESISTENTE: copie salvate PRIMA dell'azzeramento generico.
        saved_action_smoothness = self._episode_sums["action_smoothness"][ids].clone()
        saved_oscillation_raw = self._episode_sums["action_oscillation_raw"][ids].clone() / self.max_episode_length
        saved_vy_ref_abs = self._episode_sums["uniciclo_vy_ref_abs"][ids].clone() / self.max_episode_length
        saved_vy_real_abs = self._episode_sums["uniciclo_vy_real_abs"][ids].clone() / self.max_episode_length
        saved_reverse_ref = self._episode_sums["uniciclo_reverse_ref"][ids].clone() / self.max_episode_length

        # NUOVO — sezione "Reward/*": stesse somme per episodio di
        # "Episode_Reward/*", ma normalizzate per NUMERO DI STEP invece
        # che per SECONDI. E' la reward media per singolo step di
        # controllo — direttamente confrontabile con l'aritmetica "per
        # step" usata nei commenti del cfg (es. "reverse_vx a retromarcia
        # massima vale -0.24 per step"). Salvata PRIMA dell'azzeramento,
        # per lo stesso motivo del fix sopra.
        for k in self._reward_keys:
            extras[f"Reward/{k}"] = torch.mean(self._episode_sums[k][ids]) / self.max_episode_length

        for k in self._episode_sums:
            avg = torch.mean(self._episode_sums[k][ids])
            if k == "n_targets_reached":
                extras[f"Episode_Info/{k}"] = avg
            elif k in self._metric_keys:
                extras[f"Episode_Info/{k}"] = avg / self.max_episode_length
            else:
                extras[f"Episode_Reward/{k}"] = avg / self.max_episode_length_s
            self._episode_sums[k][ids] = 0.0

        for k, v in self._episode_min_max.items():
            extras[f"Episode_Info/{k}"] = torch.mean(v[ids])
            v[ids] = float("inf") if "_min" in k else float("-inf")

        extras["Episode_Info/room_x_pos"] = torch.mean(self._room_max[ids, 0])
        extras["Episode_Info/room_x_neg"] = torch.mean(-self._room_min[ids, 0])
        extras["Episode_Info/room_y_pos"] = torch.mean(self._room_max[ids, 1])
        extras["Episode_Info/room_y_neg"] = torch.mean(-self._room_min[ids, 1])
        extras["Episode_Info/room_z_max"] = torch.mean(self._room_max[ids, 2])

        extras["DoF_Mask/frac_vx"] = torch.mean(self._dof_mask[ids, 0])
        extras["DoF_Mask/frac_vy"] = torch.mean(self._dof_mask[ids, 1])
        extras["DoF_Mask/frac_vz"] = torch.mean(self._dof_mask[ids, 2])
        extras["DoF_Mask/frac_wz"] = torch.mean(self._dof_mask[ids, 3])

        pos_env = self.robot.data.root_pos_w[ids] - self._env_origins[ids]
        i0 = self._wp_idx[ids].clamp(max=self.cfg.n_waypoints - 1)
        w0_pos = self._wp_pos_queue[ids, i0]
        w0_yaw = self._wp_yaw_queue[ids, i0]
        _, _, y_f = euler_xyz_from_quat(self.robot.data.root_quat_w[ids])

        final_dist = torch.norm(pos_env - w0_pos, dim=1)
        yaw_ctrl = self._dof_mask[ids, 3] > 0.5
        final_yaw_abs = torch.abs(wrap_to_pi(y_f - w0_yaw))
        final_yaw = torch.where(yaw_ctrl, final_yaw_abs, torch.zeros_like(final_yaw_abs))

        extras["Episode_Termination/final_distance_to_target"] = torch.mean(final_dist)
        extras["Episode_Termination/final_distance_to_target_min"] = torch.min(final_dist)
        extras["Episode_Termination/final_distance_to_target_max"] = torch.max(final_dist)
        extras["Episode_Termination/final_yaw_error"] = torch.mean(final_yaw)
        alpha_yaw = 0.2
        combined_error = final_dist + alpha_yaw * final_yaw
        extras["Episode_Termination/combined_error"] = torch.mean(combined_error)

        # ===============================================================
        # FUNZIONE OBIETTIVO DELLO SWEEP — GLOBALE, TERMINI SEPARATI
        # ===============================================================
        # ERRORE e SMOOTHNESS: GLOBALI, media su TUTTI gli env di questo
        # batch di reset. Nessuna distinzione full/uniciclo/variabile/ecc.
        err_per_env = combined_error
        err_mean = torch.mean(err_per_env)
        smooth_mean = torch.mean(saved_oscillation_raw)

        # RETROMARCIA e VY: mascherate SOLO su uniciclo (full puo'
        # legittimamente usare vy, quindi non va incluso in queste medie).
        is_uniciclo_mask = self._dof_mask[ids, 1] < 0.5

        def _masked_mean(arr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if torch.any(mask):
                return torch.mean(arr[mask])
            return torch.zeros((), device=self.device)

        reverse_uniciclo_mean = _masked_mean(saved_reverse_ref, is_uniciclo_mask)
        vy_ref_uniciclo_mean = _masked_mean(saved_vy_ref_abs, is_uniciclo_mask)
        vy_real_uniciclo_mean = _masked_mean(saved_vy_real_abs, is_uniciclo_mask)

        # -- valori GREZZI, confrontabili tra trial con pesi diversi --
        extras["Objective/error_mean"] = err_mean
        extras["Objective/smoothness_mean"] = smooth_mean
        extras["Diag/uniciclo_reverse_ref_mean"] = reverse_uniciclo_mean
        extras["Diag/uniciclo_vy_ref_abs_mean"] = vy_ref_uniciclo_mean
        extras["Diag/uniciclo_vy_real_abs_mean"] = vy_real_uniciclo_mean

        # -- contributi GIA' PESATI, nella stessa unita' del composite --
        error_contrib = self.cfg.obj_w_err * err_mean
        smoothness_contrib = self.cfg.obj_w_smooth * smooth_mean
        reverse_contrib = self.cfg.obj_w_rev * reverse_uniciclo_mean
        vy_ref_contrib = self.cfg.obj_w_vy_ref * vy_ref_uniciclo_mean
        vy_real_contrib = self.cfg.obj_w_vy_real * vy_real_uniciclo_mean

        extras["Objective/error_contrib"] = error_contrib
        extras["Objective/smoothness_contrib"] = smoothness_contrib
        extras["Objective/reverse_contrib"] = reverse_contrib
        extras["Objective/vy_ref_contrib"] = vy_ref_contrib
        extras["Objective/vy_real_contrib"] = vy_real_contrib

        composite_target = (
            error_contrib + smoothness_contrib + reverse_contrib + vy_ref_contrib + vy_real_contrib
        )
        extras["Objective/composite_target"] = composite_target

        # -- storico in memoria, per il CSV su wandb (vedi get_objective_history) --
        self._objective_history["error_mean"].append(float(err_mean.item()))
        self._objective_history["smoothness_mean"].append(float(smooth_mean.item()))
        self._objective_history["reverse_uniciclo_mean"].append(float(reverse_uniciclo_mean.item()))
        self._objective_history["vy_ref_uniciclo_mean"].append(float(vy_ref_uniciclo_mean.item()))
        self._objective_history["vy_real_uniciclo_mean"].append(float(vy_real_uniciclo_mean.item()))
        self._objective_history["composite_target"].append(float(composite_target.item()))

        n_died_t = torch.count_nonzero(self.reset_terminated[ids])
        n_to_t = torch.count_nonzero(self.reset_time_outs[ids])
        n_died = int(n_died_t.item())
        n_to = int(n_to_t.item())
        extras["Episode_Termination/died"] = n_died_t.float()
        extras["Episode_Termination/time_out"] = n_to_t.float()
        extras["Episode_Termination/died_frac"] = n_died_t.float() / max(n, 1)

        for mid, name in MODE_NAMES.items():
            extras[f"Reference_Mode/frac_{name}"] = torch.mean(
                (self._reference_mode[ids] == mid).float()
            )
        extras["Reference_Mode/frac_hover"] = torch.mean(self._is_hover[ids].float())
        # NUOVO: frazione di env (sul totale) in sequenza CANONICA. Per
        # costruzione e' un sottoinsieme della modalita' variabile.
        extras["Reference_Mode/frac_canonical"] = torch.mean(self._is_canonical[ids].float())
        extras["Waypoints/mean_final_wp_idx"] = torch.mean(self._wp_idx[ids].float())
        var_mask = self._reference_mode[ids] == MODE_VARIABILE
        if torch.any(var_mask):
            extras["Waypoints/mean_final_wp_idx_variabile"] = torch.mean(
                self._wp_idx[ids][var_mask].float()
            )
            extras["Waypoints/frac_completed_variabile"] = torch.mean(
                (self._wp_idx[ids][var_mask] >= self.cfg.n_waypoints - 1).float()
            )

        self.extras["log"] = extras

        self._reset_call_counter += 1
        every = self.cfg.debug_print_every_n_resets
        if every and self._reset_call_counter % every == 0:
            e = extras

            def _v(key, default=float("nan")):
                val = e.get(key, default)
                return val.item() if isinstance(val, torch.Tensor) else val

            print(
                f"\n[pos][reset #{self._reset_call_counter}] n_env={n}\n"
                f"  reward(step) approach={_v('Reward/position_approach'):+.4f} "
                f"prec={_v('Reward/position_prec'):+.4f} "
                f"yaw={_v('Reward/yaw_error'):+.4f} "
                f"yaw_prec={_v('Reward/yaw_error_prec'):+.4f} "
                f"vy_pen(ref)={_v('Reward/uniciclo_vy_penalty'):+.4f} "
                f"vy_pen(reale)={_v('Reward/uniciclo_vy_real_penalty'):+.4f} "
                f"brake={_v('Reward/approach_brake_penalty'):+.4f} "
                f"reverse_vx={_v('Reward/reverse_vx_penalty'):+.4f} "
                f"aggressive_cmd={_v('Reward/aggressive_cmd'):+.4f}\n"
                f"  errori   pos_err={_v('Episode_Info/position_error_abs'):.3f}m "
                f"yaw_err={_v('Episode_Info/yaw_error_abs'):.3f}rad "
                f"dist_fin={_v('Episode_Termination/final_distance_to_target'):.3f}m "
                f"yaw_fin={_v('Episode_Termination/final_yaw_error'):.3f}rad\n"
                f"  OBJECTIVE (GLOBALE) error_mean={_v('Objective/error_mean'):.4f} "
                f"smoothness_mean={_v('Objective/smoothness_mean'):.4f} "
                f"composite={_v('Objective/composite_target'):.4f}\n"
                f"  OBJECTIVE contrib   error={_v('Objective/error_contrib'):.4f} "
                f"smooth={_v('Objective/smoothness_contrib'):.4f} "
                f"reverse={_v('Objective/reverse_contrib'):.4f} "
                f"vy_ref={_v('Objective/vy_ref_contrib'):.4f} "
                f"vy_real={_v('Objective/vy_real_contrib'):.4f}\n"
                f"  DIAG (solo uniciclo) vy_ref={_v('Diag/uniciclo_vy_ref_abs_mean'):.4f} "
                f"vy_reale={_v('Diag/uniciclo_vy_real_abs_mean'):.4f} "
                f"retromarcia={_v('Diag/uniciclo_reverse_ref_mean'):.4f}\n"
                f"  WAYPOINT target_raggiunti/ep={_v('Episode_Info/n_targets_reached'):.2f}/"
                f"{self.cfg.n_waypoints - 1}  "
                f"wp_idx_finale(var)={_v('Waypoints/mean_final_wp_idx_variabile'):.2f}  "
                f"coda_completata(var)={100*_v('Waypoints/frac_completed_variabile', 0.0):.0f}%\n"
                f"  fine     died={n_died} ({100*_v('Episode_Termination/died_frac'):.1f}%) "
                f"time_out={n_to} "
                f"min_clearance={_v('Episode_Info/min_clearance'):.2f}m\n"
                f"  modalita variabile={_v('Reference_Mode/frac_variabile'):.2f} "
                f"singolo={_v('Reference_Mode/frac_singolo'):.2f} "
                f"(hover={_v('Reference_Mode/frac_hover'):.2f}, "
                f"canonica={_v('Reference_Mode/frac_canonical'):.2f})\n"
                f"  DoF mask frac  vx={_v('DoF_Mask/frac_vx'):.2f} vy={_v('DoF_Mask/frac_vy'):.2f} "
                f"vz={_v('DoF_Mask/frac_vz'):.2f} wz={_v('DoF_Mask/frac_wz'):.2f}"
            )

        # -- RESET fisico --
        self.robot.reset(ids)
        super()._reset_idx(env_ids)

        if n == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[ids] = 0.0
        self._high_level_actions[ids] = 0.0
        self._prev_high_level_actions[ids] = 0.0
        self._hold_timer[ids] = 0.0
        self._advanced_now[ids] = False
        self._wp_idx[ids] = 0
        self._err_integral[ids] = 0.0

        self._sample_room(ids)
        self._sample_reference_mode(ids)
        self._sample_dof_mask(ids)

        joint_pos = self.robot.data.default_joint_pos[ids]
        joint_vel = self.robot.data.default_joint_vel[ids]
        root_state = self.robot.data.default_root_state[ids].clone()

        spawn_pos_env = self._sample_in_room(ids, self.cfg.spawn_room_margin)

        is_low = torch.rand(n, device=self.device) < self.cfg.spawn_low_prob
        low_z = sample_uniform(
            self.cfg.spawn_low_z_range[0], self.cfg.spawn_low_z_range[1], (n,), device=self.device
        )
        spawn_pos_env[:, 2] = torch.where(is_low, low_z, spawn_pos_env[:, 2])

        root_state[:, :3] = self._env_origins[ids] + spawn_pos_env

        spawn_yaw = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
        zero = torch.zeros_like(spawn_yaw)
        root_state[:, 3:7] = quat_from_euler_xyz(zero, zero, spawn_yaw)
        root_state[:, 7:] = 0.0

        self.robot.write_root_pose_to_sim(root_state[:, :7], ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, ids)

        self._build_queue(ids, spawn_pos=spawn_pos_env, spawn_yaw=spawn_yaw)

    # ===================================================================
    # DEBUG VIS
    # ===================================================================
    def _set_debug_vis_impl(self, debug_vis: bool):
        self._target_marker.set_visibility(debug_vis)
        self._room_marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        w0_pos, w0_yaw = self._current_target()
        tgt_w = w0_pos + self._env_origins
        z = torch.zeros_like(w0_yaw)
        quat = quat_from_euler_xyz(z, z, w0_yaw)
        self._target_marker.visualize(translations=tgt_w, orientations=quat)

        center = 0.5 * (self._room_min + self._room_max) + self._env_origins
        scale = self._room_max - self._room_min
        self._room_marker.visualize(translations=center, scales=scale)