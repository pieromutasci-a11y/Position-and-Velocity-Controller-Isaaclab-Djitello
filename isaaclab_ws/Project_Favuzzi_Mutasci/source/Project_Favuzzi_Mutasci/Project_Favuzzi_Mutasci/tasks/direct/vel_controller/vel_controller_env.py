# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ambiente per l'addestramento del CONTROLLORE DI VELOCITA' del drone (Crazyflie), skrl.
#
# FIX in questa versione rispetto alla precedente (non funzionante):
#
#   1. REWARD BUG (causa principale del tracking pessimo, vedi discussione): la reward
#      di vel_error divideva l'errore normalizzato per n_assi_attivi. Questo diluiva il
#      segnale di tracking proprio nel caso piu' difficile (mod 0/3, tutti gli assi
#      attivi contemporaneamente) rispetto ai casi con un solo asse attivo - l'opposto
#      di cio' che serve. Ora la reward usa la somma dell'errore normalizzato (NON
#      divisa per assi attivi), coerente con quello che faceva il vecchio env rsl_rl
#      funzionante. La divisione per n_assi_attivi resta SOLO nella valutazione del
#      curriculum (_valuta_e_promuovi), esattamente come nel vecchio env.
#
#   2. NIENTE PAVIMENTO FISICO, ma piano di riferimento visivo con griglia: prima il
#      piano era un MeshCuboidCfg 1000x1000 bianco pieno, senza texture, che con la dome
#      light a 2000 di intensita' saturava la vista (schermo bianco, drone poco
#      visibile). Ora uso l'asset standard "grid" di Isaac Lab per il riferimento
#      visivo, ma rimuovo il collider via USD subito dopo lo spawn: niente interazione
#      fisica col drone (come richiesto: questo controllore vola in spazio libero), ma
#      drone e griglia entrambi visibili in viewport.
#
#   3. DEBUG PRINT su console (oltre a self.extras["log"], che skrl al momento non
#      mostra in TensorBoard - problema separato, non ancora risolto): stampa quando un
#      gruppo di ambienti viene promosso di livello, e un riepilogo periodico di
#      curriculum/reward/errori per asse, ogni N chiamate a _reset_idx (cfg
#      debug_print_every_n_resets), cosi' hai visibilita' anche senza TensorBoard.

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from .vel_controller_env_cfg import MyDroneVelEnvCfg


class MyDroneVelEnv(DirectRLEnv):
    cfg: MyDroneVelEnvCfg

    def __init__(self, cfg: MyDroneVelEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -- corpo del drone su cui applicare spinta e momenti --
        self._body_id = self.robot.find_bodies("body")[0]
        self._robot_mass = float(self.robot.root_physx_view.get_masses()[0].sum())
        self._gravity_magnitude = float(
            torch.tensor(self.cfg.sim.gravity, device=self.device).norm().item()
        )
        self._robot_weight = self._robot_mass * self._gravity_magnitude

        # -- buffer azioni / forze --
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # -- range dei target di velocita', shape (4,) -> [vx, vy, vz, wz] --
        self._vel_range = torch.tensor(
            [
                self.cfg.max_lin_vel_xy,
                self.cfg.max_lin_vel_xy,
                self.cfg.max_lin_vel_z,
                self.cfg.max_ang_vel_z,
            ],
            device=self.device,
        )

        # -- stato target / curriculum, per-ambiente --
        self._target_vel = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_mask = torch.ones(self.num_envs, 4, device=self.device)

        self._hold_timer = torch.zeros(self.num_envs, device=self.device)
        self._hold_duration = torch.zeros(self.num_envs, device=self.device)

        n_livelli = len(self.cfg.curriculum_livelli)
        self._curriculum_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._max_livello = n_livelli - 1

        # contatore cumulativo di promozioni, mai azzerato (per Curriculum/total_promotions)
        self._total_promotions = 0

        # accumulatori per l'errore medio nel periodo di hold corrente (dopo il transiente)
        self._err_accum = torch.zeros(self.num_envs, device=self.device)
        self._err_count = torch.zeros(self.num_envs, device=self.device)

        # tabelle precompute dei pesi/parametri di curriculum per livello, shape (n_livelli, ...)
        self._cur_pesi_modalita = torch.tensor(
            [liv["pesi_modalita"] for liv in self.cfg.curriculum_livelli], device=self.device
        )  # (n_livelli, 5)
        self._cur_hold_min = torch.tensor(
            [liv["hold_min_s"] for liv in self.cfg.curriculum_livelli], device=self.device
        )
        self._cur_hold_max = torch.tensor(
            [liv["hold_max_s"] for liv in self.cfg.curriculum_livelli], device=self.device
        )
        self._cur_mode3_min_frac = torch.tensor(
            [liv["mode3_min_frac"] for liv in self.cfg.curriculum_livelli], device=self.device
        )

        # contatore globale di "step di controllo" (a decimation, non a fisica) per il
        # curriculum sull'action_rate penalty
        self._control_step_counter = 0

        # contatore delle chiamate a _reset_idx, solo per throttle dei print di debug
        self._reset_call_counter = 0

        # accumulatori per-episodio delle singole componenti di reward, solo per
        # logging/debug in TensorBoard
        self._episode_sums = {
            "vel_error": torch.zeros(self.num_envs, device=self.device),
            "action_rate": torch.zeros(self.num_envs, device=self.device),
            "unwanted_ang_vel": torch.zeros(self.num_envs, device=self.device),
            "tilt": torch.zeros(self.num_envs, device=self.device),
            "died": torch.zeros(self.num_envs, device=self.device),
        }
        self._episode_axis_error_sums = {
            "vx": torch.zeros(self.num_envs, device=self.device),
            "vy": torch.zeros(self.num_envs, device=self.device),
            "vz": torch.zeros(self.num_envs, device=self.device),
            "wz": torch.zeros(self.num_envs, device=self.device),
        }

        # inizializza i target per tutti gli ambienti
        self._sample_new_targets(torch.arange(self.num_envs, device=self.device))

    # ===================================================================
    # SETUP SCENA
    # ===================================================================
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        # Piano di riferimento visivo CON GRIGLIA, ma SENZA collisione fisica.
        #
        # Uso l'asset standard "grid" di Isaac Lab (stesso look del ground plane di
        # default, non il cubo bianco pieno di prima che saturava la vista) tramite
        # GroundPlaneCfg, poi rimuovo il collider via USD subito dopo lo spawn: questo
        # controllore vola in spazio libero e non deve MAI interagire fisicamente col
        # pavimento (compito del futuro controllore di posizione).
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        self._rimuovi_collider_pavimento("/World/ground")

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @staticmethod
    def _rimuovi_collider_pavimento(prim_path: str) -> None:
        """Rimuove ricorsivamente CollisionAPI/RigidBodyAPI dal prim del pavimento (e
        dai suoi figli, dato che l'asset "grid" standard spesso mette il collider su un
        sotto-prim). Il pavimento resta visibile ma non interagisce mai fisicamente col
        drone: e' solo un riferimento grafico per il volo in spazio libero."""
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            print(f"[vel_controller_env] ATTENZIONE: prim pavimento '{prim_path}' non trovato, "
                  f"impossibile rimuovere il collider.")
            return

        n_rimossi = 0
        for prim in Usd.PrimRange(root_prim):
            for api in (UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
                    n_rimossi += 1
        print(f"[vel_controller_env] Pavimento visivo senza collisione: rimossi {n_rimossi} "
              f"componenti fisici sotto '{prim_path}'.")

    # ===================================================================
    # AZIONI
    # ===================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # thrust totale: canale 0 in [-1,1] -> [0,1] -> [0, thrust_to_weight * peso]
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # 3 momenti, canali 1..3
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]

    def _apply_action(self) -> None:
        self.robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ===================================================================
    # OSSERVAZIONI
    # ===================================================================
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot.data.root_lin_vel_b,      # 3
                self.robot.data.root_ang_vel_b,       # 3
                self.robot.data.projected_gravity_b,  # 3
                self._actions,                        # 4
                self._target_vel,                     # 4
            ),
            dim=-1,
        )
        return {"policy": obs}

    # ===================================================================
    # REWARD
    # ===================================================================
    def _current_action_rate_scale(self) -> float:
        """Interpola linearmente la penalita' di action_rate dal valore iniziale a quello
        finale in funzione dello step di controllo globale (curriculum temporale)."""
        frac = min(1.0, self._control_step_counter / max(1, self.cfg.action_rate_curriculum_steps))
        start = self.cfg.action_rate_reward_scale_start
        end = self.cfg.action_rate_reward_scale_end
        return start + frac * (end - start)

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_b = self.robot.data.root_lin_vel_b
        ang_vel_z = self.robot.data.root_ang_vel_b[:, 2]
        vel_attuale = torch.cat((lin_vel_b, ang_vel_z.unsqueeze(-1)), dim=-1)  # (N,4)

        # errore normalizzato per il range massimo di CIASCUN asse, prima di elevare al
        # quadrato, cosi' assi con range fisici diversi contribuiscono in modo comparabile
        err_sq_norm = torch.square((vel_attuale - self._target_vel) / self._vel_range)

        # FIX: la reward usa la SOMMA dell'errore normalizzato, non la media per asse
        # attivo. Dividere per n_assi_attivi qui (come nella versione precedente)
        # diluiva il segnale di tracking proprio quando piu' assi sono attivi insieme
        # (mod 0/3) - l'opposto di quanto serve. Coerente col vecchio env rsl_rl
        # funzionante, dove la reward usava l'errore raw (non diviso) e la
        # normalizzazione per asse attivo restava riservata alla sola valutazione del
        # curriculum, vedi sotto.
        vel_error_reward = err_sq_norm.sum(dim=-1)

        ang_vel_xy = self.robot.data.root_ang_vel_b[:, :2]
        unwanted_ang_vel = torch.sum(torch.square(ang_vel_xy), dim=-1)

        gz = torch.clamp(-self.robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
        tilt_rad = torch.acos(gz)
        tilt_free_rad = math.radians(self.cfg.tilt_free_deg)
        tilt_eccesso = torch.clamp(tilt_rad - tilt_free_rad, min=0.0)

        action_rate_scale = self._current_action_rate_scale()
        action_rate = torch.sum(torch.square(self._actions - self._prev_actions), dim=-1)

        died = self._is_died()

        # i 4 termini continui sono scalati per self.step_dt. died_penalty resta senza
        # step_dt: e' un evento discreto (una tantum), non un rate.
        r_vel_error = self.cfg.vel_error_reward_scale * vel_error_reward * self.step_dt
        r_action_rate = action_rate_scale * action_rate * self.step_dt
        r_unwanted_ang_vel = self.cfg.unwanted_ang_vel_reward_scale * unwanted_ang_vel * self.step_dt
        r_tilt = self.cfg.tilt_reward_scale * torch.square(tilt_eccesso) * self.step_dt
        r_died = self.cfg.died_penalty * died.float()

        reward = r_vel_error + r_action_rate + r_unwanted_ang_vel + r_tilt + r_died

        # accumulo per-episodio delle singole componenti, solo per logging
        self._episode_sums["vel_error"] += r_vel_error
        self._episode_sums["action_rate"] += r_action_rate
        self._episode_sums["unwanted_ang_vel"] += r_unwanted_ang_vel
        self._episode_sums["tilt"] += r_tilt
        self._episode_sums["died"] += r_died

        # errore per singolo asse, raw (non normalizzato), solo per logging
        raw_axis_sq_error = torch.square(vel_attuale - self._target_vel)  # (N,4)
        axis_names = ("vx", "vy", "vz", "wz")
        for i, name in enumerate(axis_names):
            self._episode_axis_error_sums[name] += raw_axis_sq_error[:, i] * self.step_dt

        # -- valutazione per il CURRICULUM: qui SI', normalizzata per n_assi_attivi -- #
        # (errore quadratico medio PER ASSE ATTIVO, coerente col vecchio env rsl_rl:
        # serve solo a decidere la promozione di livello, non entra nella reward sopra)
        n_assi_attivi = torch.clamp(self._target_mask.sum(dim=-1), min=1.0)
        vel_error_curriculum = err_sq_norm.sum(dim=-1) / n_assi_attivi

        dopo_transiente = self._hold_timer >= self.cfg.curriculum_transiente_skip_s
        self._err_accum += torch.where(dopo_transiente, vel_error_curriculum, torch.zeros_like(vel_error_curriculum))
        self._err_count += dopo_transiente.float()

        self._control_step_counter += 1

        return reward

    # ===================================================================
    # TERMINAZIONE
    # ===================================================================
    def _is_died(self) -> torch.Tensor:
        # Il controllore di velocita' NON deve preoccuparsi del pavimento (nessun
        # collider fisico nella scena, vedi _setup_scene): l'unica condizione di morte
        # e' il tilt eccessivo.
        if not self.cfg.terminate_su_tilt_eccessivo:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        gz = torch.clamp(-self.robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
        tilt_rad = torch.acos(gz)
        return tilt_rad > math.radians(self.cfg.max_tilt_deg)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self._is_died()

        # NOVITA': se disable_target_resampling e' True (impostato da script esterni,
        # es. evaluation con riferimento costante/manuale), il resampling interno del
        # target/curriculum viene saltato del tutto. Impostare solo _hold_duration a un
        # valore alto NON basta: env.reset() lo sovrascrive comunque tramite
        # _sample_new_targets, e il livello di curriculum riparte sempre da 0 ad ogni
        # nuova istanza dell'ambiente (non e' salvato nel checkpoint) - quindi
        # hold_max_s del livello 0 (5.0s) faceva ripartire il resampling interno ogni
        # ~3.5-5s, con uno spike di tracking ad ogni scadenza (la policy vedeva per uno
        # step il nuovo target casuale prima che lo script potesse sovrascriverlo).
        if getattr(self, "disable_target_resampling", False):
            return died, time_out

        dt = self.step_dt
        self._hold_timer += dt
        scaduti = self._hold_timer >= self._hold_duration
        if torch.any(scaduti):
            self._valuta_e_promuovi(scaduti)
            self._sample_new_targets(scaduti.nonzero(as_tuple=False).squeeze(-1))

        return died, time_out

    # ===================================================================
    # CURRICULUM: valutazione successo periodo di hold + promozione livello
    # ===================================================================
    def _valuta_e_promuovi(self, env_ids_scaduti: torch.Tensor) -> None:
        idx = env_ids_scaduti.nonzero(as_tuple=False).squeeze(-1) if env_ids_scaduti.dtype == torch.bool else env_ids_scaduti
        if idx.numel() == 0:
            return

        conteggio = torch.clamp(self._err_count[idx], min=1.0)
        errore_medio = self._err_accum[idx] / conteggio
        riuscito = errore_medio < self.cfg.curriculum_soglia_successo

        self._success_streak[idx] = torch.where(
            riuscito,
            self._success_streak[idx] + 1,
            torch.zeros_like(self._success_streak[idx]),
        )

        puo_salire = (self._success_streak[idx] >= self.cfg.curriculum_successi_richiesti) & (
            self._curriculum_level[idx] < self._max_livello
        )
        livelli_prima = self._curriculum_level[idx].clone()
        self._curriculum_level[idx] = torch.where(
            puo_salire, self._curriculum_level[idx] + 1, self._curriculum_level[idx]
        )
        self._success_streak[idx] = torch.where(
            puo_salire, torch.zeros_like(self._success_streak[idx]), self._success_streak[idx]
        )

        n_promossi = int(puo_salire.sum().item())
        self._total_promotions += n_promossi

        # -- DEBUG PRINT: promozioni di livello --
        # stampa un aggregato (non per-env, con 4096 ambienti sarebbe illeggibile) ogni
        # volta che almeno un ambiente sale di livello.
        if n_promossi > 0:
            idx_promossi = puo_salire.nonzero(as_tuple=False).squeeze(-1)
            livelli_dest = (livelli_prima[idx_promossi] + 1)
            for lvl in torch.unique(livelli_dest).tolist():
                cnt = int((livelli_dest == lvl).sum().item())
                print(f"[curriculum] {cnt} ambiente/i promossi al livello {lvl} "
                      f"(errore medio soglia={self.cfg.curriculum_soglia_successo}, "
                      f"totale promozioni finora={self._total_promotions})")

        self._err_accum[idx] = 0.0
        self._err_count[idx] = 0.0

    # ===================================================================
    # GENERAZIONE TARGET (5 modalita', pesate secondo il livello di curriculum corrente)
    # ===================================================================
    def _sample_new_targets(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        livelli = self._curriculum_level[env_ids]  # (n,)

        pesi = self._cur_pesi_modalita[livelli]  # (n, 5)
        modalita = torch.multinomial(pesi, num_samples=1).squeeze(-1)  # (n,) valori in {0,1,2,3,4}

        target = torch.zeros(n, 4, device=self.device)
        mask = torch.zeros(n, 4, device=self.device)
        rango = self._vel_range  # (4,)

        # -- modalita' 0: random multi-asse, tutti gli assi attivi --
        sel = modalita == 0
        if torch.any(sel):
            m = sel.nonzero(as_tuple=False).squeeze(-1)
            target[m] = sample_uniform(-1.0, 1.0, (m.numel(), 4), self.device) * rango
            mask[m] = 1.0

        # -- modalita' 1: mask casuale, sottoinsieme di assi attivi (almeno 1) --
        sel = modalita == 1
        if torch.any(sel):
            m = sel.nonzero(as_tuple=False).squeeze(-1)
            k = m.numel()
            prob_attivo = sample_uniform(0.3, 0.9, (k, 1), self.device)
            m_mask = (torch.rand(k, 4, device=self.device) < prob_attivo).float()
            nessuno = m_mask.sum(dim=-1) == 0
            if torch.any(nessuno):
                idx_forzati = torch.randint(0, 4, (int(nessuno.sum().item()),), device=self.device)
                righe = nessuno.nonzero(as_tuple=False).squeeze(-1)
                m_mask[righe, idx_forzati] = 1.0
            valori = sample_uniform(-1.0, 1.0, (k, 4), self.device) * rango
            target[m] = valori * m_mask
            mask[m] = m_mask

        # -- modalita' 2: singolo asse attivo, gli altri a zero --
        sel = modalita == 2
        if torch.any(sel):
            m = sel.nonzero(as_tuple=False).squeeze(-1)
            k = m.numel()
            asse = torch.randint(0, 4, (k,), device=self.device)
            m_mask = torch.zeros(k, 4, device=self.device)
            m_mask[torch.arange(k, device=self.device), asse] = 1.0
            valori = sample_uniform(-1.0, 1.0, (k, 4), self.device) * rango
            target[m] = valori * m_mask
            mask[m] = m_mask

        # -- modalita' 3: combinato aggressivo, tutti gli assi attivi, ampiezza minima --
        sel = modalita == 3
        if torch.any(sel):
            m = sel.nonzero(as_tuple=False).squeeze(-1)
            k = m.numel()
            frac_min = self._cur_mode3_min_frac[livelli[sel]]  # (k,)
            segno = torch.where(
                torch.rand(k, 4, device=self.device) < 0.5,
                -torch.ones(k, 4, device=self.device),
                torch.ones(k, 4, device=self.device),
            )
            ampiezza = frac_min.unsqueeze(-1) + (1.0 - frac_min.unsqueeze(-1)) * torch.rand(k, 4, device=self.device)
            target[m] = segno * ampiezza * rango
            mask[m] = 1.0

        # -- modalita' 4: hover / piccola correzione --
        sel = modalita == 4
        if torch.any(sel):
            m = sel.nonzero(as_tuple=False).squeeze(-1)
            k = m.numel()
            is_hover_esatto = torch.rand(k, device=self.device) < 0.5
            piccola_corr = sample_uniform(-1.0, 1.0, (k, 4), self.device) * (self.cfg.hover_frac * rango)
            valori = torch.where(is_hover_esatto.unsqueeze(-1), torch.zeros(k, 4, device=self.device), piccola_corr)
            target[m] = valori
            mask[m] = 1.0

        self._target_vel[env_ids] = target
        self._target_mask[env_ids] = mask

        hold_min = self._cur_hold_min[livelli]
        hold_max = self._cur_hold_max[livelli]
        self._hold_duration[env_ids] = hold_min + torch.rand(n, device=self.device) * (hold_max - hold_min)
        self._hold_timer[env_ids] = 0.0

    # ===================================================================
    # RESET
    # ===================================================================
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # --- LOG DELLE STATISTICHE DI EPISODIO ---
        if "log" not in self.extras:
            self.extras["log"] = dict()
        for key in self._episode_sums.keys():
            episodic_avg = torch.mean(self._episode_sums[key][env_ids_t])
            self.extras["log"]["Episode_Reward/" + key] = (episodic_avg / self.cfg.episode_length_s).item()
            self._episode_sums[key][env_ids_t] = 0.0
        for name in self._episode_axis_error_sums.keys():
            axis_avg = torch.mean(self._episode_axis_error_sums[name][env_ids_t])
            self.extras["log"]["Episode_Metrics/error_" + name] = (axis_avg / self.cfg.episode_length_s).item()
            self._episode_axis_error_sums[name][env_ids_t] = 0.0

        # --- LOG DEL CURRICULUM, calcolato su TUTTI gli ambienti ---
        self.extras["log"]["Curriculum/mean_level"] = torch.mean(self._curriculum_level.float()).item()
        self.extras["log"]["Curriculum/max_level_reached"] = torch.max(self._curriculum_level).item()
        level_counts = torch.bincount(self._curriculum_level, minlength=len(self.cfg.curriculum_livelli))
        for lvl in range(len(self.cfg.curriculum_livelli)):
            frac = level_counts[lvl].float() / self.num_envs
            self.extras["log"][f"Curriculum/frac_at_level_{lvl}"] = frac.item()
        self.extras["log"]["Curriculum/total_promotions"] = float(self._total_promotions)

        # --- DEBUG PRINT: riepilogo periodico su console ---
        # bypassa il problema (non ancora risolto) di skrl che non mostra
        # self.extras["log"] in TensorBoard: qui hai visibilita' comunque.
        self._reset_call_counter += 1
        n_ogni = self.cfg.debug_print_every_n_resets
        if n_ogni and self._reset_call_counter % n_ogni == 0:
            fracs = " ".join(
                f"L{lvl}={self.extras['log'][f'Curriculum/frac_at_level_{lvl}']:.2f}"
                for lvl in range(len(self.cfg.curriculum_livelli))
            )
            print(
                f"[curriculum][step_ctrl={self._control_step_counter}] "
                f"mean_level={self.extras['log']['Curriculum/mean_level']:.3f} "
                f"max_level={self.extras['log']['Curriculum/max_level_reached']} "
                f"promozioni_tot={self._total_promotions} | {fracs}"
            )
            print(
                f"[reward][step_ctrl={self._control_step_counter}] "
                f"vel_error={self.extras['log']['Episode_Reward/vel_error']:.4f} "
                f"action_rate={self.extras['log']['Episode_Reward/action_rate']:.4f} "
                f"tilt={self.extras['log']['Episode_Reward/tilt']:.4f} "
                f"died={self.extras['log']['Episode_Reward/died']:.4f}"
            )
            print(
                f"[errori_asse][step_ctrl={self._control_step_counter}] "
                f"vx={self.extras['log']['Episode_Metrics/error_vx']:.4f} "
                f"vy={self.extras['log']['Episode_Metrics/error_vy']:.4f} "
                f"vz={self.extras['log']['Episode_Metrics/error_vz']:.4f} "
                f"wz={self.extras['log']['Episode_Metrics/error_wz']:.4f}"
            )

        # --- RESET DEL ROBOT ---
        self.robot.reset(env_ids_t)
        super()._reset_idx(env_ids)

        # al reset iniziale di TUTTI gli ambienti, randomizza episode_length_buf cosi'
        # non terminano tutti sincronizzati per timeout
        if len(env_ids_t) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        joint_pos = self.robot.data.default_joint_pos[env_ids_t]
        joint_vel = self.robot.data.default_joint_vel[env_ids_t]

        default_root_state = self.robot.data.default_root_state[env_ids_t].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids_t]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_t)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_t)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_t)

        self._actions[env_ids_t] = 0.0
        self._prev_actions[env_ids_t] = 0.0

        # il livello di curriculum e la streak NON vengono azzerati al reset dell'episodio
        self._err_accum[env_ids_t] = 0.0
        self._err_count[env_ids_t] = 0.0
        self._sample_new_targets(env_ids_t)