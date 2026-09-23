# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from ..drone_physics import apply_drone_inertial_props, log_inertial_props
from .vel_controller_env_cfg import MyDroneVelEnvCfg

class MyDroneVelEnv(DirectRLEnv):
    cfg: MyDroneVelEnvCfg

    def __init__(self, cfg: MyDroneVelEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        applied = apply_drone_inertial_props(self.robot, self.cfg.drone_inertial)
        log_inertial_props("vel_controller_env", applied, self.cfg.drone_inertial)

        self._body_id = self.robot.find_bodies("body")[0]
        self._moment_scale = torch.tensor(
            self.cfg.moment_scale, device=self.device
        ).unsqueeze(0)
        self._robot_mass = float(self.robot.root_physx_view.get_masses()[0].sum())
        self._gravity_magnitude = float(
            torch.tensor(self.cfg.sim.gravity, device=self.device).norm().item()
        )
        self._robot_weight = self._robot_mass * self._gravity_magnitude

        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._vel_range = torch.tensor(
            [
                self.cfg.max_lin_vel_xy,
                self.cfg.max_lin_vel_xy,
                self.cfg.max_lin_vel_z,
                self.cfg.max_ang_vel_z,
            ],
            device=self.device,
        )

        self._target_vel = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_mask = torch.ones(self.num_envs, 4, device=self.device)

        self._hold_timer = torch.zeros(self.num_envs, device=self.device)
        self._hold_duration = torch.zeros(self.num_envs, device=self.device)

        n_livelli = len(self.cfg.curriculum_livelli)
        self._curriculum_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._max_livello = n_livelli - 1

        self._total_promotions = 0

        self._err_accum = torch.zeros(self.num_envs, device=self.device)
        self._err_count = torch.zeros(self.num_envs, device=self.device)

        self._cur_pesi_modalita = torch.tensor(
            [liv["pesi_modalita"] for liv in self.cfg.curriculum_livelli], device=self.device
        )
        self._cur_hold_min = torch.tensor(
            [liv["hold_min_s"] for liv in self.cfg.curriculum_livelli], device=self.device
        )
        self._cur_hold_max = torch.tensor(
            [liv["hold_max_s"] for liv in self.cfg.curriculum_livelli], device=self.device
        )
        self._cur_mode3_min_frac = torch.tensor(
            [liv["mode3_min_frac"] for liv in self.cfg.curriculum_livelli], device=self.device
        )

        self._control_step_counter = 0

        self._reset_call_counter = 0

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

        self._sample_new_targets(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self._moment_scale * self._actions[:, 1:4]

    def _apply_action(self) -> None:
        self.robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._actions,
                self._target_vel,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _current_action_rate_scale(self) -> float:
        frac = min(1.0, self._control_step_counter / max(1, self.cfg.action_rate_curriculum_steps))
        start = self.cfg.action_rate_reward_scale_start
        end = self.cfg.action_rate_reward_scale_end
        return start + frac * (end - start)

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_b = self.robot.data.root_lin_vel_b
        ang_vel_z = self.robot.data.root_ang_vel_b[:, 2]
        vel_attuale = torch.cat((lin_vel_b, ang_vel_z.unsqueeze(-1)), dim=-1)

        err_sq_norm = torch.square((vel_attuale - self._target_vel) / self._vel_range)

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

        r_vel_error = self.cfg.vel_error_reward_scale * vel_error_reward * self.step_dt
        r_action_rate = action_rate_scale * action_rate * self.step_dt
        r_unwanted_ang_vel = self.cfg.unwanted_ang_vel_reward_scale * unwanted_ang_vel * self.step_dt
        r_tilt = self.cfg.tilt_reward_scale * torch.square(tilt_eccesso) * self.step_dt
        r_died = self.cfg.died_penalty * died.float()

        reward = r_vel_error + r_action_rate + r_unwanted_ang_vel + r_tilt + r_died

        self._episode_sums["vel_error"] += r_vel_error
        self._episode_sums["action_rate"] += r_action_rate
        self._episode_sums["unwanted_ang_vel"] += r_unwanted_ang_vel
        self._episode_sums["tilt"] += r_tilt
        self._episode_sums["died"] += r_died

        raw_axis_sq_error = torch.square(vel_attuale - self._target_vel)
        axis_names = ("vx", "vy", "vz", "wz")
        for i, name in enumerate(axis_names):
            self._episode_axis_error_sums[name] += raw_axis_sq_error[:, i] * self.step_dt

        n_assi_attivi = torch.clamp(self._target_mask.sum(dim=-1), min=1.0)
        vel_error_curriculum = err_sq_norm.sum(dim=-1) / n_assi_attivi

        dopo_transiente = self._hold_timer >= self.cfg.curriculum_transiente_skip_s
        self._err_accum += torch.where(dopo_transiente, vel_error_curriculum, torch.zeros_like(vel_error_curriculum))
        self._err_count += dopo_transiente.float()

        self._control_step_counter += 1

        return reward

    def _is_died(self) -> torch.Tensor:
        if not self.cfg.terminate_su_tilt_eccessivo:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        gz = torch.clamp(-self.robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
        tilt_rad = torch.acos(gz)
        return tilt_rad > math.radians(self.cfg.max_tilt_deg)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self._is_died()

        if getattr(self, "disable_target_resampling", False):
            return died, time_out

        dt = self.step_dt
        self._hold_timer += dt
        scaduti = self._hold_timer >= self._hold_duration
        if torch.any(scaduti):
            self._valuta_e_promuovi(scaduti)
            self._sample_new_targets(scaduti.nonzero(as_tuple=False).squeeze(-1))

        return died, time_out

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

    def _sample_new_targets(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        livelli = self._curriculum_level[env_ids]

        pesi = self._cur_pesi_modalita[livelli]
        modalita = torch.multinomial(pesi, num_samples=1).squeeze(-1)

        target = torch.zeros(n, 4, device=self.device)
        mask = torch.zeros(n, 4, device=self.device)
        rango = self._vel_range

        sel = modalita == 0
        if torch.any(sel):
            m = sel.nonzero(as_tuple=False).squeeze(-1)
            target[m] = sample_uniform(-1.0, 1.0, (m.numel(), 4), self.device) * rango
            mask[m] = 1.0

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

        sel = modalita == 3
        if torch.any(sel):
            m = sel.nonzero(as_tuple=False).squeeze(-1)
            k = m.numel()
            frac_min = self._cur_mode3_min_frac[livelli[sel]]
            segno = torch.where(
                torch.rand(k, 4, device=self.device) < 0.5,
                -torch.ones(k, 4, device=self.device),
                torch.ones(k, 4, device=self.device),
            )
            ampiezza = frac_min.unsqueeze(-1) + (1.0 - frac_min.unsqueeze(-1)) * torch.rand(k, 4, device=self.device)
            target[m] = segno * ampiezza * rango
            mask[m] = 1.0

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

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

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

        self.extras["log"]["Curriculum/mean_level"] = torch.mean(self._curriculum_level.float()).item()
        self.extras["log"]["Curriculum/max_level_reached"] = torch.max(self._curriculum_level).item()
        level_counts = torch.bincount(self._curriculum_level, minlength=len(self.cfg.curriculum_livelli))
        for lvl in range(len(self.cfg.curriculum_livelli)):
            frac = level_counts[lvl].float() / self.num_envs
            self.extras["log"][f"Curriculum/frac_at_level_{lvl}"] = frac.item()
        self.extras["log"]["Curriculum/total_promotions"] = float(self._total_promotions)

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

        self.robot.reset(env_ids_t)
        super()._reset_idx(env_ids)

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

        self._err_accum[env_ids_t] = 0.0
        self._err_count[env_ids_t] = 0.0
        self._sample_new_targets(env_ids_t)
