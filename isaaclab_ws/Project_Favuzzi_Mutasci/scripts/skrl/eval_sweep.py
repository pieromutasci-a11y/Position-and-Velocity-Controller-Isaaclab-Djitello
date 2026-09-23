from __future__ import annotations

import csv
import math
import os
import tempfile

import torch
import wandb

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, sample_uniform

ALPHA_YAW = 0.2
REGIME_FRAC = 0.70
W_FULL = 1.0
W_UNICICLO = 1.0

WCOMP_ERR_REGIME = 3.0
WCOMP_ERR_TRANSIT = 1.0
WCOMP_SMOOTHNESS = 20.0
WCOMP_AZIONI = 5.0
WCOMP_VY_REF = 10.0
WCOMP_VY_REAL = 10.0
WCOMP_REVERSE = 15.0

EVAL_SEED = 20260101
EVAL_NUM_ENVS = 3000
EVAL_NUM_QUEUES = 3
EVAL_MAX_STEPS_PER_QUEUE = 600

_MODES = ("hover", "singolo", "variabile")
_MASKS = ("full", "uniciclo")
_DOF_MASKS = {
    "full":     (1.0, 1.0, 1.0, 1.0),
    "uniciclo": (1.0, 0.0, 1.0, 1.0),
}

def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))

def _assign_eval_groups(n_eval: int, device: torch.device) -> dict[tuple[str, str], torch.Tensor]:
    combos = [(m, mo) for m in _MASKS for mo in _MODES]
    n_combo = len(combos)
    chunk = n_eval // n_combo
    groups = {}
    start = 0
    for i, combo in enumerate(combos):
        end = start + chunk if i < n_combo - 1 else n_eval
        groups[combo] = torch.arange(start, end, device=device, dtype=torch.long)
        start = end
    return groups

def _setup_eval_scenario(base_env, groups: dict[tuple[str, str], torch.Tensor], device: torch.device):
    torch.manual_seed(EVAL_SEED)

    all_ids = torch.cat(list(groups.values()))
    base_env._sample_room(all_ids)

    from Project_Favuzzi_Mutasci.tasks.direct.pos_controller.pos_controller_env import (
        MODE_VARIABILE,
        MODE_SINGOLO,
    )

    for (mask_name, mode_name), ids in groups.items():
        if ids.numel() == 0:
            continue

        mask_tensor = torch.tensor(_DOF_MASKS[mask_name], device=device, dtype=torch.float)
        base_env._dof_mask[ids] = mask_tensor

        if mode_name == "hover":
            base_env._reference_mode[ids] = MODE_SINGOLO
            base_env._is_hover[ids] = True
        elif mode_name == "singolo":
            base_env._reference_mode[ids] = MODE_SINGOLO
            base_env._is_hover[ids] = False
        else:
            base_env._reference_mode[ids] = MODE_VARIABILE
            base_env._is_hover[ids] = False
        base_env._is_canonical[ids] = False

    n = all_ids.numel()
    joint_pos = base_env.robot.data.default_joint_pos[all_ids]
    joint_vel = base_env.robot.data.default_joint_vel[all_ids]
    root_state = base_env.robot.data.default_root_state[all_ids].clone()

    spawn_pos_env = base_env._sample_in_room(all_ids, base_env.cfg.spawn_room_margin)
    root_state[:, :3] = base_env._env_origins[all_ids] + spawn_pos_env

    spawn_yaw = sample_uniform(-math.pi, math.pi, (n,), device=device)
    zero = torch.zeros_like(spawn_yaw)
    root_state[:, 3:7] = quat_from_euler_xyz(zero, zero, spawn_yaw)
    root_state[:, 7:] = 0.0

    base_env.robot.write_root_pose_to_sim(root_state[:, :7], all_ids)
    base_env.robot.write_root_velocity_to_sim(root_state[:, 7:], all_ids)
    base_env.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, all_ids)

    base_env._wp_idx[all_ids] = 0
    base_env._hold_timer[all_ids] = 0.0
    base_env._advanced_now[all_ids] = False
    base_env._err_integral[all_ids] = 0.0

    base_env._actions[all_ids] = 0.0
    base_env._high_level_actions[all_ids] = 0.0
    base_env._prev_high_level_actions[all_ids] = 0.0
    base_env._physics_step_counter = 0

    base_env._build_queue(all_ids, spawn_pos=spawn_pos_env, spawn_yaw=spawn_yaw)

def _regenerate_queue(base_env, ids: torch.Tensor):
    if ids.numel() == 0:
        return
    pos_env = base_env.robot.data.root_pos_w[ids] - base_env._env_origins[ids]
    _, _, yaw = euler_xyz_from_quat(base_env.robot.data.root_quat_w[ids])
    base_env._build_queue(ids, spawn_pos=pos_env, spawn_yaw=yaw)

class _Accumulators:

    def __init__(self, n_total: int, max_steps_per_wp: int, device: torch.device):
        self.device = device
        self.n_total = n_total
        self.max_steps = max_steps_per_wp

        self.err_buf = torch.zeros(n_total, max_steps_per_wp, device=device)
        self.step_in_wp = torch.zeros(n_total, dtype=torch.long, device=device)

        self.sum_err_regime = torch.zeros(n_total, device=device)
        self.sum_err_transit = torch.zeros(n_total, device=device)
        self.n_wp_completed = torch.zeros(n_total, device=device)

        self.sum_smoothness = torch.zeros(n_total, device=device)
        self.sum_azioni = torch.zeros(n_total, device=device)
        self.sum_vy_ref_abs = torch.zeros(n_total, device=device)
        self.sum_vy_real_abs = torch.zeros(n_total, device=device)
        self.sum_reverse_vx = torch.zeros(n_total, device=device)
        self.total_steps = torch.zeros(n_total, device=device)

    def record_step(self, base_env, active_mask: torch.Tensor):
        dev = self.device
        pos_env = base_env.robot.data.root_pos_w - base_env._env_origins
        w0_pos, w0_yaw = base_env._current_target()
        _, _, yaw = euler_xyz_from_quat(base_env.robot.data.root_quat_w)

        dist = torch.norm(pos_env - w0_pos, dim=1)
        yaw_err = torch.abs(_wrap_to_pi(yaw - w0_yaw))
        err_combined = dist + ALPHA_YAW * yaw_err

        idx = self.step_in_wp.clamp(max=self.max_steps - 1)
        ar = torch.arange(self.n_total, device=dev)
        self.err_buf[ar[active_mask], idx[active_mask]] = err_combined[active_mask]
        self.step_in_wp[active_mask] += 1

        is_uniciclo = base_env._dof_mask[:, 1] < 0.5

        cmd_now = base_env._high_level_actions
        cmd_prev = base_env._prev_high_level_actions

        d_act = cmd_now - cmd_prev
        smoothness_step = torch.sum(torch.square(d_act), dim=1)

        azioni_step = torch.sum(torch.square(cmd_now), dim=1)

        vy_ref_step = torch.abs(cmd_now[:, 1])
        vy_real_step = torch.abs(base_env.robot.data.root_lin_vel_b[:, 1])
        reverse_step = torch.clamp(-cmd_now[:, 0], min=0.0)

        m = active_mask.float()
        self.sum_smoothness += m * smoothness_step
        self.sum_azioni += m * azioni_step
        self.sum_vy_ref_abs += m * is_uniciclo.float() * vy_ref_step
        self.sum_vy_real_abs += m * is_uniciclo.float() * vy_real_step
        self.sum_reverse_vx += m * is_uniciclo.float() * reverse_step
        self.total_steps += m

    def on_waypoint_advanced(self, adv_ids: torch.Tensor):
        if adv_ids.numel() == 0:
            return
        for env_id in adv_ids.tolist():
            length = int(self.step_in_wp[env_id].item())
            if length <= 0:
                continue
            length = min(length, self.max_steps)
            cutoff = max(1, int(round(length * REGIME_FRAC)))
            cutoff = min(cutoff, length)
            transit_vals = self.err_buf[env_id, :cutoff]
            regime_vals = self.err_buf[env_id, cutoff:length]
            self.sum_err_transit[env_id] += transit_vals.mean()
            if regime_vals.numel() > 0:
                self.sum_err_regime[env_id] += regime_vals.mean()
            else:
                self.sum_err_regime[env_id] += transit_vals.mean()
            self.n_wp_completed[env_id] += 1

        self.err_buf[adv_ids] = 0.0
        self.step_in_wp[adv_ids] = 0

class _RepresentativeRecorder:
    def __init__(self, groups: dict[tuple[str, str], torch.Tensor]):
        self.rep_ids = {combo: ids[0].item() for combo, ids in groups.items() if ids.numel() > 0}
        self.rows: dict[tuple[str, str], list[dict]] = {combo: [] for combo in self.rep_ids}
        self._step_counter = 0

    def record_pre_step(self, base_env):
        pos_env = base_env.robot.data.root_pos_w - base_env._env_origins
        _, _, yaw = euler_xyz_from_quat(base_env.robot.data.root_quat_w)
        w0_pos, w0_yaw = base_env._current_target()

        for combo, env_id in self.rep_ids.items():
            self.rows[combo].append({
                "step": self._step_counter,
                "target_x": w0_pos[env_id, 0].item(),
                "target_y": w0_pos[env_id, 1].item(),
                "target_z": w0_pos[env_id, 2].item(),
                "target_yaw": w0_yaw[env_id].item(),
                "current_x": pos_env[env_id, 0].item(),
                "current_y": pos_env[env_id, 1].item(),
                "current_z": pos_env[env_id, 2].item(),
                "current_yaw": yaw[env_id].item(),
            })

    def record_post_step(self, base_env):
        hl_applied = base_env._high_level_actions.clamp(-1.0, 1.0)
        ref_vel_step = hl_applied * base_env._vel_ref_scale
        lin_vel_b = base_env.robot.data.root_lin_vel_b
        ang_vel_z = base_env.robot.data.root_ang_vel_b[:, 2]

        for combo, env_id in self.rep_ids.items():
            row = self.rows[combo][-1]
            row["lin_vel_x"] = lin_vel_b[env_id, 0].item()
            row["lin_vel_y"] = lin_vel_b[env_id, 1].item()
            row["lin_vel_z"] = lin_vel_b[env_id, 2].item()
            row["ang_vel_z"] = ang_vel_z[env_id].item()
            row["ref_vx"] = ref_vel_step[env_id, 0].item()
            row["ref_vy"] = ref_vel_step[env_id, 1].item()
            row["ref_vz"] = ref_vel_step[env_id, 2].item()
            row["ref_wz"] = ref_vel_step[env_id, 3].item()
            row["err_pos"] = math.sqrt(
                (row["current_x"] - row["target_x"]) ** 2
                + (row["current_y"] - row["target_y"]) ** 2
                + (row["current_z"] - row["target_z"]) ** 2
            )

        self._step_counter += 1

    def write_csv(self, control_dt: float, out_dir: str) -> list[str]:
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        fieldnames = [
            "time_s", "step",
            "target_x", "target_y", "target_z", "target_yaw",
            "current_x", "current_y", "current_z", "current_yaw",
            "err_pos",
            "lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_z",
            "ref_vx", "ref_vy", "ref_vz", "ref_wz",
        ]
        for (mask_name, mode_name), rows in self.rows.items():
            if not rows:
                continue
            path = os.path.join(out_dir, f"eval_timeseries_{mask_name}_{mode_name}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    row["time_s"] = row["step"] * control_dt
                    writer.writerow({k: row.get(k) for k in fieldnames})
            paths.append(path)
            print(f"[sweep_eval] CSV serie temporale salvato: {path}")
        return paths

@torch.inference_mode()
def _run_eval_loop(base_env, agent, groups: dict[tuple[str, str], torch.Tensor], device: torch.device,
                    recorder: "_RepresentativeRecorder | None" = None):
    n_total = base_env.num_envs
    all_ids = torch.cat(list(groups.values()))

    acc = _Accumulators(n_total, EVAL_MAX_STEPS_PER_QUEUE, device)

    queue_count = torch.zeros(n_total, dtype=torch.long, device=device)
    steps_on_queue = torch.zeros(n_total, dtype=torch.long, device=device)
    hold_steps = torch.zeros(n_total, dtype=torch.long, device=device)
    hold_steps_required = max(1, round(base_env.cfg.target_hold_time_s / base_env.step_dt))

    done_env = torch.ones(n_total, dtype=torch.bool, device=device)
    done_env[all_ids] = False

    reach_thr = base_env.cfg.target_reach_threshold
    max_total_steps = EVAL_NUM_QUEUES * EVAL_MAX_STEPS_PER_QUEUE

    step = 0
    while (~done_env[all_ids]).any() and step < max_total_steps:
        active_mask = ~done_env

        acc.record_step(base_env, active_mask)

        if recorder is not None:
            recorder.record_pre_step(base_env)

        obs = base_env._get_observations()["policy"]
        outputs = agent.act(obs, timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        actions = torch.where(active_mask.unsqueeze(-1), actions, torch.zeros_like(actions))

        base_env._pre_physics_step(actions)
        for _ in range(base_env.cfg.decimation):
            base_env._apply_action()
            base_env.scene.write_data_to_sim()
            base_env.sim.step(render=False)
            base_env.scene.update(dt=base_env.physics_dt)

        if recorder is not None:
            recorder.record_post_step(base_env)

        pos_env = base_env.robot.data.root_pos_w - base_env._env_origins
        w0_pos, w0_yaw = base_env._current_target()
        dist = torch.norm(pos_env - w0_pos, dim=1)
        n_wp = base_env.cfg.n_waypoints

        steps_on_queue[active_mask] += 1
        coda_esaurita = base_env._wp_idx >= (n_wp - 1)
        entro_soglia = coda_esaurita & (dist < reach_thr)
        hold_steps = torch.where(entro_soglia, hold_steps + 1, torch.zeros_like(hold_steps))
        assestato = hold_steps >= hold_steps_required
        timeout_coda = steps_on_queue >= EVAL_MAX_STEPS_PER_QUEUE

        adv_now = base_env._advanced_now & active_mask
        adv_ids = adv_now.nonzero(as_tuple=False).squeeze(-1)
        acc.on_waypoint_advanced(adv_ids)

        finish_queue = (assestato | timeout_coda) & active_mask
        finish_ids = finish_queue.nonzero(as_tuple=False).squeeze(-1)
        if finish_ids.numel() > 0:
            acc.on_waypoint_advanced(finish_ids)

            queue_count[finish_ids] += 1
            steps_on_queue[finish_ids] = 0
            hold_steps[finish_ids] = 0
            _regenerate_queue(base_env, finish_ids)

            done_now = queue_count[finish_ids] >= EVAL_NUM_QUEUES
            done_env[finish_ids[done_now]] = True

        step += 1

    return acc

def _aggregate(acc: _Accumulators, groups: dict[tuple[str, str], torch.Tensor]) -> dict[str, float]:
    results: dict[str, float] = {}
    cost_by_mask: dict[str, list[float]] = {"full": [], "uniciclo": []}

    comp_by_mask: dict[str, dict[str, list[float]]] = {
        "full": {"err_regime": [], "err_transit": [], "smoothness": [], "azioni": []},
        "uniciclo": {"err_regime": [], "err_transit": [], "smoothness": [], "azioni": [],
                     "vy_ref": [], "vy_real": [], "reverse_vx": []},
    }

    for (mask_name, mode_name), ids in groups.items():
        if ids.numel() == 0:
            continue

        n_wp_env = acc.n_wp_completed[ids].clamp(min=1.0)
        err_regime_per_env = acc.sum_err_regime[ids] / n_wp_env
        err_transit_per_env = acc.sum_err_transit[ids] / n_wp_env
        err_regime = err_regime_per_env.mean().item()
        err_transit = err_transit_per_env.mean().item()

        n_steps_env = acc.total_steps[ids].clamp(min=1.0)
        smoothness = (acc.sum_smoothness[ids] / n_steps_env).mean().item()
        azioni = (acc.sum_azioni[ids] / n_steps_env).mean().item()

        cost = (
            WCOMP_ERR_REGIME * err_regime
            + WCOMP_ERR_TRANSIT * err_transit
            + WCOMP_SMOOTHNESS * smoothness
            + WCOMP_AZIONI * azioni
        )

        comp_by_mask[mask_name]["err_regime"].append(err_regime)
        comp_by_mask[mask_name]["err_transit"].append(err_transit)
        comp_by_mask[mask_name]["smoothness"].append(smoothness)
        comp_by_mask[mask_name]["azioni"].append(azioni)

        if mask_name == "uniciclo":
            vy_ref = (acc.sum_vy_ref_abs[ids] / n_steps_env).mean().item()
            vy_real = (acc.sum_vy_real_abs[ids] / n_steps_env).mean().item()
            reverse_vx = (acc.sum_reverse_vx[ids] / n_steps_env).mean().item()
            cost += (
                WCOMP_VY_REF * vy_ref
                + WCOMP_VY_REAL * vy_real
                + WCOMP_REVERSE * reverse_vx
            )
            results[f"Eval/uniciclo_{mode_name}_vy_ref"] = vy_ref
            results[f"Eval/uniciclo_{mode_name}_vy_real"] = vy_real
            results[f"Eval/uniciclo_{mode_name}_reverse_vx"] = reverse_vx
            comp_by_mask["uniciclo"]["vy_ref"].append(vy_ref)
            comp_by_mask["uniciclo"]["vy_real"].append(vy_real)
            comp_by_mask["uniciclo"]["reverse_vx"].append(reverse_vx)

        results[f"Eval/{mask_name}_{mode_name}_err_regime"] = err_regime
        results[f"Eval/{mask_name}_{mode_name}_err_transit"] = err_transit
        results[f"Eval/{mask_name}_{mode_name}_smoothness"] = smoothness
        results[f"Eval/{mask_name}_{mode_name}_azioni"] = azioni
        results[f"Eval/{mask_name}_{mode_name}_cost"] = cost

        cost_by_mask[mask_name].append(cost)

    cost_full = sum(cost_by_mask["full"]) / max(len(cost_by_mask["full"]), 1)
    cost_uniciclo = sum(cost_by_mask["uniciclo"]) / max(len(cost_by_mask["uniciclo"]), 1)
    composite = W_FULL * cost_full + W_UNICICLO * cost_uniciclo

    def _mean(lst: list[float]) -> float:
        return sum(lst) / max(len(lst), 1)

    for mask_name, comps in comp_by_mask.items():
        for comp_name, vals in comps.items():
            results[f"Eval/comp_{mask_name}_{comp_name}"] = _mean(vals)

    results["Eval/cost_full"] = cost_full
    results["Eval/cost_uniciclo"] = cost_uniciclo
    results["Eval/composite_target"] = composite
    return results

def run_sweep_evaluation(base_env, agent, wandb_run=None) -> dict[str, float]:
    device = base_env.device
    n_eval = min(EVAL_NUM_ENVS, base_env.num_envs)

    groups = _assign_eval_groups(n_eval, device)
    _setup_eval_scenario(base_env, groups, device)

    agent.set_running_mode("eval")

    recorder = _RepresentativeRecorder(groups)
    acc = _run_eval_loop(base_env, agent, groups, device, recorder=recorder)

    results = _aggregate(acc, groups)

    print(
        f"\n[sweep_eval] composite_target={results['Eval/composite_target']:.4f} "
        f"(cost_full={results['Eval/cost_full']:.4f}, "
        f"cost_uniciclo={results['Eval/cost_uniciclo']:.4f})"
    )

    if wandb_run is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_paths = recorder.write_csv(base_env.step_dt, tmp_dir)
            if csv_paths:
                artifact = wandb.Artifact(
                    name=f"eval_timeseries_{wandb_run.id}",
                    type="eval_timeseries",
                    description="Serie temporale (tracking/errore/velocita') per 1 env rappresentativo "
                                "delle 6 combinazioni mask x mode, stessa granularita' dei plot di "
                                "evaluate_pos_controller_continuous.py.",
                )
                for path in csv_paths:
                    artifact.add_file(path)
                wandb_run.log_artifact(artifact)
                print(f"[sweep_eval] Artifact CSV caricato su wandb: eval_timeseries_{wandb_run.id}")

    if wandb_run is not None:
        wandb_run.log(results)

    return results
