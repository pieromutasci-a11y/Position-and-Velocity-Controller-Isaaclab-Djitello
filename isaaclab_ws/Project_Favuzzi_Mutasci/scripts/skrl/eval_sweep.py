# eval_sweep.py
#
# EVALUATION DETERMINISTICA per lo SWEEP wandb.
#
# Chiamata UNA VOLTA, nello stesso processo, subito dopo che il training di
# un trial e' finito (stesso env, stesso agent, Isaac Sim non si riavvia).
# Calcola un unico scalare (Eval/composite_target) da loggare su wandb come
# metrica ufficiale dello sweep bayesiano, al posto di Objective/composite_target
# calcolato in env.py durante il training (quello resta come diagnostica
# "in diretta", ma soffre di policy stocastica e scenari diversi ad ogni
# episodio: troppo rumoroso per guidare uno sweep bayesiano).
#
# STRUTTURA DELL'EVAL — 6 combinazioni FISSE (mask x mode):
#   (full,hover) (full,singolo) (full,variabile)
#   (uniciclo,hover) (uniciclo,singolo) (uniciclo,variabile)
#
#   "variabile" in eval e' SEMPRE puramente random: la sequenza canonica
#   (avanti/indietro/destra/sinistra) e' solo per il training, qui non si
#   usa mai (_is_canonical resta sempre False).
#
# GERARCHIA DELLA METRICA (concordata):
#   1) per ogni ENV: media sui suoi waypoint dell'errore a regime e in
#      transitorio (ogni waypoint del singolo env pesa 1 dentro quella
#      media). "regime" = ultimo 30% degli step del tratto, "transitorio"
#      = primo 70%; split ESATTO sulla durata reale di ciascun tratto.
#      Stessa logica per smoothness/azioni/vy/reverse: media sui suoi
#      step per quell'env.
#   2) media TRA GLI ENV del gruppo: ogni ENV pesa 1, indipendentemente da
#      quanti waypoint/step ha completato (un env che ne fa 10 e uno che
#      ne fa 2 contano uguale).
#   3) i pesi WCOMP_* (fissi, non sweeppati) riportano le componenti a
#      contributi confrontabili nel cost, compensando le scale grezze
#      molto diverse tra loro (stesso principio dei rew_scale_* nell'env):
#         cost(mask,mode) = WCOMP_ERR_REGIME*err_regime + WCOMP_ERR_TRANSIT*err_transit
#                         + WCOMP_SMOOTHNESS*smoothness + WCOMP_AZIONI*azioni
#                         [+ WCOMP_VY_REF*vy_ref + WCOMP_VY_REAL*vy_real
#                            + WCOMP_REVERSE*reverse_vx   SOLO per uniciclo]
#   4) cost_full     = media( cost(full,hover), cost(full,singolo), cost(full,variabile) )
#      cost_uniciclo = media( cost(uniciclo,hover), cost(uniciclo,singolo), cost(uniciclo,variabile) )
#      (media semplice sulle 3 modalita': ognuna pesa 1/3)
#   5) composite_target = W_FULL * cost_full + W_UNICICLO * cost_uniciclo
#      (nessuna cerniera/protezione asimmetrica: full e uniciclo alla pari)
#
# Tutti i pesi (ALPHA_YAW, REGIME_FRAC, W_FULL, W_UNICICLO, WCOMP_*) sono
# costanti FISSE qui sotto, NON sweeppate: definiscono COSA si vuole
# ottimizzare (la funzione obiettivo), separati dai rew_scale_* del cfg
# che definiscono COME si allena. Se lo sweep potesse toccarli, potrebbe
# "vincere" azzerando il termine piu' difficile da migliorare, invalidando
# il confronto tra trial.
#
# ===================================================================================
# NUOVO IN QUESTO GIRO:
#   - _RepresentativeRecorder: registra, per il PRIMO env id di ciascuna
#     delle 6 combinazioni (mask, mode), la stessa serie temporale che
#     evaluate_pos_controller_continuous.py plotta come .png (tracking
#     x/y/z/yaw, errore di posizione, velocita' reali vs riferimento
#     EFFETTIVAMENTE applicato). A fine eval, queste serie vengono scritte
#     come CSV (uno per combinazione) e caricate su wandb come Artifact,
#     cosi' da essere disponibili per OGNI run dello sweep, non solo per
#     l'evaluation manuale standalone.
#   - NON tocca in alcun modo la logica esistente di _Accumulators,
#     _aggregate, ne' le metriche Eval/*/Objective/* gia' loggate: e' un
#     binario di registrazione completamente parallelo e opzionale
#     (recorder=None disabilita tutto senza alcun effetto collaterale).
# ===================================================================================

from __future__ import annotations

import csv
import math
import os
import tempfile

import torch
import wandb

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, sample_uniform

# ===================================================================
# COSTANTI DELL'OBIETTIVO — FISSE, NON SWEEPPATE
# ===================================================================
ALPHA_YAW = 0.2          # peso dello yaw dentro err_regime/err_transitorio
REGIME_FRAC = 0.70       # frazione iniziale del tratto = transitorio; il resto = regime
W_FULL = 1.0             # peso di cost_full nel composite_target
W_UNICICLO = 1.0         # peso di cost_uniciclo nel composite_target

# ===================================================================
# PESI PER-COMPONENTE del cost — FISSI, NON SWEEPPATI.
#
# Le componenti grezze hanno SCALE MOLTO DIVERSE: err_regime/err_transit
# sono in metri (~0.1-0.4), smoothness e' Sum((Delta azione)^2) (~0.01-0.05),
# azioni e' Sum(azione^2) (~0.05-0.3), vy_ref/vy_real/reverse sono |grandezza|
# (~0.02-0.1). Sommate a peso 1 sarebbero DOMINATE dall'errore, e smoothness/
# vy/reverse non conterebbero quasi nulla nella scelta dello sweep. Questi
# pesi riportano ogni componente a un contributo confrontabile — STESSO
# principio dei rew_scale_* nell'env.
#
# Taratura di partenza (da regolare guardando i grafici Eval/comp_* su
# wandb: se una componente resta sempre trascurabile nel composite, alza
# il suo peso; se ne domina una sola, abbassalo):
WCOMP_ERR_REGIME = 3.0    # precisione a target: il pezzo piu' importante
WCOMP_ERR_TRANSIT = 1.0   # qualita' del transitorio (gia' di scala ~0.3-0.4)
WCOMP_SMOOTHNESS = 20.0   # oscillazioni: grezza ~0.02, va alzata per contare
WCOMP_AZIONI = 5.0        # sforzo comando: grezza ~0.1
WCOMP_VY_REF = 10.0       # solo uniciclo: deriva vy comandata (~0.05-0.1)
WCOMP_VY_REAL = 10.0      # solo uniciclo: deriva vy reale
WCOMP_REVERSE = 15.0      # solo uniciclo: retromarcia (da scoraggiare forte)

# ===================================================================
# COSTANTI DELLO SCENARIO DI EVAL — FISSE, NON SWEEPPATE
# ===================================================================
EVAL_SEED = 20260101
EVAL_NUM_ENVS = 3000              # deve essere <= num_envs del training env
EVAL_NUM_QUEUES = 3               # code (rigenerazioni waypoint) per env
EVAL_MAX_STEPS_PER_QUEUE = 600    # limite di sicurezza per singola coda

_MODES = ("hover", "singolo", "variabile")
_MASKS = ("full", "uniciclo")
_DOF_MASKS = {
    "full":     (1.0, 1.0, 1.0, 1.0),
    "uniciclo": (1.0, 0.0, 1.0, 1.0),
}


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


# ===================================================================
# ASSEGNAZIONE GRUPPI — deterministica, contigua, NON stocastica
# ===================================================================
def _assign_eval_groups(n_eval: int, device: torch.device) -> dict[tuple[str, str], torch.Tensor]:
    """Partiziona [0, n_eval) in 6 blocchi contigui fissi (mask, mode).
    Nessuna casualita' qui: la stessa combinazione riceve sempre lo stesso
    range di env id, ad ogni trial. Gli env id restano validi come indici
    dentro base_env.num_envs (n_eval <= num_envs)."""
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


# ===================================================================
# SETUP SCENARIO — seed fisso, identico ad ogni trial
# ===================================================================
def _setup_eval_scenario(base_env, groups: dict[tuple[str, str], torch.Tensor], device: torch.device):
    """Fissa stanza, DoF mask, modalita' di riferimento e coda iniziale per
    tutti gli env di eval, con un seed dedicato e riproducibile. Azzera
    anche lo stato residuo lasciato dal training (azioni correnti/precedenti,
    contatore ZOH del low-level), altrimenti il primo step dell'eval
    userebbe dati sporchi ereditati dall'ultimo batch di training."""
    torch.manual_seed(EVAL_SEED)

    all_ids = torch.cat(list(groups.values()))
    base_env._sample_room(all_ids)

    # Import assoluto: questo file vive in scripts/skrl/, NON dentro il
    # pacchetto Project_Favuzzi_Mutasci.tasks.direct.pos_controller, quindi
    # non puo' usare un import relativo (from .pos_controller_env import ...).
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
        else:  # variabile — SEMPRE random puro in eval, mai canonico
            base_env._reference_mode[ids] = MODE_VARIABILE
            base_env._is_hover[ids] = False
        base_env._is_canonical[ids] = False

    # -- reset fisico deterministico (stessa logica di _reset_idx, senza
    # pero' ricampionare room/mode/dof_mask, gia' fissati sopra) --
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

    # -- azzera stato residuo del training --
    base_env._actions[all_ids] = 0.0
    base_env._high_level_actions[all_ids] = 0.0
    base_env._prev_high_level_actions[all_ids] = 0.0
    # Il contatore di decimation del low-level e' globale (non per-env):
    # va azzerato una volta, altrimenti il primo ciclo ZOH in eval parte
    # sfasato rispetto all'inizio del tratto.
    base_env._physics_step_counter = 0

    base_env._build_queue(all_ids, spawn_pos=spawn_pos_env, spawn_yaw=spawn_yaw)


def _regenerate_queue(base_env, ids: torch.Tensor):
    """Rigenera la coda per gli env indicati, SENZA toccare room/dof_mask/
    mode (gia' fissati). 'variabile' resta sempre random puro (_is_canonical
    resta False per costruzione)."""
    if ids.numel() == 0:
        return
    pos_env = base_env.robot.data.root_pos_w[ids] - base_env._env_origins[ids]
    _, _, yaw = euler_xyz_from_quat(base_env.robot.data.root_quat_w[ids])
    base_env._build_queue(ids, spawn_pos=pos_env, spawn_yaw=yaw)


# ===================================================================
# ACCUMULATORI — dimensionati su base_env.num_envs (NON su n_eval),
# perche' tutti i tensori dell'env (dof_mask, high_level_actions, ecc.)
# hanno quella dimensione. Gli env fuori dai 6 gruppi restano a zero e
# non vengono mai letti in fase di aggregazione.
# ===================================================================
class _Accumulators:
    """Tiene sia il buffer per lo split esatto regime/transitorio del
    waypoint CORRENTE, sia le somme correnti (su tutta la durata
    dell'eval) di smoothness/azioni/vy/reverse — una riga per ENV."""

    def __init__(self, n_total: int, max_steps_per_wp: int, device: torch.device):
        self.device = device
        self.n_total = n_total
        self.max_steps = max_steps_per_wp

        # buffer per lo split esatto: errore combinato (pos + alpha*yaw)
        # per ogni step del waypoint CORRENTE. Si azzera/riparte da 0 ad
        # ogni avanzamento di waypoint.
        self.err_buf = torch.zeros(n_total, max_steps_per_wp, device=device)
        self.step_in_wp = torch.zeros(n_total, dtype=torch.long, device=device)

        # accumulatori regime/transitorio: somma delle MEDIE-per-waypoint
        # e conteggio, aggiornati SOLO quando un waypoint si completa.
        self.sum_err_regime = torch.zeros(n_total, device=device)
        self.sum_err_transit = torch.zeros(n_total, device=device)
        self.n_wp_completed = torch.zeros(n_total, device=device)

        # accumulatori per-step, su TUTTA la durata dell'eval (non per
        # waypoint): smoothness, azioni, vy_ref, vy_real, reverse_vx.
        self.sum_smoothness = torch.zeros(n_total, device=device)
        self.sum_azioni = torch.zeros(n_total, device=device)
        self.sum_vy_ref_abs = torch.zeros(n_total, device=device)
        self.sum_vy_real_abs = torch.zeros(n_total, device=device)
        self.sum_reverse_vx = torch.zeros(n_total, device=device)
        self.total_steps = torch.zeros(n_total, device=device)

    def record_step(self, base_env, active_mask: torch.Tensor):
        """Chiamata ad OGNI step della simulazione, per tutti gli env
        ancora attivi (che non hanno finito le loro num_queues code)."""
        dev = self.device
        pos_env = base_env.robot.data.root_pos_w - base_env._env_origins
        w0_pos, w0_yaw = base_env._current_target()
        _, _, yaw = euler_xyz_from_quat(base_env.robot.data.root_quat_w)

        dist = torch.norm(pos_env - w0_pos, dim=1)
        yaw_err = torch.abs(_wrap_to_pi(yaw - w0_yaw))
        err_combined = dist + ALPHA_YAW * yaw_err

        # -- scrivi nel buffer del waypoint corrente (solo env attivi) --
        idx = self.step_in_wp.clamp(max=self.max_steps - 1)
        ar = torch.arange(self.n_total, device=dev)
        self.err_buf[ar[active_mask], idx[active_mask]] = err_combined[active_mask]
        self.step_in_wp[active_mask] += 1

        is_uniciclo = base_env._dof_mask[:, 1] < 0.5

        cmd_now = base_env._high_level_actions
        cmd_prev = base_env._prev_high_level_actions

        # smoothness: Sum((Delta azione)^2) su tutti e 4 i canali.
        d_act = cmd_now - cmd_prev
        smoothness_step = torch.sum(torch.square(d_act), dim=1)

        # azioni: penalita' GENERALE su TUTTE le azioni — tutti e 4 i canali,
        # nessuna esclusione, per entrambe le maschere (misura lo sforzo di
        # comando TOTALE come metrica di eval).
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
        """Chiamata subito dopo che il flag _advanced_now e' stato True per
        gli env in adv_ids: legge il buffer, fa lo split 70/30 ESATTO sulla
        lunghezza effettiva di quel tratto, accumula la MEDIA del tratto
        (non la somma: cosi' ogni waypoint pesa 1 indipendentemente da
        quanti step e' durato), poi resetta il buffer per il prossimo wp."""
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
                # tratto cosi' corto che non esiste una fase di regime
                # distinta: usa la media del transitorio come fallback.
                self.sum_err_regime[env_id] += transit_vals.mean()
            self.n_wp_completed[env_id] += 1

        self.err_buf[adv_ids] = 0.0
        self.step_in_wp[adv_ids] = 0


# ===================================================================
# REGISTRAZIONE SERIE TEMPORALE — SOLO 1 ENV RAPPRESENTATIVO PER GRUPPO
# ===================================================================
# Indipendente da _Accumulators: non tocca nessuna metrica Eval/* gia'
# esistente. Registra, per il primo env id di ciascuno dei 6 gruppi
# (mask,mode), le stesse grandezze dei plot di
# evaluate_pos_controller_continuous.py (tracking, errore, velocita'),
# cosi' da poter esportare un CSV equivalente ai .png per ogni run.
class _RepresentativeRecorder:
    def __init__(self, groups: dict[tuple[str, str], torch.Tensor]):
        # un solo env id per combinazione: il primo del gruppo
        self.rep_ids = {combo: ids[0].item() for combo, ids in groups.items() if ids.numel() > 0}
        self.rows: dict[tuple[str, str], list[dict]] = {combo: [] for combo in self.rep_ids}
        self._step_counter = 0

    def record_pre_step(self, base_env):
        """Chiamata PRIMA di _pre_physics_step: cattura target/pos correnti,
        esattamente come fa evaluate_pos_controller_continuous.py."""
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
        """Chiamata DOPO lo step fisico: velocita' reali + riferimento
        EFFETTIVAMENTE applicato (post-gating DoF mask), stesso allineamento
        temporale usato in evaluate_pos_controller_continuous.py."""
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
        """Scrive un CSV per combinazione, colonne identiche ai tre plot
        (tracking + errore + velocita'), aggiungendo 'time_s' esplicito.
        Ritorna la lista dei path scritti."""
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


# ===================================================================
# LOOP DI SIMULAZIONE
# ===================================================================
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

    # Attivo SOLO per gli env nei 6 gruppi; tutti gli altri (se n_eval <
    # num_envs) restano "done" fin da subito e non vengono mai simulati
    # ne' letti in aggregazione.
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

        # avanzamento all'INTERNO della coda: il flag _advanced_now e'
        # gia' stato calcolato dentro _pre_physics_step -> _update_waypoint
        # di QUESTO giro, quindi riflette lo stato subito dopo lo step.
        adv_now = base_env._advanced_now & active_mask
        adv_ids = adv_now.nonzero(as_tuple=False).squeeze(-1)
        acc.on_waypoint_advanced(adv_ids)

        finish_queue = (assestato | timeout_coda) & active_mask
        finish_ids = finish_queue.nonzero(as_tuple=False).squeeze(-1)
        if finish_ids.numel() > 0:
            # l'ultimo waypoint della coda in chiusura non passa da
            # _advanced_now (non c'e' un waypoint successivo nella STESSA
            # coda): lo contiamo qui esplicitamente prima di rigenerare.
            acc.on_waypoint_advanced(finish_ids)

            queue_count[finish_ids] += 1
            steps_on_queue[finish_ids] = 0
            hold_steps[finish_ids] = 0
            _regenerate_queue(base_env, finish_ids)

            done_now = queue_count[finish_ids] >= EVAL_NUM_QUEUES
            done_env[finish_ids[done_now]] = True

        step += 1

    return acc


# ===================================================================
# AGGREGAZIONE — media-per-env, poi media-tra-env, per OGNI componente
# (ogni env pesa 1, indipendentemente da quanti waypoint/step ha fatto).
# ===================================================================
def _aggregate(acc: _Accumulators, groups: dict[tuple[str, str], torch.Tensor]) -> dict[str, float]:
    results: dict[str, float] = {}
    cost_by_mask: dict[str, list[float]] = {"full": [], "uniciclo": []}

    # Componenti disaggregate GREZZE, raccolte per maschera per poterle poi
    # mediare sulle 3 modalita' (stesso schema di cost_by_mask). Servono
    # per i grafici su wandb: mostrano QUANTO ciascun pezzo contribuisce,
    # in unita' leggibili (non pesate), cosi' da scegliere il compromesso.
    comp_by_mask: dict[str, dict[str, list[float]]] = {
        "full": {"err_regime": [], "err_transit": [], "smoothness": [], "azioni": []},
        "uniciclo": {"err_regime": [], "err_transit": [], "smoothness": [], "azioni": [],
                     "vy_ref": [], "vy_real": [], "reverse_vx": []},
    }

    for (mask_name, mode_name), ids in groups.items():
        if ids.numel() == 0:
            continue

        # -- ERRORE: media-per-env, poi media-tra-env. Per OGNI env:
        # media dei suoi waypoint (sum_err_regime[env] contiene la somma
        # delle medie-per-waypoint di QUEL env, diviso il suo numero di
        # wp). Poi si media su tutti gli env del gruppo -> ogni ENV pesa
        # 1, indipendentemente da quanti waypoint ha completato. --
        n_wp_env = acc.n_wp_completed[ids].clamp(min=1.0)
        err_regime_per_env = acc.sum_err_regime[ids] / n_wp_env
        err_transit_per_env = acc.sum_err_transit[ids] / n_wp_env
        err_regime = err_regime_per_env.mean().item()
        err_transit = err_transit_per_env.mean().item()

        # -- smoothness/azioni: stesso principio (media-per-env, poi
        # media-tra-env), su STEP invece che su waypoint. --
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

        # -- grafici PER-COMBINAZIONE (6 gruppi): componenti GREZZE (non
        # pesate) — comportamento fisico reale. 'cost' e' invece la somma
        # GIA' PESATA (cio' che contribuisce al composite_target). --
        results[f"Eval/{mask_name}_{mode_name}_err_regime"] = err_regime
        results[f"Eval/{mask_name}_{mode_name}_err_transit"] = err_transit
        results[f"Eval/{mask_name}_{mode_name}_smoothness"] = smoothness
        results[f"Eval/{mask_name}_{mode_name}_azioni"] = azioni
        results[f"Eval/{mask_name}_{mode_name}_cost"] = cost

        cost_by_mask[mask_name].append(cost)

    # -- media semplice tra le 3 modalita' (ogni modalita' pesa 1/3) --
    cost_full = sum(cost_by_mask["full"]) / max(len(cost_by_mask["full"]), 1)
    cost_uniciclo = sum(cost_by_mask["uniciclo"]) / max(len(cost_by_mask["uniciclo"]), 1)
    composite = W_FULL * cost_full + W_UNICICLO * cost_uniciclo

    # -- COMPONENTI AGGREGATE PER MASCHERA (media sulle 3 modalita'): i
    # grafici principali per scegliere il compromesso. --
    def _mean(lst: list[float]) -> float:
        return sum(lst) / max(len(lst), 1)

    for mask_name, comps in comp_by_mask.items():
        for comp_name, vals in comps.items():
            results[f"Eval/comp_{mask_name}_{comp_name}"] = _mean(vals)

    results["Eval/cost_full"] = cost_full
    results["Eval/cost_uniciclo"] = cost_uniciclo
    results["Eval/composite_target"] = composite
    return results


# ===================================================================
# ENTRY POINT
# ===================================================================
def run_sweep_evaluation(base_env, agent, wandb_run=None) -> dict[str, float]:
    """Da chiamare UNA VOLTA, nello stesso processo, subito dopo la fine
    del training di un trial (Isaac Sim ancora aperto). Ritorna il dict di
    metriche (gia' loggate su wandb_run se fornito)."""
    device = base_env.device
    n_eval = min(EVAL_NUM_ENVS, base_env.num_envs)

    groups = _assign_eval_groups(n_eval, device)
    _setup_eval_scenario(base_env, groups, device)

    agent.set_running_mode("eval")

    # NUOVO: registratore della serie temporale per 1 env rappresentativo
    # (il primo id) di ciascuna delle 6 combinazioni — completamente
    # indipendente da _Accumulators/_aggregate, nessun effetto sulle
    # metriche Eval/*/Objective/* gia' esistenti.
    recorder = _RepresentativeRecorder(groups)
    acc = _run_eval_loop(base_env, agent, groups, device, recorder=recorder)

    results = _aggregate(acc, groups)

    print(
        f"\n[sweep_eval] composite_target={results['Eval/composite_target']:.4f} "
        f"(cost_full={results['Eval/cost_full']:.4f}, "
        f"cost_uniciclo={results['Eval/cost_uniciclo']:.4f})"
    )

    # ===================================================================
    # NUOVO: CSV serie temporale (1 env per combinazione) + upload artifact
    # Non tocca in alcun modo i risultati/log gia' esistenti sopra o sotto.
    # ===================================================================
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