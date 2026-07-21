# evaluate_pos_controller_continuous.py
#
# EVALUATION del CONTROLLORE DI POSIZIONE (target SEQUENZIALI, senza reset, plot continui).
#
# LOGICA:
#   - Il drone vola in modo continuo: quando raggiunge il target corrente e resta
#     entro `reach_thr` per `settle_window_s` secondi consecutivi, viene generato
#     un NUOVO target (dentro la stanza corrente) e il drone riparte dalla posizione
#     in cui si trova, senza alcun reset.
#   - Le terminazioni per OOB (uscita stanza) e tilt eccessivo sono DISABILITATE
#     (override di `_get_dones`), cosi' l'ambiente non si auto-resetta mai durante
#     l'evaluation: il drone resta sempre visibile e continua a volare.
#   - Il marker solido della stanza (`_room_marker`, il cubo blu pieno) viene
#     nascosto; al suo posto viene disegnata una stanza WIREFRAME (senza tetto)
#     con `debug_draw`, insieme al cubo target e alle frecce di yaw (target/reale).
#   - Vengono stampati warning per: tilt eccessivo, urto/OOB muro-pavimento-soffitto.
#   - PLOT: per ciascun ambiente viene generato UN SOLO grafico continuo su tutta la
#     sequenza di target (non uno per ogni target), con x/y/z/yaw reali vs target
#     (a gradino) + un subplot con l'errore di posizione (norma) continuo nel tempo,
#     con linee verticali tratteggiate nei momenti di cambio target.

import argparse
import sys
import os
import math
import types
import torch
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# =======================================================================
# ARGPARSE
# =======================================================================
parser = argparse.ArgumentParser(description="Evaluation PosController: target sequenziali senza reset, plot continui")
parser.add_argument("--num_envs", type=int, default=1, help="Numero di ambienti")
parser.add_argument("--episodes", type=int, default=5,
                    help="Numero di TARGET SEQUENZIALI da raggiungere per ciascun ambiente")
parser.add_argument("--reach_thr", type=float, default=0.2, help="Soglia distanza per considerare il target raggiunto [m]")
parser.add_argument("--settle_window_s", type=float, default=2.0,
                    help="Finestra continua [s] entro reach_thr prima di generare il target successivo")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Project-Favuzzi-Mutasci-PosController-Direct-v0",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path al checkpoint .pt del pos_controller")
parser.add_argument("--tilt_warn_deg", type=float, default=45.0, help="Soglia [deg] di tilt oltre la quale stampare un warning")
parser.add_argument(
    "--out_dir",
    type=str,
    default="/workspace/project_workspace/Project_Favuzzi_Mutasci/scripts/skrl/result_pos_controller",
)
parser.add_argument("--plot_envs", type=int, nargs="+", default=None,
                    help="indici di ambiente da plottare. Default: tutti gli ambienti (0..num_envs-1)")
# margine di sicurezza sul numero massimo di step totali (evita loop infiniti se la policy non converge mai)
parser.add_argument("--max_steps_per_target", type=int, default=1000,
                     help="Limite di sicurezza di step per singolo target, oltre il quale si passa comunque al target successivo")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =======================================================================
# IMPORT POST-AVVIO ISAAC SIM
# =======================================================================
import gymnasium as gym
from isaacsim.util.debug_draw import _debug_draw
from isaaclab.utils.math import euler_xyz_from_quat

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import Project_Favuzzi_Mutasci.tasks  # noqa: F401


# =======================================================================
# UTILITY: CUBO (12 spigoli) — usato per il target
# =======================================================================
def costruisci_cubo(centro, lato=0.25):
    h = lato / 2.0
    v = []
    for dx in (-h, +h):
        for dy in (-h, +h):
            for dz in (-h, +h):
                v.append((centro[0] + dx, centro[1] + dy, centro[2] + dz))
    def idx(bx, by, bz): return 4 * bx + 2 * by + bz
    spigoli = []
    for by in (0, 1):
        for bz in (0, 1):
            spigoli.append((v[idx(0, by, bz)], v[idx(1, by, bz)]))
    for bx in (0, 1):
        for bz in (0, 1):
            spigoli.append((v[idx(bx, 0, bz)], v[idx(bx, 1, bz)]))
    for bx in (0, 1):
        for by in (0, 1):
            spigoli.append((v[idx(bx, by, 0)], v[idx(bx, by, 1)]))
    return spigoli


# =======================================================================
# UTILITY: FRECCIA — usata per lo yaw target/reale
# =======================================================================
def costruisci_freccia(origine, vettore, scala=1.0, lunghezza_testa=0.08, larghezza_testa=0.05):
    punta = [origine[0] + vettore[0] * scala, origine[1] + vettore[1] * scala, origine[2] + vettore[2] * scala]
    norma = math.sqrt(sum((punta[i] - origine[i]) ** 2 for i in range(3)))
    segmenti = [(tuple(origine), tuple(punta))]
    if norma < 1e-5:
        return segmenti
    direzione = [(punta[i] - origine[i]) / norma for i in range(3)]
    appoggio = (0.0, 0.0, 1.0) if abs(direzione[2]) < 0.9 else (1.0, 0.0, 0.0)
    perp = [
        direzione[1] * appoggio[2] - direzione[2] * appoggio[1],
        direzione[2] * appoggio[0] - direzione[0] * appoggio[2],
        direzione[0] * appoggio[1] - direzione[1] * appoggio[0],
    ]
    norma_perp = math.sqrt(sum(p ** 2 for p in perp)) + 1e-8
    perp = [p / norma_perp for p in perp]
    base_testa = [punta[i] - direzione[i] * lunghezza_testa for i in range(3)]
    ala1 = [base_testa[i] + perp[i] * larghezza_testa for i in range(3)]
    ala2 = [base_testa[i] - perp[i] * larghezza_testa for i in range(3)]
    segmenti.append((tuple(punta), tuple(ala1)))
    segmenti.append((tuple(punta), tuple(ala2)))
    return segmenti


# =======================================================================
# UTILITY: STANZA WIREFRAME SENZA TETTO (pavimento + 4 montanti)
# =======================================================================
def costruisci_stanza(ox, oy, oz, lx, ly, lz):
    hx, hy = lx / 2.0, ly / 2.0
    verts = []
    for dx in (-hx, +hx):
        for dy in (-hy, +hy):
            for dz in (0.0, lz):
                verts.append((ox + dx, oy + dy, oz + dz))
    def idx(bx, by, bz): return 4 * bx + 2 * by + bz
    spigoli = []
    for by in (0, 1):
        spigoli.append((verts[idx(0, by, 0)], verts[idx(1, by, 0)]))
    for bx in (0, 1):
        spigoli.append((verts[idx(bx, 0, 0)], verts[idx(bx, 1, 0)]))
    for bx in (0, 1):
        for by in (0, 1):
            spigoli.append((verts[idx(bx, by, 0)], verts[idx(bx, by, 1)]))
    return spigoli


# =======================================================================
# GENERATORE TARGET CASUALE DENTRO LA STANZA
# =======================================================================
def genera_target(room_size_xy, room_size_z, env_cfg, device):
    """Genera un target casuale DENTRO la stanza corrente (con margine dai bordi)."""
    n = room_size_xy.shape[0]
    margin = env_cfg.target_room_margin
    bound_xy = room_size_xy * margin
    bound_z = room_size_z * margin

    target_pos = torch.zeros(n, 3, device=device)
    target_pos[:, 0] = (torch.rand(n, device=device) * 2 - 1) * bound_xy
    target_pos[:, 1] = (torch.rand(n, device=device) * 2 - 1) * bound_xy
    z_min = torch.full((n,), env_cfg.min_z_pos, device=device)
    z_max = torch.clamp(bound_z, min=env_cfg.min_z_pos + 0.1)
    target_pos[:, 2] = z_min + torch.rand(n, device=device) * (z_max - z_min)

    target_yaw = (torch.rand(n, device=device) * 2 - 1) * math.pi
    return target_pos, target_yaw.unsqueeze(-1)


def _get_dones_eval(self):
    """Override: nessuna terminazione per OOB/tilt/crash. L'episodio non si resetta
    mai durante l'evaluation, cosi' il drone continua a volare senza interruzioni."""
    time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    died = torch.zeros_like(time_out)
    return died, time_out


# =======================================================================
# PLOT CONTINUO (un solo grafico per ambiente, su tutta la sequenza di target)
# =======================================================================
def plot_continuo(env_id, tempo_s, target_pos, current_pos, target_yaw, current_yaw,
                   target_changes, reach_thr, room_xy, room_z, target_count, out_dir):
    t = tempo_s.numpy()
    x, y, z = current_pos[:, 0].numpy(), current_pos[:, 1].numpy(), current_pos[:, 2].numpy()
    tgt_x, tgt_y, tgt_z = target_pos[:, 0].numpy(), target_pos[:, 1].numpy(), target_pos[:, 2].numpy()
    yaw = current_yaw.numpy()
    tgt_yaw = target_yaw.numpy()

    err = np.sqrt((x - tgt_x) ** 2 + (y - tgt_y) ** 2 + (z - tgt_z) ** 2)

    fig, axs = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    fig.suptitle(
        f"Ambiente {env_id} | sequenza continua di {target_count} target | "
        f"Stanza: L={2.0*room_xy:.2f}m  H={room_z:.2f}m",
        fontsize=13
    )

    line_real = {'color': 'royalblue', 'linestyle': '-', 'linewidth': 1.3, 'label': 'reale'}
    line_target = {'color': 'red', 'linestyle': '--', 'linewidth': 1.3, 'label': 'target'}

    def add_change_lines(ax):
        for step in target_changes:
            ax.axvline(step * (t[1] - t[0]) if len(t) > 1 else 0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

    axs[0].plot(t, tgt_x, **line_target); axs[0].plot(t, x, **line_real)
    axs[0].set_ylabel('x (m)'); axs[0].legend(loc='upper right'); axs[0].grid(True)
    add_change_lines(axs[0])

    axs[1].plot(t, tgt_y, **line_target); axs[1].plot(t, y, **line_real)
    axs[1].set_ylabel('y (m)'); axs[1].legend(loc='upper right'); axs[1].grid(True)
    add_change_lines(axs[1])

    axs[2].plot(t, tgt_z, **line_target); axs[2].plot(t, z, **line_real)
    axs[2].set_ylabel('z (m)'); axs[2].legend(loc='upper right'); axs[2].grid(True)
    add_change_lines(axs[2])

    axs[3].plot(t, tgt_yaw, **line_target); axs[3].plot(t, yaw, **line_real)
    axs[3].set_ylabel('yaw (rad)'); axs[3].legend(loc='upper right'); axs[3].grid(True)
    add_change_lines(axs[3])

    axs[4].plot(t, err, color='darkorange', linewidth=1.4, label='errore |pos - target|')
    axs[4].axhline(reach_thr, color='green', linestyle='--', linewidth=1.0, label=f'reach_thr={reach_thr:.2f}m')
    axs[4].set_ylabel('errore (m)'); axs[4].set_xlabel('Tempo (s)')
    axs[4].legend(loc='upper right'); axs[4].grid(True)
    add_change_lines(axs[4])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    percorso = os.path.join(out_dir, f"eval_plot_continuous_env_{env_id}.png")
    plt.savefig(percorso, dpi=300)
    plt.close(fig)
    print(f"[eval] Grafico continuo salvato in: {percorso}")


# =======================================================================
# MAIN
# =======================================================================
@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    # rendiamo l'episodio molto lungo: il loop e' comunque controllato manualmente
    # da noi (num target * max_steps_per_target), niente timeout intermedi
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        args_cli.episodes * args_cli.max_steps_per_target * control_dt + 5.0
    )

    print(f"[INFO] Creazione task: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    base_env = env.unwrapped
    device = env.device

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Caricamento checkpoint: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    # Disabilita il debug_vis dell'env: il room_marker (cubo blu solido) copre tutto.
    # Usiamo le nostre linee debug_draw per visualizzare la stanza in wireframe.
    env_cfg.debug_vis = False
    base_env.set_debug_vis(False)
    try:
        base_env._room_marker.set_visibility(False)
    except Exception:
        print("[WARN] Non sono riuscito a nascondere _room_marker: se il cubo blu resta "
              "visibile, verifica il nome corretto dell'attributo nell'env.")

    obs, _ = env.reset()

    # Nessuna terminazione per OOB/tilt/crash: l'ambiente non si resetta mai da solo.
    base_env._get_dones = types.MethodType(_get_dones_eval, base_env)

    room_size_xy = base_env._room_size_xy.clone()
    room_size_z = base_env._room_size_z.clone()
    print(f"[INFO] room_size_xy (semi): {room_size_xy.cpu().tolist()}")
    print(f"[INFO] room_size_z  (h):   {room_size_z.cpu().tolist()}")

    # -- Primo target casuale -- #
    target_pos, target_yaw = genera_target(room_size_xy, room_size_z, env_cfg, device)
    base_env._target_pos[:] = target_pos
    base_env._target_yaw[:] = target_yaw

    N = args_cli.num_envs
    target_count = torch.zeros(N, dtype=torch.long)
    settle_counter = torch.zeros(N, dtype=torch.long)
    steps_on_target = torch.zeros(N, dtype=torch.long)  # limite di sicurezza per target
    done_env = torch.zeros(N, dtype=torch.bool)
    was_tilting = torch.zeros(N, dtype=torch.bool)
    was_out_of_room = torch.zeros(N, dtype=torch.bool)

    settle_steps = max(1, int(round(args_cli.settle_window_s / control_dt)))

    storia_target_pos = [[] for _ in range(N)]
    storia_current_pos = [[] for _ in range(N)]
    storia_target_yaw = [[] for _ in range(N)]
    storia_current_yaw = [[] for _ in range(N)]
    storia_tempo = [[] for _ in range(N)]
    target_changes = [[] for _ in range(N)]

    draw = _debug_draw.acquire_debug_draw_interface()

    print(f"[INFO] Avvio simulazione: {args_cli.episodes} target sequenziali per ambiente "
          f"(soglia raggiungimento: {args_cli.reach_thr}m, settle: {args_cli.settle_window_s}s)...")

    global_step = 0
    max_total_steps = args_cli.episodes * args_cli.max_steps_per_target

    with torch.inference_mode():
        while (~done_env).any() and global_step < max_total_steps:

            drone_pos_w = base_env.robot.data.root_pos_w
            env_origins = base_env._env_origins
            drone_pos_env = drone_pos_w - env_origins
            dist = torch.norm(drone_pos_env - target_pos, dim=1)

            drone_quat_w = base_env.robot.data.root_quat_w
            _, _, drone_yaw = euler_xyz_from_quat(drone_quat_w)
            proj_grav_b = base_env.robot.data.projected_gravity_b

            # -------------------- warning tilt / muro (edge-triggered) --------------------
            for env_id in range(N):
                if done_env[env_id]:
                    continue
                pos = drone_pos_env[env_id]

                hit_wall = torch.any(torch.abs(pos[:2]) > room_size_xy[env_id]).item()
                hit_floor = pos[2].item() < 0.0
                hit_ceiling = pos[2].item() > room_size_z[env_id].item()
                out_now = hit_wall or hit_floor or hit_ceiling
                if out_now and not was_out_of_room[env_id]:
                    if hit_wall:
                        print(f"[WARN][FUORI STANZA] Env {env_id} | step {global_step} | MURO: "
                              f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m")
                    if hit_floor:
                        print(f"[WARN][FUORI STANZA] Env {env_id} | step {global_step} | PAVIMENTO: z={pos[2]:.2f}m")
                    if hit_ceiling:
                        print(f"[WARN][FUORI STANZA] Env {env_id} | step {global_step} | SOFFITTO: "
                              f"z={pos[2]:.2f}m (max={room_size_z[env_id]:.2f}m)")
                was_out_of_room[env_id] = out_now

                cos_tilt = (-proj_grav_b[env_id, 2]).clamp(-1.0, 1.0).item()
                tilt_deg = math.degrees(math.acos(cos_tilt))
                tilting_now = tilt_deg > args_cli.tilt_warn_deg
                if tilting_now and not was_tilting[env_id]:
                    print(f"[WARN][TILT] Env {env_id} | step {global_step} | tilt={tilt_deg:.1f} deg")
                was_tilting[env_id] = tilting_now

            # -------------------- assestamento sul target corrente --------------------
            for env_id in range(N):
                if done_env[env_id]:
                    continue
                steps_on_target[env_id] += 1
                if dist[env_id].item() < args_cli.reach_thr:
                    settle_counter[env_id] += 1
                else:
                    settle_counter[env_id] = 0

                reached = settle_counter[env_id] >= settle_steps
                timed_out_on_target = steps_on_target[env_id] >= args_cli.max_steps_per_target
                if reached or timed_out_on_target:
                    if timed_out_on_target and not reached:
                        print(f"[WARN] Env {env_id}: limite di sicurezza ({args_cli.max_steps_per_target} step) "
                              f"raggiunto sul target #{target_count[env_id].item()} senza assestamento, passo al successivo.")
                    new_pos, new_yaw = genera_target(
                        room_size_xy[env_id:env_id+1], room_size_z[env_id:env_id+1], env_cfg, device
                    )
                    target_pos[env_id] = new_pos[0]
                    target_yaw[env_id] = new_yaw[0]
                    target_count[env_id] += 1
                    settle_counter[env_id] = 0
                    steps_on_target[env_id] = 0
                    target_changes[env_id].append(len(storia_tempo[env_id]))
                    print(f"[INFO] Env {env_id}: target #{target_count[env_id].item() - 1} raggiunto "
                          f"(t={global_step*control_dt:.2f}s) -> nuovo target #{target_count[env_id].item()}: "
                          f"({target_pos[env_id,0]:.2f}, {target_pos[env_id,1]:.2f}, {target_pos[env_id,2]:.2f})m")

                    if target_count[env_id] >= args_cli.episodes:
                        done_env[env_id] = True

            base_env._target_pos[:] = target_pos
            base_env._target_yaw[:] = target_yaw

            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])

            # --- DEBUG VISIVO --- #
            draw.clear_lines()
            for env_id in range(N):
                ox, oy, oz = env_origins[env_id].cpu().tolist()

                lx = (2.0 * room_size_xy[env_id]).item()
                ly = (2.0 * room_size_xy[env_id]).item()
                lz = room_size_z[env_id].item()
                for (p1, p2) in costruisci_stanza(ox, oy, oz, lx, ly, lz):
                    draw.draw_lines([p1], [p2], [(1.0, 1.0, 1.0, 0.5)], [2.0])

                target_pos_w = (target_pos[env_id] + env_origins[env_id]).cpu().tolist()
                for (p1, p2) in costruisci_cubo(target_pos_w, lato=0.25):
                    draw.draw_lines([p1], [p2], [(1.0, 0.0, 0.0, 1.0)], [3.0])

                tgt_yaw_val = target_yaw[env_id, 0].item()
                dir_target = [math.cos(tgt_yaw_val), math.sin(tgt_yaw_val), 0.0]
                for (p1, p2) in costruisci_freccia(target_pos_w, dir_target, scala=0.4):
                    draw.draw_lines([p1], [p2], [(1.0, 1.0, 0.0, 1.0)], [3.5])

                cur_yaw_val = drone_yaw[env_id].item()
                drone_origin = drone_pos_w[env_id].cpu().tolist()
                drone_origin[2] += 0.05
                dir_drone = [math.cos(cur_yaw_val), math.sin(cur_yaw_val), 0.0]
                for (p1, p2) in costruisci_freccia(drone_origin, dir_drone, scala=0.4):
                    draw.draw_lines([p1], [p2], [(0.0, 0.4, 1.0, 1.0)], [3.5])

            # -------------------- log dati per plot --------------------
            for env_id in range(N):
                if done_env[env_id]:
                    continue
                storia_target_pos[env_id].append(target_pos[env_id].clone().cpu())
                storia_current_pos[env_id].append(drone_pos_env[env_id].clone().cpu())
                storia_target_yaw[env_id].append(target_yaw[env_id, 0].clone().cpu())
                storia_current_yaw[env_id].append(drone_yaw[env_id].clone().cpu())
                storia_tempo[env_id].append(global_step * control_dt)

            obs, _, _, _, _ = env.step(actions)
            global_step += 1

    draw.clear_lines()
    env.close()

    print(f"\n[INFO] Target raggiunti per env: {target_count.tolist()}")

    # =======================================================================
    # PLOT CONTINUO (un solo file per ambiente)
    # =======================================================================
    os.makedirs(args_cli.out_dir, exist_ok=True)
    envs_to_plot = args_cli.plot_envs if args_cli.plot_envs is not None else list(range(N))

    for env_id in envs_to_plot:
        if len(storia_tempo[env_id]) == 0:
            continue
        tempo_s = torch.tensor(storia_tempo[env_id])
        tgt_pos = torch.stack(storia_target_pos[env_id])
        cur_pos = torch.stack(storia_current_pos[env_id])
        tgt_yaw_t = torch.stack(storia_target_yaw[env_id])
        cur_yaw_t = torch.stack(storia_current_yaw[env_id])

        plot_continuo(
            env_id, tempo_s, tgt_pos, cur_pos, tgt_yaw_t, cur_yaw_t,
            target_changes[env_id], args_cli.reach_thr,
            float(room_size_xy[env_id]), float(room_size_z[env_id]),
            int(target_count[env_id].item()), args_cli.out_dir,
        )

    # =======================================================================
    # REPORT FINALE
    # =======================================================================
    err_pos_all, err_yaw_all = [], []
    for env_id in range(N):
        if len(storia_tempo[env_id]) == 0:
            continue
        cur_pos = torch.stack(storia_current_pos[env_id])
        tgt_pos = torch.stack(storia_target_pos[env_id])
        cur_yaw_t = torch.stack(storia_current_yaw[env_id])
        tgt_yaw_t = torch.stack(storia_target_yaw[env_id])
        err_pos_all.append(torch.norm(cur_pos - tgt_pos, dim=-1))
        yaw_diff = cur_yaw_t - tgt_yaw_t
        err_yaw_all.append(torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs())

    if err_pos_all:
        err_pos = torch.mean(torch.cat(err_pos_all)).item()
        err_yaw = torch.mean(torch.cat(err_yaw_all)).item()
        print("\n" + "=" * 50)
        print(f"  RISULTATI")
        print(f"  Target raggiunti totali: {int(target_count.sum().item())}")
        print(f"  Errore medio posizione: {err_pos:.4f} m")
        print(f"  Errore medio yaw:       {err_yaw:.4f} rad ({math.degrees(err_yaw):.2f} deg)")
        print("=" * 50)


if __name__ == "__main__":
    main()
    simulation_app.close()