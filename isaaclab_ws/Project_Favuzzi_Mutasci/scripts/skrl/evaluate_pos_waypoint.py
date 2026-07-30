# evaluate_pos_controller_continuous.py
#
# EVALUATION del CONTROLLORE DI POSIZIONE — MyDronePosEnv
# (coda di waypoint + stanza asimmetrica a 6 muri + clearance + DoF mask).
#
# Adattato alla versione CORRENTE di pos_controller_env.py / pos_controller_env_cfg.py:
#
#   - DOF_MASKS aggiornate alle 3 maschere 3D del training (cfg.dof_mask_set):
#         full             (1,1,1,1)   onnidirezionale
#         uniciclo         (1,0,1,1)   vx + vz + wz, no strafe
#         planare_olonomo  (1,1,1,0)   vx + vy + vz, no rotazione
#
#   - _build_queue richiede SEMPRE spawn_pos/spawn_yaw non-None. Per planare_olonomo
#     (wz=0) li usa per congelare il target YAW allo yaw corrente del drone al momento
#     della rigenerazione della coda. Per full e uniciclo lo yaw del target e' libero
#     (ma spawn_pos/spawn_yaw devono comunque essere passati per hover e per
#     compatibilita' con la firma).
#
#   - Il riferimento comandato per il plot delle velocita' viene ricostruito da
#     `base_env._high_level_actions` (l'azione DOPO il gating hard della DoF mask
#     in _pre_physics_step), NON dall'uscita grezza della policy: sui canali
#     disattivati il drone riceve sempre e solo zero, quindi il grafico deve
#     mostrare zero, non il rumore mai eseguito della policy.
#
#   - AUTO-DETECT ARCHITETTURA: la dimensione dei layer nascosti (policy_units_l1/
#     l2/l3 usati in training_manual.py per gli sweep) viene ricavata AUTOMATICAMENTE
#     ispezionando le shape dei pesi salvati nel checkpoint, PRIMA di costruire il
#     Runner. Questo evita di dover passare/ricordare a mano gli iperparametri di
#     rete usati dalla specifica run dello sweep che si vuole testare: basta puntare
#     al checkpoint giusto con --checkpoint e lo script si adatta da solo.
#
#   - MODALITA' SPAWN (--spawn_mode):
#         'free'             (default) comportamento invariato: spawn dal reset
#                            standard dell'ambiente (curriculum spawn_low_prob),
#                            nessun vincolo sull'ultimo target.
#         'takeoff_landing'  il drone viene forzato a spawnare vicino al
#                            pavimento (decollo) SUBITO DOPO il reset iniziale,
#                            e l'ultimo waypoint dell'ULTIMA coda pursuita
#                            (quella che porta a done_env=True) viene forzato
#                            ad un'altezza bassa (atterraggio), mantenendo x,y
#                            gia' campionati cosi' che il drone debba solo
#                            scendere verticalmente per "atterrare". Utile per
#                            valutare visivamente il comportamento di
#                            decollo/atterraggio della policy in un contesto
#                            altrimenti identico alla modalita' 'free'.
#
# Selezione da terminale:
#   --mode  hover | singolo | variabile
#   --mask  full | uniciclo | planare_olonomo
#   --spawn_mode  free | takeoff_landing
#
# La maschera scelta viene applicata a TUTTI gli ambienti e resta FISSA per
# l'intera valutazione (non ricampionata mai, a differenza del training dove
# _sample_dof_mask gira ad ogni reset).
#
# Durata controllata da --num_queues (quante code di n_waypoints target
# completare per ambiente) e --max_steps_per_queue (limite di sicurezza per
# singola coda).

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
# MASCHERE DoF disponibili (nome -> [vx, vy, vz, wz])
# DEVONO combaciare ESATTAMENTE con cfg.dof_mask_set del training.
# =======================================================================
DOF_MASKS = {
    "full":            (1.0, 1.0, 1.0, 1.0),
    "uniciclo":        (1.0, 0.0, 1.0, 1.0),  # vx + vz + wz, no vy
    "planare_olonomo": (1.0, 1.0, 1.0, 0.0),  # vx + vy + vz, no wz (yaw congelato a spawn)
}

# =======================================================================
# ARGPARSE
# =======================================================================
parser = argparse.ArgumentParser(
    description="Evaluation PosController (coda waypoint, stanza asimmetrica, DoF mask), continuo senza reset."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--num_queues", type=int, default=5,
    help="Numero di CODE (rigenerazioni di n_waypoints target) da eseguire per ambiente. "
         "E' il principale controllo sulla DURATA totale della valutazione: piu' code = "
         "piu' tempo simulato (l'episodio non finisce mai da solo, dato che OOB/tilt sono "
         "disabilitati e non c'e' timeout episodico in eval).",
)
parser.add_argument(
    "--task",
    type=str,
    default="Template-Project-Favuzzi-Mutasci-PosController-Direct-v0",
)
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path al checkpoint .pt del pos_controller (alto livello)",
)
parser.add_argument("--agent", type=str, default="skrl_cfg_entry_point")
parser.add_argument(
    "--tilt_warn_deg", type=float, default=None,
    help="Soglia [deg] di tilt per warning. Default: usa cfg.max_tilt_deg dell'env",
)
parser.add_argument(
    "--out_dir", type=str,
    default="/workspace/project_workspace/Project_Favuzzi_Mutasci/scripts/skrl/result_pos_controller",
)
parser.add_argument(
    "--plot_envs", type=int, nargs="+", default=None,
    help="indici di ambiente da plottare. Default: tutti gli ambienti (0..num_envs-1)",
)
parser.add_argument(
    "--max_steps_per_queue", type=int, default=2000,
    help="Limite di sicurezza di step per singola coda: se il drone non converge sull'ultimo "
         "waypoint entro questo numero di step high-level, la coda viene rigenerata comunque "
         "(evita che una policy bloccata/non convergente faccia durare la simulazione "
         "all'infinito). Con step_dt tipico di 0.04s (25Hz), 2000 step = 80s per coda.",
)
parser.add_argument(
    "--mode", type=str, default="variabile", choices=["hover", "singolo", "variabile"],
    help="Modalita' di riferimento per l'evaluation: "
         "'hover' (resta fermo dove si trova, coda ripetuta sulla posizione attuale), "
         "'singolo' (un solo target per coda, ripetuto su tutti i waypoint), "
         "'variabile' (waypoint tutti distinti in sequenza)",
)
parser.add_argument(
    "--mask", type=str, default="full", choices=list(DOF_MASKS.keys()),
    help="Maschera DoF [vx,vy,vz,wz] FISSA per tutta l'evaluation (tutti gli env): "
         f"{', '.join(f'{k}={v}' for k, v in DOF_MASKS.items())}",
)
parser.add_argument(
    "--spawn_mode", type=str, default="free", choices=["free", "takeoff_landing"],
    help="'free': spawn/coda invariati rispetto al comportamento standard. "
         "'takeoff_landing': il drone viene forzato a spawnare vicino al "
         "pavimento subito dopo il reset iniziale (decollo), e l'ultimo "
         "waypoint dell'ultima coda pursuita viene forzato ad altezza bassa "
         "(atterraggio), mantenendo invariate le coordinate x,y gia' "
         "campionate per quel waypoint.",
)
parser.add_argument(
    "--landing_z", type=float, default=None,
    help="Altezza [m] del target di atterraggio in modalita' takeoff_landing. "
         "Default: usa cfg.min_z_pos dell'ambiente (il pavimento della stanza).",
)
parser.add_argument(
    "--eval_hold_time_s", type=float, default=None,
    help="Sovrascrive target_hold_time_s SOLO per l'evaluation.",
)
parser.add_argument(
    "--eval_reach_threshold", type=float, default=None,
    help="Sovrascrive target_reach_threshold SOLO per l'evaluation (soglia di distanza [m]).",
)
parser.add_argument(
    "--eval_reach_yaw_threshold", type=float, default=None,
    help="Sovrascrive target_reach_yaw_threshold SOLO per l'evaluation (soglia di yaw [rad]).",
)
parser.add_argument(
    "--no_auto_arch", action="store_true",
    help="Disabilita l'auto-detect dell'architettura dal checkpoint e usa quella "
         "di default definita in agent_cfg (utile per debug).",
)

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
# UTILITY: STANZA WIREFRAME ASIMMETRICA
# =======================================================================
def costruisci_stanza_asimmetrica(ox, oy, oz, room_min, room_max):
    x0, y0, z0 = room_min
    x1, y1, z1 = room_max
    verts = []
    for dx in (x0, x1):
        for dy in (y0, y1):
            for dz in (z0, z1):
                verts.append((ox + dx, oy + dy, oz + dz))

    def idx(bx, by, bz):
        return 4 * bx + 2 * by + bz

    spigoli = []
    for by in (0, 1):
        spigoli.append((verts[idx(0, by, 0)], verts[idx(1, by, 0)]))
    for bx in (0, 1):
        spigoli.append((verts[idx(bx, 0, 0)], verts[idx(bx, 1, 0)]))
    for bx in (0, 1):
        for by in (0, 1):
            spigoli.append((verts[idx(bx, by, 0)], verts[idx(bx, by, 1)]))
    return spigoli


def costruisci_freccia(origine, vettore, scala=1.0, lunghezza_testa=0.08, larghezza_testa=0.05):
    punta = [
        origine[0] + vettore[0] * scala,
        origine[1] + vettore[1] * scala,
        origine[2] + vettore[2] * scala,
    ]
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


def _get_dones_eval(self):
    """Nessuna terminazione per OOB/tilt/crash: il drone non si resetta mai.
    NOTA: e' proprio questo che rende necessario un controllo ESPLICITO della durata
    (vedi --num_queues / --max_steps_per_queue), perche' non c'e' alcun timeout
    episodico naturale che fermi la simulazione da solo."""
    time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    died = torch.zeros_like(time_out)
    return died, time_out


def genera_nuova_coda(base_env, env_ids, mode, device):
    """Rigenera la coda per gli ambienti indicati, SENZA toccare la stanza ne' la
    DoF mask (nessun reset). Il drone continua dalla posizione in cui si trova.

    _build_queue richiede SEMPRE spawn_pos/spawn_yaw non-None:
      - per planare_olonomo (wz=0): lo yaw corrente del drone viene usato per
        congelare il target YAW allo yaw corrente al momento della
        rigenerazione della coda (il drone non puo' ruotare, quindi il target
        yaw deve essere quello attuale, non uno casuale irraggiungibile).
      - per full e uniciclo: spawn_pos/spawn_yaw servono solo per hover (coda
        ripetuta sulla posa corrente); lo yaw del target e' libero.
    In tutti i casi si calcola la posa corrente del drone e la si passa."""
    from Project_Favuzzi_Mutasci.tasks.direct.pos_controller.pos_controller_env import (
        MODE_VARIABILE,
        MODE_SINGOLO,
    )

    pos_env = base_env.robot.data.root_pos_w[env_ids] - base_env._env_origins[env_ids]
    _, _, yaw = euler_xyz_from_quat(base_env.robot.data.root_quat_w[env_ids])

    if mode == "hover":
        base_env._reference_mode[env_ids] = MODE_SINGOLO
        base_env._is_hover[env_ids] = True
    elif mode == "singolo":
        base_env._reference_mode[env_ids] = MODE_SINGOLO
        base_env._is_hover[env_ids] = False
    else:  # variabile
        base_env._reference_mode[env_ids] = MODE_VARIABILE
        base_env._is_hover[env_ids] = False

    base_env._build_queue(env_ids, spawn_pos=pos_env, spawn_yaw=yaw)


def applica_dof_mask_fissa(base_env, mask_name, env_ids, device):
    """Forza la DoF mask scelta da terminale su TUTTI gli ambienti indicati.
    A differenza del training (_sample_dof_mask, campionata per episodio), qui la
    maschera e' FISSA e identica per l'intera valutazione, per poter isolare il
    comportamento della policy su un singolo sottoinsieme di gradi di liberta'."""
    vals = DOF_MASKS[mask_name]
    mask_tensor = torch.tensor(vals, device=device, dtype=torch.float)
    base_env._dof_mask[env_ids] = mask_tensor


def forza_spawn_da_terra(base_env, env_ids, device):
    """Sovrascrive la posa root del drone per gli ambienti indicati portando
    l'altezza vicino al pavimento (decollo), mantenendo invariate x,y e
    l'orientazione correnti, e azzerando la velocita' lineare/angolare.

    Usato SOLO in modalita' --spawn_mode takeoff_landing, subito dopo il
    reset iniziale dell'ambiente (che altrimenti spawnerebbe con il
    curriculum standard spawn_low_prob/spawn_room_margin usato in training).
    """
    n = env_ids.numel()
    pos_env_current = base_env.robot.data.root_pos_w[env_ids] - base_env._env_origins[env_ids]
    quat_current = base_env.robot.data.root_quat_w[env_ids].clone()

    z_low = torch.empty(n, device=device).uniform_(
        base_env.cfg.spawn_low_z_range[0], base_env.cfg.spawn_low_z_range[1]
    )
    pos_env_current = pos_env_current.clone()
    pos_env_current[:, 2] = z_low

    pose = torch.cat(
        [pos_env_current + base_env._env_origins[env_ids], quat_current], dim=-1
    )
    vel_zero = torch.zeros(n, 6, device=device)

    base_env.robot.write_root_pose_to_sim(pose, env_ids)
    base_env.robot.write_root_velocity_to_sim(vel_zero, env_ids)


def forza_atterraggio_ultima_coda(base_env, env_ids, landing_z, device):
    """Sovrascrive SOLO l'ultimo waypoint (indice n_waypoints-1) della coda
    appena generata per gli ambienti indicati, forzandone l'altezza a
    landing_z. Le coordinate x,y del waypoint restano quelle GIA' campionate
    da _build_queue: il drone deve quindi solo scendere verticalmente
    sull'ultimo target per "atterrare", non spostarsi lateralmente.

    Usato SOLO in modalita' --spawn_mode takeoff_landing, applicato
    esclusivamente alla coda che verra' effettivamente pursuita fino alla
    fine (quella il cui completamento porta done_env=True per l'ambiente).
    """
    n_wp = base_env.cfg.n_waypoints
    base_env._wp_pos_queue[env_ids, n_wp - 1, 2] = landing_z


def rileva_architettura_da_checkpoint(checkpoint_path):
    """Ispeziona il checkpoint .pt e ricava AUTOMATICAMENTE la lista delle
    dimensioni dei layer nascosti (equivalente a policy_units_l1/l2/l3 usati
    in train_manual.py per gli sweep), leggendo direttamente le shape dei
    pesi salvati. Cosi' l'evaluation si adatta da sola a QUALSIASI run dello
    sweep, senza bisogno di conoscere a priori gli iperparametri di rete usati.

    Ritorna una lista tipo [256, 256, 32] (una entry per ogni Linear layer
    dentro 'net_container', nell'ordine in cui compaiono), oppure None se la
    struttura del checkpoint non viene riconosciuta (in tal caso lo script
    ricadra' sull'architettura di default definita in agent_cfg).
    """
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[AUTO][WARN] Impossibile aprire il checkpoint per l'ispezione: {e}")
        return None

    if not isinstance(raw, dict):
        return None

    # Il checkpoint skrl e' un dict {nome_modulo: state_dict, ...} (es. "policy",
    # "value", "optimizer", ...). Cerchiamo il primo modulo che contiene chiavi
    # riconducibili a 'net_container' (il blocco Sequential di layer nascosti
    # condiviso da policy/value nel modello SharedModel).
    state_dict = None
    for key, value in raw.items():
        if isinstance(value, dict) and any("net_container" in k for k in value.keys()):
            state_dict = value
            break

    if state_dict is None:
        return None

    # Estrae i pesi dei Linear layer dentro net_container (le chiavi tipo
    # 'net_container.0.weight', 'net_container.2.weight', ... — gli indici
    # pari sono i Linear, quelli dispari le attivazioni, che non hanno pesi).
    layer_entries = []
    for k, v in state_dict.items():
        if k.startswith("net_container") and k.endswith(".weight") and hasattr(v, "dim") and v.dim() == 2:
            try:
                idx = int(k.split(".")[1])
            except (IndexError, ValueError):
                continue
            layer_entries.append((idx, v.shape[0]))  # out_features del layer

    if not layer_entries:
        return None

    layer_entries.sort(key=lambda x: x[0])
    structure = [out_f for _, out_f in layer_entries]

    # L'ultimo Linear dentro net_container e' tipicamente il layer nascosto
    # finale prima delle teste policy/value (policy_layer / value_layer), che
    # vengono ricostruite automaticamente da skrl in base a 'structure' e non
    # vanno incluse qui.
    return structure


# =======================================================================
# PLOT CONTINUO
# =======================================================================
def plot_tracking(
    env_id, tempo_s, target_pos, current_pos, target_yaw, current_yaw,
    queue_changes, room_min, room_max, n_queues, mode, mask_name, out_dir,
):
    """Grafico 1: tracking di posizione (x,y,z) e yaw rispetto al target."""
    t = tempo_s.numpy()
    x = current_pos[:, 0].numpy()
    y = current_pos[:, 1].numpy()
    z = current_pos[:, 2].numpy()
    tgt_x = target_pos[:, 0].numpy()
    tgt_y = target_pos[:, 1].numpy()
    tgt_z = target_pos[:, 2].numpy()
    yaw = current_yaw.numpy()
    tgt_yaw = target_yaw.numpy()

    fig, axs = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    room_str = (
        f"x=[{room_min[0]:.2f},{room_max[0]:.2f}] "
        f"y=[{room_min[1]:.2f},{room_max[1]:.2f}] "
        f"z=[{room_min[2]:.2f},{room_max[2]:.2f}]"
    )
    mvx, mvy, mvz, mwz = DOF_MASKS[mask_name]
    fig.suptitle(
        f"Ambiente {env_id} | {n_queues} code | mode={mode} | "
        f"mask={mask_name} [{mvx:.0f},{mvy:.0f},{mvz:.0f},{mwz:.0f}] | "
        f"Stanza: {room_str}",
        fontsize=11,
    )

    line_real = {"color": "royalblue", "linestyle": "-", "linewidth": 1.3, "label": "reale"}
    line_target = {"color": "red", "linestyle": "--", "linewidth": 1.3, "label": "target"}

    def add_change_lines(ax):
        for tc in queue_changes:
            ax.axvline(tc, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    axs[0].plot(t, tgt_x, **line_target)
    axs[0].plot(t, x, **line_real)
    axs[0].set_ylabel("x (m)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)
    add_change_lines(axs[0])

    axs[1].plot(t, tgt_y, **line_target)
    axs[1].plot(t, y, **line_real)
    axs[1].set_ylabel("y (m)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)
    add_change_lines(axs[1])

    axs[2].plot(t, tgt_z, **line_target)
    axs[2].plot(t, z, **line_real)
    axs[2].set_ylabel("z (m)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True)
    add_change_lines(axs[2])

    axs[3].plot(t, tgt_yaw, **line_target)
    axs[3].plot(t, yaw, **line_real)
    axs[3].set_ylabel("yaw (rad)")
    axs[3].set_xlabel("Tempo (s)")
    axs[3].legend(loc="upper right")
    axs[3].grid(True)
    # planare_olonomo: yaw congelato a spawn -> target yaw costante = yaw iniziale
    if mwz < 0.5:
        axs[3].set_facecolor("#ffecec")
        axs[3].set_title("[yaw non controllabile — target congelato a spawn]", fontsize=8, color="red")
    add_change_lines(axs[3])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    percorso = os.path.join(
        out_dir, f"eval_plot_tracking_env_{env_id}_{mode}_{mask_name}.png"
    )
    plt.savefig(percorso, dpi=300)
    plt.close(fig)
    print(f"[eval] Grafico tracking salvato in: {percorso}")


def plot_errore(
    env_id, tempo_s, target_pos, current_pos,
    queue_changes, reach_thr, n_queues, mode, mask_name, out_dir,
):
    """Grafico 2: andamento dell'errore di posizione |pos - target|.
    Due viste sovrapposte in un'unica figura: scala log (mostra sia i grandi
    transitori sia i piccoli residui a regime) e zoom lineare (range fisso
    piccolo, per leggere chiaramente la precisione di assestamento)."""
    t = tempo_s.numpy()
    x = current_pos[:, 0].numpy()
    y = current_pos[:, 1].numpy()
    z = current_pos[:, 2].numpy()
    tgt_x = target_pos[:, 0].numpy()
    tgt_y = target_pos[:, 1].numpy()
    tgt_z = target_pos[:, 2].numpy()
    err = np.sqrt((x - tgt_x) ** 2 + (y - tgt_y) ** 2 + (z - tgt_z) ** 2)

    mvx, mvy, mvz, mwz = DOF_MASKS[mask_name]

    fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        f"Ambiente {env_id} — Errore di posizione | {n_queues} code | mode={mode} | "
        f"mask={mask_name} [{mvx:.0f},{mvy:.0f},{mvz:.0f},{mwz:.0f}]",
        fontsize=11,
    )

    def add_change_lines(ax):
        for tc in queue_changes:
            ax.axvline(tc, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    # -- vista 1: scala LOG, tutta la traiettoria --
    err_plot = np.maximum(err, 1e-4)  # evita log(0)
    axs[0].plot(t, err_plot, color="darkorange", linewidth=1.4, label="errore |pos - target|")
    axs[0].axhline(
        reach_thr, color="green", linestyle="--", linewidth=1.0,
        label=f"reach_thr={reach_thr:.2f}m",
    )
    axs[0].set_yscale("log")
    axs[0].set_ylabel("errore (m, log)")
    axs[0].set_title("Vista completa (scala logaritmica)", fontsize=9)
    axs[0].legend(loc="upper right")
    axs[0].grid(True, which="both")
    add_change_lines(axs[0])

    # -- vista 2: ZOOM lineare, range fisso piccolo --
    zoom_max = max(reach_thr * 3.0, 0.3)
    axs[1].plot(t, err, color="darkorange", linewidth=1.4, label="errore |pos - target|")
    axs[1].axhline(
        reach_thr, color="green", linestyle="--", linewidth=1.0,
        label=f"reach_thr={reach_thr:.2f}m",
    )
    axs[1].set_ylim(0, zoom_max)
    axs[1].set_ylabel(f"errore (m, zoom 0-{zoom_max:.2f})")
    axs[1].set_xlabel("Tempo (s)")
    axs[1].set_title("Zoom sulla precisione a regime (picchi tagliati)", fontsize=9)
    axs[1].legend(loc="upper right")
    axs[1].grid(True)
    add_change_lines(axs[1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    percorso = os.path.join(
        out_dir, f"eval_plot_errore_env_{env_id}_{mode}_{mask_name}.png"
    )
    plt.savefig(percorso, dpi=300)
    plt.close(fig)
    print(f"[eval] Grafico errore salvato in: {percorso}")


def plot_velocita(
    env_id, tempo_s, lin_vel_b, ang_vel_z, ref_vel, vel_scales,
    queue_changes, mode, mask_name, out_dir,
):
    """ref_vel: (T,4) riferimento EFFETTIVAMENTE APPLICATO (post-gating DoF mask),
    NON l'azione grezza della policy."""
    t = tempo_s.numpy()
    vx = lin_vel_b[:, 0].numpy()
    vy = lin_vel_b[:, 1].numpy()
    vz = lin_vel_b[:, 2].numpy()
    wz = ang_vel_z.numpy()
    ref = ref_vel.numpy()

    fig, axs = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    mvx, mvy, mvz_m, mwz = DOF_MASKS[mask_name]
    fig.suptitle(
        f"Ambiente {env_id} — Velocita' reali vs riferimento EFFETTIVAMENTE APPLICATO "
        f"(mode={mode}, mask={mask_name} [{mvx:.0f},{mvy:.0f},{mvz_m:.0f},{mwz:.0f}])",
        fontsize=11,
    )
    nomi = ["vx_b (m/s)", "vy_b (m/s)", "vz_b (m/s)", "wz (rad/s)"]
    reali = [vx, vy, vz, wz]
    canali_attivi = [mvx, mvy, mvz_m, mwz]

    def add_change_lines(ax):
        for tc in queue_changes:
            ax.axvline(tc, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    for i in range(4):
        scale = vel_scales[i]
        axs[i].plot(t, reali[i], color="royalblue", linewidth=1.3, label="reale")
        axs[i].plot(t, ref[:, i], color="red", linestyle="--", linewidth=1.1, label="riferimento applicato")
        axs[i].axhline(scale, color="green", linestyle=":", linewidth=1.0, label=f"limite ±{scale:.2f}")
        axs[i].axhline(-scale, color="green", linestyle=":", linewidth=1.0)
        axs[i].axhline(0.0, color="gray", linewidth=0.6, linestyle="-")
        titolo_asse = nomi[i] + ("" if canali_attivi[i] > 0.5 else "  [DISATTIVO]")
        axs[i].set_ylabel(titolo_asse)
        if canali_attivi[i] < 0.5:
            axs[i].set_facecolor("#ffecec")
        axs[i].legend(loc="upper right", fontsize=8)
        axs[i].grid(True)
        add_change_lines(axs[i])
    axs[-1].set_xlabel("Tempo (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    percorso = os.path.join(
        out_dir, f"eval_plot_velocita_env_{env_id}_{mode}_{mask_name}.png"
    )
    plt.savefig(percorso, dpi=300)
    plt.close(fig)
    print(f"[eval] Grafico velocita' salvato in: {percorso}")


# =======================================================================
# MAIN
# =======================================================================
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_vis = True
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.eval_hold_time_s is not None:
        print(
            f"[EVAL OVERRIDE] target_hold_time_s: "
            f"{env_cfg.target_hold_time_s}s -> {args_cli.eval_hold_time_s}s"
        )
        env_cfg.target_hold_time_s = args_cli.eval_hold_time_s
    if args_cli.eval_reach_threshold is not None:
        print(
            f"[EVAL OVERRIDE] target_reach_threshold: "
            f"{env_cfg.target_reach_threshold}m -> {args_cli.eval_reach_threshold}m"
        )
        env_cfg.target_reach_threshold = args_cli.eval_reach_threshold
    if args_cli.eval_reach_yaw_threshold is not None:
        print(
            f"[EVAL OVERRIDE] target_reach_yaw_threshold: "
            f"{env_cfg.target_reach_yaw_threshold}rad -> {args_cli.eval_reach_yaw_threshold}rad"
        )
        env_cfg.target_reach_yaw_threshold = args_cli.eval_reach_yaw_threshold

    print(f"[INFO] Creazione ambiente: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    base_env = env.unwrapped
    device = env.device
    control_dt = base_env.step_dt

    # -- AUTO-DETECT architettura rete dal checkpoint --
    # Cosi' non serve piu' conoscere a mano policy_units_l1/l2/l3 della run
    # specifica dello sweep che si vuole valutare: lo script legge le shape
    # dei pesi salvati e ricostruisce un agent_cfg compatibile PRIMA di
    # istanziare il Runner (che altrimenti costruirebbe una rete con le
    # dimensioni di default, causando un size mismatch al load()).
    if not args_cli.no_auto_arch:
        structure = rileva_architettura_da_checkpoint(args_cli.checkpoint)
        if structure:
            print(f"[AUTO] Architettura rete rilevata dal checkpoint: {structure}")
            agent_cfg["models"]["policy"]["network"][0]["layers"] = structure
            agent_cfg["models"]["value"]["network"][0]["layers"] = structure
        else:
            print(
                "[AUTO][WARN] Impossibile rilevare l'architettura dal checkpoint "
                "(struttura non riconosciuta): uso quella di default in agent_cfg. "
                "Se il load() fallisce per size mismatch, verifica manualmente "
                "policy_units_l1/l2/l3 su wandb per questa run."
            )
    else:
        print("[AUTO] Auto-detect disabilitato (--no_auto_arch): uso agent_cfg di default.")

    runner = Runner(env, agent_cfg)
    print(f"[INFO] Caricamento checkpoint: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()

    # Nessuna terminazione per OOB/tilt/crash: l'ambiente non si resetta mai da solo.
    base_env._get_dones = types.MethodType(_get_dones_eval, base_env)

    try:
        base_env._room_marker.set_visibility(False)
    except Exception:
        print("[WARN] Non sono riuscito a nascondere _room_marker.")

    N = args_cli.num_envs
    n_wp = env_cfg.n_waypoints
    reach_thr = env_cfg.target_reach_threshold
    tilt_warn_rad = (
        math.radians(args_cli.tilt_warn_deg)
        if args_cli.tilt_warn_deg is not None
        else math.radians(env_cfg.max_tilt_deg)
    )

    landing_z = args_cli.landing_z if args_cli.landing_z is not None else env_cfg.min_z_pos

    all_ids = torch.arange(N, device=device)

    # -- FORZA la maschera DoF scelta da terminale, DOPO il reset iniziale --
    applica_dof_mask_fissa(base_env, args_cli.mask, all_ids, device)
    print(
        f"[INFO] DoF mask fissata per tutta l'evaluation: "
        f"{args_cli.mask} = {DOF_MASKS[args_cli.mask]}"
    )

    # -- MODALITA' SPAWN: takeoff_landing forza il decollo da terra --
    if args_cli.spawn_mode == "takeoff_landing":
        forza_spawn_da_terra(base_env, all_ids, device)
        print(
            f"[INFO] spawn_mode='takeoff_landing': drone forzato a spawnare "
            f"vicino al pavimento (z in {env_cfg.spawn_low_z_range}), "
            f"landing_z={landing_z:.2f}m"
        )
    else:
        print("[INFO] spawn_mode='free': spawn/coda invariati.")

    genera_nuova_coda(base_env, all_ids, args_cli.mode, device)

    # Contatore di code GENERATE per ambiente (diverso da queue_count, che
    # conta le code COMPLETATE). Serve per identificare l'ultima coda che
    # verra' effettivamente pursuita fino alla fine, cosi' da poterne forzare
    # l'ultimo waypoint ad altezza di atterraggio in modalita' takeoff_landing.
    codas_generate = torch.ones(N, dtype=torch.long)  # la coda iniziale conta come generata

    if args_cli.spawn_mode == "takeoff_landing" and args_cli.num_queues == 1:
        # caso limite: l'unica coda e' gia' quella iniziale
        forza_atterraggio_ultima_coda(base_env, all_ids, landing_z, device)

    queue_count = torch.zeros(N, dtype=torch.long)
    steps_on_queue = torch.zeros(N, dtype=torch.long)
    hold_steps = torch.zeros(N, dtype=torch.long)
    hold_steps_required = max(1, round(env_cfg.target_hold_time_s / control_dt))
    print(
        f"[INFO] target_hold_time_s={env_cfg.target_hold_time_s}s -> "
        f"{hold_steps_required} step consecutivi richiesti entro reach_thr "
        f"prima di rigenerare la coda."
    )
    done_env = torch.zeros(N, dtype=torch.bool)
    was_tilting = torch.zeros(N, dtype=torch.bool)
    was_out_of_room = torch.zeros(N, dtype=torch.bool)

    storia_target_pos = [[] for _ in range(N)]
    storia_current_pos = [[] for _ in range(N)]
    storia_target_yaw = [[] for _ in range(N)]
    storia_current_yaw = [[] for _ in range(N)]
    storia_tempo = [[] for _ in range(N)]
    storia_lin_vel_b = [[] for _ in range(N)]
    storia_ang_vel_z = [[] for _ in range(N)]
    storia_ref_vel = [[] for _ in range(N)]
    queue_changes = [[] for _ in range(N)]

    draw = _debug_draw.acquire_debug_draw_interface()

    durata_stimata_s = args_cli.num_queues * args_cli.max_steps_per_queue * control_dt
    print(
        f"[INFO] Avvio simulazione: mode='{args_cli.mode}' | mask='{args_cli.mask}' | "
        f"spawn_mode='{args_cli.spawn_mode}' | "
        f"{args_cli.num_queues} code sequenziali di {n_wp} waypoint per ambiente "
        f"(reach_thr={reach_thr}m) | "
        f"step_dt={control_dt:.4f}s ({1/control_dt:.1f}Hz) | "
        f"durata MASSIMA teorica: {durata_stimata_s:.1f}s "
        f"({args_cli.num_queues}×{args_cli.max_steps_per_queue} step)"
    )

    global_step = 0
    max_total_steps = args_cli.num_queues * args_cli.max_steps_per_queue

    with torch.inference_mode():
        while (~done_env).any() and global_step < max_total_steps:

            drone_pos_w = base_env.robot.data.root_pos_w
            env_origins = base_env._env_origins
            drone_pos_env = drone_pos_w - env_origins

            drone_quat_w = base_env.robot.data.root_quat_w
            _, _, drone_yaw = euler_xyz_from_quat(drone_quat_w)
            proj_grav_b = base_env.robot.data.projected_gravity_b

            w0_pos, w0_yaw = base_env._current_target()
            dist = torch.norm(drone_pos_env - w0_pos, dim=1)
            clearance = base_env._clearance(drone_pos_env)

            # -- warning OOB e tilt --
            for env_id in range(N):
                if done_env[env_id]:
                    continue

                out_now = torch.any(clearance[env_id] < 0.0).item()
                if out_now and not was_out_of_room[env_id]:
                    c = clearance[env_id].cpu().tolist()
                    pos = drone_pos_env[env_id].cpu().tolist()
                    print(
                        f"[WARN][FUORI STANZA] Env {env_id} | step {global_step} | "
                        f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})m | "
                        f"clearance(+x,+y,+z,-x,-y,-z)={['%.2f' % v for v in c]}"
                    )
                was_out_of_room[env_id] = out_now

                cos_tilt = (-proj_grav_b[env_id, 2]).clamp(-1.0, 1.0).item()
                tilt_deg = math.degrees(math.acos(cos_tilt))
                tilting_now = math.radians(tilt_deg) > tilt_warn_rad
                if tilting_now and not was_tilting[env_id]:
                    print(
                        f"[WARN][TILT] Env {env_id} | step {global_step} | tilt={tilt_deg:.1f} deg"
                    )
                was_tilting[env_id] = tilting_now

            # -- avanzamento code --
            for env_id in range(N):
                if done_env[env_id]:
                    continue
                steps_on_queue[env_id] += 1

                coda_esaurita = base_env._wp_idx[env_id].item() >= (n_wp - 1)
                entro_soglia = coda_esaurita and dist[env_id].item() < reach_thr

                # Il drone deve restare CONTINUATIVAMENTE entro reach_thr per
                # target_hold_time_s (hold_steps_required step) prima che la
                # coda venga considerata completata: un singolo passaggio
                # veloce sotto soglia non basta piu' a far scattare il target
                # successivo. Se il drone esce dalla soglia, il contatore si
                # azzera e l'hold riparte da zero.
                if entro_soglia:
                    hold_steps[env_id] += 1
                else:
                    hold_steps[env_id] = 0

                assestato_su_ultimo = hold_steps[env_id].item() >= hold_steps_required
                timeout_coda = steps_on_queue[env_id].item() >= args_cli.max_steps_per_queue

                if assestato_su_ultimo or timeout_coda:
                    if timeout_coda and not assestato_su_ultimo:
                        print(
                            f"[WARN] Env {env_id}: limite di sicurezza "
                            f"({args_cli.max_steps_per_queue} step) raggiunto sulla coda "
                            f"#{queue_count[env_id].item()}, rigenero comunque."
                        )
                    ids_t = torch.tensor([env_id], device=device)
                    genera_nuova_coda(base_env, ids_t, args_cli.mode, device)
                    applica_dof_mask_fissa(base_env, args_cli.mask, ids_t, device)
                    codas_generate[env_id] += 1

                    # Se la coda appena generata sara' l'ultima effettivamente
                    # pursuita (quella il cui completamento porta
                    # done_env=True), e siamo in modalita' takeoff_landing,
                    # forza il suo ultimo waypoint ad altezza di atterraggio.
                    if (
                        args_cli.spawn_mode == "takeoff_landing"
                        and codas_generate[env_id].item() == args_cli.num_queues
                    ):
                        forza_atterraggio_ultima_coda(base_env, ids_t, landing_z, device)
                        print(
                            f"[INFO] Env {env_id}: ultima coda (landing) generata, "
                            f"ultimo waypoint forzato a z={landing_z:.2f}m"
                        )

                    queue_count[env_id] += 1
                    steps_on_queue[env_id] = 0
                    hold_steps[env_id] = 0
                    queue_changes[env_id].append(len(storia_tempo[env_id]) * control_dt)
                    new_w0_pos, new_w0_yaw = base_env._current_target()
                    print(
                        f"[INFO] Env {env_id}: coda #{queue_count[env_id].item() - 1} completata "
                        f"(t={global_step*control_dt:.2f}s) -> nuova coda "
                        f"#{queue_count[env_id].item()}, "
                        f"primo target: ({new_w0_pos[env_id,0]:.2f}, "
                        f"{new_w0_pos[env_id,1]:.2f}, {new_w0_pos[env_id,2]:.2f})m"
                    )

                    if queue_count[env_id] >= args_cli.num_queues:
                        done_env[env_id] = True

            w0_pos, w0_yaw = base_env._current_target()

            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])

            # -- debug draw: stanza wireframe + frecce yaw --
            draw.clear_lines()
            for env_id in range(N):
                if done_env[env_id]:
                    continue
                ox, oy, oz = env_origins[env_id].cpu().tolist()
                rmin = base_env._room_min[env_id].cpu().tolist()
                rmax = base_env._room_max[env_id].cpu().tolist()
                for p1, p2 in costruisci_stanza_asimmetrica(ox, oy, oz, rmin, rmax):
                    draw.draw_lines([p1], [p2], [(1.0, 1.0, 1.0, 0.5)], [2.0])

                # freccia gialla = yaw target
                target_pos_w = (w0_pos[env_id] + env_origins[env_id]).cpu().tolist()
                tgt_yaw_val = w0_yaw[env_id].item()
                dir_target = [math.cos(tgt_yaw_val), math.sin(tgt_yaw_val), 0.0]
                for p1, p2 in costruisci_freccia(target_pos_w, dir_target, scala=0.4):
                    draw.draw_lines([p1], [p2], [(1.0, 1.0, 0.0, 1.0)], [3.5])

                # freccia blu = yaw drone
                cur_yaw_val = drone_yaw[env_id].item()
                drone_origin = drone_pos_w[env_id].cpu().tolist()
                drone_origin[2] += 0.05
                dir_drone = [math.cos(cur_yaw_val), math.sin(cur_yaw_val), 0.0]
                for p1, p2 in costruisci_freccia(drone_origin, dir_drone, scala=0.4):
                    draw.draw_lines([p1], [p2], [(0.0, 0.4, 1.0, 1.0)], [3.5])

            # -- registrazione storia --
            for env_id in range(N):
                if done_env[env_id]:
                    continue
                storia_target_pos[env_id].append(w0_pos[env_id].clone().cpu())
                storia_current_pos[env_id].append(drone_pos_env[env_id].clone().cpu())
                storia_target_yaw[env_id].append(w0_yaw[env_id].clone().cpu())
                storia_current_yaw[env_id].append(drone_yaw[env_id].clone().cpu())
                storia_tempo[env_id].append(global_step * control_dt)

            obs, _, _, _, _ = env.step(actions)

            # Riferimento EFFETTIVAMENTE APPLICATO al low-level: letto da
            # _high_level_actions (gia' passato dal gating hard della DoF mask in
            # _pre_physics_step). NON usare 'actions' grezze: sui canali disattivati
            # mostrerebbero un riferimento fittizio mai eseguito fisicamente.
            #
            # NOTA ALLINEAMENTO TEMPORALE: lin_vel_b/ang_vel_z vengono registrate QUI,
            # DOPO env.step(), insieme al riferimento che le ha appena generate. Se
            # venissero lette prima dello step (come nella versione precedente) si
            # otterrebbe uno sfasamento di un campione tra riferimento e velocita'
            # reale: storia_ref_vel[i] risulterebbe la causa di storia_lin_vel_b[i+1]
            # invece che di storia_lin_vel_b[i], facendo sembrare il drone "in ritardo"
            # anche con un controllo perfetto e istantaneo.
            hl_applied = base_env._high_level_actions.clamp(-1.0, 1.0)
            ref_vel_step = hl_applied * base_env._vel_ref_scale
            for env_id in range(N):
                if done_env[env_id]:
                    continue
                storia_lin_vel_b[env_id].append(
                    base_env.robot.data.root_lin_vel_b[env_id].clone().cpu()
                )
                storia_ang_vel_z[env_id].append(
                    base_env.robot.data.root_ang_vel_b[env_id, 2].clone().cpu()
                )
                storia_ref_vel[env_id].append(ref_vel_step[env_id].clone().cpu())

            global_step += 1

    draw.clear_lines()
    env.close()

    print(f"\n[INFO] Code completate per env: {queue_count.tolist()}")
    print(
        f"[INFO] Durata effettiva simulata: "
        f"{global_step * control_dt:.1f}s ({global_step} step high-level)"
    )

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

        plot_tracking(
            env_id, tempo_s, tgt_pos, cur_pos, tgt_yaw_t, cur_yaw_t,
            queue_changes[env_id],
            base_env._room_min[env_id].cpu().tolist(),
            base_env._room_max[env_id].cpu().tolist(),
            int(queue_count[env_id].item()), args_cli.mode, args_cli.mask, args_cli.out_dir,
        )

        plot_errore(
            env_id, tempo_s, tgt_pos, cur_pos,
            queue_changes[env_id], reach_thr,
            int(queue_count[env_id].item()), args_cli.mode, args_cli.mask, args_cli.out_dir,
        )

        lin_vel_b = torch.stack(storia_lin_vel_b[env_id])
        ang_vel_z = torch.stack(storia_ang_vel_z[env_id])
        ref_vel = torch.stack(storia_ref_vel[env_id])
        vel_scales = [
            env_cfg.target_lin_vel_xy_scale,
            env_cfg.target_lin_vel_xy_scale,
            env_cfg.target_lin_vel_z_scale,
            env_cfg.target_yaw_vel_scale,
        ]
        plot_velocita(
            env_id, tempo_s, lin_vel_b, ang_vel_z, ref_vel, vel_scales,
            queue_changes[env_id], args_cli.mode, args_cli.mask, args_cli.out_dir,
        )

    # =======================================================================
    # REPORT FINALE
    # =======================================================================
    err_pos_all, err_yaw_all = [], []
    err_pos_regime_all = []
    for env_id in range(N):
        if len(storia_tempo[env_id]) == 0:
            continue
        cur_pos = torch.stack(storia_current_pos[env_id])
        tgt_pos = torch.stack(storia_target_pos[env_id])
        cur_yaw_t = torch.stack(storia_current_yaw[env_id])
        tgt_yaw_t = torch.stack(storia_target_yaw[env_id])
        err_pos_ep = torch.norm(cur_pos - tgt_pos, dim=-1)
        err_pos_all.append(err_pos_ep)
        yaw_diff = cur_yaw_t - tgt_yaw_t
        err_yaw_all.append(torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).abs())

        # -- errore "a regime": ultimo 30% di ogni tratto tra due cambi di target
        # (o dell'intera coda, se non ci sono stati cambi), quando il drone dovrebbe
        # gia' essersi assestato. E' la metrica di PRECISIONE, a differenza della
        # media su tutta la traiettoria che e' dominata dai transitori. --
        bounds = [0] + [
            int(round(tc / control_dt)) for tc in queue_changes[env_id]
        ] + [len(err_pos_ep)]
        regime_idx = []
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            if b <= a:
                continue
            start_regime = a + int(round((b - a) * 0.7))
            regime_idx.extend(range(start_regime, b))
        if regime_idx:
            err_pos_regime_all.append(err_pos_ep[regime_idx])

    if err_pos_all:
        err_pos = torch.mean(torch.cat(err_pos_all)).item()
        err_yaw = torch.mean(torch.cat(err_yaw_all)).item()
        mvx, mvy, mvz, mwz = DOF_MASKS[args_cli.mask]
        print("\n" + "=" * 55)
        print(f"  RISULTATI  (mode={args_cli.mode}, mask={args_cli.mask} [{mvx:.0f},{mvy:.0f},{mvz:.0f},{mwz:.0f}], "
              f"spawn_mode={args_cli.spawn_mode})")
        print(f"  Code completate totali:  {int(queue_count.sum().item())}")
        print(f"  Errore medio posizione (INTERA traiettoria, incl. transitori): {err_pos:.4f} m")
        if err_pos_regime_all:
            err_pos_regime_cat = torch.cat(err_pos_regime_all)
            print(
                f"  Errore medio posizione A REGIME (ultimo 30% di ogni tratto): "
                f"{torch.mean(err_pos_regime_cat).item():.4f} m "
                f"(max: {torch.max(err_pos_regime_cat).item():.4f} m)"
            )
        if mwz > 0.5:
            print(f"  Errore medio yaw:        {err_yaw:.4f} rad ({math.degrees(err_yaw):.2f} deg)")
        else:
            print(f"  Errore medio yaw:        N/A (wz=0, yaw non controllabile)")
        print("=" * 55)


if __name__ == "__main__":
    main()
    simulation_app.close()