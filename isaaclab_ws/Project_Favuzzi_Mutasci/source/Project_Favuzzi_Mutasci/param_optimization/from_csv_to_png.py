import argparse
import csv
import math
import os
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt

DOF_MASKS = {
    "full":            (1.0, 1.0, 1.0, 1.0),
    "uniciclo":        (1.0, 0.0, 1.0, 1.0),
    "planare_olonomo": (1.0, 1.0, 1.0, 0.0),
}

DEFAULT_VEL_SCALES = {
    "vx": 1.0,
    "vy": 1.0,
    "vz": 1.0,
    "wz": 1.5,
}

def _leggi_csv(path):
    data = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for k in fieldnames:
            data[k] = []
        for row in reader:
            for k in fieldnames:
                v = row[k]
                data[k].append(float(v) if v not in (None, "") else float("nan"))
    for k in data:
        data[k] = np.asarray(data[k], dtype=np.float64)
    return data

def _rileva_cambi_target(data, tol=1e-6):
    tx, ty, tz, tyaw = data["target_x"], data["target_y"], data["target_z"], data["target_yaw"]
    t = data["time_s"]
    changes = []
    for i in range(1, len(t)):
        if (
            abs(tx[i] - tx[i - 1]) > tol
            or abs(ty[i] - ty[i - 1]) > tol
            or abs(tz[i] - tz[i - 1]) > tol
            or abs(tyaw[i] - tyaw[i - 1]) > tol
        ):
            changes.append(t[i])
    return changes

def plot_tracking(data, mask_name, mode_name, out_dir):
    t = data["time_s"]
    x, y, z, yaw = data["current_x"], data["current_y"], data["current_z"], data["current_yaw"]
    tgt_x, tgt_y, tgt_z, tgt_yaw = data["target_x"], data["target_y"], data["target_z"], data["target_yaw"]

    mvx, mvy, mvz, mwz = DOF_MASKS.get(mask_name, (1.0, 1.0, 1.0, 1.0))
    queue_changes = _rileva_cambi_target(data)

    fig, axs = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle(
        f"Eval sweep — mask={mask_name} [{mvx:.0f},{mvy:.0f},{mvz:.0f},{mwz:.0f}] | mode={mode_name} "
        f"(env rappresentativo)",
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
    if mwz < 0.5:
        axs[3].set_facecolor("#ffecec")
        axs[3].set_title("[yaw non controllabile — target congelato a spawn]", fontsize=8, color="red")
    add_change_lines(axs[3])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    percorso = os.path.join(out_dir, f"eval_plot_tracking_{mask_name}_{mode_name}.png")
    plt.savefig(percorso, dpi=300)
    plt.close(fig)
    print(f"[csv2png] Grafico tracking salvato in: {percorso}")

def plot_errore(data, mask_name, mode_name, reach_thr, out_dir):
    t = data["time_s"]
    err = data["err_pos"]
    queue_changes = _rileva_cambi_target(data)

    fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        f"Eval sweep — Errore di posizione | mask={mask_name} | mode={mode_name} "
        f"(env rappresentativo)",
        fontsize=11,
    )

    def add_change_lines(ax):
        for tc in queue_changes:
            ax.axvline(tc, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    err_plot = np.maximum(err, 1e-4)
    axs[0].plot(t, err_plot, color="darkorange", linewidth=1.4, label="errore |pos - target|")
    axs[0].axhline(reach_thr, color="green", linestyle="--", linewidth=1.0, label=f"reach_thr={reach_thr:.2f}m")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("errore (m, log)")
    axs[0].set_title("Vista completa (scala logaritmica)", fontsize=9)
    axs[0].legend(loc="upper right")
    axs[0].grid(True, which="both")
    add_change_lines(axs[0])

    zoom_max = max(reach_thr * 3.0, 0.3)
    axs[1].plot(t, err, color="darkorange", linewidth=1.4, label="errore |pos - target|")
    axs[1].axhline(reach_thr, color="green", linestyle="--", linewidth=1.0, label=f"reach_thr={reach_thr:.2f}m")
    axs[1].set_ylim(0, zoom_max)
    axs[1].set_ylabel(f"errore (m, zoom 0-{zoom_max:.2f})")
    axs[1].set_xlabel("Tempo (s)")
    axs[1].set_title("Zoom sulla precisione a regime (picchi tagliati)", fontsize=9)
    axs[1].legend(loc="upper right")
    axs[1].grid(True)
    add_change_lines(axs[1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    percorso = os.path.join(out_dir, f"eval_plot_errore_{mask_name}_{mode_name}.png")
    plt.savefig(percorso, dpi=300)
    plt.close(fig)
    print(f"[csv2png] Grafico errore salvato in: {percorso}")

def plot_velocita(data, mask_name, mode_name, vel_scales, out_dir):
    t = data["time_s"]
    vx, vy, vz, wz = data["lin_vel_x"], data["lin_vel_y"], data["lin_vel_z"], data["ang_vel_z"]
    ref = np.stack([data["ref_vx"], data["ref_vy"], data["ref_vz"], data["ref_wz"]], axis=1)

    mvx, mvy, mvz_m, mwz = DOF_MASKS.get(mask_name, (1.0, 1.0, 1.0, 1.0))
    queue_changes = _rileva_cambi_target(data)

    fig, axs = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle(
        f"Eval sweep — Velocita' reali vs riferimento EFFETTIVAMENTE APPLICATO "
        f"(mask={mask_name} [{mvx:.0f},{mvy:.0f},{mvz_m:.0f},{mwz:.0f}], mode={mode_name})",
        fontsize=11,
    )
    nomi = ["vx_b (m/s)", "vy_b (m/s)", "vz_b (m/s)", "wz (rad/s)"]
    reali = [vx, vy, vz, wz]
    canali_attivi = [mvx, mvy, mvz_m, mwz]
    scale_list = [vel_scales["vx"], vel_scales["vy"], vel_scales["vz"], vel_scales["wz"]]

    def add_change_lines(ax):
        for tc in queue_changes:
            ax.axvline(tc, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    for i in range(4):
        scale = scale_list[i]
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
    percorso = os.path.join(out_dir, f"eval_plot_velocita_{mask_name}_{mode_name}.png")
    plt.savefig(percorso, dpi=300)
    plt.close(fig)
    print(f"[csv2png] Grafico velocita' salvato in: {percorso}")

def _parse_nome_combo(filename):
    base = os.path.basename(filename)
    if not base.startswith("eval_timeseries_") or not base.endswith(".csv"):
        return None
    core = base[len("eval_timeseries_"):-len(".csv")]
    for mask_name in sorted(DOF_MASKS.keys(), key=len, reverse=True):
        prefix = mask_name + "_"
        if core.startswith(prefix):
            mode_name = core[len(prefix):]
            return mask_name, mode_name
    return None

def _scarica_artifact_wandb(artifact_path: str, download_root: str) -> str:
    try:
        import wandb
    except ImportError:
        print(
            "[csv2png][ERRORE] Il pacchetto 'wandb' non e' installato in questo "
            "ambiente Python. Installalo (pip install wandb) oppure usa --csv_dir "
            "puntando a una cartella con i CSV gia' scaricati manualmente."
        )
        sys.exit(1)

    print(f"[csv2png] Connessione a wandb, download artifact: {artifact_path}")
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    local_dir = artifact.download(root=download_root)
    print(f"[csv2png] Artifact scaricato in: {local_dir}")
    return local_dir

def main():
    parser = argparse.ArgumentParser(
        description="Converte i CSV di eval_sweep.py (eval_timeseries_{mask}_{mode}.csv) "
                    "negli stessi 3 grafici .png (tracking, errore, velocita') prodotti da "
                    "evaluate_pos_controller_continuous.py, uno per ogni combinazione trovata."
    )
    parser.add_argument(
        "--csv_dir", type=str, default=None,
        help="Cartella LOCALE contenente gia' i file eval_timeseries_{mask}_{mode}.csv. "
             "Alternativa a --wandb_artifact: uno dei due e' obbligatorio.",
    )
    parser.add_argument(
        "--wandb_artifact", type=str, default=None,
        help="Path dell'Artifact wandb da cui SCARICARE i CSV direttamente dal server, "
             "es. 'pieromutasci-politecnico-di-bari/tuning_pos_controller/"
             "eval_timeseries_<run_id>:latest'. Alternativa a --csv_dir.",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Cartella di output per i .png. Default: la sottocartella 'graph' della cartella "
             "di questo script (es. param_optimization/graph/), cosi' i .png finiscono li' senza "
             "bisogno di specificare nulla.",
    )
    parser.add_argument(
        "--reach_thr", type=float, default=0.15,
        help="Soglia [m] di raggiungimento target (target_reach_threshold), "
             "usata solo per disegnare la linea verde nei grafici errore.",
    )
    parser.add_argument("--vel_scale_vx", type=float, default=DEFAULT_VEL_SCALES["vx"])
    parser.add_argument("--vel_scale_vy", type=float, default=DEFAULT_VEL_SCALES["vy"])
    parser.add_argument("--vel_scale_vz", type=float, default=DEFAULT_VEL_SCALES["vz"])
    parser.add_argument("--vel_scale_wz", type=float, default=DEFAULT_VEL_SCALES["wz"])
    args = parser.parse_args()

    if not args.csv_dir and not args.wandb_artifact:
        parser.error("Specifica --csv_dir (cartella locale) oppure --wandb_artifact (download da wandb).")
    if args.csv_dir and args.wandb_artifact:
        parser.error("Usa SOLO uno tra --csv_dir e --wandb_artifact, non entrambi.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out_dir = os.path.join(script_dir, "graph")

    out_dir = args.out_dir if args.out_dir is not None else default_out_dir
    os.makedirs(out_dir, exist_ok=True)

    vel_scales = {
        "vx": args.vel_scale_vx,
        "vy": args.vel_scale_vy,
        "vz": args.vel_scale_vz,
        "wz": args.vel_scale_wz,
    }

    if args.wandb_artifact:
        with tempfile.TemporaryDirectory() as download_root:
            csv_dir = _scarica_artifact_wandb(args.wandb_artifact, download_root)
            _elabora_csv(csv_dir, out_dir, args.reach_thr, vel_scales)
        return
    else:
        csv_dir = args.csv_dir
        _elabora_csv(csv_dir, out_dir, args.reach_thr, vel_scales)

def _elabora_csv(csv_dir, out_dir, reach_thr, vel_scales):
    csv_files = [
        f for f in os.listdir(csv_dir)
        if f.startswith("eval_timeseries_") and f.endswith(".csv")
    ]
    if not csv_files:
        print(f"[csv2png] Nessun file 'eval_timeseries_*.csv' trovato in: {csv_dir}")
        return

    for fname in sorted(csv_files):
        combo = _parse_nome_combo(fname)
        if combo is None:
            print(f"[csv2png][WARN] Nome file non riconosciuto, salto: {fname}")
            continue
        mask_name, mode_name = combo
        path = os.path.join(csv_dir, fname)
        print(f"[csv2png] Elaboro {fname} (mask={mask_name}, mode={mode_name})...")

        data = _leggi_csv(path)
        if len(data.get("time_s", [])) == 0:
            print(f"[csv2png][WARN] CSV vuoto, salto: {fname}")
            continue

        plot_tracking(data, mask_name, mode_name, out_dir)
        plot_errore(data, mask_name, mode_name, reach_thr, out_dir)
        plot_velocita(data, mask_name, mode_name, vel_scales, out_dir)

    print(f"\n[csv2png] Completato. {len(csv_files)} combinazione/i elaborata/e -> {out_dir}")

if __name__ == "__main__":
    main()
