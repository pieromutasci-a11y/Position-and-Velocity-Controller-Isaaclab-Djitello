# train_manual.py
#
# Variante di train.py con supporto WANDB SWEEP per il tuning automatico dei
# pesi reward del PosController. skrl.utils.runner.torch.Runner costruisce
# gia' le reti a partire da un dizionario (agent_cfg["models"]["policy"]
# ["network"][0]["layers"]); PPO e architettura NON sono sweeppati in questa
# versione (restano quelli di agent_cfg, gia' buoni) — lo sweep si concentra
# sui pesi della reward, in linea con l'obiettivo GLOBALE definito nell'env
# (vedi pos_controller_env_cfg.py / pos_controller_env.py).
#
# Uso identico a train.py per un training normale; con --wandb_sweep si attiva
# la lettura degli overrides da wandb.config (durante uno sweep agent).
#
# NOVITA' rispetto alla versione precedente:
#   - Metrica sweep: Objective/composite_target (globale, vedi env),
#     non piu' combined_error_smooth / combined_error_maxmode.
#   - Rimossi CLI args obsoleti (rew_scale_vel_mask_penalty, non esiste piu'
#     nel cfg attuale).
#   - Aggiunti CLI args per tutti i pesi reward specifici di uniciclo:
#     rew_scale_vy_penalty_uniciclo, rew_scale_vy_real_penalty_uniciclo,
#     rew_scale_reverse_vx, rew_scale_aggressive_cmd,
#     rew_scale_approach_brake, approach_brake_dist, yaw_gate_dist_uniciclo.
#   - A fine training, se in modalita' sweep, lo storico delle componenti
#     dell'obiettivo (env.get_objective_history()) viene scritto come 6 file
#     CSV dentro un wandb.Artifact e caricato con run.log_artifact(...).
#     NESSUN file CSV viene persistito sull'host: wandb.Artifact.new_file()
#     scrive direttamente nello staging interno gestito da wandb, che viene
#     caricato e reso disponibile nel run — non in una cartella locale che
#     gestiamo noi.

import argparse
import csv
import sys
import os
import random
from datetime import datetime

from isaaclab.app import AppLauncher

# =======================================================================
# ARGPARSE
# =======================================================================
parser = argparse.ArgumentParser(
    description="Train PosController con skrl (supporto wandb sweep, obiettivo globale)."
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument(
    "--task",
    type=str,
    default="Template-Project-Favuzzi-Mutasci-PosController-Direct-v0",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--agent", type=str, default="skrl_cfg_entry_point")

# --- WandB Sweep ---
parser.add_argument(
    "--wandb_sweep", action="store_true", default=False,
    help="Abilita la modalita' sweep wandb.",
)
parser.add_argument("--wandb_entity", type=str, default="vitt-politecnico-di-bari")
parser.add_argument("--wandb_project", type=str, default="drone-pos-tuning")
parser.add_argument(
    "--max_timesteps", type=int, default=None,
    help="Timesteps totali per il trial (override del trainer)",
)

# --- Reward / Env Tuning (nomi = attributi esatti di MyDronePosEnvCfg) ---
# -- posizione / yaw (base) --
parser.add_argument("--rew_scale_alive", type=float, default=None)
parser.add_argument("--rew_scale_position_approach", type=float, default=None)
parser.add_argument("--reward_exp_beta", type=float, default=None)
parser.add_argument("--rew_scale_position_prec", type=float, default=None)
parser.add_argument("--reward_exp_beta_prec", type=float, default=None)
parser.add_argument("--rew_scale_yaw_error", type=float, default=None)
parser.add_argument("--rew_scale_yaw_prec", type=float, default=None)
parser.add_argument("--reward_exp_beta_yaw_prec", type=float, default=None)

# -- regolarizzazione / anti-oscillazione --
parser.add_argument("--rew_scale_action_smoothness", type=float, default=None)
parser.add_argument("--rew_scale_reg_ang_vel_xy", type=float, default=None)
parser.add_argument("--rew_scale_reg_ang_vel_wz", type=float, default=None)
parser.add_argument("--rew_scale_aggressive_cmd", type=float, default=None)

# -- uniciclo: vy, retromarcia, anti-overshoot --
parser.add_argument("--rew_scale_reverse_vx", type=float, default=None)
parser.add_argument("--rew_scale_vy_penalty_uniciclo", type=float, default=None)
parser.add_argument("--rew_scale_vy_real_penalty_uniciclo", type=float, default=None)
parser.add_argument("--rew_scale_approach_brake", type=float, default=None)
parser.add_argument("--approach_brake_dist", type=float, default=None)
parser.add_argument("--yaw_gate_dist_uniciclo", type=float, default=None)

# -- eventi impulsivi / soglie waypoint --
parser.add_argument("--rew_scale_target_reached", type=float, default=None)
parser.add_argument("--oob_reward", type=float, default=None)
parser.add_argument("--tilt_death_reward", type=float, default=None)
parser.add_argument("--target_reach_threshold", type=float, default=None)
parser.add_argument("--target_reach_yaw_threshold", type=float, default=None)
parser.add_argument("--target_hold_time_s", type=float, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =======================================================================
# IMPORT POST-AVVIO ISAAC SIM
# =======================================================================
import gymnasium as gym
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import Project_Favuzzi_Mutasci.tasks  # noqa: F401

if args_cli.wandb_sweep:
    import wandb


def apply_env_override(env_cfg, key, val):
    """Sovrascrive un attributo di env_cfg se esiste, con log a schermo."""
    if hasattr(env_cfg, key):
        setattr(env_cfg, key, val)
        print(f"[SWEEP] env_cfg.{key} = {val}")
    else:
        print(f"[SWEEP][WARN] env_cfg non ha l'attributo '{key}', ignorato.")


def sweep_val(key, args_cli):
    """Priorita': wandb.config > CLI arg > None (nessun override)."""
    v = wandb.config.get(key)
    if v is not None:
        return v
    return getattr(args_cli, key, None)


def _find_base_env(env):
    """Risale la catena di wrapper (SkrlVecEnvWrapper, gymnasium, ecc.)
    cercando l'env base che espone get_objective_history(). Prova gli
    attributi piu' comuni usati dai wrapper per riferirsi all'env interno.
    """
    seen = set()
    stack = [env]
    while stack:
        e = stack.pop()
        if id(e) in seen or e is None:
            continue
        seen.add(id(e))
        if hasattr(e, "get_objective_history"):
            return e
        for attr in ("unwrapped", "_env", "env"):
            inner = getattr(e, attr, None)
            if inner is not None and id(inner) not in seen:
                stack.append(inner)
    return None


def _upload_objective_csv_to_wandb(env, run):
    """Scrive lo storico delle componenti dell'obiettivo come CSV dentro un
    wandb.Artifact e lo carica sul run. Nessun file viene persistito
    sull'host: wandb.Artifact.new_file() gestisce internamente lo staging.
    """
    base_env = _find_base_env(env)
    if base_env is None:
        print("[WARN] Nessun env con get_objective_history() trovato, CSV obiettivo non caricati.")
        return

    history = base_env.get_objective_history()
    if not history or all(len(v) == 0 for v in history.values()):
        print("[WARN] Storico obiettivo vuoto, nessun CSV caricato.")
        return

    artifact = wandb.Artifact(f"objective-csv-{run.id}", type="objective_csv")
    n_files = 0
    for metric_name, values in history.items():
        with artifact.new_file(f"{metric_name}.csv", mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(["reset_index", metric_name])
            for i, v in enumerate(values):
                writer.writerow([i, v])
        n_files += 1

    run.log_artifact(artifact)
    print(f"[INFO] Caricati {n_files} CSV delle componenti dell'obiettivo su wandb (artifact objective-csv-{run.id}).")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )

    experiment_name = args_cli.experiment_name
    run = None

    if args_cli.wandb_sweep:
        print("[INFO] Inizializzazione WandB per lo sweep...")
        run = wandb.init(
            entity=args_cli.wandb_entity,
            project=args_cli.wandb_project,
            sync_tensorboard=True,
        )

        if args_cli.seed is None:
            args_cli.seed = random.randint(0, 10000)
        env_cfg.seed = args_cli.seed
        agent_cfg["seed"] = args_cli.seed
        wandb.config.update({"seed": args_cli.seed}, allow_val_change=True)

        experiment_name = (
            f"SWEEP_{run.name}" if experiment_name is None
            else os.path.join(experiment_name, run.name)
        )

        # ---- Reward / Env overrides ----
        env_keys = [
            "rew_scale_alive",
            "rew_scale_position_approach", "reward_exp_beta",
            "rew_scale_position_prec", "reward_exp_beta_prec",
            "rew_scale_yaw_error",
            "rew_scale_yaw_prec", "reward_exp_beta_yaw_prec",
            "rew_scale_action_smoothness",
            "rew_scale_reg_ang_vel_xy",
            "rew_scale_reg_ang_vel_wz",
            "rew_scale_aggressive_cmd",
            "rew_scale_reverse_vx",
            "rew_scale_vy_penalty_uniciclo",
            "rew_scale_vy_real_penalty_uniciclo",
            "rew_scale_approach_brake",
            "approach_brake_dist",
            "yaw_gate_dist_uniciclo",
            "rew_scale_target_reached",
            "oob_reward", "tilt_death_reward",
            "target_reach_threshold", "target_reach_yaw_threshold", "target_hold_time_s",
        ]
        for key in env_keys:
            v = sweep_val(key, args_cli)
            if v is not None:
                apply_env_override(env_cfg, key, v)

        # ---- Durata del trial ----
        max_ts = sweep_val("max_timesteps", args_cli)
        if max_ts is not None:
            agent_cfg["trainer"]["timesteps"] = int(max_ts)
            print(f"[SWEEP] trainer.timesteps = {int(max_ts)}")

    # ---- logging dirs ----
    log_root_path = os.path.abspath(
        os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_torch"
    if experiment_name:
        log_dir += f"_{experiment_name}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.max_iterations and not args_cli.wandb_sweep:
        agent_cfg["trainer"]["timesteps"] = (
            args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        )

    # ---- env + runner ----
    print(f"[INFO] Creazione ambiente: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)

    if args_cli.checkpoint:
        print(f"[INFO] Caricamento checkpoint: {args_cli.checkpoint}")
        runner.agent.load(args_cli.checkpoint)

    print("[INFO] Avvio training...")
    runner.run()

    # ---- CSV obiettivo su wandb (PRIMA di env.close(), l'env deve essere ancora vivo) ----
    if args_cli.wandb_sweep and run is not None:
        try:
            _upload_objective_csv_to_wandb(env, run)
        except Exception as e:
            print(f"[ERROR] Upload CSV obiettivo fallito: {e}")

    env.close()

    # ---- upload best model su wandb ----
    if args_cli.wandb_sweep and run is not None:
        try:
            best_path = os.path.join(log_root_path, log_dir, "checkpoints", "best_agent.pt")
            if os.path.exists(best_path):
                artifact = wandb.Artifact(f"model-{run.id}", type="model")
                artifact.add_file(best_path)
                run.log_artifact(artifact)
                print(f"[INFO] Caricato best_agent artifact su wandb: {best_path}")
            else:
                print(f"[WARN] best_agent.pt non trovato in: {best_path}")
        except Exception as e:
            print(f"[ERROR] Upload artifact fallito: {e}")
        run.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()