import argparse
import sys
import os
import math
import torch
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluation skrl: tracking riferimenti casuali per VelController")
parser.add_argument("--num_envs", type=int, default=1,help="Numero di ambienti da analizzare")
parser.add_argument("--num_steps", type=int, default=1000, help="Numero di step di controllo da registrare")
parser.add_argument("--hold_s", type=float, default=2.0, help="Durata di ogni gradino casuale, in secondi")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Project-Favuzzi-Mutasci-VelController-Direct-v0",
    help="Nome del task registrato in gymnasium"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/workspace/isaaclab/logs/skrl/vel_controller/2026-07-09_16-01-03_ppo_torch/checkpoints/best_agent.pt" ,
    help="Percorso assoluto al file del checkpoint di skrl (.pt)"
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="/workspace/project_workspace/Project_Favuzzi_Mutasci/scripts/skrl/result_vel_controller",
    help="Cartella dove salvare i grafici"
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="Seed per il generatore di target casuali (indipendente dal seed di skrl). "
         "Se omesso, viene usato un seed derivato dall'orologio di sistema, quindi ogni "
         "esecuzione produce target diversi."
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaacsim.util.debug_draw import _debug_draw
from isaaclab.utils.math import quat_apply

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import Project_Favuzzi_Mutasci.tasks

def costruisci_freccia(origine, vettore, scala=1.0, lunghezza_testa=0.06, larghezza_testa=0.035):
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

class RandomTargetGenerator:
    def __init__(self, num_envs, device, env_cfg, control_dt, hold_s, seed=None):
        self.num_envs = num_envs
        self.device = device
        self.control_dt = control_dt
        self.hold_steps = max(1, int(round(hold_s / control_dt)))

        self.amps = torch.tensor(
            [env_cfg.max_lin_vel_xy, env_cfg.max_lin_vel_xy, env_cfg.max_lin_vel_z, env_cfg.max_ang_vel_z],
            device=device
        )

        self._rng = torch.Generator(device=device if device != "cpu" else "cpu")
        if seed is not None:
            self._rng.manual_seed(seed)
        else:
            self._rng.seed()

        self.steps_until_change = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_target = torch.zeros(num_envs, 4, device=device)
        self.transition_steps = []

        self._resample_targets(torch.arange(num_envs, device=device))
        self.steps_until_change[:] = self.hold_steps

    def _resample_targets(self, env_ids):
        n = env_ids.numel()
        rand_frac = torch.rand(n, 4, device=self.device, generator=self._rng) * 2.0 - 1.0
        self.current_target[env_ids] = rand_frac * self.amps.unsqueeze(0)

    def get(self, step):
        due = (self.steps_until_change <= 0).nonzero(as_tuple=False).squeeze(-1)
        if due.numel() > 0:
            self._resample_targets(due)
            self.steps_until_change[due] = self.hold_steps
            self.transition_steps.append(step)

        target = self.current_target.clone()
        self.steps_until_change -= 1
        return target

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.num_steps * control_dt + 1.0)
    env_cfg.terminate_su_tilt_eccessivo = False

    print(f"[INFO] Creazione task: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    base_env = env.unwrapped
    device = env.device

    base_env.disable_target_resampling = True
    base_env._hold_duration[:] = 999999.0
    base_env._hold_timer[:] = 0.0

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Caricamento checkpoint da: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    generator = RandomTargetGenerator(
        args_cli.num_envs, device, env_cfg, control_dt, args_cli.hold_s, seed=args_cli.seed
    )
    obs, _ = env.reset()

    storia_target_vel, storia_current_vel = [], []
    draw = _debug_draw.acquire_debug_draw_interface()
    SCALA_VEL_LIN, SCALA_VEL_ANG = 0.5, 0.4

    print(f"[INFO] Avvio simulazione per {args_cli.num_steps} step...")

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            target = generator.get(step)
            base_env._target_vel[:] = target
            base_env._target_mask[:] = 1.0

            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])

            current_vel = torch.cat(
                [base_env.robot.data.root_lin_vel_b, base_env.robot.data.root_ang_vel_b[:, 2].unsqueeze(-1)], dim=-1
            )

            draw.clear_lines()
            posizioni = base_env.robot.data.root_pos_w
            orientamenti = base_env.robot.data.root_quat_w
            target_vel_correnti = base_env._target_vel

            for env_id in range(args_cli.num_envs):
                origine = posizioni[env_id].cpu().tolist()
                origine[2] += 0.15
                quat = orientamenti[env_id]

                vel_target_body = target_vel_correnti[env_id, :3]
                vel_reale_body = current_vel[env_id, :3]
                vel_target_world = quat_apply(quat.unsqueeze(0), vel_target_body.unsqueeze(0)).squeeze(0).cpu().tolist()
                vel_reale_world = quat_apply(quat.unsqueeze(0), vel_reale_body.unsqueeze(0)).squeeze(0).cpu().tolist()

                for (p1, p2) in costruisci_freccia(origine, vel_target_world, scala=SCALA_VEL_LIN):
                    draw.draw_lines([p1], [p2], [(0.0, 1.0, 0.0, 1.0)], [3.0])
                for (p1, p2) in costruisci_freccia(origine, vel_reale_world, scala=SCALA_VEL_LIN):
                    draw.draw_lines([p1], [p2], [(1.0, 0.0, 0.0, 1.0)], [3.0])

                wz_target = target_vel_correnti[env_id, 3].item()
                wz_reale = current_vel[env_id, 3].item()

                for (p1, p2) in costruisci_freccia(origine, [0.0, 0.0, wz_target], scala=SCALA_VEL_ANG):
                    draw.draw_lines([p1], [p2], [(1.0, 1.0, 0.0, 1.0)], [3.0])
                for (p1, p2) in costruisci_freccia(origine, [0.0, 0.0, wz_reale], scala=SCALA_VEL_ANG):
                    draw.draw_lines([p1], [p2], [(0.0, 0.0, 1.0, 1.0)], [3.0])

            storia_target_vel.append(base_env._target_vel.clone().cpu())
            storia_current_vel.append(current_vel.clone().cpu())

            obs, _, _, _, _ = env.step(actions)

    draw.clear_lines()
    env.close()

    storia_target_vel = torch.stack(storia_target_vel)
    storia_current_vel = torch.stack(storia_current_vel)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    nomi_assi_vel = ["vx (m/s)", "vy (m/s)", "vz (m/s)", "wz (rad/s)"]
    tempo_s = torch.arange(args_cli.num_steps).float() * control_dt

    for env_id in range(args_cli.num_envs):
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        fig.suptitle(f"Ambiente {env_id}: Tracking riferimenti casuali (SKRL)")

        for i in range(4):
            axes[i].plot(tempo_s.numpy(), storia_target_vel[:, env_id, i].numpy(), label="target", linestyle="--")
            axes[i].plot(tempo_s.numpy(), storia_current_vel[:, env_id, i].numpy(), label="reale")

            for t_step in generator.transition_steps:
                axes[i].axvline(t_step * control_dt, color="gray", alpha=0.3, linewidth=0.8)

            axes[i].set_ylabel(nomi_assi_vel[i])
            axes[i].legend(loc="upper right")
            axes[i].grid(True)

        axes[-1].set_xlabel("Tempo (s)")
        plt.tight_layout()
        percorso = os.path.join(args_cli.out_dir, f"tracking_random_env_{env_id}.png")
        plt.savefig(percorso)
        plt.close(fig)
        print(f"[INFO] Grafico salvato: {percorso}")

    errore_assoluto = torch.mean(torch.abs(storia_current_vel - storia_target_vel)).item()
    print(f"\n[RISULTATO] Errore medio assoluto: {errore_assoluto:.4f}")

if __name__ == "__main__":
    main()
    simulation_app.close()
