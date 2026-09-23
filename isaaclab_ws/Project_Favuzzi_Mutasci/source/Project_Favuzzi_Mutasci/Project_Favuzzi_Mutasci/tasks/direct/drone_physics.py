# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.utils import configclass

@configclass
class DroneInertialCfg:

    enabled: bool = True

    total_mass: float = 0.087

    inertia_diag: tuple[float, float, float] = (1.04e-4, 1.11e-4, 2.07e-4)

    inertia_scale: float = 1.0

    rotor_mass: float = 1.0e-5

    body_name: str = "body"

TELLO_INERTIAL_CFG = DroneInertialCfg()

CRAZYFLIE_INERTIAL_CFG = DroneInertialCfg(enabled=False)

def apply_drone_inertial_props(robot, cfg: DroneInertialCfg) -> dict[str, float]:
    view = robot.root_physx_view
    masses = view.get_masses().clone()
    inertias = view.get_inertias().clone()

    body_idx = robot.find_bodies(cfg.body_name)[0][0]
    n_envs, n_bodies = masses.shape
    rotor_idx = [i for i in range(n_bodies) if i != body_idx]

    if not cfg.enabled:
        return {
            "total_mass": float(masses[0].sum()),
            "ixx": float(inertias[0, body_idx, 0]),
            "iyy": float(inertias[0, body_idx, 4]),
            "izz": float(inertias[0, body_idx, 8]),
        }

    body_mass = cfg.total_mass - len(rotor_idx) * cfg.rotor_mass
    if body_mass <= 0.0:
        raise ValueError(
            f"[drone_physics] rotor_mass={cfg.rotor_mass} troppo grande: le {len(rotor_idx)} "
            f"eliche pesano piu' di total_mass={cfg.total_mass}."
        )

    old_rotor_masses = masses[:, rotor_idx].clone()
    masses[:, body_idx] = body_mass
    masses[:, rotor_idx] = cfg.rotor_mass

    ixx, iyy, izz = (v * cfg.inertia_scale for v in cfg.inertia_diag)
    body_inertia = torch.tensor(
        [ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz], dtype=inertias.dtype
    )
    inertias[:, body_idx, :] = body_inertia

    ratio = (cfg.rotor_mass / old_rotor_masses.clamp(min=1e-12)).unsqueeze(-1)
    inertias[:, rotor_idx, :] = inertias[:, rotor_idx, :] * ratio

    env_ids = torch.arange(n_envs, device="cpu")
    view.set_masses(masses, env_ids)
    view.set_inertias(inertias, env_ids)

    robot.data.default_mass[:] = masses.to(robot.data.default_mass.device)
    robot.data.default_inertia[:] = inertias.to(robot.data.default_inertia.device)

    return {"total_mass": cfg.total_mass, "ixx": ixx, "iyy": iyy, "izz": izz}

def log_inertial_props(tag: str, applied: dict[str, float], cfg: DroneInertialCfg) -> None:
    stato = "TELLO override" if cfg.enabled else "asset nativo (nessun override)"
    print(
        f"[{tag}] proprieta' inerziali: {stato} | "
        f"massa={applied['total_mass'] * 1000.0:.1f} g | "
        f"Ixx={applied['ixx']:.3e}  Iyy={applied['iyy']:.3e}  Izz={applied['izz']:.3e} kg*m^2"
    )
