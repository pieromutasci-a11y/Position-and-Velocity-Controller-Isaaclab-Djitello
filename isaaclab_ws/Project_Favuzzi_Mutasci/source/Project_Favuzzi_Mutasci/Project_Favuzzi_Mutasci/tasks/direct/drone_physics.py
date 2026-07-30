# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# ===================================================================================
# PROPRIETA' INERZIALI DEL DRONE — OVERRIDE CRAZYFLIE -> DJI/RYZE TELLO
# ===================================================================================
#
# PERCHE' QUESTO MODULO ESISTE
# ----------------------------
# La scena continua a usare l'asset USD del Crazyflie (cf2x.usd) per la GEOMETRIA
# (mesh, corpi, giunti dei rotori), ma le sue proprieta' INERZIALI sono quelle di un
# drone da 27 g, mentre il drone reale su cui va deployata la policy e' un DJI/Ryze
# Tello da ~87 g. Massa e tensore d'inerzia diversi = dinamica diversa = policy che
# non trasferisce. Qui sovrascriviamo massa e inerzia a runtime, senza toccare l'URDF
# / USD e senza importare un nuovo asset.
#
# PERCHE' A RUNTIME E NON VIA `spawn.mass_props`
# ----------------------------------------------
# `UsdFileCfg.mass_props` (sim_utils.MassPropertiesCfg) NON va bene per due motivi:
#   1. e' applicata con `apply_nested`: assegnerebbe la STESSA massa a ogni prim con
#      UsdPhysics.MassAPI sotto il robot (corpo + 4 eliche), quindi mass=0.087
#      darebbe un drone da 5 x 0.087 = 0.435 kg, non 0.087 kg;
#   2. espone solo `mass` e `density`, NON il tensore d'inerzia. Con la sola massa,
#      PhysX riscalerebbe l'inerzia del Crazyflie proporzionalmente, mantenendone la
#      GEOMETRIA (rapporti Ixx:Iyy:Izz e "raggio giratore" del Crazyflie), che non e'
#      quella del Tello. La sola densita' e' ancora peggio: viene ignorata quando la
#      massa e' definita esplicitamente (la massa ha precedenza in UsdPhysics.MassAPI),
#      e quando non lo e' viene applicata all'approssimazione di collisione del mesh,
#      quindi non e' un modo controllabile di fissare 87 g.
# Percio' scriviamo direttamente nella vista tensoriale PhysX (`root_physx_view`), che
# espone sia `set_masses` sia `set_inertias` per-corpo.
#
# DOVE VA LA MASSA
# ----------------
# In questi env spinta e momenti NON sono generati dai rotori: sono un wrench lumped
# applicato al link `body` (vedi `set_external_force_and_torque(..., body_ids=self._body_id)`
# in vel_controller_env / pos_controller_env). I 4 link elica sono puramente
# cosmetici (girano a velocita' di giunto costante per l'effetto visivo). Quindi la
# scelta corretta e' concentrare tutta la massa del veicolo sul link `body` e lasciare
# alle eliche una massa trascurabile: cosi' il tensore che assegniamo al `body` E'
# esattamente il tensore d'inerzia dell'intero veicolo, senza doppi conteggi e senza
# contributi di Steiner delle eliche da sottrarre.
#
# DA DOVE VENGONO I NUMERI DEL TELLO
# ----------------------------------
# Massa: 87 g (valore misurato dall'utente; la scheda tecnica Ryze dichiara 80 g con
# eliche e batteria — se pesi il TUO esemplare, metti quel valore in `total_mass`).
#
# Inertia: il Tello non pubblica il tensore d'inerzia, quindi lo stimiamo con il
# modello piu' onesto e riproducibile possibile — parallelepipedo pieno OMOGENEO con
# le dimensioni esterne del velivolo (98 x 92.5 x 41 mm):
#
#     Ixx = m/12 * (b^2 + c^2)
#     Iyy = m/12 * (a^2 + c^2)
#     Izz = m/12 * (a^2 + b^2)          con a=x=0.098, b=y=0.0925, c=z=0.041
#
# che da' (m = 0.087 kg):
#
#     Ixx = 7.422e-5,  Iyy = 8.182e-5,  Izz = 1.3166e-4   [kg m^2]
#
# Il modello omogeneo tende a SOVRASTIMARE un po' l'inerzia (la massa reale e'
# concentrata al centro: batteria + PCB), soprattutto su Izz. Se in futuro fai un test
# di identificazione (es. pendolo bifilare) sostituisci i tre numeri in
# `TELLO_INERTIAL_CFG`: e' l'unica cosa da cambiare. `inertia_scale` permette di
# smussare globalmente la stima senza toccare i rapporti fra gli assi.
#
# ATTENZIONE — QUESTO MODULO NON BASTA PER IL SIM-TO-REAL
# -------------------------------------------------------
# Cambiare massa e inerzia cambia la DINAMICA, ma non ricalibra da solo gli attuatori.
# Vedi le note su `thrust_to_weight` e `moment_scale` in fondo al file.
# ===================================================================================

from __future__ import annotations

import torch

from isaaclab.utils import configclass


@configclass
class DroneInertialCfg:
    """Override delle proprieta' inerziali del drone, applicato dopo lo spawn."""

    enabled: bool = True
    """Se False la funzione non tocca nulla: l'asset resta il Crazyflie originale
    (utile per rieseguire vecchi checkpoint con la dinamica con cui erano addestrati)."""

    total_mass: float = 0.087
    """Massa TOTALE del veicolo in kg (corpo + eliche + batteria)."""

    inertia_diag: tuple[float, float, float] = (1.04e-4, 1.11e-4, 2.07e-4)
    """Diagonale (Ixx, Iyy, Izz) del tensore d'inerzia del veicolo intero, in kg*m^2,
    espressa negli assi corpo e rispetto al centro di massa. I termini fuori diagonale
    sono assunti nulli (velivolo simmetrico rispetto ai piani xz e yz).

    Stima a MASSE CONCENTRATE (non piu' box pieno omogeneo): batteria + PCB come slab
    98x92.5x40mm al centro, + 4 masse puntiformi da 7.5 g (motore+elica+braccio)
    posizionate a (+-0.05, +-0.05, 0) m, la posizione ESATTA dei motori nell'URDF
    (tello.urdf, joint m*_joint). Il box pieno omogeneo sottostimava l'inerzia
    ignorando che ~30 g di massa (i 4 gruppi motore) stanno a 50mm dal centro invece
    che spalmati nel corpo; il box "ingombro eliche" (180x180x50) la sovrastimava
    all'opposto, trattando anche l'aria intorno alle pale come massa piena.
    Sensitivity: motore 6g -> (9.45e-5, 1.02e-4, 1.86e-4); motore 9g -> (1.14e-4,
    1.21e-4, 2.28e-4). Usa `inertia_scale` per esplorare questo range."""

    inertia_scale: float = 1.0
    """Fattore moltiplicativo globale su `inertia_diag`. Serve per fare sensitivity
    analysis / domain randomization grossolana sulla stima d'inerzia senza cambiare i
    rapporti fra gli assi."""

    rotor_mass: float = 1.0e-5
    """Massa (kg) assegnata a CIASCUN link elica. Deve essere trascurabile rispetto a
    `total_mass` ma > 0: PhysX non accetta corpi dinamici a massa nulla. Con 1e-5 kg le
    4 eliche pesano insieme 4e-5 kg, cioe' lo 0.046% del totale."""

    body_name: str = "body"
    """Nome del link che porta la massa e su cui gli env applicano il wrench."""


TELLO_INERTIAL_CFG = DroneInertialCfg()
"""Proprieta' inerziali del DJI/Ryze Tello (vedi derivazione in testa al file)."""

CRAZYFLIE_INERTIAL_CFG = DroneInertialCfg(enabled=False)
"""Nessun override: lascia le proprieta' native dell'asset cf2x.usd."""


def apply_drone_inertial_props(robot, cfg: DroneInertialCfg) -> dict[str, float]:
    """Sovrascrive massa e tensore d'inerzia del drone nella simulazione PhysX.

    Va chiamata DOPO `super().__init__()` dell'env (cioe' dopo che la scena e' stata
    creata e la sim resettata): prima di quel momento `robot.root_physx_view` non
    esiste ancora.

    Oltre alla vista PhysX aggiorna anche `robot.data.default_mass` /
    `robot.data.default_inertia`, cosi' eventuali event term di domain randomization
    (che randomizzano *a partire dai default*) partono dai valori del Tello e non da
    quelli del Crazyflie.

    Args:
        robot: l'`Articulation` del drone.
        cfg: le proprieta' da applicare.

    Returns:
        Un dizionario con i valori effettivamente applicati (massa totale e inerzia),
        utile per il logging. Se `cfg.enabled` e' False riporta i valori nativi
        dell'asset, senza modificarli.
    """
    view = robot.root_physx_view
    # le API tensoriali di massa/inerzia di PhysX lavorano su CPU
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

    # -- masse: tutto sul corpo, briciole sulle eliche --
    # Le eliche vengono riscalate PRIMA di leggerne l'inerzia, cosi' il rapporto qui
    # sotto usa ancora la massa originale dell'asset.
    old_rotor_masses = masses[:, rotor_idx].clone()
    masses[:, body_idx] = body_mass
    masses[:, rotor_idx] = cfg.rotor_mass

    # -- inerzia del corpo: tensore diagonale del veicolo intero --
    ixx, iyy, izz = (v * cfg.inertia_scale for v in cfg.inertia_diag)
    body_inertia = torch.tensor(
        [ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz], dtype=inertias.dtype
    )
    inertias[:, body_idx, :] = body_inertia

    # -- inerzia delle eliche: stessa geometria, riscalata col rapporto delle masse --
    # (l'inerzia di un corpo rigido e' lineare nella massa a geometria costante)
    ratio = (cfg.rotor_mass / old_rotor_masses.clamp(min=1e-12)).unsqueeze(-1)
    inertias[:, rotor_idx, :] = inertias[:, rotor_idx, :] * ratio

    # stessa convenzione degli event term di Isaac Lab: indici su CPU
    env_ids = torch.arange(n_envs, device="cpu")
    view.set_masses(masses, env_ids)
    view.set_inertias(inertias, env_ids)

    # allinea i default usati dall'eventuale domain randomization
    robot.data.default_mass[:] = masses.to(robot.data.default_mass.device)
    robot.data.default_inertia[:] = inertias.to(robot.data.default_inertia.device)

    return {"total_mass": cfg.total_mass, "ixx": ixx, "iyy": iyy, "izz": izz}


def log_inertial_props(tag: str, applied: dict[str, float], cfg: DroneInertialCfg) -> None:
    """Stampa un riepilogo di cosa e' stato applicato, cosi' non resta il dubbio di
    aver addestrato con la dinamica sbagliata guardando solo i log di training."""
    stato = "TELLO override" if cfg.enabled else "asset nativo (nessun override)"
    print(
        f"[{tag}] proprieta' inerziali: {stato} | "
        f"massa={applied['total_mass'] * 1000.0:.1f} g | "
        f"Ixx={applied['ixx']:.3e}  Iyy={applied['iyy']:.3e}  Izz={applied['izz']:.3e} kg*m^2"
    )


# ===================================================================================
# NOTE DI CALIBRAZIONE — LEGGERE PRIMA DI RIADDESTRARE
# ===================================================================================
#
# 1) `thrust_to_weight`: VALORE FINALE = 1.8 (era 1.9, tarato sul Crazyflie).
#    Gli env calcolano la spinta come
#        thrust = thrust_to_weight * (massa * g) * (a0 + 1)/2
#    quindi la spinta scala automaticamente con la massa nuova e l'azione di hover
#    resta la stessa; 1.9 pero' era il rapporto spinta/peso del CRAZYFLIE. Il Tello ha
#    T/W teorico ~2.0 a batteria carica (motori 8520, spinta statica ~0.43 N/motore,
#    W=0.853 N), ma una 1S LiPo sotto carico e a meta' scarica e' piu' vicino a 1.7-1.8:
#    1.8 e' una stima leggermente conservativa (sbagliare per difetto e' il verso
#    sicuro: una policy che chiede meno spinta di quanta il drone ne abbia vola, il
#    contrario cade).
#
# 2) `moment_scale`: VALORE FINALE = (0.021, 0.021, 0.0085) per (roll, pitch, yaw),
#    NON piu' uno scalare unico (era 0.01 su tutti e tre gli assi, tarato sul
#    Crazyflie e comunque gia' sovrastimato: superava anche il massimo assoluto del
#    Crazyflie del 22%).
#
#    Braccio: d = 0.05 m, letto DIRETTAMENTE dall'URDF (tello.urdf, origin dei 4
#    joint motore, "(+-0.05, +-0.05, 0)"), non piu' una stima geometrica.
#
#    Roll/pitch — coppia HOVER-PRESERVING, non il massimo assoluto: in sim spinta e
#    momento sono applicati come wrench disaccoppiato, ma nella realta' la coppia si
#    paga in margine di spinta (un motore non puo' scendere sotto zero). Il valore
#    fisicamente sensato e' quindi quello ottenibile SENZA perdere quota:
#        hover/motore = W/4 = 0.213 N
#        Delta = spinta_max/motore - hover/motore = 0.213 N (con T/W=2.0 a pieno)
#        tau_roll = 2 * Delta * d = 2 * 0.213 * 0.05 = 0.0213 N*m
#    (il massimo assoluto, raggiungibile solo rinunciando alla quota, sarebbe 0.0427).
#
#    Yaw — meta' di roll/pitch: la coppia di imbardata viene dalla reazione di
#    trascinamento delle eliche (drag torque), non dalla spinta differenziale; e' un
#    meccanismo fisicamente diverso e piu' debole. Applicare lo stesso moment_scale
#    a tutti e tre gli assi (come faceva il codice originale) raddoppia l'autorita' in
#    yaw rispetto al drone reale — proprio l'asse critico per il task uniciclo.
#
#    Effetto sull'accelerazione angolare massima (alpha = tau/I):
#        Crazyflie (moment_scale=0.01 uniforme):  0.01/1.4e-5    ~ 714 rad/s^2
#        Tello (moment_scale vettoriale sopra):   0.0213/1.04e-4 ~ 205 rad/s^2 (roll/pitch)
#                                                  0.0085/2.07e-4 ~  41 rad/s^2 (yaw)
#    Il drone sara' visibilmente piu' pigro: aspettati di dover ritarare
#    rew_scale_aggressive_cmd / rew_scale_action_smoothness, tarati sulla reattivita'
#    (eccessiva) del Crazyflie.
#
#    Questi tre numeri hanno un'incertezza onesta di ~+-20% (spinta motore e sag
#    batteria non misurati direttamente): se vuoi robustezza al sim-to-real invece di
#    scommettere sul valore centrale, randomizzali per-env ad ogni reset invece di
#    tenerli fissi.
#
# 3) ORDINE DI RIADDESTRAMENTO — obbligatorio.
#    `pos_controller` carica una policy di velocita' CONGELATA
#    (cfg.low_level_policy_path) addestrata sulla dinamica del Crazyflie. Cambiando
#    massa e inerzia quella policy non e' piu' valida. Bisogna:
#        a. riaddestrare `vel_controller` con le proprieta' del Tello;
#        b. aggiornare `low_level_policy_path` col nuovo checkpoint;
#        c. riaddestrare `pos_controller`.
#    I checkpoint esistenti in pos_controller_checkpoints/ e vel_controller_checkpoints/
#    restano validi SOLO con `enabled=False`.
#
# 4) COSA ARRIVA DAVVERO SUL TELLO.
#    L'SDK del Tello accetta comandi `rc a b c d` (rollio/beccheggio/quota/imbardata
#    normalizzati), non spinta e momenti: sul drone reale il livello basso e' il
#    firmware DJI. Quindi cio' che deployerai e' l'uscita di `pos_controller` (i
#    riferimenti di velocita'), mentre `vel_controller` in simulazione fa da SURROGATO
#    del loop interno del Tello. Massa e inerzia corrette servono a rendere quel
#    surrogato credibile, ma il match finale dipende da quanto la risposta in velocita'
#    ad anello chiuso della sim somiglia a quella vera del Tello (ritardo, tempo di
#    salita, sovraelongazione): vale la pena misurarla con qualche step di velocita'
#    sul drone reale e tarare di conseguenza.
# ===================================================================================
