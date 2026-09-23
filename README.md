# Position and Velocity Controller — Isaac Lab / DJI Tello

Controllore di posizione per il drone DJI Tello, addestrato con reinforcement learning
(PPO, tramite [skrl](https://github.com/Toni-SM/skrl)) in **NVIDIA Isaac Lab** e
deployato su drone reale via **ROS 2** (motion capture Vicon). Questo branch (`essential`)
contiene solo il codice e i due checkpoint effettivamente usati, senza log di training,
sweep di iperparametri o asset intermedi.

Questo README riassume l'intero lavoro seguendo la relazione del progetto
*"Reinforcement Learning-Based Position Control for the DJI Tello Drone: Training in
Isaac Lab and Deployment on ROS2"* (Favuzzi, Mutasci — Politecnico di Bari, A.Y.
2025-2026). Per i dettagli completi (formule, figure, risultati) fare riferimento a
quel documento; qui sono riportati architettura, spazi di osservazione/azione e le
reward **con i coefficienti effettivamente in vigore nel codice**.

## Idea di fondo

Il Tello espone come unica interfaccia di controllo un comando RC normalizzato
(roll, pitch, throttle, yaw) che il firmware interpreta come riferimento di velocità
corpo-fissa: non è possibile comandare direttamente spinta o momenti. Qualsiasi
controllore appreso deve quindi vivere sopra il livello di velocità.

La soluzione è una gerarchia a due livelli, addestrata in ordine (prima il livello di
velocità, poi quello di posizione) e **nidificata dentro la stessa simulazione fisica**:

| Livello | Frequenza | Cosa produce |
|---|---|---|
| Simulazione fisica | 100 Hz | integrazione della dinamica |
| Controllore di velocità (basso livello) | 50 Hz | spinta e momenti |
| Controllore di posizione (alto livello) | 25 Hz | riferimento di velocità |

```
(p*, ψ*) --[pos_controller, πθ]--> (vx, vy, vz, ωz) --[vel_controller, πθ_LL congelato]--> (Mx, My, Mz, T)
```

Il punto centrale: il controllore di velocità è un **artefatto solo-simulazione**. Non
viene mai esportato e non ha equivalente sul drone reale, dove il loop di
velocità/assetto è chiuso dal firmware del Tello. Il suo unico scopo è esporre, durante
il training del livello di posizione, una risposta di velocità realistica (con transitorio,
overshoot, ritardo) invece dell'ipotesi ideale "la velocità comandata è raggiunta
istantaneamente" — è questo che rende il controllore di posizione robusto quando, sul
drone reale, il loop di velocità è chiuso dal firmware e non da una rete neurale.
**Solo il controllore di posizione viene deployato sul drone reale.**

## NVIDIA Isaac Lab

Isaac Lab simula migliaia di copie indipendenti dello stesso ambiente in parallelo
sulla GPU (qui: 4096 env), con stato, osservazioni e reward calcolati come operazioni
tensoriali su tutte le istanze insieme — è questo che rende praticabile il training PPO
(milioni di step in poche ore). Il training usa la libreria **skrl** con l'algoritmo
**PPO** (Proximal Policy Optimization). Non essendo disponibile un modello CAD/URDF
del Tello, la geometria visiva/di collisione è quella di un Crazyflie esistente, ma le
proprietà che guidano la dinamica (massa ≈87g, tensore d'inerzia, thrust-to-weight,
scala dei momenti) sono impostate sui valori reali del Tello (vedi
`drone_physics.py`).

---

## 1. Controllore di velocità (basso livello)

File: [vel_controller_env.py](isaaclab_ws/Project_Favuzzi_Mutasci/source/Project_Favuzzi_Mutasci/Project_Favuzzi_Mutasci/tasks/direct/vel_controller/vel_controller_env.py) · [vel_controller_env_cfg.py](isaaclab_ws/Project_Favuzzi_Mutasci/source/Project_Favuzzi_Mutasci/Project_Favuzzi_Mutasci/tasks/direct/vel_controller/vel_controller_env_cfg.py)

Dato un riferimento di velocità corpo-fissa, impara a produrre spinta e momenti che lo
tracciano. Vola in spazio libero (nessuna stanza, nessun ostacolo).

### Osservazione (17 componenti, in ordine)

| Blocco | Dim. | Contenuto |
|---|---|---|
| velocità lineare corpo | 3 | `root_lin_vel_b` |
| velocità angolare corpo | 3 | `root_ang_vel_b` |
| proiezione gravità corpo | 3 | vettore unitario, codifica l'assetto senza angoli |
| azione precedente | 4 | l'output della rete al passo precedente (memoria per la smoothness) |
| riferimento di velocità target | 4 | `(vx*, vy*, vz*, ωz*)` |

### Azione (4 componenti, in [-1, 1])

| Comp. | Significato | Comando fisico |
|---|---|---|
| `a0` | spinta collettiva normalizzata | `T = 1.8·mg·(a0+1)/2 ∈ [0, 1.8·mg]` |
| `a1` | momento di roll | `Mx = 0.021·a1` N·m |
| `a2` | momento di pitch | `My = 0.021·a2` N·m |
| `a3` | momento di yaw | `Mz = 0.0085·a3` N·m (metà autorità di roll/pitch) |

### Reward (5 termini, `vel_controller_env.py:_get_rewards`)

`r = rtrack + rrate + rang + rtilt + rdie`, i primi 4 continui (moltiplicati per `Δt`),
l'ultimo impulsivo.

| Termine | Formula | Coefficiente |
|---|---|---|
| Tracking velocità | `-scale·Δt·Σ((v-v*)/v_max)²` | `vel_error_reward_scale = -7.0` |
| Action rate (con curriculum) | `crate(k)·Δt·‖a-a_prev‖²`, `crate` da -0.01 a -0.1 in 40000 step | `action_rate_reward_scale_start/end = -0.01 / -0.1` |
| Velocità angolare indesiderata (roll/pitch) | `-scale·Δt·(ωx²+ωy²)` | `unwanted_ang_vel_reward_scale = -0.05` |
| Tilt | `-scale·Δt·max(0, θ-35°)²` | `tilt_reward_scale = -2.0` (zona libera fino a 35°) |
| Perdita d'assetto (impulsivo) | `-penalty` se `θ > 65°` | `died_penalty = -10.0` |

### Generazione dei target e curriculum

Ad ogni cambio di riferimento viene scelta a caso una delle 5 modalità (multi-asse
casuale, maschera parziale, asse singolo, combinata aggressiva, hover/piccola
correzione), con probabilità che dipendono dal **livello di curriculum** corrente
dell'environment (0→4, per-env, mai retrocesso):

| Livello | Probabilità modi (0,1,2,3,4) | Hold [s] | Ampiezza minima modo 3 |
|---|---|---|---|
| 0 | 0.00 0.00 0.95 0.00 0.05 | 3.5–5.0 | 0.30 |
| 1 | 0.475 0.00 0.475 0.00 0.05 | 3.0–4.0 | 0.30 |
| 2 | 0.40 0.15 0.30 0.10 0.05 | 2.0–3.0 | 0.40 |
| 3 | 0.35 0.15 0.20 0.25 0.05 | 1.5–2.5 | 0.50 |
| 4 | 0.35 0.15 0.15 0.30 0.05 | 1.0–2.5 | 0.60 |

Promozione al livello successivo dopo **5 hold period consecutivi** con errore medio
normalizzato sotto soglia (`curriculum_soglia_successo = 0.02`); un solo fallimento
azzera il contatore. Episodio: 10s (500 step a 50Hz), termina anticipatamente solo per
tilt > 65°.

---

## 2. Controllore di posizione (alto livello)

File: [pos_controller_env.py](isaaclab_ws/Project_Favuzzi_Mutasci/source/Project_Favuzzi_Mutasci/Project_Favuzzi_Mutasci/tasks/direct/pos_controller/pos_controller_env.py) · [pos_controller_env_cfg.py](isaaclab_ws/Project_Favuzzi_Mutasci/source/Project_Favuzzi_Mutasci/Project_Favuzzi_Mutasci/tasks/direct/pos_controller/pos_controller_env_cfg.py)

Osserva la posa del drone e la coda di waypoint da raggiungere, e decide istante per
istante quale riferimento di velocità chiedere al livello inferiore (**congelato**,
pesi non aggiornati durante questo training). Vola dentro una stanza randomizzata.

### Ambiente

Ad ogni reset: stanza a parallelepipedo con i 4 semi-assi orizzontali campionati
**indipendentemente** in [1,4] m (così l'origine non è mai al centro di una stanza
simmetrica) e soffitto in [1,4] m; pavimento fisso a 0.1 m. Spawn/target ristretti
all'80% della stanza. Con probabilità 0.3 lo spawn è "da terra" (quota in [0.12, 0.25]
m), per allenare anche il decollo.

| Costante | Valore |
|---|---|
| Env paralleli | 4096 |
| Episodio | 30 s (750 step a 25 Hz) |
| Waypoint per coda | 4 |
| Orizzonte di preview | 4 (1 feedback + 3 feedforward) |
| Soglia di raggiungimento | 0.15 m posizione, 0.20 rad yaw, per 1.2 s consecutivi |
| Terminazione | fuori stanza, o tilt > 65° |
| Leaky integrator | τ = 5 s, saturato a ±1 |

### Integrazione del controllore congelato

Dal checkpoint del vel_controller si estraggono pesi, architettura (ricostruita dalle
dimensioni dei pesi) e le statistiche di normalizzazione (media/varianza) — i 17 input
vengono standardizzati con le stesse statistiche accumulate durante il suo training
**prima** di ogni chiamata (`õ = (o-μ)/√(σ²+ε)`). L'azione del pos_controller viene
clampata a [-1,1] e scalata per `(1.0, 1.0, 1.0, 1.5)` (lo stesso range con cui è stato
addestrato il vel_controller) prima di diventare il blocco "riferimento richiesto"
dell'input del livello congelato: l'ordine e la scala di quell'input non sono
opzionali, sono un vincolo di compatibilità col checkpoint caricato.

### Il compito: coda di waypoint

4 target fissati all'inizio dell'episodio (mai modificati dopo), ciascuno posizione+yaw
completo, da raggiungere in ordine; raggiunto il 4°, il drone resta in hover fino a
fine episodio. Ad ogni reset viene scelta una **modalità di riferimento**:

| Modalità | Frequenza | Coda risultante |
|---|---|---|
| variabile | 80% | 4 pose distinte da percorrere in sequenza |
| singolo | 14% | 1 pose ripetuta in tutti i 4 slot (isola il regime steady-state) |
| hover | 6% | la pose di spawn ripetuta in tutti i 4 slot |

Dentro il 35% degli episodi "variabile" (≈28% del totale) la coda casuale è sostituita
dalla **sequenza canonica**: avanti (+x), indietro (-x), destra (-y), sinistra (+y),
ciascuno spostato di 50cm–distanza al muro rispetto allo spawn.

### La maschera Degrees-of-Freedom (DOF mask)

Ad ogni reset viene scelto con pari probabilità uno dei due **regimi di moto**:

| Regime | vx | vy | vz | ωz | Frequenza |
|---|---|---|---|---|---|
| omnidirezionale (`full`) | ✓ | ✓ | ✓ | ✓ | 50% |
| uniciclo | ✓ | ✗ | ✓ | ✓ | 50% |

Il regime uniciclo esiste per simulare la "cecità laterale" di una camera fissa
frontale: non potendo vedere di lato/dietro, il drone non deve tradurre lateralmente,
solo ruotare e poi avanzare, come un veicolo non-olonomo. **La maschera non tocca mai
lo spazio delle azioni** (nessun clip/zero forzato): il comportamento uniciclo è
interamente imparato tramite la reward. La maschera è un input dell'osservazione — la
rete impara così **due politiche in un solo set di pesi**, indicizzate dalla maschera.

### Osservazione (52 componenti, raggruppate)

| Gruppo | Dim. | Domanda a cui risponde |
|---|---|---|
| posizione nella stanza | 3 | dove sono |
| yaw (sin, cos) | 2 | come sono orientato |
| velocità lineare corpo | 3 | come mi sto muovendo |
| velocità angolare corpo | 3 | come sto ruotando |
| proiezione gravità corpo | 3 | quanto sono inclinato |
| comando precedente | 4 | cosa ho appena chiesto |
| errore + preview coda | 20 | dove devo andare, ora e dopo |
| distanze dai 6 muri | 6 | quanto spazio ho intorno |
| maschera DOF | 4 | come mi è permesso muovermi |
| integrale d'errore | 4 | che errore sto portando con me |
| **Totale** | **52** | |

Posizione/errore/preview sono nel frame della stanza; velocità/gravità sono nel frame
corpo (coerenti con l'output, anch'esso corpo-fisso). Yaw e ogni errore angolare sono
codificati come (sin, cos), non come angolo grezzo, per evitare la discontinuità a
±π. Il blocco di preview (20 = 4 blocchi da 5) contiene 1 blocco di **feedback** (errore
verso il waypoint attivo) + 3 blocchi **feedforward** (spostamento tra waypoint futuri
consecutivi, non errori: restano stabili mentre il drone si avvicina al target attivo,
cambiano solo quando la coda avanza) — è quello che permette alla policy di anticipare.

### Azione (4 componenti, in [-1,1])

Stessa forma del vel_controller: prodotta dalla policy gaussiana, clampata a [-1,1] e
scalata per `(1.0, 1.0, 1.0, 1.5)` prima di diventare il riferimento per il livello
congelato.

### Reward (16 termini, `pos_controller_env.py:_get_rewards` — coefficienti attuali)

Continui (moltiplicati per `Δt = 0.04s`, salvo dove indicato) + impulsivi (eventi
singoli, non scalati per `Δt`).

| Termine | Formula (sintetica) | Coefficiente attuale |
|---|---|---|
| Avvicinamento posizione | `+scale·e^(-β·d)` | `rew_scale_position_approach=25.0`, `reward_exp_beta=0.5` |
| Precisione posizione | `+scale·e^(-β·d)` | `rew_scale_position_prec=35.0`, `reward_exp_beta_prec=8.0` |
| Errore yaw (con gate distanza in uniciclo) | `-scale·eψ·w_gate` | `rew_scale_yaw_error=-8.0`, gate a `yaw_gate_dist_uniciclo=0.18` m |
| Precisione yaw (solo full) | `+scale·e^(-β·eψ)` | `rew_scale_yaw_prec=8.0`, `reward_exp_beta_yaw_prec=7.0` |
| Chattering yaw (peso cresce vicino al target in full) | `-scale·ωz²·w_prox` | `rew_scale_reg_ang_vel_wz=-0.03` |
| Reg. velocità ang. roll/pitch | `-scale·(ωx²+ωy²)` | `rew_scale_reg_ang_vel_xy=-0.00` (disattivato) |
| Comandi aggressivi (esclude wz in uniciclo) | `-scale·Σci·ai²` | `rew_scale_aggressive_cmd=-8.0` |
| Smoothness azioni | `-scale·‖Δa‖²` | `rew_scale_action_smoothness=-25.0` |
| Vivo (costante) | `+scale` | `rew_scale_alive=0.2` |
| vy richiesta (solo uniciclo) | `-scale·avy²` | `rew_scale_vy_penalty_uniciclo=-300.0` |
| vy reale (solo uniciclo) | `-scale·vyb²` | `rew_scale_vy_real_penalty_uniciclo=-150.0` |
| Retromarcia (solo uniciclo) | `-scale·max(0,-avx)²` | `rew_scale_reverse_vx=-70.0` |
| Frenata in approccio (solo uniciclo, attiva sotto 0.5m) | `-scale·w_brake·(vxb²+vzb²)` | `rew_scale_approach_brake=-15.0`, `approach_brake_dist=0.5` |
| Waypoint raggiunto (impulsivo) | `+bonus` | `rew_scale_target_reached=80.0` |
| Uscita dai limiti (impulsivo) | `-penalty` | `oob_reward=-50.0` |
| Tilt eccessivo (impulsivo) | `-penalty` | `tilt_death_reward=-50.0` |

I 5 coefficienti sopra (`rew_scale_position_prec`, `reward_exp_beta_prec`,
`rew_scale_yaw_prec`, `reward_exp_beta_yaw_prec`, `rew_scale_action_smoothness`) sono
allineati sia al checkpoint effettivamente deployato in ROS
(`pos_controller_checkpoints/2026-07-31_13-02-43_ppo_torch`) sia alla relazione di
progetto; tutti gli altri termini coincidevano già.

I pesi `obj_w_*` in fondo alla cfg **non** fanno parte di questa reward: alimentano
solo un obiettivo composito loggato per il tuning degli iperparametri via sweep wandb
(strumentazione non presente in questo branch), non influenzano la policy addestrata.

---

## 3. Deployment su drone reale (ROS 2)

Solo il controllore di posizione viene deployato. La pipeline (non in questo branch,
progetto ROS 2 separato) è organizzata come 4 nodi indipendenti che comunicano solo via
topic:

```
Vicon (motion capture) ──> target_handler ────┐
                       └──> observation_handler ──> policy_handler ──> vel_command_handler ──> Tello (RC/UDP)
```

- **`target_handler`** genera/avanza la coda di waypoint (stessi 5 modi + criterio di
  raggiungimento del training).
- **`observation_handler`** ricostruisce l'osservazione a 52 componenti dai dati Vicon
  (velocità per differenze finite filtrate, velocità angolare dalla cinematica di
  Eulero, proiezione gravità, integrale leaky) — **stesso ordine esatto** usato in
  simulazione (verificato: vedi sezione precedente).
- **`policy_handler`** carica il checkpoint (`best_agent.pt` + statistiche di
  normalizzazione) ed esegue l'inferenza a 25 Hz.
- **`vel_command_handler`** è l'unico nodo con accesso al drone: scala l'azione, la
  converte in comando RC, e applica i meccanismi di sicurezza (watchdog su
  osservazioni/comandi scaduti → hover, failsafe batteria, atterraggio automatico su
  perdita prolungata del tracking, arming esplicito pre-decollo).

---

## Struttura di questo branch

```
.
├── run.sh                          # avvio del container Docker Isaac Lab
├── exec.sh                         # entra nel container già in esecuzione
├── pos_controller_checkpoints/2026-07-31_13-02-43_ppo_torch/
│   └── checkpoints/best_agent.pt   # checkpoint deployato per l'inferenza reale (ROS)
├── vel_controller_checkpoints/2026-07-30_08-29-15_ppo_torch/
│   └── checkpoints/best_agent.pt   # frozen low-level policy, caricato da pos_controller_env_cfg.py
└── isaaclab_ws/
    ├── train_pos_controller.sh / train_vel_controller.sh
    ├── evaluate_pos_controller.sh / evaluate_vel_controller_*.sh
    └── Project_Favuzzi_Mutasci/            # pacchetto Isaac Lab (extension)
        ├── scripts/
        │   ├── list_envs.py, random_agent.py, zero_agent.py     # utility di verifica ambiente
        │   └── skrl/
        │       ├── train.py, play.py                            # training/replay skrl generici
        │       └── evaluate_pos_waypoint.py,
        │           evaluate_vel_controller_{constant,custom,variable}_ref.py
        └── source/Project_Favuzzi_Mutasci/
            ├── setup.py, pyproject.toml, config/, docs/          # metadata del pacchetto
            └── Project_Favuzzi_Mutasci/
                └── tasks/direct/
                    ├── drone_physics.py                          # override inerziale Crazyflie->Tello
                    ├── vel_controller/  (env, cfg, agents/skrl_ppo_cfg.yaml)
                    └── pos_controller/  (env, cfg, agents/skrl_ppo_cfg.yaml)
```

I due checkpoint sono gli unici presenti: sono quelli effettivamente caricati a runtime
(`low_level_policy_path` in `pos_controller_env_cfg.py` per il vel_controller; il
checkpoint deployato via ROS per il pos_controller), ciascuno con il proprio
`params/{agent,env}.yaml` originale.

## Come usarlo

```bash
./run.sh                              # avvia il container Docker con Isaac Lab
./exec.sh                             # entra in un secondo terminale nel container già attivo

# dentro il container
python scripts/list_envs.py           # verifica che i task siano registrati
python scripts/skrl/train.py --task=Template-Project-Favuzzi-Mutasci-VelController-Direct-v0
python scripts/skrl/train.py --task=Template-Project-Favuzzi-Mutasci-PosController-Direct-v0
python scripts/skrl/play.py  --task=Template-Project-Favuzzi-Mutasci-PosController-Direct-v0 --checkpoint <path/best_agent.pt>

python scripts/skrl/evaluate_pos_waypoint.py --checkpoint pos_controller_checkpoints/2026-07-31_13-02-43_ppo_torch/checkpoints/best_agent.pt \
    --mode variabile --mask full
```

I wrapper `.sh` in `isaaclab_ws/` (`train_*.sh`, `evaluate_*.sh`) fissano gli argomenti
più comuni per i due task.

## Riferimento

Favuzzi V., Mutasci P., *"Reinforcement Learning-Based Position Control for the DJI
Tello Drone: Training in Isaac Lab and Deployment on ROS2"*, Project Work — Mobile
Robotics, Politecnico di Bari, A.Y. 2025-2026.
