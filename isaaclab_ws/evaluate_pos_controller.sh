#!/bin/bash

# =============================================================================
# evaluate_pos_controller.sh
# Lancia l'evaluation multi-target del controllore di posizione.
# Il drone insegue target casuali dentro la stanza: quando ne raggiunge uno
# (dist < reach_thr) ne viene generato automaticamente un altro.
# =============================================================================

# -- Checkpoint del CONTROLLORE DI POSIZIONE da valutare -- #
CHECKPOINT="/workspace/isaaclab/logs/skrl/pos_controller/2026-07-10_14-22-00_ppo_torch/checkpoints/best_agent.pt"

# -- Cartella dove salvare i grafici -- #
OUT_DIR="/workspace/project_workspace/Project_Favuzzi_Mutasci/scripts/skrl/result_pos_controller"

# -- Parametri simulazione -- #
NUM_ENVS=1
NUM_STEPS=1000

# -- Soglia distanza per considerare il target raggiunto [m] -- #
REACH_THR=0.2

# =============================================================================

/workspace/isaaclab/isaaclab.sh -p \
    /workspace/project_workspace/Project_Favuzzi_Mutasci/scripts/skrl/evaluate_pos_controller.py \
    --task Template-Project-Favuzzi-Mutasci-PosController-Direct-v0 \
    --checkpoint "$CHECKPOINT" \
    --out_dir "$OUT_DIR" \
    --num_envs $NUM_ENVS \
    --num_steps $NUM_STEPS \
    --reach_thr $REACH_THR