#!/bin/bash
# launch_sweep.sh
#
# Uso: ./launch_sweep.sh <SWEEP_ID> [NUM_ENVS]
# Esempio: ./launch_sweep.sh pieromutasci-politecnico-di-bari/drone-pos-tuning/abc123 512
#
# SWEEP_ID si ottiene lanciando prima:
#   /workspace/isaaclab/_isaac_sim/python.sh -m wandb sweep sweep_v1.yaml
# che stampa qualcosa tipo "wandb: Created sweep with ID: abc123"
# e poi il path completo entity/project/sweep_id.
#
# NUM_ENVS (opzionale, default 512): viene letto da train_sweep.py tramite
# la variabile d'ambiente SKRL_NUM_ENVS per sovrascrivere env_cfg.scene.num_envs
# senza dover toccare il cfg per ogni run — utile per limitare la VRAM
# durante lo sweep.
SWEEP_ID=$1
NUM_ENVS=${2:-512}
if [ -z "$SWEEP_ID" ]; then
    echo "Uso: ./launch_sweep.sh <SWEEP_ID> [NUM_ENVS]"
    echo "Esempio: ./launch_sweep.sh pieromutasci-politecnico-di-bari/drone-pos-tuning/abc123 512"
    exit 1
fi
echo "=========================================="
echo "Lancio WandB Agent per lo Sweep: $SWEEP_ID"
echo "  NUM_ENVS = $NUM_ENVS"
echo "=========================================="
export WANDB_ENTITY=pieromutasci-politecnico-di-bari
export WANDB_PROJECT=drone-pos-tuning
export SKRL_NUM_ENVS=$NUM_ENVS
/workspace/isaaclab/_isaac_sim/python.sh -m wandb agent "$SWEEP_ID"