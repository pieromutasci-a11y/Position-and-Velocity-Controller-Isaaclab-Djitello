#!/bin/bash
# launch_sweep.sh
#
# Uso: ./launch_sweep.sh <SWEEP_ID> [NUM_ENVS]
# Esempio: ./launch_sweep.sh vitt-politecnico-di-bari/drone-pos-tuning/abc123 512
#
# SWEEP_ID si ottiene lanciando prima:
#   /workspace/isaaclab/_isaac_sim/python.sh -m wandb sweep sweep_v1.yaml
# che stampa qualcosa tipo "wandb: Created sweep with ID: abc123"
# e poi il path completo entity/project/sweep_id.

SWEEP_ID=$1
NUM_ENVS=${2:-512}

if [ -z "$SWEEP_ID" ]; then
    echo "Uso: ./launch_sweep.sh <SWEEP_ID> [NUM_ENVS]"
    echo "Esempio: ./launch_sweep.sh vitt-politecnico-di-bari/drone-pos-tuning/abc123 512"
    exit 1
fi

echo "=========================================="
echo "Lancio WandB Agent per lo Sweep: $SWEEP_ID"
echo "  NUM_ENVS = $NUM_ENVS"
echo "=========================================="

export WANDB_ENTITY=vitt-politecnico-di-bari
export WANDB_PROJECT=drone-pos-tuning
export SKRL_NUM_ENVS=$NUM_ENVS

/workspace/isaaclab/_isaac_sim/python.sh -m wandb agent "$SWEEP_ID"