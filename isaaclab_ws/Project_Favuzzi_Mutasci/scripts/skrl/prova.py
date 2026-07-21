import torch
ckpt = torch.load("/workspace/project_workspace/Project_Favuzzi_Mutasci/source/Project_Favuzzi_Mutasci/param_optimization/logs/skrl/pos_controller/2026-07-18_01-31-16_ppo_torch_POS_SWEEP_V2/jolly-sweep-18/checkpoints/best_agent.pt", map_location="cpu")
print(list(ckpt["policy"].keys()))