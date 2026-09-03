import torch
import warp as wp
from mushroom_rl.environments.mujoco_warp_envs import Go2Walk

env = Go2Walk(
    num_envs=4, n_substeps=1, n_intermediate_steps=10, use_graph_capture=False
)
mask = torch.ones(4, dtype=torch.bool, device="cuda:0")
env.reset_all(mask)

for i in range(40):
    env.step_all(mask, torch.zeros(4, 12, device="cuda:0"))

qvel = wp.to_torch(env._data_wp.qvel)

print("freejoint lin vel (qvel[0:3]) :", qvel[0, 0:3].tolist())
print("freejoint ang vel (qvel[3:6]) :", qvel[0, 3:6].tolist())
print(
    "base_vel_world ang [:3]       :", env._read_data("base_vel_world")[0, :3].tolist()
)
print(
    "base_vel_world lin [3:]       :", env._read_data("base_vel_world")[0, 3:].tolist()
)
print("base_vel_local  ang [:3]      :", env._base_ang_vel()[0].tolist())
print("base_vel_local  lin [3:]      :", env._base_lin_vel()[0].tolist())

jv = env._joint_vel()
print("joint_vel max abs             :", jv.abs().max().item())
print("prev_joint_vel max abs        :", env._prev_joint_vel.abs().max().item())
print("delta max abs                 :", (jv - env._prev_joint_vel).abs().max().item())
print("dt                            :", env.dt)
