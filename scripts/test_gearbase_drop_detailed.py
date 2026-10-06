"""Trace gear position every single physics step while it falls onto the GearBase axis."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import shelf_retrieval.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    obj = env.unwrapped.scene["object"]
    dev = env.unwrapped.device

    prev_z = None
    with torch.inference_mode():
        for t in range(args_cli.steps):
            actions = torch.zeros(env.action_space.shape, device=dev)
            actions[:, 6] = -1.0
            env.step(actions)
            pos = obj.data.root_pos_w[0].cpu().numpy()
            vel = obj.data.root_lin_vel_w[0].cpu().numpy()
            flag = ""
            if prev_z is not None and abs(pos[1] - 0.0) > 0.005 and prev_z - pos[2] < 0.001:
                flag = "  <-- settled here, y already off-axis"
            print(f"t={t:3d} pos=[{pos[0]:+.4f} {pos[1]:+.4f} {pos[2]:+.4f}] "
                  f"vel=[{vel[0]:+.3f} {vel[1]:+.3f} {vel[2]:+.3f}]{flag}", flush=True)
            prev_z = pos[2]

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
