"""Let gear fall onto GearBase axis and report where it settles."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
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

    with torch.inference_mode():
        for t in range(150):
            actions = torch.zeros(env.action_space.shape, device=dev)
            actions[:, 6] = -1.0  # gripper open, arm stays at default pose
            env.step(actions)
            if t % 15 == 0:
                pos = obj.data.root_pos_w[0].cpu().numpy()
                print(f"t={t:3d} gear pos: {pos.round(4)}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
