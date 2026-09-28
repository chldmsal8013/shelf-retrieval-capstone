"""Probe: how far does the EE move per unit action for each arm joint?"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--disable_fabric", action="store_true", default=False)
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

NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
DELTA = 0.3
SETTLE = 80


def run(env, arm_action):
    with torch.inference_mode():
        env.reset()
    dev = env.unwrapped.device
    a = torch.zeros(env.action_space.shape, device=dev)
    a[:, :6] = torch.tensor(arm_action, device=dev)
    a[:, 6] = -1.0  # gripper open
    with torch.inference_mode():
        for _ in range(SETTLE):
            env.step(a)
    ee = env.unwrapped.scene["ee_frame"].data.target_pos_w[0, 0].clone()
    gear = env.unwrapped.scene["object"].data.root_pos_w[0].clone()
    return ee.cpu(), gear.cpu()


def main():
    import traceback
    try:
        _main()
    except Exception:
        traceback.print_exc()
        raise


def _main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=cfg)
    base_ee, gear = run(env, [0.0] * 6)
    print(f"BASE EE {base_ee.numpy().round(4)} | GEAR {gear.numpy().round(4)}")
    print(f"NEED (gear - EE) {(gear - base_ee).numpy().round(4)}")
    for i, n in enumerate(NAMES):
        act = [0.0] * 6
        act[i] = DELTA
        ee, _ = run(env, act)
        per_unit = ((ee - base_ee) / DELTA).numpy().round(3)
        print(f"{n:7s} +{DELTA}: EE moves per 1.0 action = {per_unit}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
