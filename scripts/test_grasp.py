"""Scripted grasp test: home -> hover -> grasp pose -> close -> lift (no PPO)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted grasp test.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import shelf_retrieval.tasks  # noqa: F401

# pan, lift, elbow, wrist1, wrist2, wrist3 (target joint = default + 0.5 * action)
HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HOVER = [-0.29, 1.798, -2.609, 0.811, 0.0, 0.0]
GRASP = [-0.29, 1.625, -1.909, 0.285, 0.0, 0.0]
LIFT = HOVER  # same xy, EE z back up to 0.30


def lerp(a, b, k):
    k = max(0.0, min(1.0, k))
    return [x + (y - x) * k for x, y in zip(a, b)]


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    dev = env.unwrapped.device
    obj = env.unwrapped.scene["object"]
    ee_frame = env.unwrapped.scene["ee_frame"]

    t = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if t < 60:  # home -> hover, open
                arm, grip = lerp(HOME, HOVER, t / 50), -1.0
            elif t < 110:  # hover -> grasp, open
                arm, grip = lerp(HOVER, GRASP, (t - 60) / 30), -1.0
            elif t < 150:  # close
                arm, grip = GRASP, 1.0
            else:  # lift while closed
                arm, grip = lerp(GRASP, LIFT, (t - 150) / 40), 1.0

            actions = torch.zeros(env.action_space.shape, device=dev)
            actions[:, :6] = torch.tensor(arm, device=dev)
            actions[:, 6] = grip
            _, _, term, trunc, _ = env.step(actions)

            if t % 10 == 0:
                gear = obj.data.root_pos_w[0]
                ee = ee_frame.data.target_pos_w[0, 0]
                print(
                    f"t {t:3d} | grip {grip:+.0f} | gear z {gear[2].item():.4f} | "
                    f"EE {ee.cpu().numpy().round(3)} | gear-EE {(gear - ee).cpu().numpy().round(3)}",
                    flush=True,
                )
            t = 0 if bool((term | trunc).any()) else t + 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
