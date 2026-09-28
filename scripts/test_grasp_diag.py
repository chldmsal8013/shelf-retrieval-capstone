"""Scripted grasp test with joint diagnostics: home -> hover -> grasp -> close -> lift (no PPO)."""

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
HOVER = [-0.292, 1.635, -2.106, 0.472, 0.0, 0.0]
GRASP = [-0.292, 1.570, -1.597, 0.027, 0.0, 0.0]
LIFT = HOVER  # same xy, EE z back up to 0.30

ARM_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
DIAG_STEPS = {100, 140, 170, 200, 230}


def lerp(a, b, k):
    k = max(0.0, min(1.0, k))
    return [x + (y - x) * k for x, y in zip(a, b)]


def fmt(x):
    return " ".join(f"{v:+.3f}" for v in x.detach().cpu().tolist())


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    dev = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    ids = robot.find_joints(ARM_NAMES, preserve_order=True)[0]

    lim = robot.data.soft_joint_pos_limits[0, ids]
    print("JT limits lower:", fmt(lim[:, 0]), flush=True)
    print("JT limits upper:", fmt(lim[:, 1]), flush=True)
    try:
        print("JT effort limit:", fmt(robot.actuators["arm"].effort_limit[0]), flush=True)
    except Exception as e:  # noqa: BLE001
        print("JT effort limit: n/a", e, flush=True)

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
            if t in DIAG_STEPS:
                tgt = robot.data.joint_pos_target[0, ids]
                act = robot.data.joint_pos[0, ids]
                tq = robot.data.applied_torque[0, ids]
                print(f"JT t={t}", flush=True)
                print("JT   target :", fmt(tgt), flush=True)
                print("JT   actual :", fmt(act), flush=True)
                print("JT   error  :", fmt(tgt - act), flush=True)
                print("JT   torque :", fmt(tq), flush=True)
                fj = robot.find_joints(["finger_joint"])[0][0]
                print("JT   finger_joint:", f"{robot.data.joint_pos[0, fj].item():.3f}", flush=True)
            t = 0 if bool((term | trunc).any()) else t + 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
