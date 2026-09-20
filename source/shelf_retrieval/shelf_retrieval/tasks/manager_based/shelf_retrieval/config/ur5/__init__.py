import gymnasium as gym

from .joint_pos_env_cfg import UR5ShelfRetrievalEnvCfg

gym.register(
    id="Isaac-ShelfRetrieval-Home-UR5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR5ShelfRetrievalEnvCfg",
        "rsl_rl_cfg_entry_point": "shelf_retrieval.tasks.manager_based.shelf_retrieval.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
