from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

FACTORY_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
from ... import mdp
from ...shelf_retrieval_env_cfg import ShelfRetrievalEnvCfg


UR5_ROBOTIQ_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path="/home/choi/ur5_robotiq_2f85.usd",
        activate_contact_sensors=False,  # contact sensor 안 쓰므로 끔
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": -1.571,
            "wrist_2_joint": -1.571,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "right_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            stiffness=400.0,
            damping=60.0,
        ),
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=1650.0,
            velocity_limit_sim=10.0,
            stiffness=17.0,
            damping=0.2,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=["right_inner_finger_joint", "left_inner_finger_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=10.0,
            stiffness=2.0,
            damping=0.01,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["right_outer_knuckle_joint",
                              "right_inner_finger_knuckle_joint",
                              "left_inner_finger_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class UR5ShelfRetrievalEnvCfg(ShelfRetrievalEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum = None

        self.scene.robot = UR5_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            scale=0.4,
            use_default_offset=False,
            offset=0.4,
        )

        self.commands.object_pose.body_name = "wrist_3_link"

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.13]),
                ),
            ],
        )

        # 물체는 씬에서 이미 우리가 정한 크기/위치(reset 범위)로 정의했으니, 여기선 spawn만 지정
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.75, 0.0, 0.20], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{FACTORY_ASSET_DIR}/factory_gear_medium.usd",
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=3666.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.005),
            ),
        )


@configclass
class UR5ShelfRetrievalPregraspEnvCfg(UR5ShelfRetrievalEnvCfg):
    """Step 2: identical to the home-reset env; ONLY the robot reset pose differs.

    Reset pose = gripper open, EE ~10 cm above the gear (found by the scripted grasp test).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = self.scene.robot.replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={
                    "shoulder_pan_joint": -0.146,
                    "shoulder_lift_joint": -0.895,
                    "elbow_joint": 0.659,
                    "wrist_1_joint": -1.335,
                    "wrist_2_joint": -1.571,
                    "wrist_3_joint": 0.0,
                    "finger_joint": 0.0,
                    "right_outer_knuckle_joint": 0.0,
                    "right_inner_finger_joint": 0.0,
                    "right_inner_finger_knuckle_joint": 0.0,
                    "left_inner_finger_knuckle_joint": 0.0,
                    "left_inner_finger_joint": 0.0,
                },
            )
        )


# ======================= reward v2 tasks =======================
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402


def _apply_reward_v2(cfg):
    """Reward v2: lifting only counts while the gear is held between the fingers."""
    finger = SceneEntityCfg("robot", joint_names=["finger_joint"])
    # the mid-air goal inherited from Lift is irrelevant here and rewards knocking the gear up
    cfg.rewards.object_goal_tracking.weight = 0.0
    cfg.rewards.object_goal_tracking_fine_grained.weight = 0.0
    cfg.rewards.lifting_object = RewTerm(
        func=mdp.object_lifted_and_grasped,
        weight=15.0,
        params={"minimal_height": 0.245, "finger_cfg": finger},
    )
    cfg.rewards.grasp_hold = RewTerm(func=mdp.object_grasped, weight=3.0, params={"finger_cfg": finger})
    cfg.rewards.grasp_lift_progress = RewTerm(
        func=mdp.object_grasp_lift_progress,
        weight=5.0,
        params={"rest_height": 0.195, "max_rise": 0.05, "finger_cfg": finger},
    )


@configclass
class UR5ShelfRetrievalHomeV2EnvCfg(UR5ShelfRetrievalEnvCfg):
    """Home reset + reward v2."""

    def __post_init__(self):
        super().__post_init__()
        _apply_reward_v2(self)


@configclass
class UR5ShelfRetrievalPregraspV2EnvCfg(UR5ShelfRetrievalPregraspEnvCfg):
    """Pre-grasp reset + reward v2."""

    def __post_init__(self):
        super().__post_init__()
        _apply_reward_v2(self)
