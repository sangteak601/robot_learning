import math

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from envs.robots.so101 import SO101_CFG
from envs.observations.robot_observations import get_ee_pose, get_jacobian
from envs.observations.camera_observations import get_rgb_image

@configclass
class SceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    so101: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "a_1": 0.0,
                "a_2": 0.0,
                "a_3": 0.0,
                "a_4": math.pi / 2.0,
                "a_5": math.pi / 2.0,
                "a_6": 0.0,
            }
        )
    )

    # camera
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_cam",
        update_period=0.1,
        height=480,
        width=640,
        spawn=sim_utils.PinholeCameraCfg(focal_length=5),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -0.1, 0.0), rot=(0.0, 0.0, 0.985, 0.174)),
    )

    # cube
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

@configclass
class ActionsCfg:
    joint_position = mdp.JointPositionActionCfg(
        asset_name="so101",
        joint_names=["a_1", "a_2", "a_3", "a_4", "a_5", "a_6"],
        use_default_offset=False
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        concatenate_terms = False
        joint_positions = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("so101")})
        ee_pose = ObsTerm(func=get_ee_pose, params={"asset_cfg": SceneEntityCfg("so101"), "ee_name": "gripper"})
        images = ObsTerm(func=get_rgb_image, params={"sensor_cfg": SceneEntityCfg("wrist_cam")})
    @configclass
    class InfoCfg(ObsGroup):
        concatenate_terms = False
        cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube")})
        jacobian = ObsTerm(func=get_jacobian, params={
            "asset_cfg": SceneEntityCfg("so101"),
            "ee_name": "gripper",
            "joint_names": ["a_1", "a_2", "a_3", "a_4", "a_5"]
        })
    policy = PolicyCfg()
    info = InfoCfg()
@configclass
class EventCfg:
    reset_joint_positions = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("so101"),
            "position_range": (1.0, 1.0),
            "velocity_range": (1.0, 1.0),
        },
    )
    reset_cube_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.25, -0.15),
                "z": (0.025, 0.025),
                "yaw": (-math.pi/3.0, math.pi/3.0)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
            },
        },
    )

@configclass
class CubePickEnvCfg(ManagerBasedEnvCfg):

    # Scene settings
    scene = SceneCfg(num_envs=4, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz
