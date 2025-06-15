import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ws/assets/so101_new_calib/so101_new_calib.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True
        ),
    ),
    actuators={
        "so101": ImplicitActuatorCfg(
            joint_names_expr=["a_1", "a_2", "a_3", "a_4", "a_5", "a_6"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=10000.0,
            damping=10.0,
        ),
    },
)
