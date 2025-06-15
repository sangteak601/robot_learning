from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

def get_ee_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ee_name: str = "ee"):
    asset: Articulation = env.scene[asset_cfg.name]
    ee_ids, _ = asset.find_bodies(name_keys=ee_name)
    return asset.data.body_state_w[:, ee_ids[0], 0:7]

def get_jacobian(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ee_name: str = "ee", joint_names: list[str] = ["joint"]):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(name_keys=joint_names)
    ee_ids, _ = asset.find_bodies(name_keys=ee_name)
    ee_id = ee_ids[0]
    jacobian_id = ee_id
    if asset.is_fixed_base:
        jacobian_id -= 1  # fixed base is not included in jacobian
    asset.root_physx_view.get_jacobians()[:, jacobian_id, :, joint_ids]
    return asset.root_physx_view.get_jacobians()[:, jacobian_id, :, joint_ids]