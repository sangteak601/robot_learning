from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import Camera
from isaaclab.managers import SceneEntityCfg

def get_rgb_image(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")):
    sensor: Camera = env.scene[sensor_cfg.name]
    return sensor.data.output["rgb"]