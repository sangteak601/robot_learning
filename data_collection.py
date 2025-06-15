import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Data Collection")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
import datetime
import pickle
from isaaclab.envs import ManagerBasedEnv
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_mul
from envs.cube_pick_env import CubePickEnvCfg
from policy.cartesian_script_policy import CartesianScriptPolicy, PoseWithTime

def ik(ee_pose, jacobian, joint_pos, target_pose):
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device="cuda")
    diff_ik_controller.reset()
    diff_ik_controller.set_command(target_pose)
    return diff_ik_controller.compute(ee_pose[:,:3], ee_pose[:,3:], jacobian, joint_pos)

def main():
    # parse the arguments
    env_cfg = CubePickEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    init_pose = torch.zeros_like(env.action_manager.action)
    init_pose[:, 3] = math.pi / 2.0

    # data structure
    data = []

    # simulate physics
    step = 0
    num_episodes = 0
    episode_length = 100
    scripted_policy = None
    while simulation_app.is_running() and num_episodes < episode_length:
        # reset
        if step % 100 == 0:
            step = 0
            num_episodes += 1
            if num_episodes % 10 == 0:
                print("[INFO]: Saving data...")
                pickle_path = "dataset_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
                with open(pickle_path, "wb") as file:
                    pickle.dump(data, file)
                data = []  # reset data after saving
            env.reset()
            obs, _ = env.step(init_pose)
            ee_pose = obs["policy"]["ee_pose"][0]
            cube_pos = obs["info"]["cube_pos"][0]
            cube_quat = obs["info"]["cube_quat"][0]
            target_quat = quat_mul(ee_pose[3:7], cube_quat)
            scripted_policy = CartesianScriptPolicy(
                target_poses=[
                    PoseWithTime(t=0.0, x=ee_pose[0], y=ee_pose[1], z=ee_pose[2], qw=ee_pose[3], qx=ee_pose[4], qy=ee_pose[5], qz=ee_pose[6]),
                    PoseWithTime(t=1.0, x=cube_pos[0], y=cube_pos[1], z=cube_pos[2] + 0.2, qw=target_quat[0], qx=target_quat[1], qy=target_quat[2], qz=target_quat[3]),
                    PoseWithTime(t=2.0, x=cube_pos[0], y=cube_pos[1], z=cube_pos[2] + 0.1, qw=target_quat[0], qx=target_quat[1], qy=target_quat[2], qz=target_quat[3]),
                ],
                time_step=env_cfg.sim.dt * env_cfg.decimation,
            )
            print("[INFO]: Resetting environment... episode:", num_episodes)

        pose_with_time = scripted_policy.get_action(step)
        target_pose = torch.tensor(
            [[pose_with_time.x, pose_with_time.y, pose_with_time.z, pose_with_time.qw, pose_with_time.qx, pose_with_time.qy, pose_with_time.qz]],
            device="cuda"
        )
        joint_positions = ik(obs["policy"]["ee_pose"], obs["info"]["jacobian"], obs["policy"]["joint_positions"][:,:-1], target_pose)
        if step < 90:
            joint_positions = torch.cat((joint_positions, torch.tensor([[3.14]], device="cuda")), dim=1)
        else:
            joint_positions = torch.cat((joint_positions, torch.tensor([[0.0]], device="cuda")), dim=1)
        # step the environment
        obs, _ = env.step(joint_positions)
        data.append({
            "step": step,
            "observation": obs,
            "action": joint_positions,
        })
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()