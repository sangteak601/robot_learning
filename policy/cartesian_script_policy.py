import math
import torch
from dataclasses import dataclass

from isaaclab.utils.math import quat_slerp

@dataclass
class PoseWithTime:
    t: float
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float

class CartesianScriptPolicy:
    def __init__(self, target_poses, time_step, num_envs=1):
        """
        target_poses: List of intermediate poses and final pose in the format PoseWithTime
        time_step: Time step for the trajectory generation
        """
        self._target_poses = target_poses
        self._time_step = time_step
        self._num_envs = num_envs
        self._trajectory = self._generate_trajectory()

    def _generate_trajectory(self):
        """
        Generate a trajectory from the target poses.
        Each pose is represented as a PoseWithTime.
        """
        trajectory = []
        for i in range(len(self._target_poses) - 1):
            start_pose = self._target_poses[i]
            end_pose = self._target_poses[i + 1]
            num_steps = math.ceil((end_pose.t - start_pose.t) / self._time_step)
            for step in range(num_steps):
                quat_slerped = quat_slerp(
                    torch.tensor([start_pose.qw, start_pose.qx, start_pose.qy, start_pose.qz], device='cuda'),
                    torch.tensor([end_pose.qw, end_pose.qx, end_pose.qy, end_pose.qz], device='cuda'),
                    step / num_steps,
                )
                interpolated_pose = PoseWithTime(
                    t=start_pose.t + step * self._time_step,
                    x=start_pose.x + step * (end_pose.x - start_pose.x) / num_steps,
                    y=start_pose.y + step * (end_pose.y - start_pose.y) / num_steps,
                    z=start_pose.z + step * (end_pose.z - start_pose.z) / num_steps,
                    qw=quat_slerped[0],
                    qx=quat_slerped[1],
                    qy=quat_slerped[2],
                    qz=quat_slerped[3],
                )
                trajectory.append(interpolated_pose)
        return trajectory

    def get_action(self, current_step):
        """
        Get the action for the current step.
        Returns PoseWithTime representing the pose.
        """
        if current_step < 0 or current_step >= len(self._trajectory):
            raise ValueError("Current step is out of bounds.")

        return self._trajectory[current_step]