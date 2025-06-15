import pickle
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import login


def main():
    login()
    # Parameters
    resume = True
    task_description = "Cube pick task with SO101 robot"
    time_step = 1.0/50.0

    # Load pick1 dataset
    raw_data = None
    with open("dataset_2025-06-14_20-12-11.pkl", "rb") as f:
        raw_data = pickle.load(f)

    if resume:
        dataset = LeRobotDataset(
            repo_id="stlee601/so101_cube_pick1",
        )
        print("Dataset loaded from hub.")
    else:
        dataset = LeRobotDataset.create(
            repo_id="stlee601/so101_cube_pick1",
            fps=50,
            features=
            {
                "action":
                {
                    "dtype": "float32",
                    "shape": raw_data[0]["action"][0].shape,
                },
                "observation.joint_positions":
                {
                    "dtype": "float32",
                    "shape": raw_data[0]["observation"]["policy"]["joint_positions"][0].shape,
                },
                "observation.wrist_rgb":
                {
                    "dtype": "video",
                    "shape": raw_data[0]["observation"]["policy"]["images"][0].shape,
                },
            },
            image_writer_processes=0,
            image_writer_threads=4
        )

    # Add data to dataset
    for data in raw_data:
        dataset.add_frame(
            frame=
            {
                "action": data["action"][0].cpu().numpy(),
                "observation.joint_positions": data["observation"]["policy"]["joint_positions"][0].cpu().numpy(),
                "observation.wrist_rgb": data["observation"]["policy"]["images"][0].cpu().numpy(),
            },
            task=task_description,
            timestamp=data["step"] * time_step,
        )
        # Save dataset
        if data["step"] == 99:
            dataset.save_episode()
            dataset.clear_episode_buffer()

    dataset.push_to_hub()

if __name__ == "__main__":
    main()