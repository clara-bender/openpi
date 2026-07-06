from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np

class Dataset():
    def __init__(self, repo_name, task, fps_collect):
        self.task = task
        dataset_path = HF_LEROBOT_HOME / repo_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        if dataset_path.exists(): 
            self.dataset = LeRobotDataset(
                root=dataset_path,
                repo_id=repo_name,
            )
            print("Adding to existing dataset, waiting for signal.")
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=repo_name,
                robot_type="xarm",
                fps=fps_collect,
                features={
                    "exterior_image_1_left": {
                        "dtype": "image",
                        "shape": (240, 320, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "exterior_image_2_left": { # this one is not used, put it as zeros or something
                        "dtype": "image",
                        "shape": (240, 320, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "wrist_image_left": {
                        "dtype": "image",
                        "shape": (240, 320, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "joint_position": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["joint_position"],
                    },
                    "gripper_position": {
                        "dtype": "float32",
                        "shape": (1,),
                        "names": ["gripper_position"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (7,),  # We will use joint *velocity* actions here (6D) + gripper position (1D)
                        "names": ["actions"],
                    },
                },
            )

    def collect(self, observation_queue):

        while not observation_queue.empty():
            obs = observation_queue.get()
            tripod_camera = obs["observation/exterior_image_1_left"]
            extra_camera = obs["observation/exterior_image_2_left"]
            wrist_camera = obs["observation/wrist_image_left"]
            gripper_pos = obs["observation/gripper_position"]
            servo_state = obs["observation/joint_position"]

            total_state = np.concatenate((servo_state,np.array([gripper_pos],dtype=np.float32)))

            if self.prev_data is not None:
                self.dataset.add_frame(
                    {
                        "joint_position": self.prev_data["joints"],
                        "gripper_position": self.prev_data["gripper"],
                        "actions": total_state,  # This is the "future" state reached
                        "exterior_image_1_left": self.prev_data["base"],
                        "exterior_image_2_left": self.prev_data["base2"],
                        "wrist_image_left": self.prev_data["wrist"],
                        "task": self.task,
                    }
                )
                self.frames_recorded += 1

            self.prev_data = {
                "joints": total_state[:6],
                "gripper": total_state[-1:],
                "wrist": wrist_camera,
                "base": tripod_camera,
                "base2": extra_camera,
            }

        self.dataset.save_episode()