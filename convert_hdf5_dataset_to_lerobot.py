"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import os
import tyro
import numpy as np
from PIL import Image


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


REPO_NAME = "christian/2000_demo_six_env_mimicgen"  # Name of the output dataset, also used for the Hugging Face Hub
# RAW_DATASET_NAMES = [
#     "libero_10_no_noops",
#     "libero_goal_no_noops",
#     "libero_object_no_noops",
#     "libero_spatial_no_noops",
# ]  # For simplicity we will combine multiple Libero datasets into one training dataset
home_path = "/home/stud_scherer/mimicgen_datasets"

DATASET_PATHS = [
    os.path.join(home_path, "coffee_2k.hdf5"),
    os.path.join(home_path, "hammer_cleanup_2k.hdf5"),
    os.path.join(home_path, "mug_cleanup_2k.hdf5"),
    os.path.join(home_path, "threading_2k.hdf5"),
    os.path.join(home_path, "nut_assembly_2k.hdf5"),
]
TASK_DESCRIPTIONS = [
    "pick capsule. put in machine. close lid",
    "put hammer inside drawer",
    "put mug inside drawer",
    "thread stick through hole",
    "put square on square stick. put ring on cylinder"
]
# LEROBOT_HOME = "/home/scherer/.cache/huggingface/lerobot/"


def main():
    # Clean up any existing dataset in the output directory
    # output_path = os.path.join(LEROBOT_HOME, REPO_NAME)
    # if os.path.exists(output_path):
    #     shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for i, dataset_path in enumerate(DATASET_PATHS):
        raw_dataset = h5py.File(dataset_path)
        for num_episodes, episode_key in enumerate(raw_dataset["data"].keys()):
            if num_episodes >= 2000:
                break
            episode = raw_dataset["data"][episode_key]
            if episode["actions"].shape[0] < 100:
                continue
            for step in range(episode["actions"].shape[0]):
                state_keys = ["robot0_eef_pos", "robot0_eef_quat"]
                state = np.concatenate(
                    [episode["obs"][key] for key in state_keys], axis=1
                )[step]

                # gripper = float(int(episode["obs"]["robot0_gripper_qpos"][step]))
                gripper = float(int(
                    (
                        episode["obs"]["robot0_gripper_qpos"][step][0]
                        - episode["obs"]["robot0_gripper_qpos"][step][1]
                    )
                    < 0.05
                ))
                state = np.concatenate([state, [gripper]])

                dataset.add_frame(
                    {
                        "image": resize_with_pad(episode["obs"]["agentview_image"][step], 224, 224).astype(np.uint8),
                        "wrist_image": resize_with_pad(episode["obs"]["robot0_eye_in_hand_image"][step], 224, 224).astype(np.uint8),
                        "state": state.astype(np.float32),
                        "actions": episode["actions"][step].astype(np.float32),
                        "task": TASK_DESCRIPTIONS[i],
                    }
                )
            dataset.save_episode()


if __name__ == "__main__":
    tyro.cli(main)

