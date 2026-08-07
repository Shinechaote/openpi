"""Parallel HDF5 -> LeRobot conversion for the MimicGen datasets.

Same output as `convert_hdf5_dataset_to_lerobot.py`, but:

1. Episode arrays are read from HDF5 in one shot instead of per step. The images are
   gzip-compressed with chunks of (28, 28, 56, 1), so a per-frame read decompresses each
   chunk 28 times over. Bulk reading alone measured ~16x faster.
2. Episodes are pre-indexed, so every episode's global episode index and global frame
   offset are known before any conversion happens. Workers can therefore write final,
   globally-numbered parquet files independently, and the merge is just concatenating
   metadata -- no parquet is ever rewritten.
3. Frames are encoded straight to PNG bytes inside the parquet, skipping LeRobot's
   write-PNG-to-disk / read-back / delete round trip.
4. Per-episode work never touches the growing dataset, avoiding the O(N^2) `rglob` asserts
   and the unbounded `concatenate_datasets` accumulation in `LeRobotDataset.save_episode`.

Usage:
    uv run convert_hdf5_dataset_to_lerobot_parallel.py --num-workers 12
"""

import dataclasses
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import datasets
import h5py
import jsonlines
import numpy as np
import tyro
from lerobot.common.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARQUET_PATH,
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    INFO_PATH,
    get_hf_features_from_features,
    serialize_dict,
    write_info,
    write_json,
)

# This script writes the v2.1 on-disk layout: one parquet per episode under
# data/chunk-{chunk:03d}/, and jsonl metadata (episodes, episodes_stats, tasks). Every path
# and constant below is taken from the installed lerobot, so upgrading lerobot to a release
# that emits v3.0 would silently change the output format instead of failing.
TARGET_CODEBASE_VERSION = "v2.1"
if CODEBASE_VERSION != TARGET_CODEBASE_VERSION:
    raise RuntimeError(
        f"Installed lerobot writes {CODEBASE_VERSION} datasets, but this script must produce "
        f"{TARGET_CODEBASE_VERSION}. Pin lerobot back to a {TARGET_CODEBASE_VERSION} revision "
        "(openpi pins rev 0cf8648) or port this script to the newer layout."
    )

REPO_NAME = "christian/2000_demo_six_env_mimicgen"
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
    "put square on square stick. put ring on cylinder",
]

FPS = 30
MAX_EPISODES_PER_FILE = 2000
MIN_EPISODE_LENGTH = 100
STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat"]

FEATURES = {
    "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}

# 256 MB chunk cache so a bulk episode read never evicts a chunk it still needs.
H5_KWARGS = {"rdcc_nbytes": 256 * 1024**2, "rdcc_nslots": 1_000_003}


@dataclasses.dataclass(frozen=True)
class EpisodeJob:
    """Everything a worker needs to write one final parquet file."""

    dataset_idx: int
    episode_key: str
    length: int
    episode_index: int  # global
    frame_offset: int  # global index of this episode's first frame
    task_index: int


def build_index() -> list[EpisodeJob]:
    """Scan the HDF5 files for episode lengths only, then assign global indices.

    Reading `actions.shape` is a metadata lookup, so this pass costs seconds, not hours.
    Episode ordering matches the sequential script: files in order, `data` keys in order.
    """
    jobs: list[EpisodeJob] = []
    episode_index = 0
    frame_offset = 0
    for dataset_idx, path in enumerate(DATASET_PATHS):
        with h5py.File(path, "r") as raw_dataset:
            for num_episodes, episode_key in enumerate(raw_dataset["data"].keys()):
                if num_episodes >= MAX_EPISODES_PER_FILE:
                    break
                length = raw_dataset["data"][episode_key]["actions"].shape[0]
                if length < MIN_EPISODE_LENGTH:
                    continue
                jobs.append(
                    EpisodeJob(
                        dataset_idx=dataset_idx,
                        episode_key=episode_key,
                        length=length,
                        episode_index=episode_index,
                        frame_offset=frame_offset,
                        task_index=dataset_idx,
                    )
                )
                episode_index += 1
                frame_offset += length
    return jobs


def read_episode(episode: h5py.Group, length: int) -> dict[str, np.ndarray]:
    """Bulk-read one episode. Every array is fetched with a single `[:]`."""
    obs = episode["obs"]
    images = obs["agentview_image"][:]
    wrist_images = obs["robot0_eye_in_hand_image"][:]
    eef = np.concatenate([obs[key][:] for key in STATE_KEYS], axis=1)
    gripper_qpos = obs["robot0_gripper_qpos"][:]
    actions = episode["actions"][:]

    gripper = ((gripper_qpos[:, 0] - gripper_qpos[:, 1]) < 0.05).astype(np.float32)
    state = np.concatenate([eef, gripper[:, None]], axis=1).astype(np.float32)

    # The recorded arrays can be longer than `actions`; keep them aligned with the length
    # the index pass used so global frame offsets stay exact.
    return {
        "image": images[:length],
        "wrist_image": wrist_images[:length],
        "state": state[:length],
        "actions": actions[:length].astype(np.float32),
    }


def image_stats_from_array(images: np.ndarray) -> dict[str, np.ndarray]:
    """`compute_episode_stats` equivalent for images we hold in memory rather than on disk.

    LeRobot's `sample_images` reads back the PNG files it just wrote; we skip writing them,
    so reproduce the same sampling and normalization directly on the array.
    """
    indices = sample_indices(len(images))
    sampled = np.stack([auto_downsample_height_width(images[i].transpose(2, 0, 1)) for i in indices])
    stats = get_feature_stats(sampled, axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def compute_stats(frames: dict[str, np.ndarray], extra: dict[str, np.ndarray]) -> dict[str, dict]:
    stats = {}
    for key, ft in FEATURES.items():
        if ft["dtype"] == "image":
            stats[key] = image_stats_from_array(frames[key])
        else:
            stats[key] = get_feature_stats(frames[key], axis=0, keepdims=False)
    for key, value in extra.items():
        stats[key] = get_feature_stats(value, axis=0, keepdims=value.ndim == 1)
    return stats


def write_episode_parquet(root: Path, job: EpisodeJob, frames: dict[str, np.ndarray], hf_features) -> dict:
    length = job.length
    extra = {
        "timestamp": (np.arange(length) / FPS).astype(np.float32),
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, job.episode_index, dtype=np.int64),
        "index": np.arange(job.frame_offset, job.frame_offset + length, dtype=np.int64),
        "task_index": np.full(length, job.task_index, dtype=np.int64),
    }

    episode_dict = {
        # `datasets.Image()` encodes each array to PNG bytes directly into the table, so no
        # image files are ever written to disk.
        "image": list(frames["image"]),
        "wrist_image": list(frames["wrist_image"]),
        "state": frames["state"].tolist(),
        "actions": frames["actions"].tolist(),
        **{k: v.tolist() for k, v in extra.items()},
    }
    ep_dataset = datasets.Dataset.from_dict(episode_dict, features=hf_features, split="train")

    chunk = job.episode_index // DEFAULT_CHUNK_SIZE
    path = root / DEFAULT_PARQUET_PATH.format(episode_chunk=chunk, episode_index=job.episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(path)

    return compute_stats(frames, extra)


def run_worker(worker_id: int, root: Path, shard_dir: Path, jobs: list[EpisodeJob]) -> None:
    hf_features = get_hf_features_from_features({**FEATURES, **_default_features()})
    episodes_path = shard_dir / f"ep_{worker_id:03d}.jsonl"
    stats_path = shard_dir / f"stats_{worker_id:03d}.jsonl"

    # Jobs are grouped by source file so each file is opened once per worker.
    by_dataset: dict[int, list[EpisodeJob]] = {}
    for job in jobs:
        by_dataset.setdefault(job.dataset_idx, []).append(job)

    done = 0
    with (
        jsonlines.open(episodes_path, "w", flush=True) as ep_writer,
        jsonlines.open(stats_path, "w", flush=True) as stats_writer,
    ):
        for dataset_idx, dataset_jobs in by_dataset.items():
            with h5py.File(DATASET_PATHS[dataset_idx], "r", **H5_KWARGS) as raw_dataset:
                data = raw_dataset["data"]
                for job in dataset_jobs:
                    frames = read_episode(data[job.episode_key], job.length)
                    ep_stats = write_episode_parquet(root, job, frames, hf_features)
                    ep_writer.write(
                        {
                            "episode_index": job.episode_index,
                            "tasks": [TASK_DESCRIPTIONS[job.dataset_idx]],
                            "length": job.length,
                        }
                    )
                    stats_writer.write(
                        {"episode_index": job.episode_index, "stats": serialize_dict(ep_stats)}
                    )
                    done += 1
                    if done % 25 == 0:
                        print(f"[worker {worker_id}] {done}/{len(jobs)} episodes", flush=True)
    print(f"[worker {worker_id}] done ({done} episodes)", flush=True)


def _default_features() -> dict:
    from lerobot.common.datasets.utils import DEFAULT_FEATURES

    return DEFAULT_FEATURES


def split_jobs(jobs: list[EpisodeJob], num_workers: int) -> list[list[EpisodeJob]]:
    """Greedy longest-first packing so workers finish at roughly the same time."""
    shards: list[list[EpisodeJob]] = [[] for _ in range(num_workers)]
    loads = [0] * num_workers
    for job in sorted(jobs, key=lambda j: -j.length):
        target = min(range(num_workers), key=lambda w: loads[w])
        shards[target].append(job)
        loads[target] += job.length
    return shards


def merge_metadata(root: Path, shard_dir: Path, jobs: list[EpisodeJob]) -> None:
    """Concatenate per-worker metadata in episode order and finalize info.json/stats.json."""
    episodes: dict[int, dict] = {}
    stats_rows: dict[int, dict] = {}
    for path in sorted(shard_dir.glob("ep_*.jsonl")):
        with jsonlines.open(path, "r") as reader:
            for row in reader:
                episodes[row["episode_index"]] = row
    for path in sorted(shard_dir.glob("stats_*.jsonl")):
        with jsonlines.open(path, "r") as reader:
            for row in reader:
                stats_rows[row["episode_index"]] = row

    expected = {job.episode_index for job in jobs}
    missing = expected - set(episodes)
    if missing:
        raise RuntimeError(f"{len(missing)} episodes were not written, e.g. {sorted(missing)[:5]}")

    with jsonlines.open(root / EPISODES_PATH, "w") as writer:
        for episode_index in sorted(episodes):
            writer.write(episodes[episode_index])
    with jsonlines.open(root / EPISODES_STATS_PATH, "w") as writer:
        for episode_index in sorted(stats_rows):
            writer.write(stats_rows[episode_index])

    total_frames = sum(job.length for job in jobs)
    total_episodes = len(jobs)
    info = json.loads((root / INFO_PATH).read_text())
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = (total_episodes - 1) // DEFAULT_CHUNK_SIZE + 1
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_info(info, root)

    # Same aggregation LeRobot does incrementally in `meta.save_episode`.
    unserialized = []
    for episode_index in sorted(stats_rows):
        stats = stats_rows[episode_index]["stats"]
        unserialized.append(
            {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in stats.items()}
        )
    write_json(serialize_dict(aggregate_stats(unserialized)), root / "meta/stats.json")


def main(num_workers: int = 12, overwrite: bool = False) -> None:
    root = HF_LEROBOT_HOME / REPO_NAME
    if root.exists():
        if not overwrite:
            raise SystemExit(f"{root} already exists; pass --overwrite to replace it.")
        shutil.rmtree(root)

    print("Indexing episodes...", flush=True)
    jobs = build_index()
    print(f"{len(jobs)} episodes, {sum(j.length for j in jobs)} frames", flush=True)

    # Create the dataset skeleton once (info.json + tasks.jsonl) with every task registered
    # up front, so all workers agree on task indices without coordinating.
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=FPS,
        features=FEATURES,
        image_writer_threads=0,
        image_writer_processes=0,
    )
    for task in TASK_DESCRIPTIONS:
        dataset.meta.add_task(task)
    for dataset_idx, task in enumerate(TASK_DESCRIPTIONS):
        assert dataset.meta.get_task_index(task) == dataset_idx, "task indices must match dataset order"
    del dataset

    shard_dir = root / "meta_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shards = split_jobs(jobs, num_workers)
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=run_worker, args=(worker_id, root, shard_dir, shard), daemon=False)
        for worker_id, shard in enumerate(shards)
        if shard
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} worker(s) failed with exit codes {failed}")

    print("Merging metadata...", flush=True)
    merge_metadata(root, shard_dir, jobs)
    shutil.rmtree(shard_dir)
    print(f"Done: {root}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
