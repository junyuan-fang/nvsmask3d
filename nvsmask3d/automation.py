"""Eval automation scrip"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import GPUtil

run_dataset = "scannetpp"

# scenes to run from dataset
SCENE_NAMES = [
    "7b6477cb95",
    "c50d2d1d42",
    "cc5237fd77",
    "acd95847c5",
    "fb5a96b1a2",
    "a24f64f7fb",
    "1ada7a0617",
    "5eb31827b7",
    "3e8bba0176",
    "3f15a9266d",
    "21d970d8de",
    "5748ce6f01",
    "c4c04e6d6c",
    "7831862f02",
    "bde1e479ad",
    "38d58a7a31",
    "5ee7c22ba0",
    "f9f95681fd",
    "3864514494",
    "40aec5fffa",
    "13c3e046d7",
    "e398684d27",
    "a8bf42d646",
    "45b0dac5e3",
    "31a2c91c43",
    "e7af285f7d",
    "286b55a2bf",
    "7bc286c1b6",
    "f3685d06a9",
    "b0a08200c9",
    "825d228aec",
    "a980334473",
    "f2dc06b1d2",
    "5942004064",
    "25f3b7a318",
    "bcd2436daf",
    "f3d64c30f8",
    "0d2ee665be",
    "3db0a1c8f3",
    "ac48a9b736",
    "c5439f4607",
    "578511c8a9",
    "d755b3d9d8",
    "99fa5c25e1",
    "09c1414f1b",
    "5f99900f09",
    "9071e139d9",
    "6115eddb86",
    "27dd4da69e",
    "c49a8c6cff",
]

LOAD_CONFIGS = [
    f"outputs/{scene}_dslr_colmap/nvsmask3d/config.yml" for scene in SCENE_NAMES
]


@dataclass
class BenchmarkConfig:
    """Baseline benchmark config"""

    # function to run
    function: str = "nvsmask3d/script/eval_scannetpp.py"
    # path to data
    dataset: str = "scannetpp"
    kind: str = "crop"
    sam: bool = False
    wandb_mode: str = "disabled"
    project_name: str = "crop"
    experiment_type: str = "rgb"
    scene_name: str = None  # Accept either str or List
    load_config: str = None  # Accept either str or List
    interpolate_n_camera: int = 0
    top_k: int = 15
    excluded_gpus: set = field(default_factory=set)
    output_dir: str = None


scannetpp_config = BenchmarkConfig(
    sam=False,
    kind="crop",
    scene_name=SCENE_NAMES[0],
    load_config=LOAD_CONFIGS[0],
    experiment_type="rgb",
    interpolate_n_camera=0,
)

configs_to_run = [scannetpp_config]


def eval_scene(gpu, config: BenchmarkConfig, dry_run):
    # print("------------------------------------------------------------------------------------------------------")
    # load_configs_str = ' '.join(config.load_configs)
    # scene_names_str = ' '.join(config.scene_names)
    # print("load_configs_str:", load_configs_str)
    # print("scene_names_str:", scene_names_str)

    """Train a single scene with config on current gpu"""
    sam_flag = "--sam" if config.sam else ""
    output_dir = f"results/sam_{config.sam}_interp_cam_{config.interpolate_n_camera}"
    # 生成命令
    cmd = (
        f"OMP_NUM_THREADS=4 "
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"python {config.function} "
        f"--load_config {config.load_config} "
        f"--dataset {config.dataset} "
        f"--scene_name {config.scene_name} "
        f"--experiment_type {config.experiment_type} "
        f"{sam_flag} "
        f"--project_name {config.project_name} "
        f"--wandb_mode {config.wandb_mode} "
        f"--kind {config.kind} "
        f"--output_dir {output_dir} "
        f"--interpolate_n_camera {config.interpolate_n_camera}"
    )
    print("Generated command:", cmd)  # Debugging print
    # output_file = f"{config.dataset} SAM:{config.sam} topk:{config.top_k} mode:{config.experiment_type} CAMERA_INTERP:{config.interpolate_n_camera} kind:{config.kind}" + ".txt"
    # run_command_and_save_output(cmd, output_file)

    if not dry_run:
        os.system(cmd)

    return True


def worker(config, gpu, dry_run):
    """This worker function starts a job and returns when it's done."""
    print(f"Starting {config.function} job on GPU {gpu} with")
    eval_scene(gpu, config, dry_run)
    print(f"Finished {config.function} job on GPU {gpu} with \n")


def dispatch_jobs(jobs, executor, dry_run):
    future_to_job = {}
    reserved_gpus = set()
    print("Jobs to dispatch:", jobs)
    while jobs or future_to_job:
        all_available_gpus = set(
            GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5)
        )
        available_gpus = list(all_available_gpus - reserved_gpus)

        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, job, gpu, dry_run)
            future_to_job[future] = (gpu, job)
            reserved_gpus.add(gpu)
            print(f"Dispatched job on GPU {gpu}")

        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)
            gpu = job[0]
            reserved_gpus.discard(gpu)
            print(f"Job {job} has finished, releasing GPU {gpu}")

        time.sleep(5)

    print("All jobs have been processed.")


def main(dry_run: False):
    """Launch batch_configs in serial but process each config in parallel (multi gpu)"""
    jobs = configs_to_run

    # Run multiple gpu scripts
    # Using ThreadPoolExecutor to manage the thread pool
    with ThreadPoolExecutor(max_workers=8) as executor:
        dispatch_jobs(jobs, executor, dry_run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--dry-run", help="run command in dry run mode", action="store_true"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
