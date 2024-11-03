# Benchmark script

import glob
import os
import time
from typing import Optional
from typing import List, Union
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import GPUtil

run_dataset = "scannetpp"  # "mip_360"

# path to datasets
data_dirs = {
    "scannetpp": "data/360_v2",
    "replica": "data/bilarf_data/bilarf_data/testscenes",
}

# scenes to run from dataset
scenes = {
    "scannetpp": [
        '7b6477cb95', 'c50d2d1d42', 'cc5237fd77', 'acd95847c5', 'fb5a96b1a2', 'a24f64f7fb',
        '1ada7a0617', '5eb31827b7', '3e8bba0176', '3f15a9266d', '21d970d8de', '5748ce6f01',
        'c4c04e6d6c', '7831862f02', 'bde1e479ad', '38d58a7a31', '5ee7c22ba0', 'f9f95681fd',
        '3864514494', '40aec5fffa', '13c3e046d7', 'e398684d27', 'a8bf42d646', '45b0dac5e3',
        '31a2c91c43', 'e7af285f7d', '286b55a2bf', '7bc286c1b6', 'f3685d06a9', 'b0a08200c9',
        '825d228aec', 'a980334473', 'f2dc06b1d2', '5942004064', '25f3b7a318', 'bcd2436daf',
        'f3d64c30f8', '0d2ee665be', '3db0a1c8f3', 'ac48a9b736', 'c5439f4607', '578511c8a9',
        'd755b3d9d8', '99fa5c25e1', '09c1414f1b', '5f99900f09', '9071e139d9', '6115eddb86',
        '27dd4da69e', 'c49a8c6cff'
    ],
    "replica": [
        "office0",
        "office1",
        "office2",
        "office3",
        "office4",
        "room1",
        "room2",
        "room0",
    ],
}
REPLICA_LOAD_CONFIGS = [

                "outputs/office0/nvsmask3d/config.yml",
                "outputs/office1/nvsmask3d/config.yml",
                "outputs/office2/nvsmask3d/config.yml",
                "outputs/office3/nvsmask3d/config.yml",
                "outputs/office4/nvsmask3d/config.yml",
                "outputs/room1/nvsmask3d/config.yml",
                "outputs/room2/nvsmask3d/config.yml",
                "outputs/room0/nvsmask3d/config.yml",

            ]

@dataclass
class BenchmarkConfig:
    """Baseline benchmark config"""
    # trainer to run
    function: str = "nvsmask3d/script/eval_config.py"
    # path to data
    dataset : str = "replica"
    kind : str = "crop"
    sam : bool = False
    wandb_mode : str = "disabled"
    project_name : str = "crop"
    experiment_type : str = "rgb"   
    scene_names: Optional[List[str]]= None  # Accept either str or List
    load_configs: Optional[List[str]] = None  # Accept either str or List
    dry_run: bool = False
    excluded_gpus: set = field(default_factory=set)

# Configurations of different options
replica_config = BenchmarkConfig(load_configs = REPLICA_LOAD_CONFIGS, scene_names = scenes["replica"])
scannetpp_config = BenchmarkConfig(load_configs = "outputs/7b6477cb95_dslr_colmap/nvsmask3d/config.yml")

configs_to_run = [  
    replica_config
]
# Jobs to run or different "configs" to run
# configs_to_run = [
#     first_config,
#     second_config
# ]


SKIP_TRAIN = False


# def train_scene(gpu, config: BenchmarkConfig):
#     """Train a single scene with config on current gpu"""
#     # additional user set model configs
#     model_config_args = " ".join(f"{k} {v}" for k, v in config.model_configs.items())

#     if not SKIP_TRAIN:
#         # train without eval
#         cmd = f"OMP_NUM_THREADS=4 " \
#                 f"CUDA_VISIBLE_DEVICES={gpu} " \
#                 f"python {config.function} " \
#                 f"--load-configs {config.load_configs} " \
#                 f"--dataset {config.dataset} " \
#                 f"--scene-names {config.scene_names} " \
#                 f"--experiment_type {config.experiment_type} " \
#                 f"--sam {config.sam} " \
#                 f"--project_name {config.project_name} " \
#                 f"--wandb_mode {config.wandb_mode} " \
#                 f"--kind {config.kind}"

#         if not config.dry_run:
#             os.system(cmd)
#     return True

def train_scene(gpu, config: BenchmarkConfig):
    print("------------------------------------------------------------------------------------------------------")
    load_configs_str = ' '.join(config.load_configs)
    scene_names_str = ' '.join(config.scene_names)
    print("load_configs_str:", load_configs_str)
    print("scene_names_str:", scene_names_str)

    """Train a single scene with config on current gpu"""
    cmd = f"OMP_NUM_THREADS=4 " \
          f"CUDA_VISIBLE_DEVICES={gpu} " \
          f"python {config.function} " \
          f"--load-configs {load_configs_str} " \
          f"--dataset {config.dataset} " \
          f"--scene-names {scene_names_str} " \
          f"--experiment_type {config.experiment_type} " \
          f"--sam {config.sam} " \
          f"--project_name {config.project_name} " \
          f"--wandb_mode {config.wandb_mode} " \
          f"--kind {config.kind}"

    print("Generated command:", cmd)  # Debugging print
    if not config.dry_run:
        os.system(cmd)


def worker(config, gpu):
    """This worker function starts a job and returns when it's done."""
    print(f"Starting {config.function} job on GPU {gpu} with")
    train_scene(gpu, config)
    print(f"Finished {config.function} job on GPU {gpu} with \n")


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()
    print("Jobs to dispatch:", jobs)
    while jobs or future_to_job:
        print("Checking for available GPUs...")
        all_available_gpus = set(
            GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5)
        )
        available_gpus = list(all_available_gpus - reserved_gpus)
        print("Available GPUs:", available_gpus)
        print("Reserved GPUs:", reserved_gpus)

        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, job, gpu)
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


def main():
    """Launch batch_configs in serial but process each config in parallel (multi gpu)"""

    # list: [(garden, data), ()]
    jobs = configs_to_run

    # Run multiple gpu scripts
    # Using ThreadPoolExecutor to manage the thread pool
    with ThreadPoolExecutor(max_workers=8) as executor:
        dispatch_jobs(jobs, executor)


if __name__ == "__main__":
    main()