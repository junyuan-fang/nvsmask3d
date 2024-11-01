# Benchmark script

import glob
import os
import time
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
    "scannetpp": (
        '7b6477cb95', 'c50d2d1d42', 'cc5237fd77', 'acd95847c5', 'fb5a96b1a2', 'a24f64f7fb',
        '1ada7a0617', '5eb31827b7', '3e8bba0176', '3f15a9266d', '21d970d8de', '5748ce6f01',
        'c4c04e6d6c', '7831862f02', 'bde1e479ad', '38d58a7a31', '5ee7c22ba0', 'f9f95681fd',
        '3864514494', '40aec5fffa', '13c3e046d7', 'e398684d27', 'a8bf42d646', '45b0dac5e3',
        '31a2c91c43', 'e7af285f7d', '286b55a2bf', '7bc286c1b6', 'f3685d06a9', 'b0a08200c9',
        '825d228aec', 'a980334473', 'f2dc06b1d2', '5942004064', '25f3b7a318', 'bcd2436daf',
        'f3d64c30f8', '0d2ee665be', '3db0a1c8f3', 'ac48a9b736', 'c5439f4607', '578511c8a9',
        'd755b3d9d8', '99fa5c25e1', '09c1414f1b', '5f99900f09', '9071e139d9', '6115eddb86',
        '27dd4da69e', 'c49a8c6cff'
    ),
    "replica": (
        "office0",
        "office1",
        "office2",
        "office3",
        "office4",
        "room1",
        "room2",
        "room0",
    ),
}


@dataclass
class BenchmarkConfig:
    """Baseline benchmark config"""

    # trainer to run
    function: str = "eval_config_run.py"
    # path to data
    load_config : str = "outputs/7b6477cb95_dslr_colmap/nvsmask3d/config.yml"
    dry_run: bool = True
    excluded_gpus: set = field(default_factory=set)

# Configurations of different options
first_config = BenchmarkConfig(
    load_config = "outputs/7b6477cb95_dslr_colmap/nvsmask3d/config.yml")
second_config = first_config

# Jobs to run or different "configs" to run
configs_to_run = [
    first_config,
    second_config
]

SKIP_TRAIN = False


def train_scene(gpu, config: BenchmarkConfig):
    """Train a single scene with config on current gpu"""
    # additional user set model configs
    model_config_args = " ".join(f"{k} {v}" for k, v in config.model_configs.items())

    if not SKIP_TRAIN:
        # train without eval
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python {config.function} --load-config {config.load_config}"
        print(cmd)

        if not config.dry_run:
            os.system(cmd)

    return True


def worker(config, gpu):
    """This worker function starts a job and returns when it's done."""
    print(f"Starting {config.function} job on GPU {gpu} with")
    train_scene(gpu, config)
    print(f"Finished {config.function} job on GPU {gpu} with \n")


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet
    print(jobs)
    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(
            GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1, maxLoad=0.1)
        )
        available_gpus = list(all_available_gpus - reserved_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(
                worker, job, gpu
            )  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)
            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(
                future
            )  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., releasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
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