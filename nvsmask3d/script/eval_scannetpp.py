import torch
from pathlib import Path
from typing import Optional
from nvsmask3d.script.nvsmask3d_eval_single import ComputeForAP
import wandb
import ast
from typing import List


from tqdm import tqdm


def scannetpp_run_experiment(
    scene_name: str = None,
    load_config: str = None,
    experiment_type: str = "rgb",
    dataset: str = "scannetpp",
    sam: bool = False,
    algorithm: int = 0,
    project_name: str = None,
    wandb_mode: str = "disabled",  # online,offline,disabled
    kind: str = "crop",
    top_k: int = 15, 
    interpolate_n_camera: int = 0,
    output_dir: str = None,
):  
    if sam:
        print("SAM is enabled.")
    else:
        print("SAM is disabled.")
    # Map for experiment configuration
    config_map = {
        "rgb": {"gt_camera_rgb": True, "gt_camera_gaussian": False, "interpolate_n_rgb_camera": 1, "interpolate_n_gaussian_camera": 0},
        "gaussian": {"gt_camera_rgb": False, "gt_camera_gaussian": True, "interpolate_n_rgb_camera": 0, "interpolate_n_gaussian_camera": 1},
        "mix": {"gt_camera_rgb": True, "gt_camera_gaussian": True, "interpolate_n_rgb_camera": 1, "interpolate_n_gaussian_camera": 1}
    }
    if experiment_type not in config_map:
        raise ValueError("Invalid experiment type. Choose from 'rgb', 'gaussian', or 'mix'.")

    config = config_map[experiment_type]
    run_name_for_wandb = f"{dataset} SAM:{sam} algo:{algorithm} topk:{top_k} mode:{experiment_type} CAMERA_INTERP:{interpolate_n_camera} kind:{kind}"

    # Create an instance of ComputeForAP with appropriate arguments
    compute_ap = ComputeForAP(
        sam=sam,
        top_k=top_k,  # Replace this with the correct value if needed
        occlusion_aware=True,  # Adjust based on actual needs
        interpolate_n_camera=interpolate_n_camera,  # Adjust as needed
        project_name=project_name,
        run_name_for_wandb=run_name_for_wandb,  # Customize as needed
        inference_dataset=dataset,
        algorithm=algorithm,
        wandb_mode=wandb_mode,
        kind=kind,
        scene_name=scene_name,
        load_config=load_config,
        output_dir=output_dir,  # Customize as needed
        **config
    )
    wandb.init(
            project=project_name, 
            name=run_name_for_wandb, 
            mode= wandb_mode,)
    # Run the main computation
    compute_ap.main()
    wandb.finish()
if __name__ == "__main__":
    import tyro
    tyro.cli(scannetpp_run_experiment) 
