import torch
from pathlib import Path
from typing import Optional
from nvsmask3d.script.nvsmask3d_eval import ComputeForAP
import wandb
from tqdm import tqdm

# Visibility score functions
VISIBILITY_SCORES = {
    "visible_points": {
        "fn": lambda num_visible_points, bounding_box_area: num_visible_points,
        "description": "visible_points"
    },
    "visible_points*bounding_box_area": {
        "fn": lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        "description": "visible_points*bounding_box_area"
    },
}

# Experiment class
class Experiment:
    def __init__(self,
                 load_config: Path,
                 top_k: int = 15,
                 sam: bool = False,
                 dataset: str = "replica",
                 visibility_score_key: str = "visible_points",
                 occlusion_aware: Optional[bool] = True,
                 interpolate_n_camera: Optional[int] = 0,
                 interpolate_n_rgb_camera: Optional[int] = 1,
                 interpolate_n_gaussian_camera: Optional[int] = 1,
                 gt_camera_rgb: Optional[bool] = True,
                 gt_camera_gaussian: Optional[bool] = True,
                 project_name: str = "depth_corrected",
                 run_name_for_wandb: Optional[str] = None,
                 algorithm: int = 0):
        # Initialize experiment configuration
        self.load_config = load_config
        self.top_k = top_k
        self.sam = sam
        self.visibility_score_key = visibility_score_key
        self.visibility_score_fn = VISIBILITY_SCORES[visibility_score_key]["fn"]
        self.visibility_score_description = VISIBILITY_SCORES[visibility_score_key]["description"]
        self.occlusion_aware = occlusion_aware
        self.interpolate_n_camera = interpolate_n_camera
        self.interpolate_n_rgb_camera = interpolate_n_rgb_camera
        self.interpolate_n_gaussian_camera = interpolate_n_gaussian_camera
        self.gt_camera_rgb = gt_camera_rgb
        self.gt_camera_gaussian = gt_camera_gaussian
        self.project_name = project_name
        self.algorithm = algorithm
        self.run_name_for_wandb = self.generate_run_name() if run_name_for_wandb is None else run_name_for_wandb
        self.dataset = dataset

    def generate_run_name(self) -> str:
        """Generate a WandB run name based on experiment parameters."""
        mode_info = f"MODE: {'rgb' if self.gt_camera_rgb else ''}-{'masked_gaussian' if self.gt_camera_gaussian else ''}"
        interpolation_str = f"CAMERA_INTERP: {self.interpolate_n_camera}"
        visibility_score_str = f"VIS: {self.visibility_score_description}"
        occlusion_str = f"OCC_AWARE: {self.occlusion_aware}"
        return f"SAM:{self.sam} algo:{self.algorithm} topk:{self.top_k} {mode_info} {interpolation_str} {visibility_score_str} {occlusion_str}"

    def run(self):
        """Run the experiment and log results with WandB."""
        wandb.init(project=self.project_name, name=self.run_name_for_wandb, config={
            "top_k": self.top_k,
            "visibility_score_fn": self.visibility_score_fn,
            "occlusion_aware": self.occlusion_aware,
            "interpolate_n_camera": self.interpolate_n_camera,
            "interpolate_n_rgb_camera": self.interpolate_n_rgb_camera,
            "interpolate_n_gaussian_camera": self.interpolate_n_gaussian_camera,
            "gt_camera_rgb": self.gt_camera_rgb,
            "gt_camera_gaussian": self.gt_camera_gaussian,
            "project_name": self.project_name,
            "run_name_for_wandb": self.run_name_for_wandb,
            "visibility_score_key": self.visibility_score_key,
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "sam": self.sam
        })

        compute_ap = ComputeForAP(
            load_config=self.load_config,
            sam=self.sam,
            top_k=self.top_k,
            visibility_score=self.visibility_score_fn,
            occlusion_aware=self.occlusion_aware,
            interpolate_n_camera=self.interpolate_n_camera,
            interpolate_n_rgb_camera=self.interpolate_n_rgb_camera,
            interpolate_n_gaussian_camera=self.interpolate_n_gaussian_camera,
            gt_camera_rgb=self.gt_camera_rgb,
            gt_camera_gaussian=self.gt_camera_gaussian,
            project_name=self.project_name,
            run_name_for_wandb=self.run_name_for_wandb,
            inference_dataset=self.dataset,
            algorithm=self.algorithm,
        )

        compute_ap.main()
        wandb.finish()


# Experiment configurations
def create_experiments(gt_camera_rgb: bool, gt_camera_gaussian: bool, interpolate_rgb: int, interpolate_gaussian: int, algorithm: int = 0, sam: bool = False, dataset = "replica"):
    """Creates experiments based on parameters."""
    return [
        Experiment(
            load_config=Path("nvsmask3d/data/replica"),
            top_k=15,
            gt_camera_rgb=gt_camera_rgb,
            gt_camera_gaussian=gt_camera_gaussian,
            interpolate_n_camera=i,
            interpolate_n_rgb_camera=1 if gt_camera_rgb else 0,
            interpolate_n_gaussian_camera=1 if gt_camera_gaussian else 0,
            visibility_score_key="visible_points",
            occlusion_aware=True,
            algorithm=algorithm,
            sam = sam,
            dataset = dataset
        ) for i in range(5)
    ]

# Run selected experiments
def run_experiments(experiment_type: str, dataset = "replica", sam = False, algorithm = 0):
    if experiment_type == "rgb":
        experiments = create_experiments(gt_camera_rgb=True, gt_camera_gaussian=False, interpolate_rgb=1, interpolate_gaussian=0, dataset=dataset, sam = sam, algorithm=algorithm)
    elif experiment_type == "gaussian":
        experiments = create_experiments(gt_camera_rgb=False, gt_camera_gaussian=True, interpolate_rgb=0, interpolate_gaussian=1, dataset=dataset, sam = sam, algorithm=algorithm)
    elif experiment_type == "mix":
        experiments = create_experiments(gt_camera_rgb=True, gt_camera_gaussian=True, interpolate_rgb=1, interpolate_gaussian=1, dataset=dataset, sam = sam, algorithm=algorithm)
    else:
        raise ValueError("Invalid experiment type. Choose from 'rgb', 'gaussian', or 'mix'.")

    for experiment in tqdm(experiments):
        experiment.run()

if __name__ == "__main__":
    #run_experiments("rgb")  # Modify to "gaussian" or "mix" as needed
    run_experiments(experiment_type= "gaussian", dataset = "replica", sam = False, algorithm=0)