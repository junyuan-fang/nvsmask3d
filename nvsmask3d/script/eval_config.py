from pathlib import Path
from nvsmask3d.script.nvsmask3d_eval import ComputeForAP
import wandb
from tqdm import tqdm

# Visibility score functions
VISIBILITY_SCORES = {
    "visible_points": {
        "fn": lambda num_visible_points, bounding_box_area: num_visible_points,
        "description": "visible_points",
    },
    "visible_points*bounding_box_area": {
        "fn": lambda num_visible_points, bounding_box_area: num_visible_points
        * bounding_box_area,
        "description": "visible_points*bounding_box_area",
    },
}
SCENE_NAMES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room1",
    "room2",
    "room0",
]

LOAD_CONFIGS = [
    "outputs/office0/nvsmask3d/config.yml",
    "outputs/office1/nvsmask3d/config.yml",
    "outputs/office2/nvsmask3d/config.yml",
    "outputs/office3/nvsmask3d/config.yml",
    "outputs/office4/nvsmask3d/config.yml",
    "outputs/room1/nvsmask3d/config.yml",
    "outputs/room2/nvsmask3d/config.yml",
    "outputs/room0/nvsmask3d/config.yml",
]


class Experiment:
    def __init__(
        self,
        scene_names: list,
        load_configs: list,
        project_name: str,
        gt_camera_rgb: bool,
        gt_camera_gaussian: bool,
        visibility_score_key: str = "visible_points",
        dataset: str = "replica",
        occlusion_aware: bool = True,
        top_k: int = 15,
        interpolate_n_camera: int = 0,
        algorithm: int = 0,
        sam: bool = False,
        wandb_mode: str = "online",
        kind="crop",
        interp_kind = "same",
        **kwargs,
    ):
        self.top_k = top_k
        self.sam = sam
        self.visibility_score_fn = VISIBILITY_SCORES[visibility_score_key]["fn"]
        self.visibility_score_description = VISIBILITY_SCORES[visibility_score_key][
            "description"
        ]
        self.occlusion_aware = occlusion_aware
        self.interpolate_n_camera = interpolate_n_camera
        self.interpolate_n_rgb_camera = 1 if gt_camera_rgb else 0
        self.interpolate_n_gaussian_camera = 1 if gt_camera_gaussian else 0
        self.gt_camera_rgb = gt_camera_rgb
        self.gt_camera_gaussian = gt_camera_gaussian
        self.project_name = project_name
        self.algorithm = algorithm
        self.dataset = dataset
        self.wandb_mode = wandb_mode
        self.visibility_score_key = visibility_score_key
        self.kind = kind
        self.scene_names = scene_names
        self.load_configs = load_configs
        self.interp_kind = interp_kind
        self.run_name_for_wandb = self.generate_run_name()  # bottom always

    def generate_run_name(self) -> str:
        mode_info = f"MODE: {'rgb' if self.gt_camera_rgb else ''}-{'masked_gaussian' if self.gt_camera_gaussian else ''}"
        return f"{self.dataset} SAM:{self.sam} algo:{self.algorithm} topk:{self.top_k} {mode_info} CAMERA_INTERP:{self.interpolate_n_camera} VIS:{self.visibility_score_description} OCC_AWARE:{self.occlusion_aware} kind:{self.kind} interp_kind:{self.interp_kind}"

    def run(self):
        """Run the experiment and log results with WandB."""
        wandb.init(
            project=self.project_name,
            name=self.run_name_for_wandb,
            mode=self.wandb_mode,
            config={
                "top_k": self.top_k,
                "visibility_score_fn": self.visibility_score_fn,
                "occlusion_aware": self.occlusion_aware,
                "interpolate_n_camera": self.interpolate_n_camera,
                "interpolate_n_rgb_camera": self.interpolate_n_rgb_camera,
                "interpolate_n_gaussian_camera": self.interpolate_n_gaussian_camera,
                "gt_camera_rgb": self.gt_camera_rgb,
                "gt_camera_gaussian": self.gt_camera_gaussian,
                "project_name": self.project_name,
                "visibility_score_key": self.visibility_score_key,
                "algorithm": self.algorithm,
                "dataset": self.dataset,
                "sam": self.sam,
            },
        )

        compute_ap = ComputeForAP(
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
            wandb_mode=self.wandb_mode,
            kind=self.kind,
            interp_kind=self.interp_kind,
            scene_names=self.scene_names,
            load_configs=self.load_configs,
        )

        compute_ap.main()
        wandb.finish()


def create_experiments(
    scene_names: list,
    load_configs: list,
    project_name: str,
    experiment_type: str,
    dataset: str,
    sam: bool,
    algorithm: int,
    wandb_mode: str = "online",
    kind="crop",
    interp_kind = "same",
):
    """Creates experiments based on experiment type."""
    config_map = {
        "rgb": {"gt_camera_rgb": True, "gt_camera_gaussian": False},
        "gaussian": {"gt_camera_rgb": False, "gt_camera_gaussian": True},
        "mix": {"gt_camera_rgb": True, "gt_camera_gaussian": True},
    }

    if experiment_type not in config_map:
        raise ValueError(
            "Invalid experiment type. Choose from 'rgb', 'gaussian', or 'mix'."
        )

    config = config_map[experiment_type]
    return [
        Experiment(
            load_config=Path("nvsmask3d/data/replica"),
            project_name=project_name,
            visibility_score_key="visible_points",
            dataset=dataset,
            sam=sam,
            algorithm=algorithm,
            wandb_mode=wandb_mode,
            interpolate_n_camera=i,
            scene_names=scene_names,
            load_configs=load_configs,
            kind=kind,
            interp_kind=interp_kind,
            **config,
            # TODO
        )
        for i in range(1, 4)
    ]


def run_experiments(
    scene_names: list = SCENE_NAMES,
    load_configs: list = LOAD_CONFIGS,
    experiment_type: str = None,
    dataset: str = "replica",
    sam: bool = False,
    algorithm: int = 0,
    project_name: str = "depth_corrected",
    wandb_mode: str = "online",
    kind="crop",
    interp_kind = "same",
):
    experiments = create_experiments(
        scene_names,
        load_configs,
        project_name,
        experiment_type,
        dataset,
        sam,
        algorithm,
        wandb_mode,
        kind,
        interp_kind,
    )
    for experiment in tqdm(experiments):
        experiment.run()


if __name__ == "__main__":
    run_experiments(
        experiment_type="rgb",
        dataset="replica",
        sam=False,
        algorithm=0,
        project_name="crop",
        wandb_mode="disabled",
        kind="crop",
        interp_kind = "masked_gaussian", #average
    )
    # run_experiments(
    #     experiment_type="rgb",
    #     dataset="replica",
    #     sam=False,
    #     algorithm=0,
    #     project_name="crop",
    #     wandb_mode="disabled",
    #     kind="crop",
    #     interp_kind = "same", #average
    # )
    # run_experiments(
    #     experiment_type="rgb",
    #     dataset="replica",
    #     sam=False,
    #     algorithm=-1,
    #     project_name="crop",
    #     wandb_mode="disabled",
    #     kind="crop",
    #     interp_kind = "same", #average
    # )
    
    # run_experiments(
    #     experiment_type="rgb",
    #     dataset="replica",
    #     sam=False,
    #     algorithm=1,
    #     project_name="crop",
    #     wandb_mode="disabled",
    #     kind="crop",
    #     interp_kind = "masked_gaussian", #average
    # )
    
    # run_experiments(
    #     experiment_type="rgb",
    #     dataset="replica",
    #     sam=False,
    #     algorithm=0,
    #     project_name="crop",
    #     wandb_mode="disabled",
    #     kind="crop",
    #     interp_kind = "masked_gaussian",
    # )
    # run_experiments(
    #     experiment_type="rgb",
    #     dataset="replica",
    #     sam=True,
    #     algorithm=0,
    #     project_name="crop",
    #     wandb_mode="disabled",
    #     kind="crop",
    #     interp_kind = "same",
    # )
    # run_experiments(
    #     experiment_type="rgb",
    #     dataset="replica",
    #     sam=False,
    #     algorithm=0,
    #     project_name="crop",
    #     wandb_mode="disabled",
    #     kind="crop",
    #     interp_kind = "same",
    # )

    # online,offline,disabled
    # run_experiments(experiment_type="rgb", dataset="replica", sam=False, algorithm=0, project_name="blur", wandb_mode="disabled", kind="crop")# online,offline,disabled
    # run_experiments(experiment_type="rgb", dataset="replica", sam=False, algorithm=0, project_name="blur", wandb_mode="disabled", kind="blur")# online,offline,disabled

    # run_experiments(experiment_type="rgb", dataset="replica", sam=True, algorithm=0, project_name="rgb", wandb_mode="disabled")# online,offline,disabled

    # run_experiments(experiment_type="gaussian", dataset="replica", sam=False, algorithm=0, project_name="gaussian", wandb_mode="disabled")# online,offline,disabled
    # run_experiments(experiment_type="rgb", dataset="replica", sam=True, algorithm=0, project_name="SAM rgb", wandb_mode="disabled", kind = "crop")# online,offline,disabled
