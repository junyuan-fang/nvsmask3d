import torch
from pathlib import Path
from typing import Optional
from nvsmask3d.script.nvsmask3d_eval import ComputeForAP
import wandb
import ast
from typing import List


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
SCENE_NAMES = ['7b6477cb95', 'c50d2d1d42', 'cc5237fd77', 'acd95847c5', 'fb5a96b1a2', 'a24f64f7fb',
            '1ada7a0617', '5eb31827b7', '3e8bba0176', '3f15a9266d', '21d970d8de', '5748ce6f01',
            'c4c04e6d6c', '7831862f02', 'bde1e479ad', '38d58a7a31', '5ee7c22ba0', 'f9f95681fd',
            '3864514494', '40aec5fffa', '13c3e046d7', 'e398684d27', 'a8bf42d646', '45b0dac5e3',
            '31a2c91c43', 'e7af285f7d', '286b55a2bf', '7bc286c1b6', 'f3685d06a9', 'b0a08200c9',
            '825d228aec', 'a980334473', 'f2dc06b1d2', '5942004064', '25f3b7a318', 'bcd2436daf',
            'f3d64c30f8', '0d2ee665be', '3db0a1c8f3', 'ac48a9b736', 'c5439f4607', '578511c8a9',
            'd755b3d9d8', '99fa5c25e1', '09c1414f1b', '5f99900f09', '9071e139d9', '6115eddb86',
            '27dd4da69e', 'c49a8c6cff']

LOAD_CONFIGS = [ f"nvsmask3d/outputs/{scene}_dslr_colmap/nvsmask3d/config.yml" for scene in SCENE_NAMES
            ]

class Experiment:
    def __init__(self, scene_names: Optional[List[str]], load_configs: Optional[List[str]], project_name: str, gt_camera_rgb: bool, gt_camera_gaussian: bool,
                 visibility_score_key: str = "visible_points", dataset: str = "replica", occlusion_aware: bool = True,
                 top_k: int = 15, interpolate_n_camera: int = 0, algorithm: int = 0, sam: bool = False, wandb_mode: str = "online", kind = "crop",**kwargs):
        
        self.top_k = top_k
        self.sam = sam
        self.visibility_score_fn = VISIBILITY_SCORES[visibility_score_key]["fn"]
        self.visibility_score_description = VISIBILITY_SCORES[visibility_score_key]["description"]
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
        self.run_name_for_wandb = self.generate_run_name()#bottom always


    def generate_run_name(self) -> str:
        mode_info = f"MODE: {'rgb' if self.gt_camera_rgb else ''}-{'masked_gaussian' if self.gt_camera_gaussian else ''}"
        return f"{self.dataset} SAM:{self.sam} algo:{self.algorithm} topk:{self.top_k} {mode_info} CAMERA_INTERP:{self.interpolate_n_camera} VIS:{self.visibility_score_description} OCC_AWARE:{self.occlusion_aware} kind:{self.kind}"

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
                "sam": self.sam
            }
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
            kind = self.kind,
            scene_names=self.scene_names,
            load_configs=self.load_configs
            
        )

        compute_ap.main()
        wandb.finish()


def create_experiments(scene_names: Optional[List[str]], load_configs: Optional[List[str]], project_name: str, experiment_type: str, dataset: str, sam: bool, algorithm: int, wandb_mode: str = "online", kind = "crop"):
    """Creates experiments based on experiment type."""
    config_map = {
        "rgb": {"gt_camera_rgb": True, "gt_camera_gaussian": False},
        "gaussian": {"gt_camera_rgb": False, "gt_camera_gaussian": True},
        "mix": {"gt_camera_rgb": True, "gt_camera_gaussian": True}
    }

    if experiment_type not in config_map:
        raise ValueError("Invalid experiment type. Choose from 'rgb', 'gaussian', or 'mix'.")

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
            **config
        ) for i in range(0,5)
    ]


def run_experiments(scene_names: Optional[List[str]] = SCENE_NAMES, load_configs: Optional[List[str]] = LOAD_CONFIGS, experiment_type: str = "rgb", dataset: str = "replica", sam: bool = False, algorithm: int = 0, project_name: str = "depth_corrected", wandb_mode: str = "online", kind = "crop"):
    # 将字符串解析为列表
    scene_names = ast.literal_eval(scene_names) if isinstance(scene_names, str) else scene_names
    load_configs = ast.literal_eval(load_configs) if isinstance(load_configs, str) else load_configs

    experiments = create_experiments(scene_names, load_configs, project_name, experiment_type, dataset, sam, algorithm, wandb_mode, kind)
    for experiment in tqdm(experiments):
        experiment.run()

def scannetpp_run_experiments(scene_names: Optional[List[str]] = SCENE_NAMES, load_configs: Optional[List[str]] = LOAD_CONFIGS, experiment_type: str = "rgb", dataset: str = "replica", sam: bool = False, algorithm: int = 0, project_name: str = "depth_corrected", wandb_mode: str = "online", kind = "crop"):
    # 将字符串解析为列表
    scene_names = ast.literal_eval(scene_names) if isinstance(scene_names, str) else scene_names
    load_configs = ast.literal_eval(load_configs) if isinstance(load_configs, str) else load_configs

    experiments = create_experiments(scene_names, load_configs, project_name, experiment_type, dataset, sam, algorithm, wandb_mode, kind)
    for experiment in tqdm(experiments):
        experiment.run()
    
    #

if __name__ == "__main__": 
    # import tyro
    # tyro.cli(run_experiments)  
    scannetpp_run_experiments(experiment_type="rgb", dataset="scannetpp", sam=False, algorithm=0, project_name="rgb", wandb_mode="disabled", kind="crop")
    #run_experiments(experiment_type="rgb", dataset="replica", sam=False, algorithm=0, project_name="blur", wandb_mode="disabled", kind="blur")# online,offline,disabled
    #run_experiments(experiment_type="rgb", dataset="", sam=False, algorithm=0, project_name="blur", wandb_mode="disabled", kind="blur")# online,offline,disabled

    #run_experiments(experiment_type="rgb", dataset="replica", sam=True, algorithm=0, project_name="rgb", wandb_mode="disabled")# online,offline,disabled

    #run_experiments(experiment_type="gaussian", dataset="replica", sam=False, algorithm=0, project_name="gaussian", wandb_mode="disabled")# online,offline,disabled
    #run_experiments(experiment_type="rgb", dataset="replica", sam=True, algorithm=0, project_name="SAM rgb", wandb_mode="disabled", kind = "crop")# online,offline,disabled