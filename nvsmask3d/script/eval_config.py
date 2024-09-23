import torch
from pathlib import Path
from typing import Callable, Optional
from nvsmask3d.script.nvsmask3d_eval import ComputeForAP  # 导入你的 ComputeForAP 类
import wandb
VISIBILITY_SCORES = {
    "visible_points": {
        "fn": lambda num_visible_points, bounding_box_area: num_visible_points,
        "description": "visible_points"
    },
    "visible_points*bounding box area": {
        "fn": lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        "description": "visible_points*bounding box area"
    },
    # 可以添加更多的visibility score函数
}
# 定义实验类
class Experiment:
    def __init__(self,
                 load_config: Path,
                 top_k: int = 15,
                 visibility_score_key: str = "visible_points",
                 occlusion_aware: Optional[bool] = True,
                 interpolate_n_camera: Optional[int] = 0,
                 interpolate_n_rgb_camera: Optional[int] = 1,
                 interpolate_n_gaussian_camera: Optional[int] = 1,
                 gt_camera_rgb: Optional[bool] = True,
                 gt_camera_gaussian: Optional[bool] = True,
                 project_name: str = "nvsmask3d_evaluation",#"zeroshot_enhancement",
                 run_name_for_wandb: Optional[str] = None,
                 algorithm: int = 0):
        # 初始化实验配置
        self.load_config = load_config
        self.top_k = top_k
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
    def generate_run_name(self) -> str:
        """根据实验参数生成合理的 WandB run name."""

        mode_info = f"MODE: {'rgb' if self.gt_camera_rgb else ''}-{'masked_gaussian' if self.gt_camera_gaussian else ''}"
        interpolation_str = f"CAMERA_INTERP: " \
                    f"{str(self.interpolate_n_camera) + 'rgb' if self.interpolate_n_rgb_camera*self.interpolate_n_camera> 0 else ''}-" \
                    f"{str(self.interpolate_n_camera) + '-masked gaussian' if self.interpolate_n_gaussian_camera*self.interpolate_n_camera > 0 else ''}"
        visibility_score_str = f"VIS: {self.visibility_score_description}"
        occlusion_str = f"OCC_AWARE: {str(self.occlusion_aware)}"


        
        # 生成实验名字
        run_name = f"algo:{self.algorithm} topk:{self.top_k}  {mode_info}  {interpolation_str}  {visibility_score_str}  {occlusion_str}"
        return run_name
    # 运行实验的方法
    def run(self):
        # Initialize a new WandB run for each experiment
        wandb.init(
            # mode="disabled",
            project=self.project_name,
            name=self.run_name_for_wandb,
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
                "run_name_for_wandb": self.run_name_for_wandb,
                "visibility_score_key": self.visibility_score_key,
                "algorithm": self.algorithm
            }
        )
        # 初始化 ComputeForAP 实例
        compute_ap = ComputeForAP(
            load_config=self.load_config,
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
            inference_dataset = "replica",
        )

        # 运行 ComputeForAP 的 main 方法
        compute_ap.main()
        # Finish the current WandB run so the next one is a fresh run
        wandb.finish()

# 创建实验配置
    # Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
    #     occlusion_aware=True,
    #     interpolate_n_camera=1,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
    #     occlusion_aware=True,
    #     interpolate_n_camera=1,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=0,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
    #     occlusion_aware=False,
    #     interpolate_n_camera=1,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=0,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points ,
    #     occlusion_aware=False,
    #     interpolate_n_camera=1,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=0,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points,
    #     occlusion_aware=False,
    #     interpolate_n_camera=1,
    #     interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True
    # ),

    #masked gaussian
gaussian_experiment=[

        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=False,
        gt_camera_gaussian=True,
        interpolate_n_camera=0,
        interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=False,
        gt_camera_gaussian=True,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=False,
        gt_camera_gaussian=True,
        interpolate_n_camera=2,
        interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=False,
        gt_camera_gaussian=True,
        interpolate_n_camera=3,
        interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0

    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=False,
        gt_camera_gaussian=True,
        interpolate_n_camera=4,
        interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0

    ),
    

            
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=False,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=6,
    #     interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points",
    #     occlusion_aware=True,
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=False,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=7,
    #     interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points",
    #     occlusion_aware=True,
    # ),

    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=False,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=0,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points",
    #     occlusion_aware=False,
    # )
]

#rgb
rgb_experiment=[
    
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=0,
        interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=2,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=3,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=4,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=0
    ),
                Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=0,
        interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=1
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=1
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=2,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=1
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=3,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=1
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=False,
        interpolate_n_camera=4,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        algorithm=1
    ),
]

mix_experiment=[
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=True,
        interpolate_n_camera=0,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        visibility_score_key="visible_points",
        occlusion_aware=True,
        ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=True,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        visibility_score_key="visible_points",
        occlusion_aware=True,
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=True,
        interpolate_n_camera=2,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        visibility_score_key="visible_points",
        occlusion_aware=True,
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=True,
        interpolate_n_camera=3,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        visibility_score_key="visible_points",
        occlusion_aware=True,
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        gt_camera_rgb=True,
        gt_camera_gaussian=True,
        interpolate_n_camera=4,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        visibility_score_key="visible_points",
        occlusion_aware=True,
    ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=1,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points*bounding box area",
    #     occlusion_aware=True,
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=2,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points*bounding box area",
    #     occlusion_aware=True,
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=3,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points*bounding box area",
    #     occlusion_aware=True,
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=4,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points*bounding box area",
    #     occlusion_aware=True,
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=1,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points",
    #     occlusion_aware=True,
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=2,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points",
    #     occlusion_aware=True,
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=3,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points",
    #     occlusion_aware=True,
    # ),
    #     Experiment(
    #     load_config=Path("nvsmask3d/data/replica"),
    #     top_k=15,
    #     gt_camera_rgb=True,
    #     gt_camera_gaussian=True,
    #     interpolate_n_camera=4,
    #     interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
    #     interpolate_n_gaussian_camera=1,
    #     visibility_score_key="visible_points",
    #     occlusion_aware=True,
    # ),
]

# def get_experiments():
#     return experiments    
def get_rgb_experiment():
    return rgb_experiment
def get_gaussian_experiment():
    return gaussian_experiment
def get_mix_experiment():
    return mix_experiment
# def get_rgb_gaussian_experiment():
#     return rgb_gaussian_experiment