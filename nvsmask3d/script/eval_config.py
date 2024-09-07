import torch
from pathlib import Path
from typing import Callable, Optional
from nvsmask3d.script.nvsmask3d_eval import ComputeForAP  # 导入你的 ComputeForAP 类
import tqdm

# 定义实验类
class Experiment:
    def __init__(self,
                 load_config: Path,
                 top_k: int = 15,
                 visibility_score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda num_visible_points, bounding_box_area: num_visible_points,
                 occlusion_aware: Optional[bool] = True,
                 interpolate_n_camera: Optional[int] = 0,
                 interpolate_n_rgb_camera: Optional[int] = 1,
                 interpolate_n_gaussian_camera: Optional[int] = 1,
                 gt_camera_rgb: Optional[bool] = True,
                 gt_camera_gaussian: Optional[bool] = True,
                 project_name: str = "nvsmask3d_evaluation",
                 run_name_for_wandb: Optional[str] = None):
        # 初始化实验配置
        self.load_config = load_config
        self.top_k = top_k
        self.visibility_score_fn = visibility_score_fn
        self.occlusion_aware = occlusion_aware
        self.interpolate_n_camera = interpolate_n_camera
        self.interpolate_n_rgb_camera = interpolate_n_rgb_camera
        self.interpolate_n_gaussian_camera = interpolate_n_gaussian_camera
        self.gt_camera_rgb = gt_camera_rgb
        self.gt_camera_gaussian = gt_camera_gaussian
        self.project_name = project_name
        self.run_name_for_wandb = self.generate_run_name() if run_name_for_wandb is None else run_name_for_wandb

    def generate_run_name(self) -> str:
        """根据实验参数生成合理的 WandB run name."""
        occlusion_str = "occ-aware" if self.occlusion_aware else "no-occ"
        rgb_camera_str = f"rgb-cam-{self.interpolate_n_rgb_camera}" if self.gt_camera_rgb else "no-rgb-cam"
        gaussian_camera_str = f"gaussian-cam-{self.interpolate_n_gaussian_camera}" if self.gt_camera_gaussian else "no-gaussian-cam"
        interpolation_str = f"interp-{self.interpolate_n_camera}-cam"
        
        # 生成实验名字
        run_name = f"{self.project_name}_topk-{self.top_k}_{occlusion_str}_{interpolation_str}_{rgb_camera_str}_{gaussian_camera_str}"
        return run_name
    # 运行实验的方法
    def run(self):
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

# 创建实验配置
experiments = [
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
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=False,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        gt_camera_rgb=True,
        gt_camera_gaussian=True
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points ,
        occlusion_aware=False,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=0,
        gt_camera_rgb=True,
        gt_camera_gaussian=True
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points,
        occlusion_aware=False,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=0,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=True
    ),
    #masked gaussian
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=True,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=False,
        gt_camera_gaussian=True
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=False,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=False,
        gt_camera_gaussian=True
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=True,
        interpolate_n_camera=2,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=False,
        gt_camera_gaussian=True
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=True,
        interpolate_n_camera=3,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=False,
        gt_camera_gaussian=True
    ),

        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=True,
        interpolate_n_camera=4,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=False,
        gt_camera_gaussian=True
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points ,
        occlusion_aware=True,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=False,
        gt_camera_gaussian=True
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points ,
        occlusion_aware=False,
        interpolate_n_camera=0,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=False,
        gt_camera_gaussian=True
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points ,
        occlusion_aware=False,
        interpolate_n_camera=0,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=False,
        gt_camera_gaussian=True
    ),
    #rgb
            Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=True,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=False
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=False,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=False
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=True,
        interpolate_n_camera=2,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=False
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=True,
        interpolate_n_camera=3,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=False
    ),

        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=True,
        interpolate_n_camera=4,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=False
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points ,
        occlusion_aware=True,
        interpolate_n_camera=1,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=False
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points ,
        occlusion_aware=False,
        interpolate_n_camera=0,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=False
    ),
        Experiment(
        load_config=Path("nvsmask3d/data/replica"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points ,
        occlusion_aware=False,
        interpolate_n_camera=0,
        interpolate_n_rgb_camera=1,#based on interpolate_n_camera, this will only be used as 0 or 1 first.
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=False
    ),

        
        
]

# 批量运行实验
for experiment in tqdm.tqdm(experiments):
    experiment.run()    