from typing import Callable, Optional
from pathlib import Path
import torch
import wandb

class Experiment:
    def __init__(self,
                 experiment_name: str,
                 load_config: Path,
                 output_path: Path,
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
        self.experiment_name = experiment_name
        self.load_config = load_config
        self.output_path = output_path
        self.top_k = top_k
        self.visibility_score_fn = visibility_score_fn
        self.occlusion_aware = occlusion_aware
        self.interpolate_n_camera = interpolate_n_camera
        self.interpolate_n_rgb_camera = interpolate_n_rgb_camera
        self.interpolate_n_gaussian_camera = interpolate_n_gaussian_camera
        self.gt_camera_rgb = gt_camera_rgb
        self.gt_camera_gaussian = gt_camera_gaussian
        self.project_name = project_name
        self.run_name_for_wandb = run_name_for_wandb or f"{experiment_name}_run"

    # 运行实验的方法
    def run(self):
        # 初始化 WandB 日志
        wandb.init(project=self.project_name, name=self.run_name_for_wandb)

        # 模拟 num_visible_points 和 bounding_box_area
        num_visible_points = torch.tensor(10.0)
        bounding_box_area = torch.tensor(0.8)

        # 计算 visibility_score
        visibility_score = self.visibility_score_fn(num_visible_points, bounding_box_area)

        # 输出配置和计算结果
        print(f"Running experiment: {self.experiment_name}")
        print(f"Visibility score: {visibility_score}")
        print(f"Occlusion Aware: {self.occlusion_aware}")
        print(f"Interpolate N Camera: {self.interpolate_n_camera}")
        print(f"Interpolate N RGB Camera: {self.interpolate_n_rgb_camera}")
        print(f"Interpolate N Gaussian Camera: {self.interpolate_n_gaussian_camera}")
        print(f"GT Camera RGB: {self.gt_camera_rgb}")
        print(f"GT Camera Gaussian: {self.gt_camera_gaussian}")

        # 模拟将结果保存到输出文件
        with open(self.output_path, 'w') as f:
            f.write(f"Visibility score: {visibility_score}\n")
            f.write(f"Occlusion Aware: {self.occlusion_aware}\n")
            f.write(f"Interpolate N Camera: {self.interpolate_n_camera}\n")
            f.write(f"Interpolate N RGB Camera: {self.interpolate_n_rgb_camera}\n")
            f.write(f"Interpolate N Gaussian Camera: {self.interpolate_n_gaussian_camera}\n")
            f.write(f"GT Camera RGB: {self.gt_camera_rgb}\n")
            f.write(f"GT Camera Gaussian: {self.gt_camera_gaussian}\n")
        
        # 将结果记录到 WandB
        wandb.log({
            "visibility_score": visibility_score,
            "occlusion_aware": self.occlusion_aware,
            "interpolate_n_camera": self.interpolate_n_camera,
            "interpolate_n_rgb_camera": self.interpolate_n_rgb_camera,
            "interpolate_n_gaussian_camera": self.interpolate_n_gaussian_camera,
            "gt_camera_rgb": self.gt_camera_rgb,
            "gt_camera_gaussian": self.gt_camera_gaussian
        })
        wandb.finish()

# 创建实验配置
experiments = [
    Experiment(
        experiment_name="experiment_1",
        load_config=Path("nvsmask3d/data/replica"),
        output_path=Path("output_1.txt"),
        top_k=15,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points * bounding_box_area,
        occlusion_aware=True,
        interpolate_n_camera=2,
        interpolate_n_rgb_camera=1,
        interpolate_n_gaussian_camera=1,
        gt_camera_rgb=True,
        gt_camera_gaussian=False
    ),
    Experiment(
        experiment_name="experiment_2",
        load_config=Path("nvsmask3d/data/replica"),
        output_path=Path("output_2.txt"),
        top_k=10,
        visibility_score_fn=lambda num_visible_points, bounding_box_area: num_visible_points,
        occlusion_aware=False,
        interpolate_n_camera=3,
        interpolate_n_rgb_camera=2,
        interpolate_n_gaussian_camera=0,
        gt_camera_rgb=False,
        gt_camera_gaussian=True
    )
]

# 批量运行实验
for experiment in experiments:
    experiment.run()
