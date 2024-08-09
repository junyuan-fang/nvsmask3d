"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nvsmask3d.nvsmask3d_datamanager import (
    NVSMask3dDataManagerConfig,
)
from nvsmask3d.nvsmask3d_model import NVSMask3dModelConfig
from nvsmask3d.nvsmask3d_pipeline import (
    NvsMask3dPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nvsmask3d.dataparsers.scannet_dataparser import ScanNetDataParserConfig
from nvsmask3d.encoders.open_clip_encoder import OpenCLIPNetworkConfig
from pathlib import Path


NvsMask3d = MethodSpecification(
    config=TrainerConfig(
        method_name="nvsmask3d",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=20000,
        mixed_precision=False,
        pipeline=NvsMask3dPipelineConfig(
            datamanager=NVSMask3dDataManagerConfig(
                dataparser=ScanNetDataParserConfig(
                    load_3D_points=True,
                    load_every=1,
                    train_split_fraction=0.9,
                    mask_path=Path(
                        "/home/wangs9/junyuan/openmask3d/output/2024-08-08-13-27-09-scene0000_00_/scene0011_00_vh_clean_2_masks.pt"
                    )
                    # Path('/home/wangs9/junyuan/openmask3d/output/2024-08-07-20-48-45-scene0000_00_/scene0011_00_masks.pt'),
                    # Path('/home/wangs9/junyuan/openmask3d/output/2024-07-23-11-44-44-scene0000_00_/scene0000_00__masks.pt'),
                    # data=Path("/home/wangs9/junyuan/openmask3d/data/scene0000_00_/"),
                ),
                cache_images_type="uint8",
            ),
            model=NVSMask3dModelConfig(
                lock_means=True,
                # warmup_length = 7500
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-B-16",
                clip_model_pretrained="laion2b_s34b_b88k",
                clip_n_dims=512,
            ),
            #  You can swap the type of input encoder by specifying different NetworkConfigs, the one below uses OpenAI CLIP, the one above uses OpenCLIP
            # network=CLIPNetworkConfig(
            #     clip_model_type="ViT-B/16", clip_n_dims=512
            # )
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer+wandb",
    ),
    description="Nerfstudio method nvsmask3d.",
)
