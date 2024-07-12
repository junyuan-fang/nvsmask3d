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
    TemplatePipelineConfig,
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
from nerfstudio.data.dataparsers.scannet_dataparser import ScanNetDataParserConfig

NvsMask3d = MethodSpecification(
    config=TrainerConfig(
        method_name="nvsmask3d",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=TemplatePipelineConfig(
            datamanager=NVSMask3dDataManagerConfig(
                dataparser=ScanNetDataParserConfig()
            ),
            model=NVSMask3dModelConfig(
                cull_alpha_thresh=0.005,
                continue_cull_post_densification=False,
            ),
        ),
         optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6, max_steps=30000
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
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            }
         },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio method nvsmask3d.",
)



ScanNetDataparser = DataParserSpecification(config=ScanNetDataParserConfig())

