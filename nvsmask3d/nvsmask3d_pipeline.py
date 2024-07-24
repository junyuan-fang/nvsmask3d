"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


from nvsmask3d.nvsmask3d_datamanager import NVSMask3dDataManagerConfig
from nvsmask3d.nvsmask3d_model import NVSMask3dModel, NVSMask3dModelConfig

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

@dataclass
class NvsMask3dPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NvsMask3dPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=lambda: NVSMask3dDataManagerConfig())#NVSMask3dDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=lambda: NVSMask3dModelConfig())#NVSMask3dModelConfig()
    """specifies the model config"""


class NvsMask3dPipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: NvsMask3dPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):

            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts,pts_rgb)
        self.datamanager.to(device)
        
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,   
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,# add seed points from metadata

        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                NVSMask3dModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])
        
    #     if(
    #         hasattr(self.datamanager, "train_dataparser_outputs")
    #         and "points3D_cls_num" in self.datamanager.train_dataparser_outputs.metadata
    #     ):  
    #         max_cls_num = self.datamanager.train_dataparser_outputs.metadata["points3D_cls_num"]

    #         self.segmant_gaussian = ViewerSlider (name="Segment Gaussian by the class agnostic ID ", min_value=0, max_value=max_cls_num-1, step=1, default_value=0, disabled=False,cb_hook=self._update_masked_scene_with_cls, visible=True)
    
    # def _update_masked_scene_with_cls(self, number: ViewerSlider):
    #     if number.value > self.datamanager.train_dataparser_outputs.metadata["points3D_cls_num"] - 1:
    #         number.value = self.datamanager.train_dataparser_outputs.metadata["points3D_cls_num"] - 1
    #         return   
    #     elif number.value < 0:
    #         number.value = 0
    #         return