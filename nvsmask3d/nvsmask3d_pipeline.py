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
from nvsmask3d.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
from nvsmask3d.encoders.open_clip_encoder import OpenCLIPNetworkConfig

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
    network: BaseImageEncoderConfig = OpenCLIPNetworkConfig()
    """specifies the vision-language network config"""

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
        self.image_encoder: BaseImageEncoder = config.network.setup(test_mode=test_mode)
        
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, 
            test_mode=test_mode, 
            world_size=world_size, 
            local_rank=local_rank
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
        #print(self.datamanager.train_dataset.device)#is cpu, why?
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            image_encoder=self.image_encoder,   
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,# add seed points from metadata
            cameras = self.datamanager.train_dataset.cameras if test_mode == "train" else None,
            test_mode = test_mode,
            #image_file_names = self.datamanager.train_dataset.image_filenames #for testing

        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                NVSMask3dModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])
        

