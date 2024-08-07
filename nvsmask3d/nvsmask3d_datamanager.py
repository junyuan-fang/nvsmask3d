"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union
from copy import deepcopy

import torch

from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)


@dataclass
class NVSMask3dDataManagerConfig(FullImageDatamanagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: NVSmask3dDataManager)
    
    
    camera_res_scale_factor: float = 1.0
    """Rescale cameras"""


class NVSmask3dDataManager(FullImageDatamanager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: NVSMask3dDataManagerConfig

    def __init__(
        self,
        config: NVSMask3dDataManagerConfig,
        device: Union[torch.device, str] = "gpu",
        test_mode: Literal["test", "val", "inference", "train", "all"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, 
            device=device, 
            test_mode=test_mode, 
            world_size=world_size, 
            local_rank=local_rank, 
            **kwargs
        )
        metadata = self.train_dataparser_outputs.metadata
        if test_mode == "all":
            self.all_dataparser = self.dataparser.get_dataparser_outputs(split="all")
            self.whole_dataset = self.create_whole_dataset()
        
    def create_whole_dataset(self):
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.all_dataparser,
            scale_factor=self.config.camera_res_scale_factor,
        )