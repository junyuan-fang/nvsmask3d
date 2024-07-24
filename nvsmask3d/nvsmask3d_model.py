"""
Template Model File

Currently this subclasses the splacfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type
import torch
import os
# 设置 TORCH_CUDA_ARCH_LIST 环境变量
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5;8.0"

# 启用 TensorFloat32
torch.set_float32_matmul_precision('high')
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
    
from torch.nn import Parameter
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from pytorch_msssim import SSIM
from nerfstudio.utils.colors import get_color
import math
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.cameras.cameras import Cameras




from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.models.splatfacto import (
    RGB2SH,
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
)
from nerfstudio.viewer.viewer_elements import *

from nerfstudio.viewer.viewer_elements import *


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

def num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25


@dataclass
class NVSMask3dModelConfig(SplatfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: NVSMask3dModel)
    
    random_init: bool = False
    #use_scale_regularization: bool = true
    #max_gauss_ratio: float = 1.5
    #refine_every: int = 100 # we don't cull or densify gaussians
    #warmup_length = 500





class NVSMask3dModel(SplatfactoModel):
    """Template Model."""

    config: NVSMask3dModelConfig
    
    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        self.metadata = metadata
        super().__init__(*args, **kwargs)
    #########viewer sliders#####
        try:
            self.max_cls_num = self.metadata["points3D_cls_num"]
        except:
            self.max_cls_num = 0
            
        self.segmant_gaussian = ViewerSlider (name="Segment Gaussian by the class agnostic ID ", min_value=0, max_value=self.max_cls_num-1, step=1, default_value=0, disabled=False,cb_hook=self._update_masked_scene_with_cls, visible=True)
    
    def _update_masked_scene_with_cls(self, number: ViewerSlider) -> None:
        if number.value > self.metadata["points3D_cls_num"]:
            number.value = self.metadata["points3D_cls_num"]
            return   
        elif number.value < 0:
            number.value = 0
            return
        self.cls_num = number.value
    
    
    
    def populate_modules(self):
        super().populate_modules()
        
    # def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
    #     """Takes in a camera and returns a dictionary of outputs.

    #     Args:
    #         camera: The camera(s) for which output images are rendered. It should have
    #         all the needed information to compute the outputs.

    #     Returns:
    #         Outputs of model. (ie. rendered colors)
    #     """
    #     outputs = super().get_outputs(camera)
    #     return outputs
        
    ##we don't update means
    
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # param_groups = {
        #     name: [param]
        #     for name, param in self.gauss_params.items()
        #     if name != "means"
        # }
        # return param_groups
        return {
            name: [self.gauss_params[name]]
            for name in ["scales", "quats", "features_dc", "features_rest", "opacities"]#exclude means
        }
    
    #we don't cull or densify gaussians
    def refinement_after(self, optimizers: Optimizers, step): 
        #self.binarize_opacities()
        return
    
    def binarize_opacities(self):
        with torch.no_grad():
            self.gauss_params['opacities'].data = (self.gauss_params['opacities'] > 0.5).float()