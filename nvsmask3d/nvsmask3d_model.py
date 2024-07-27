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
from nvsmask3d.encoders.image_encoder import BaseImageEncoder

from nerfstudio.utils.colors import get_color
import math
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.cameras.cameras import Cameras



from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
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
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))#off #SO3xR3 #SE3
    """Config of the camera optimizer to use"""
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
        self.metadata = metadata
        self.cls_index = 0
        super().__init__(seed_points=seed_points, *args,**kwargs)
        self.max_cls_num = max(0,self.points3D_cls_num)
        self.positives = self.negatives = ["object", "things", "stuff", "texture"]
        #viewers
        
        self.segmant_gaussian = ViewerSlider(
            name="Segment Gaussians by the class agnostic ID",
            min_value=0,
            max_value=self.max_cls_num - 1,
            step=1,
            default_value=0,
            disabled=False,
            cb_hook=self._update_masked_scene_with_cls,
            visible=True
        )
        
        self.output_text = ViewerText(
            name="Output",
            default_value="Results will be displayed here",
            disabled=True,  # Make it non-interactive
            visible=True,
            hint="Output will be displayed here"
        )
        
        self.segment_gaussian_positives = ViewerButton(
            name="Segment Gaussians with Positives", 
            cb_hook=self._segment_gaussians, 
            visible=True)

    def _segment_gaussians(self, element):
        self.output_text.value = "Segmenting Gaussians..."
        return
    
    def _update_masked_scene_with_cls(self, number: ViewerSlider) -> None:
        if number.value > self.metadata["points3D_cls_num"]:
            number.value = self.metadata["points3D_cls_num"]
            return   
        elif number.value < 0:
            number.value = 0
            return
        self.cls_index = number.value
    
    
    def populate_modules(self):
        super().populate_modules()
        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]

        
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        # Apply mask
        points3D_mask = self.points3D_mask  # Assumes this function returns a mask of appropriate shape
        if points3D_mask is not None:
            mask_indices = points3D_mask[:, self.cls_index] > 0
            opacities_masked = opacities_crop[mask_indices]
            means_masked = means_crop[mask_indices]
            features_dc_masked = features_dc_crop[mask_indices]
            features_rest_masked = features_rest_crop[mask_indices]
            scales_masked = scales_crop[mask_indices]
            quats_masked = quats_crop[mask_indices]

            colors_masked = torch.cat((features_dc_masked[:, None, :], features_rest_masked), dim=1)
        else:
            opacities_masked = opacities_crop
            means_masked = means_crop
            features_dc_masked = features_dc_crop
            features_rest_masked = features_rest_crop
            scales_masked = scales_crop
            quats_masked = quats_crop

            colors_masked = colors_crop

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            colors_masked = torch.sigmoid(colors_masked).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10, 
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
        )

        render_masked, alpha_masked, info_masked = rasterization(
            means=means_masked,
            quats=quats_masked / quats_masked.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_masked),
            opacities=torch.sigmoid(opacities_masked).squeeze(-1),
            colors=colors_masked,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
        )

        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        rgb_mask = render_masked[:, ..., :3] + (1 - alpha_masked) * background
        rgb_mask = torch.clamp(rgb_mask, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)

            depth_mask = render_masked[:, ..., 3:4]
            depth_mask = torch.where(alpha_masked > 0, depth_mask, depth_mask.detach().max()).squeeze(0)
        else:
            depth_im = None
            depth_mask = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "rgb_mask": rgb_mask.squeeze(0),  # type: ignore
            "depth_mask": depth_mask,  # type: ignore
        }

        
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
            
    @property
    def points3D_mask(self):
        return self.metadata["points3D_mask"]
    @property
    def points3D_cls_num(self):
        return self.metadata["points3D_cls_num"]