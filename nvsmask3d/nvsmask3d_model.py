"""
Template Model File

Currently this subclasses the splacfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type
import torch
import os
from PIL import Image
import torchvision.transforms as transforms

# 设置 TORCH_CUDA_ARCH_LIST 环境变量
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0"

# 启用 TensorFloat32
torch.set_float32_matmul_precision("high")
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

# from nvsmask3d.utils.camera_utils import Cameras


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
from nvsmask3d.utils.camera_utils import (
    object_optimal_k_camera_poses,
    compute_new_camera_pose_from_object_uv,
    get_camera_pose_in_opencv_convention,
    make_cameras,
    object_optimal_k_camera_poses_bounding_box,
    interpolate_camera_poses_with_camera_trajectory,
)


@dataclass
class NVSMask3dModelConfig(SplatfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: NVSMask3dModel)

    random_init: bool = False
    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(mode="off")
    )  # off #SO3xR3 #SE3
    """Config of the camera optimizer to use"""
    add_means: bool = True
    # use_scale_regularization: bool = true
    # max_gauss_ratio: float = 1.5
    # refine_every: int = 100 # we don't cull or densify gaussians
    # warmup_length = 500
    use_scale_regularization: bool = True
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    
    max_gauss_ratio: float = 5
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """

class NVSMask3dModel(SplatfactoModel):
    """Template Model."""

    config: NVSMask3dModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        metadata: Optional[Dict] = None,
        cameras: Optional[Cameras] = None,
        test_mode: Literal[
            "test", "val", "inference", "train", "all_replica", "all_scannet"
        ] = "val",
        image_file_names,  # test
        **kwargs,
    ):
        self.metadata = metadata
        self.cameras = cameras
        self.cls_index = 0
        self.image_file_names = image_file_names  # debug

        self.test_mode = test_mode
        super().__init__(seed_points=seed_points, *args, **kwargs)
        self.max_cls_num = max(0, self.points3D_cls_num) if self.points3D_cls_num else 2
        self.inference = False

        self.split_indices = {}
        self.initial_gaussians_mask = (
            torch.ones(seed_points[0].shape[0], dtype=torch.bool, device=self.device)
            if seed_points
            else None
        )
        # Initialize cache
        self._mask_cache = {}

        # viewers
        self.segmant_gaussian = ViewerSlider(
            name="Segment Gaussians by the class agnostic ID",
            min_value=0,
            max_value=self.max_cls_num - 1,
            step=1,
            default_value=0,
            disabled=False,
            cb_hook=self._update_masked_scene_with_cls,
            visible=True,
        )

        self.output_text = ViewerText(
            name="Output",
            default_value="Results will be displayed here",
            disabled=True,  # Make it non-interactive
            visible=(
                True if self.test_mode == "train" or "all" in self.test_mode else False
            ),
            hint="Output will be displayed here",
        )

        self.segment_gaussian_positives = ViewerButton(
            name="Segment Gaussians with Positives",
            cb_hook=self._segment_gaussians,
            visible=(
                True if self.test_mode == "train" or "all" in self.test_mode else False
            ),
        )
        
        self.segment_gaussian_positives_with_nvs = ViewerButton(
            name="Segment Gaussians with Positives + NVS",
            cb_hook=self._segment_gaussians_with_nvs,
            visible=(
                True if self.test_mode == "train" or "all" in self.test_mode else False
            ),
        )
        
    def _segment_gaussians_with_nvs(self,element):
        """
        Use only 2 cameras, interpolate extra cameras with NVS.
        """
        self.output_text.value = "Segmenting Gaussians..."
                
        optimized_camera_to_world = self.cameras.camera_to_worlds.to(
            "cuda"
        )  # shape (M, 4, 4)
        optimized_camera_to_world = get_camera_pose_in_opencv_convention(optimized_camera_to_world)

        # Move intrinsics to the GPU
        K = self.cameras.get_intrinsics_matrices().to(device=self.device)  # shape (M, 3, 3) in default, each K's elements are the same
        W, H = int(self.cameras.width[0].item()), int(self.cameras.height[0].item())
        optimal_camera_indices, bounding_boxes = object_optimal_k_camera_poses_bounding_box(  # object_optimal_k_camera_poses(
            seed_points_0=self.seed_points[0].cuda(),
            optimized_camera_to_world=optimized_camera_to_world,
            K=K,
            W=W,
            H=H,
            boolean_mask=self.points3D_mask[:, self.cls_index],
            k_poses=2,
        )  # image_file_names= self.image_file_names)#seedpoints, mask -> cuda, numpy
        
        #only select 2 optimal cameras for testing
        optimal_camera_indices = optimal_camera_indices[:2]
        bounding_boxes = bounding_boxes[:2]
        
        #prepare data
        best_camera_poses = self.cameras.camera_to_worlds[optimal_camera_indices].to(device=self.device)
        interpolated_poses = interpolate_camera_poses_with_camera_trajectory(best_camera_poses, K, W, H, bounding_boxes, 3)
        interpolated_cameras = make_cameras(self.cameras[0:1], interpolated_poses)
        outputs = []
        
        for i_index,i in enumerate(optimal_camera_indices):
            with Image.open(self.image_file_names[i]) as img:
                img = transforms.ToTensor()(img).cuda()
                from nvsmask3d.utils.utils import save_img
                save_img(img.permute(1, 2, 0), f"tests/input_{i_index}.png")
            min_u, min_v, max_u, max_v = bounding_boxes[i_index]
            cropped_image = img[:, int(min_v.item()):int(max_v.item()), int(min_u.item()):int(max_u.item())]
            save_img(
                cropped_image.permute(1, 2, 0), f"tests/best_{i_index}.png"
            )
        
        poses = best_camera_poses#get_camera_pose_in_opencv_convention(best_camera_poses)
        pose_a = poses[0]
        min_u_a, min_v_a, max_u_a, max_v_a = bounding_boxes[0]
        object_uv = torch.tensor([(min_u_a + max_u_a) / 2, (min_v_a + max_v_a) / 2], dtype=torch.float32)
        pose_a = compute_new_camera_pose_from_object_uv(camera_pose = pose_a, object_uv=object_uv, K = K[i], image_width=W, image_height=H)
        camera_index = optimal_camera_indices[0]
        camera = make_cameras(self.cameras[camera_index:camera_index+1], pose_a.unsqueeze(0))
        nvs_mask_img = self.get_outputs(camera)["rgb"]
        from nvsmask3d.utils.utils import save_img
        save_img(nvs_mask_img, f"tests/nvs_mask_img_{0}.png")
        import pdb; pdb.set_trace()
            

        # for index in range(interpolated_cameras.shape[0]):
        #     single_camera = interpolated_cameras[index : index + 1]
        #     assert single_camera.shape[0] == 1, "Only one camera at a time"
        #     nvs_mask_img = self.get_outputs(single_camera)["rgb"]  # ["rgb_mask"]  # (H,W,3)
        #     # Convert to integers for slicing
        #     # min_u, min_v, max_u, max_v = map(int, [min_u, min_v, max_u, max_v])
        #     # # Ensure the indices are within image bounds
        #     # min_u, min_v = max(0, min_u), max(0, min_v)
        #     # max_u, max_v = min(W, max_u), min(H, max_v)

        #     # Crop the image using valid indices
        #     # cropped_nvs_mask_image = nvs_mask_img[
        #     #     min_v:max_v, min_u:max_u
        #     # ].permute(2, 0, 1)
        #     outputs.append(nvs_mask_img.permute(2, 0, 1))
        #     #outputs.append(cropped_image)
        #     ###################save rendered image#################
        #     from nvsmask3d.utils.utils import save_img

        #     save_img(nvs_mask_img, f"tests/nvs_mask_img_{index}.png")

        #     ######################################################
        # # output = torch.stack(outputs)
        # # (B,H,W,3)->(B,C,H,W) no more
        # # output = output.permute(0, 3, 1, 2)
        # texts = self.image_encoder.classify_images(outputs)

        # self.output_text.value = texts  #''.join(output)
        # return

    def _segment_gaussians(self,element):
        self.output_text.value = "Segmenting Gaussians..."
        # get optimal cameraposes use mask proposal and poses
        
        camera = self.cameras
        optimized_camera_to_world = camera.camera_to_worlds.to(
            "cuda"
        )  # shape (M, 4, 4)
        optimized_camera_to_world = get_camera_pose_in_opencv_convention(optimized_camera_to_world)

        # Move intrinsics to the GPU
        K = camera.get_intrinsics_matrices().to("cuda")  # shape (M, 3, 3)
        W, H = int(camera.width[0].item()), int(camera.height[0].item())
        (
            optimal_camera_indices,
            bounding_boxes,
        ) = object_optimal_k_camera_poses_bounding_box(  # object_optimal_k_camera_poses(
            seed_points_0=self.seed_points[0].cuda(),
            optimized_camera_to_world=optimized_camera_to_world,
            K=K,
            W=W,
            H=H,
            boolean_mask=self.points3D_mask[:, self.cls_index],
            k_poses=2,
        )  # image_file_names= self.image_file_names)#seedpoints, mask -> cuda, numpy
        outputs = []
        print("optimal_camera_indices", optimal_camera_indices)
        for index, pose_index in enumerate(optimal_camera_indices):
            pose_index = pose_index.item()
            single_camera = self.cameras[pose_index : pose_index + 1]
            assert single_camera.shape[0] == 1, "Only one camera at a time"
            nvs_mask_img = self.get_outputs(single_camera)["rgb_mask"]  # ["rgb_mask"]  # (H,W,3)
            # instead, use the original image
            with Image.open(self.image_file_names[pose_index]) as img:
                img = transforms.ToTensor()(img).cuda()  # (C,H,W)
            # crop images with bounding box
            min_u, min_v, max_u, max_v = bounding_boxes[index]
            if any(
                map(lambda x: torch.isinf(x) or x < 0, [min_u, min_v, max_u, max_v])
            ):
                print(f"Skipping cropping for image {pose_index} due to invalid bounding box")
                cropped_image = img  # Use the whole image if bbox is invalid
                outputs.append(cropped_image)

            else:
                # Convert to integers for slicing
                min_u, min_v, max_u, max_v = map(int, [min_u, min_v, max_u, max_v])
                C, H, W = img.shape
                # Ensure the indices are within image bounds
                min_u, min_v = max(0, min_u), max(0, min_v)
                max_u, max_v = min(W, max_u), min(H, max_v)

                # Crop the image using valid indices
                cropped_image = img[:, min_v:max_v, min_u:max_u]
                cropped_nvs_mask_image = nvs_mask_img[
                    min_v:max_v, min_u:max_u
                ].permute(2, 0, 1)
                outputs.append(cropped_nvs_mask_image)
                outputs.append(cropped_image)
            ###################save rendered image#################
            from nvsmask3d.utils.utils import save_img
            save_img(
                img.permute(1, 2, 0), f"tests/output_{index}.png"
            )  # function need (H,W,3)
            save_img(cropped_image.permute(1, 2, 0), f"tests/cropped_image_{index}.png")
            save_img(nvs_mask_img, f"tests/nvs_mask_img_{index}.png")
            save_img(
                cropped_nvs_mask_image.permute(1, 2, 0),
                f"tests/cropped_nvs_mask_image_{index}.png",
            )
            ######################################################
        # output = torch.stack(outputs)
        # (B,H,W,3)->(B,C,H,W) no more
        # output = output.permute(0, 3, 1, 2)
        texts = self.image_encoder.classify_images(outputs)

        self.output_text.value = texts  #''.join(output)
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
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
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

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )

        # Apply mask
        if (
            self.points3D_mask is not None
        ):  # Assumes this function returns a mask of appropriate shape
            mask_indices = self.points3D_mask[
                :, self.cls_index
            ]  # self.get_densified_mask_indices(self.cls_index) if self.config.add_means else self.points3D_mask[:, self.cls_index]
            opacities_masked = opacities_crop[mask_indices]
            means_masked = means_crop[mask_indices]
            features_dc_masked = features_dc_crop[mask_indices]
            features_rest_masked = features_rest_crop[mask_indices]
            scales_masked = scales_crop[mask_indices]
            quats_masked = quats_crop[mask_indices]
            colors_masked = torch.cat(
                (features_dc_masked[:, None, :], features_rest_masked), dim=1
            )
        else:
            opacities_masked = opacities_crop
            means_masked = means_crop
            features_dc_masked = features_dc_crop
            features_rest_masked = features_rest_crop
            scales_masked = scales_crop
            quats_masked = quats_crop
            colors_masked = colors_crop

        BLOCK_WIDTH = (
            16  # this controls the tile size of rasterization, 16 is a good default
        )
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world).to(device=self.device)
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
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            colors_masked = torch.sigmoid(colors_masked).squeeze(
                1
            )  # [N, 1, 3] -> [N, 3]
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
            depth_im = torch.where(
                alpha > 0, depth_im, depth_im.detach().max()
            ).squeeze(0)

            # depth_mask = render_masked[:, ..., 3:4]
            # depth_mask = torch.where(alpha_masked > 0, depth_mask, depth_mask.detach().max()).squeeze(0)
        else:
            depth_im = None
            # depth_mask = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "rgb_mask": rgb_mask.squeeze(0),  # type: ignore
            # "depth_mask": depth_mask,  # type: ignore
        }

    
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # param_groups = {
        #     name: [param]
        #     for name, param in self.gauss_params.items()
        #     if name != "means"
        # }
        # return param_groups
        if not self.config.add_means:
            return {
                name: [self.gauss_params[name]]
                for name in [
                    "scales",
                    "quats",
                    "features_dc",
                    "features_rest",
                    "opacities",
                ]
            }
        else:
            return {
                name: [self.gauss_params[name]]
                for name in [
                    "means",
                    "scales",
                    "quats",
                    "features_dc",
                    "features_rest",
                    "opacities",
                ]
            }

    # we don't cull or densify gaussians
    def refinement_after(self, optimizers: Optimizers, step):
        if not self.config.add_means:
            # self.binarize_opacities()
            return
        else:
            assert step == self.step
            if self.step <= self.config.warmup_length:
                return
            with torch.no_grad():
                # Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
                # then cull
                # only split/cull if we've seen every image since opacity reset
                reset_interval = (
                    self.config.reset_alpha_every * self.config.refine_every
                )
                do_densification = (
                    self.step < self.config.stop_split_at
                    and self.step % reset_interval
                    > self.num_train_data + self.config.refine_every
                )
                if do_densification:
                    # then we densify
                    assert (
                        self.xys_grad_norm is not None
                        and self.vis_counts is not None
                        and self.max_2Dsize is not None
                    )
                    avg_grad_norm = (
                        (self.xys_grad_norm / self.vis_counts)
                        * 0.5
                        * max(self.last_size[0], self.last_size[1])
                    )
                    high_grads = (
                        avg_grad_norm > self.config.densify_grad_thresh
                    ).squeeze()
                    splits = (
                        self.scales.exp().max(dim=-1).values
                        > self.config.densify_size_thresh
                    ).squeeze()
                    if self.step < self.config.stop_screen_size_at:
                        splits |= (
                            self.max_2Dsize > self.config.split_screen_size
                        ).squeeze()
                    splits &= high_grads
                    nsamps = self.config.n_split_samples
                    split_params = self.split_gaussians(splits, nsamps)
                    self.update_split_indices(splits, nsamps)  ##############

                    dups = (
                        self.scales.exp().max(dim=-1).values
                        <= self.config.densify_size_thresh
                    ).squeeze()
                    dups &= high_grads
                    dup_params = self.dup_gaussians(dups)
                    import pdb

                    pdb.set_trace()
                    self.update_duplication_indices(dups)  ##############
                    for name, param in self.gauss_params.items():
                        self.gauss_params[name] = torch.nn.Parameter(
                            torch.cat(
                                [param.detach(), split_params[name], dup_params[name]],
                                dim=0,
                            )
                        )
                    # append zeros to the max_2Dsize tensor
                    self.max_2Dsize = torch.cat(
                        [
                            self.max_2Dsize,
                            torch.zeros_like(split_params["scales"][:, 0]),
                            torch.zeros_like(dup_params["scales"][:, 0]),
                        ],
                        dim=0,
                    )

                    split_idcs = torch.where(splits)[0]
                    self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                    dup_idcs = torch.where(dups)[0]
                    self.dup_in_all_optim(optimizers, dup_idcs, 1)

                    # After a guassian is split into two new gaussians, the original one should also be pruned.
                    splits_mask = torch.cat(
                        (
                            splits,
                            torch.zeros(
                                nsamps * splits.sum() + dups.sum(),
                                device=self.device,
                                dtype=torch.bool,
                            ),
                        )
                    )

                    deleted_mask = self.cull_gaussians(splits_mask)
                elif (
                    self.step >= self.config.stop_split_at
                    and self.config.continue_cull_post_densification
                ):
                    deleted_mask = self.cull_gaussians()
                else:
                    # if we donot allow culling post refinement, no more gaussians will be pruned.
                    deleted_mask = None

                if deleted_mask is not None:
                    self.remove_from_all_optim(optimizers, deleted_mask)

                if (
                    self.step < self.config.stop_split_at
                    and self.step % reset_interval == self.config.refine_every
                ):
                    # Reset value is set to be twice of the cull_alpha_thresh
                    reset_value = self.config.cull_alpha_thresh * 2.0
                    self.opacities.data = torch.clamp(
                        self.opacities.data,
                        max=torch.logit(
                            torch.tensor(reset_value, device=self.device)
                        ).item(),
                    )
                    # reset the exp of optimizer
                    optim = optimizers.optimizers["opacities"]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                    param_state["exp_avg_sq"] = torch.zeros_like(
                        param_state["exp_avg_sq"]
                    )

                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None

    def update_split_indices(self, splits, nsamps):
        """Update split indices for new gaussians."""
        original_indices = torch.where(splits)[0]
        start_index = self.means.size(0)
        for i, original_index in enumerate(original_indices):
            original_index = original_index.item()
            new_indices = torch.arange(
                start_index + i * nsamps,
                start_index + (i + 1) * nsamps,
                device=self.device,
            )
            if original_index in self.split_indices:
                self.split_indices[original_index] = torch.cat(
                    (self.split_indices[original_index], new_indices)
                )
            else:
                self.split_indices[original_index] = new_indices
        self.dup_index_start = start_index + (i + 1) * nsamps

    def update_duplication_indices(self, dups):
        """Update duplication indices for new gaussians."""
        original_indices = torch.where(dups)[0]
        start_index = self.dup_index_start  # self.means.size(0)

        import pdb

        pdb.set_trace()

        new_indices = torch.arange(
            start_index, start_index + len(original_indices), device=self.device
        )
        for i, original_index in enumerate(original_indices):
            original_index = original_index.item()
            new_index = new_indices[i].unsqueeze(0)
            if original_index in self.split_indices:
                self.split_indices[original_index] = torch.cat(
                    (self.split_indices[original_index], new_index)
                )
            else:
                self.split_indices[original_index] = new_index

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        if self.config.add_means and self.seed_points is not None:
            n_bef = self.num_points
            # Compute cull mask based on opacity threshold
            culls = (
                torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh
            ).squeeze()
            below_alpha_count = torch.sum(culls).item()
            toobigs_count = 0
            if extra_cull_mask is not None:
                culls = culls | extra_cull_mask

            if self.config.add_means:
                # Exclude initial Gaussians from culling
                culls &= ~self.initial_gaussians_mask
            if self.step > self.config.refine_every * self.config.reset_alpha_every:
                # cull huge ones
                toobigs = (
                    torch.exp(self.scales).max(dim=-1).values
                    > self.config.cull_scale_thresh
                ).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    # cull big screen space
                    if self.max_2Dsize is not None:
                        toobigs = (
                            toobigs
                            | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
                        )
                culls = culls | toobigs
                toobigs_count = torch.sum(toobigs).item()
            for name, param in self.gauss_params.items():
                self.gauss_params[name] = torch.nn.Parameter(param[~culls])

            if self.config.add_means:
                # Remove culled indices from split tracking
                for culled_index in torch.where(culls)[0]:
                    culled_index = culled_index.item()
                    if culled_index in self.split_indices:
                        del self.split_indices[culled_index]

                # Update mask
                self.initial_gaussians_mask = self.initial_gaussians_mask[~culls].to(
                    self.device
                )

            CONSOLE.log(
                f"Culled {n_bef - self.num_points} gaussians "
                f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
            )

            return culls
        else:
            super().cull_gaussians(extra_cull_mask)

    def split_gaussians(self, split_mask, samps):  # for gaussians to big
        # Split existing logic...
        # original_indices = torch.where(split_mask)[0]#.cpu().numpy()
        out = super().split_gaussians(split_mask, samps)
        # Only update the mask for newly added Gaussians, we don't want to delete initial means
        if self.seed_points is not None:
            num_new = out["means"].size(0)
            self.initial_gaussians_mask = torch.cat(
                [
                    self.initial_gaussians_mask.to(self.device),
                    torch.zeros((num_new), dtype=torch.bool, device=self.device),
                ]
            )

            # # Track split indices
            # start_index = self.means.size(0)
            # for i, original_index in enumerate(original_indices):
            #     original_index = original_index.item()
            #     new_indices = torch.arange(
            #         start_index + i * samps, start_index + (i + 1) * samps, device=self.device
            #     )
            #     if original_index in self.split_indices:
            #         self.split_indices[original_index] = torch.cat(
            #             (self.split_indices[original_index], new_indices)
            #         )
            #     else:
            #         self.split_indices[original_index] = new_indices

        return out

    def dup_gaussians(self, dup_mask):  # for gaussians to small
        # Duplicate existing logic...
        # original_indices = torch.where(dup_mask)[0]#.cpu().numpy()
        new_dups = super().dup_gaussians(dup_mask)

        # Only update the mask for newly duplicated Gaussians
        if self.seed_points is not None:
            num_new = new_dups["means"].size(0)
            self.initial_gaussians_mask = torch.cat(
                [
                    self.initial_gaussians_mask,
                    torch.zeros((num_new), dtype=torch.bool, device=self.device),
                ]
            )

            # # Track duplicated indices
            # start_index = self.means.size(0)
            # new_indices = torch.arange(start_index, start_index + len(original_indices), device=self.device)
            # for i, original_index in enumerate(original_indices):
            #     import pdb; pdb.set_trace()
            #     original_index = original_index.item()
            #     new_index = new_indices[i].unsqueeze(0)
            #     if original_index in self.split_indices:
            #         self.split_indices[original_index] = torch.cat(
            #             (self.split_indices[original_index], new_index)
            #         )
            #     else:
            #         self.split_indices[original_index] = new_index

        return new_dups

    def binarize_opacities(self):
        with torch.no_grad():
            self.gauss_params["opacities"].data = (
                self.gauss_params["opacities"] > 0.5
            ).float()

    def get_split_indices_batch(
        self, indices: List[int]
    ) -> Dict[int, Optional[List[int]]]:
        """Retrieve split indices for a batch of original indices."""
        return {index: self.split_indices.get(index, None) for index in indices}

    def get_densified_mask_indices(self, cls_index) -> torch.Tensor:
        """Densify masks using the input masks."""
        # if cls_index in self._mask_cache and self._mask_cache[cls_index].shape[0] == self.means.shape[0]:
        #     return self._mask_cache[cls_index]

        # Get the original mask indices
        mask_indices = torch.nonzero(
            self.points3D_mask[:, cls_index], as_tuple=False
        ).squeeze()

        # Initialize an updated mask as a tensor of zeros, aligning with self.means
        updated_mask = torch.zeros(
            self.means.shape[0], dtype=torch.bool, device=self.device
        )

        # Efficiently set the specified indices to True using scatter_
        updated_mask.scatter_(0, mask_indices, True)

        # Convert the dictionary keys to a tensor for split indices
        split_indices_keys = torch.tensor(
            list(self.split_indices.keys()), device=self.device
        )

        # Determine which mask indices intersect with split_indices keys
        is_intersected = torch.isin(mask_indices, split_indices_keys)
        intersected_keys = mask_indices[is_intersected]

        # Update the mask by setting True for all additional indices from split operations
        if intersected_keys.numel() > 0:
            # Concatenate additional indices from the split_indices tensor dictionary

            additional_indices = torch.cat(
                [self.split_indices[idx.item()] for idx in intersected_keys]
            )

            # Set the mask to True at these additional indices using scatter_
            updated_mask.scatter_(0, additional_indices, True)

        # Cache the result for the current cls_index
        # self._mask_cache[cls_index] = updated_mask

        return updated_mask

    @property
    def points3D_mask(self):
        points3D_mask = self.metadata.get("points3D_mask").bool().to(self.device)
        return points3D_mask

    @property
    def points3D_cls_num(self):
        if "points3D_cls_num" in self.metadata:
            return self.metadata["points3D_cls_num"]
        else:
            return None
