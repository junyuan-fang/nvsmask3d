# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
ns-eval for_ap --load_config nvsmask3d/data/replica
"""
from __future__ import annotations
import wandb
from PIL import Image
import torchvision.transforms as transforms
from nvsmask3d.utils.utils import save_img
from nvsmask3d.encoders.sam_encoder import SAMNetworkConfig, SamNetWork
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Literal, Optional, Tuple, Union, Callable
import torch
from nvsmask3d.utils.camera_utils import (
    get_camera_pose_in_opencv_convention,
    object_optimal_k_camera_poses_bounding_box,
    object_optimal_k_camera_poses_2D_mask,
    interpolate_camera_poses_with_camera_trajectory,
    make_cameras,
    compute_camera_pose_bounding_boxes,
    compute_camera_pose_2D_masks,
    get_points_projected_uv_and_depth
    
)
from nerfstudio.models.splatfacto import SplatfactoModel
from tqdm import tqdm
from typing_extensions import Annotated
import numpy as np
from nvsmask3d.eval.scannet200.eval_semantic_instance import (
    evaluate as evaluate_scannet200,
)
from nvsmask3d.eval.replica.eval_semantic_instance import evaluate as evaluate_replica
from nvsmask3d.eval.replica.eval_semantic_instance import (
    CLASS_LABELS as REPLICA_CLASSES,
)
import tyro
import wandb
import threading
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nvsmask3d.utils.utils import concat_images_vertically, concat_images_horizontally,log_evaluation_results_to_wandb

@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(
            output_path=self.render_output_path, get_std=True
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


@dataclass
class ComputeForAP:  # pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 177), (177,), (177,))
    """Load a checkpoint, compute some pred_scores and pred_classes for latter AP computation."""
    # use : ns-eval for_ap --load_config nvsmask3d/data/replica

    # Path to config YAML file.
    load_config: Path = Path("nvsmask3d/data/replica")
    top_k: int = 15
    visibility_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda num_visible_points, bounding_box_area: num_visible_points*bounding_box_area
    occlusion_aware: Optional[bool] = True
    interpolate_n_camera: Optional[int] = 1
    interpolate_n_rgb_camera: Optional[int] = 1
    interpolate_n_gaussian_camera: Optional[int] = 1
    gt_camera_rgb: Optional[bool] = True
    gt_camera_gaussian: Optional[bool] = True
    project_name: str = "zeroshot_enhancement"
    run_name_for_wandb: Optional[str] = "test"
    algorithm: int = 0
    sam: bool = True
    # inference
    inference_dataset: Literal["scannet200", "replica"] = "replica"

    #if run_name_for_wandb == "test":
    pretrain_embeddings = torch.load("../../hanlin/pretrain_embeddings.pt", map_location="cuda")#20000,768


    def main(self) -> None:
        if self.run_name_for_wandb == "test":
            wandb.init(project=self.project_name, name=self.run_name_for_wandb)
        # 假设从配置或命令行参数中读取 project_name 和 run_name

        gt_dir = self.load_config / "ground_truth"
        if self.inference_dataset == "replica":
            scene_names = [

                "office0",
                "office1",
                "office2",
                "office3",
                "office4",
                "room1",
                "room2",
                "room0",

            ]
            test_mode = "all replica"

            load_configs = [

                "outputs/office0/nvsmask3d/2024-08-14_204330/config.yml",
                "outputs/office1/nvsmask3d/2024-08-14_204330/config.yml",
                "outputs/office2/nvsmask3d/2024-08-14_205100/config.yml",
                "outputs/office3/nvsmask3d/2024-08-14_205128/config.yml",
                "outputs/office4/nvsmask3d/2024-08-14_210152/config.yml",
                "outputs/room1/nvsmask3d/2024-08-14_211248/config.yml",
                "outputs/room2/nvsmask3d/2024-08-14_211851/config.yml",
                "outputs/room0/nvsmask3d/2024-08-14_210501/config.yml",

            ]

            # ["outputs/office0/nvsmask3d/2024-08-12_215616/config.yml",
            # "outputs/office1/nvsmask3d/2024-08-12_170536/config.yml",
            # "outputs/office2/nvsmask3d/2024-08-12_173744/config.yml",
            # "outputs/office3/nvsmask3d/2024-08-12_173744/config.yml",
            # "outputs/office4/nvsmask3d/2024-08-12_180405/config.yml",
            # "outputs/room0/nvsmask3d/2024-08-12_180418/config.yml",
            # "outputs/room1/nvsmask3d/2024-08-12_182825/config.yml",
            # "outputs/room2/nvsmask3d/2024-08-12_182844/config.yml"]

        preds = {}
        # scene_names = ["scene0011_00"]  # hard coded for now
        with torch.no_grad():
            # for each scene
            for i, scene_name in tqdm(
                enumerate(scene_names), desc="Evaluating", total=len(scene_names)
            ):
                config, pipeline, checkpoint_path, _ = eval_setup(
                    Path(load_configs[i]),
                    test_mode=test_mode,
                )
                model = pipeline.model
                # scene_id = scene_name[5:]
                seed_points_0 = model.seed_points[0].cuda()  # shape (N, 3)

                pred_classes =  self.pred_classes_with_sam(
                                    model=model,
                                    class_agnostic_3d_mask=model.points3D_mask,
                                    seed_points_0=seed_points_0,
                                    scene_name=scene_name,
                                ) if self.sam  else self.pred_classes(
                                    model=model,
                                    class_agnostic_3d_mask=model.points3D_mask,
                                    seed_points_0=seed_points_0,
                                    scene_name=scene_name,
                                )
                pred_masks = model.points3D_mask.cpu().numpy()  # move to cpu
                pred_scores = np.ones(pred_classes.shape[0])
                # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
                print(
                    f"pred_masks.shape, pred_scores.shape, pred_classes.shape {pred_masks.shape, pred_scores.shape, pred_classes.shape}"
                )
                preds[scene_name] = {
                    "pred_masks": pred_masks,
                    "pred_scores": pred_scores,
                    "pred_classes": pred_classes,
                }
            if self.inference_dataset == "replica":
                inst_AP = evaluate_replica(
                    preds, gt_dir, output_file="output.txt", dataset="replica"
                )
                log_evaluation_results_to_wandb(inst_AP,self.run_name_for_wandb)
    def pred_classes_with_sam(self, model, class_agnostic_3d_mask, seed_points_0, scene_name=""):
        """
        Args:
        model (NVSMask3DModel): The model to use for inference
        class_agnostic_3d_mask (torch.Tensor): The class-agnostic 3D mask (N, num_cls)
        seed_points_0 (torch.Tensor): The seed points (N,3)
        k_poses (int): The number of poses to render
        """
        sam_config = SAMNetworkConfig()
        sam_network = SamNetWork(sam_config)
        # Move camera transformations to the GPU
    
        camera_to_world_opengl = model.cameras.camera_to_worlds.to(
            "cuda"
        )  # shape (M, 4, 4)
        camera_to_world_opencv = get_camera_pose_in_opencv_convention(camera_to_world_opengl)

        # Move intrinsics to the GPU
        K = model.cameras.get_intrinsics_matrices().to("cuda")  # shape (M, 3, 3)
        W, H = int(model.cameras.width[0].item()), int(model.cameras.height[0].item())

        # Convert class-agnostic mask to a boolean tensor and move to GPU
        # boolean_masks = torch.from_numpy(class_agnostic_3d_mask).bool().to('cuda')  # shape (N, 166)

        cls_num = class_agnostic_3d_mask.shape[1]
        pred_classes = np.full(cls_num, 0)  # -1)
        
        # Loop through each mask/ object
        for i in range(cls_num):
            # set instance
            model.cls_index = i
            boolean_mask = class_agnostic_3d_mask[:, i]
            masked_seed_points = seed_points_0[boolean_mask]  # shape (N_masked, 3)
            best_camera_indices, valid_u, valid_v = object_optimal_k_camera_poses_2D_mask( 
                seed_points_0=seed_points_0,
                optimized_camera_to_world=camera_to_world_opencv,
                K=K,
                W=W,
                H=H,
                boolean_mask=boolean_mask,  # select i_th mask
                depth_filenames=model.metadata["depth_filenames"] if self.occlusion_aware else None,
                depth_scale=model.depth_scale,
                k_poses=self.top_k,
                score_fn=self.visibility_score
            )
            if best_camera_indices.shape[0] == 0 or len(valid_u) == 0 or len(valid_v) == 0:
                print(
                    f"Skipping inference for mask {i} due to no valid camera poses, assign",
                )
                continue

            rgb_outputs = []
            masked_gaussian_outputs = []
            # Interpolate camera poses and bounding boxes
            if self.interpolate_n_camera*self.interpolate_n_rgb_camera > 0 or self.interpolate_n_camera*self.interpolate_n_gaussian_camera > 0:
                #camera interpolation
                interpolated_poses = interpolate_camera_poses_with_camera_trajectory(
                    camera_to_world_opengl[best_camera_indices],
                    masked_seed_points,
                    self.interpolate_n_camera,
                    # model=model,#
                    # j = i
                )# (pose-1) * step
                interpolated_cameras = make_cameras(model.cameras[0:1], interpolated_poses) 

                #get masks
                # interp_valid_u, interp_valid_v = compute_camera_pose_2D_masks(
                #     seed_points_0=model.seed_points[0].cuda(),
                #     optimized_camera_to_world=get_camera_pose_in_opencv_convention(interpolated_poses),
                #     K=interpolated_cameras.get_intrinsics_matrices().to(device="cuda"),
                #     W=W,
                #     H=H,
                #     boolean_mask=boolean_mask
                # )
                # import pdb;pdb.set_trace()
                # interpolated_poses_bounding_boxes = compute_camera_pose_bounding_boxes(
                #     seed_points_0=model.seed_points[0].cuda(),
                #     optimized_camera_to_world=get_camera_pose_in_opencv_convention(interpolated_poses),
                #     K=interpolated_cameras.get_intrinsics_matrices().to(device="cuda"),
                #     W=W,
                #     H=H,
                #     boolean_mask=boolean_mask
                # )
                #restore img for wandb
                # Project points to 2D image coordinates for all camera poses
                u, v, z = get_points_projected_uv_and_depth(masked_seed_points, get_camera_pose_in_opencv_convention(interpolated_poses), K = interpolated_cameras.get_intrinsics_matrices().to(device="cuda"))
                valid_points = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)  # shape (M, N_masked)

                interpolated_images = []
                #interpolated_images_label_map = []
                # Render NVS images, crop, save, and add to outputs
                for interpolation_index in range(interpolated_cameras.shape[0]):
                    camera = interpolated_cameras[interpolation_index:interpolation_index+1]
                    valid = valid_points[interpolation_index]  # 形状：(N,)

                    if valid_points[interpolation_index].any():
                        u_i = u[interpolation_index][valid]
                        v_i = v[interpolation_index][valid]
                        
                        # 转换为整数索引
                        u_i = u_i.long()
                        v_i = v_i.long()
                        
                        # 确保索引在图像范围内
                        u_i = torch.clamp(u_i, 0, W - 1)
                        v_i = torch.clamp(v_i, 0, H - 1)
                        
                        min_u = u_i.min()
                        max_u = u_i.max()
                        min_v = v_i.min()
                        max_v = v_i.max()
                    else:# 如果没有有效点，
                        continue

                    # Check if bounding box is valid
                    if min_u < max_u and min_v < max_v:
                        nvs_mask_img_pil = None
                        nvs_img_pil = None
                        # nvs_img_label_map = None
                        # nvs_mask_img_label_map = None
                        if self.interpolate_n_rgb_camera > 0:
                            #Get output dimensions to validate bounding box
                            nvs_img = model.get_outputs(camera)["rgb"]  # (H, W, 3)
                            cropped_nvs_img = nvs_img[min_v:max_v, min_u:max_u]
                            cropped_nvs_img = cropped_nvs_img.permute(2, 0, 1) # (H, W, 3)
                            rgb_outputs.append(cropped_nvs_img)  # (C, H, W) ######################################################rgb##################################################################
                            cropped_nvs_img = cropped_nvs_img.cpu()#for wandb
                            # nvs_img_label_map = model.image_encoder.return_image_map(cropped_nvs_img)#for wandb
                            nvs_img_pil = transforms.ToPILImage()(cropped_nvs_img)#for wandb
                            #############debug################
                            # Save the cropped image
                            try:
                                sparse_map = torch.zeros((H, W, 3), dtype=torch.float32, device="cuda")
                                sparse_map[ v_i, u_i] = 1
                                from nvsmask3d.utils.utils import save_img
                                save_img(sparse_map, f"tests/sparse_map_object{i}nvs_image_{interpolation_index}.png")
                                
                                save_img(nvs_img, f"tests/object{i}nvs_image_{interpolation_index}.png")
                                save_img(cropped_nvs_img.permute(1,2,0), f"tests/object{i}cropped_nvs_image_{interpolation_index}.png")
                            except Exception as e:
                                import pdb;pdb.set_trace()
                                print(f"Failed to save image {interpolation_index}: {e}")
                                continue  
                            ##################################
                        if self.interpolate_n_gaussian_camera > 0:
                            # # Process and crop the nvs mask image, seems will make inference worse
                            nvs_mask_img = model.get_outputs(camera)["rgb_mask"]  # (H, W, 3)
                            cropped_nvs_mask_image = nvs_mask_img[min_v:max_v, min_u:max_u].permute(2, 0, 1)  # (C, H, W)
                            masked_gaussian_outputs.append(cropped_nvs_mask_image)###########################################################################gaussian#####################################################################
                            cropped_nvs_mask_image = cropped_nvs_mask_image.cpu()
                            #nvs_mask_img_label_map = model.image_encoder.return_image_map(cropped_nvs_mask_image)#for wandb
                            nvs_mask_img_pil = transforms.ToPILImage()(cropped_nvs_mask_image)#for wandb
                            # Combine GT image and mask horizontally
                        combined_nvs_image = concat_images_vertically([nvs_img_pil, nvs_mask_img_pil])#for wandb
                        # combined_nvs_image_label_map = concat_images_vertically([nvs_img_label_map, nvs_mask_img_label_map])#for wandb
                        interpolated_images.append(combined_nvs_image)#for wandb
                        # interpolated_images_label_map.append(combined_nvs_image_label_map)#for wandb
                    else:
                        print(f"Invalid bounding box for image {interpolation_index}: "
                            f"min_u={min_u}, max_u={max_u}, min_v={min_v}, max_v={max_v}")
            #################################################################################################
            #gt camera pose
            if self.gt_camera_rgb or self.gt_camera_gaussian:
                gt_images = []
                # gt_images_label_map = []
                
                for index, pose_index in enumerate(best_camera_indices):
                    torch.manual_seed(int(index))
                    pose_index = pose_index.item()

                    single_camera = model.cameras[pose_index : pose_index + 1]
                    assert single_camera.shape[0] == 1, "Only one camera at a time"
                    # img = model.get_outputs(single_camera)["rgb_mask"]#["rgb"]#
                    with Image.open(model.image_file_names[pose_index]) as img:
                        img = transforms.ToTensor()(img).cuda()  # (C,H,W)
                    if valid_u[index].shape[0] == 0 or valid_v[index].shape[0] == 0:
                        # print(f"Skipping inference for object {i} pose {index} due to no valid camera poses, assign")
                        continue
                    proposal_points_coords_2d = torch.stack((valid_u[index].long(), valid_v[index].long()), dim=1)  # (N, 2)


                    assert(len(proposal_points_coords_2d.shape) == 2)
                    sam_network.set_image(img)#3,H,W
                    mask_i = sam_network.get_best_mask(proposal_points_coords_2d)
                    if mask_i.sum() == 0:
                        # print(f"Skipping inference for object {i} pose {index} due to no valid camera poses, assign")
                        continue
                    
                    #multilevel mask
                    for level in range(3): # num_levels = 3
                        min_u, min_v, max_u, max_v = sam_network.mask2box_multi_level(mask_i, level, expansion_ratio = 0.1)
                        _, H, W = img.shape
                        min_u = max(0, min_u)
                        min_v = max(0, min_v)
                        max_u = min(W, max_u)
                        max_v = min(H, max_v)
                        
                        if min_u < max_u and min_v < max_v:
                            gt_img_pil = None
                            gt_mask_img_pil = None
                            # gt_mask_img_label_map = None  
                            # gt_img_pil_label_map = None

                            # 如果有效，则裁剪图像
                            if self.gt_camera_rgb:
                                cropped_image = img[:, min_v:max_v, min_u:max_u]
                                rgb_outputs.append(cropped_image)#######################################################################################rgb#####################
                                cropped_image = cropped_image.cpu()#for wandb
                                # gt_img_pil_label_map = model.image_encoder.return_image_map(cropped_image) #for wandb
                                gt_img_pil = transforms.ToPILImage()(cropped_image)#for wandb
                            if self.gt_camera_gaussian:
                                nvs_mask_img = model.get_outputs(single_camera)["rgb_mask"]  # ["rgb_mask"]  # (H,W,3)
                                cropped_nvs_mask_image = nvs_mask_img[min_v:max_v, min_u:max_u].permute(2, 0, 1)
                                masked_gaussian_outputs.append(cropped_nvs_mask_image)#############################################################################gaussian###################
                                cropped_nvs_mask_image = cropped_nvs_mask_image.cpu()#for wandb
                                # gt_mask_img_label_map = model.image_encoder.return_image_map(cropped_nvs_mask_image)#for wandb
                                gt_mask_img_pil = transforms.ToPILImage()(cropped_nvs_mask_image) #for wandb
                            # Combine GT image and mask horizontally
                            combined_gt_image = concat_images_vertically([gt_img_pil, gt_mask_img_pil])
                            # combined_gt_image_label_map = concat_images_vertically([gt_img_pil_label_map, gt_mask_img_label_map])
                            gt_images.append(combined_gt_image)
                            # gt_images_label_map.append(combined_gt_image_label_map)

            # Clear intermediate memory before encoding
            if 'img' in locals():
                del img
            if 'cropped_image' in locals():
                del cropped_image
            if 'cropped_nvs_mask_image' in locals():
                del cropped_nvs_mask_image
            if 'nvs_mask_img' in locals():
                del nvs_mask_img
            torch.cuda.empty_cache()

            if len(rgb_outputs)+len(masked_gaussian_outputs) == 0:
                print(f"Skipping inference for mask {i} due to no valid camera poses")
                continue
            # output is a list, which has tensors of the shape (C,H,W)
            
            
            with torch.no_grad():
                T = 1.0# refer to temperature
                if len(rgb_outputs) > 0:
                    rgb_features = model.image_encoder.encode_batch_list_image(
                        rgb_outputs
                    )  
                    rgb_logits = torch.mm(
                        rgb_features, model.image_encoder.pos_embeds.T
                    )  
                    #pretrained text prompt
                    rgb_logits_pretrain_text = torch.mm(rgb_features, self.pretrain_embeddings.T)
                if len(masked_gaussian_outputs) > 0:
                    mask_features = model.image_encoder.encode_batch_list_image(
                        masked_gaussian_outputs
                    )  
                    mask_logits = torch.mm(
                        mask_features, model.image_encoder.pos_embeds.T
                    )  
                    mask_logits_pretrain_text = torch.mm(mask_features, self.pretrain_embeddings.T)
                # if self.run_name_for_wandb == "test":
                if self.algorithm == 0:
                #aggregate similarity scores 你目前是将批次中的相似度分数进行求和（sum），这可能会导致信息丢失，尤其是在增强视图之间存在较大差异的情况下。
                    scores = rgb_logits.sum(dim=0)  # Shape: (200,) for scannet200 
                if self.algorithm == 1:
                    weights_mask = None
                    weights_rgb = None
                    # if self.run_name_for_wandb == "test":
                    if len(masked_gaussian_outputs) > 0:
                        correction_mask = mask_logits_pretrain_text.mean(dim=1, keepdim=True)
                        weights_mask = mask_logits.max(dim=1, keepdim=True).values # accriss text prompt Mx1
                        weights_mask = torch.softmax((weights_mask - correction_mask) / T, dim=0) 
                    
                    if len(rgb_outputs) > 0:
                        correction_rgb = rgb_logits_pretrain_text.mean(dim=1, keepdim=True)
                        weights_rgb = rgb_logits.max(dim=1, keepdim=True).values  # accriss text prompt
                        weights_rgb = torch.softmax((weights_rgb - correction_rgb) / T, dim=0)
                        
                    #concat weights
                    if weights_mask is not None and weights_rgb is not None:
                        weights = torch.cat([weights_mask, weights_rgb], dim=0) 
                        all_logits = torch.cat([mask_logits, rgb_logits], dim=0)
                    elif weights_mask is not None:
                        weights = weights_mask
                        all_logits = mask_logits
                    else:
                        weights = weights_rgb
                        all_logits = rgb_logits    

                    weighted_logits = all_logits * weights
                    scores = torch.sum(weighted_logits, dim=0)

                # if algorithm == 2:
                #     E_pretrain_text = torch.mean(similarity_scores_pretrain_text, dim=1)  # (B,)
                #     assert E_pretrain_text.shape == (B,)
                #     logit_normalized = similarity_scores - ( E_pretrain_text).unsqueeze(1)# (B,C)
                #     weights = torch.softmax(logit_normalized/T, dim=0)
                #     scores = torch.sum(similarity_scores * weights, dim=0)
                max_ind = torch.argmax(scores).item()
                
                # else:
                #     #aggregate similarity scores 你目前是将批次中的相似度分数进行求和（sum），这可能会导致信息丢失，尤其是在增强视图之间存在较大差异的情况下。
                #     scores = similarity_scores.sum(dim=0)  # Shape: (200,) for scannet200 
                #     max_ind = torch.argmax(scores).item()
                
                #mean scores
                # mean_scores = similarity_scores.mean(dim=0)  # Shape: (200,)
                # max_ind = torch.argmax(mean_scores).item()
                
                #max pooling
                # scores, _ = similarity_scores.max(dim=0)  # Shape: (200,)
                # max_ind = torch.argmax(scores).item()

                
                # max_ind_remapped = model.image_encoder.label_mapper[max_ind], replica no need remapping
                pred_classes[i] = max_ind  # max_ind_remapped
                
                
                # Log interpolated images
                if 'interpolated_images' in locals() and len(interpolated_images) > 0:
                    final_interpolated_image = concat_images_horizontally(interpolated_images)
                    wandb.log({f"Interpolated Scene: {scene_name}": wandb.Image(final_interpolated_image, caption=f"Interpolated Image for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})

                # Log GT images
                if 'gt_images' in locals() and len(gt_images) > 0:
                    final_gt_image = concat_images_horizontally(gt_images)
                    wandb.log({f"GT Scene: {scene_name}": wandb.Image(final_gt_image, caption=f"GT Camera Pose for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})

                # del rgb_outputs, masked_gaussian_outputs, rgb_logits, mask_logits, rgb_logits_pretrain_text, mask_logits_pretrain_text, weights, all_logits, scores, weights_mask, weights_rgb, correction_mask, correction_rgb, weighted_logits
                # if rgb_features in locals():
                #     del rgb_features
                # if mask_features in locals():
                #     del mask_features
                # torch.cuda.empty_cache()
                
                # if 'interpolated_images_label_map' in locals() and len(interpolated_images_label_map) > 0:
                #     final_interpolated_image_label_map = concat_images_horizontally(interpolated_images_label_map)
                #     wandb.log({f"Interpolated Scene: {scene_name}": wandb.Image(final_interpolated_image_label_map, caption=f"Interpolated Image Label Map for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})
                
                # if 'gt_images_label_map' in locals() and len(gt_images_label_map) > 0:
                #     final_gt_image_label_map = concat_images_horizontally(gt_images_label_map)
                #     wandb.log({f"GT Scene: {scene_name}": wandb.Image(final_gt_image_label_map, caption=f"GT Camera Pose Label Map for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})

        return pred_classes
    
    def pred_classes(self, model, class_agnostic_3d_mask, seed_points_0, scene_name=""):
        """
        Args:
        model (NVSMask3DModel): The model to use for inference
        class_agnostic_3d_mask (torch.Tensor): The class-agnostic 3D mask (N, num_cls)
        seed_points_0 (torch.Tensor): The seed points (N,3)
        k_poses (int): The number of poses to render
        """
        # Move camera transformations to the GPU

        camera_to_world_opengl = model.cameras.camera_to_worlds.to(
            "cuda"
        )  # shape (M, 4, 4)
        camera_to_world_opencv = get_camera_pose_in_opencv_convention(camera_to_world_opengl)

        # Move intrinsics to the GPU
        K = model.cameras.get_intrinsics_matrices().to("cuda")  # shape (M, 3, 3)
        W, H = int(model.cameras.width[0].item()), int(model.cameras.height[0].item())

        # Convert class-agnostic mask to a boolean tensor and move to GPU
        # boolean_masks = torch.from_numpy(class_agnostic_3d_mask).bool().to('cuda')  # shape (N, 166)

        cls_num = class_agnostic_3d_mask.shape[1]
        pred_classes = np.full(cls_num, 0)  # -1)
        
        # Loop through each mask
        for i in range(cls_num):
            # set instance
            model.cls_index = i
            boolean_mask = class_agnostic_3d_mask[:, i]
            (
                best_camera_indices,
                bounding_boxes,
            ) = object_optimal_k_camera_poses_bounding_box(
                seed_points_0=seed_points_0,
                optimized_camera_to_world=camera_to_world_opencv,
                K=K,
                W=W,
                H=H,
                boolean_mask=boolean_mask,  # select i_th mask
                depth_filenames=model.metadata["depth_filenames"] if self.occlusion_aware else None,
                depth_scale=model.depth_scale,
                k_poses=self.top_k,
                score_fn=self.visibility_score
            )
            if best_camera_indices.shape[0] == 0:
                print(
                    f"Skipping inference for mask {i} due to no valid camera poses, assign",
                )
                continue

            rgb_outputs = []
            masked_gaussian_outputs = []
            # Interpolate camera poses and bounding boxes
            if self.interpolate_n_camera*self.interpolate_n_rgb_camera > 0 or self.interpolate_n_camera*self.interpolate_n_gaussian_camera > 0:
                interpolated_poses = interpolate_camera_poses_with_camera_trajectory(
                    camera_to_world_opengl[best_camera_indices],
                    seed_points_0[boolean_mask],
                    self.interpolate_n_camera,
                )# opencv convention
                interpolated_cameras = make_cameras(model.cameras[0:1], interpolated_poses)

                interpolated_poses_bounding_boxes = compute_camera_pose_bounding_boxes(
                    seed_points_0=model.seed_points[0].cuda(),
                    optimized_camera_to_world=get_camera_pose_in_opencv_convention(interpolated_poses),
                    K=interpolated_cameras.get_intrinsics_matrices().to(device="cuda"),
                    W=W,
                    H=H,
                    boolean_mask=boolean_mask
                )
                #restore img for wandb
                interpolated_images = []
                #interpolated_images_label_map = []
                # Render NVS images, crop, save, and add to outputs
                for interpolation_index in range(interpolated_cameras.shape[0]):
                    camera = interpolated_cameras[interpolation_index:interpolation_index+1]
                    try:
                        min_u, min_v, max_u, max_v = interpolated_poses_bounding_boxes[interpolation_index]
                    except:
                        print(f"Failed to get bounding box for image {interpolation_index}")
                        continue
                    # Unpack bounding box
                    min_u = 0 if min_u == float('-inf') else min_u
                    min_v = 0 if min_v == float('-inf') else min_v
                    max_u = W if max_u == float('inf') else max_u
                    max_v = H if max_v == float('inf') else max_v
                    
                    min_u, min_v, max_u, max_v = map(int, [min_u, min_v, max_u, max_v])

                    # Clamp values to ensure they are within the valid range
                    min_u = max(0, min(min_u, W - 1))
                    min_v = max(0, min(min_v, H - 1))
                    max_u = max(0, min(max_u, W))
                    max_v = max(0, min(max_v, H))

                    # Check if bounding box is valid
                    if min_u < max_u and min_v < max_v:
                        nvs_mask_img_pil = None
                        nvs_img_pil = None
                        # nvs_img_label_map = None
                        # nvs_mask_img_label_map = None
                        if self.interpolate_n_rgb_camera > 0:
                            #Get output dimensions to validate bounding box
                            nvs_img = model.get_outputs(camera)["rgb"]  # (H, W, 3)
                            cropped_nvs_img = nvs_img[min_v:max_v, min_u:max_u]
                            cropped_nvs_img = cropped_nvs_img.permute(2, 0, 1) # (H, W, 3)
                            rgb_outputs.append(cropped_nvs_img)  # (C, H, W) ######################################################rgb##################################################################
                            cropped_nvs_img = cropped_nvs_img.cpu()#for wandb
                            # nvs_img_label_map = model.image_encoder.return_image_map(cropped_nvs_img)#for wandb
                            nvs_img_pil = transforms.ToPILImage()(cropped_nvs_img)#for wandb
                            #############debug################
                            # Save the cropped image
                            # try:
                            #     save_img(cropped_nvs_img, f"tests/cropped_nvs_image_{interpolation_index}.png")
                            # except Exception as e:
                            #     print(f"Failed to save image {interpolation_index}: {e}")
                            #     continue  
                            ##################################
                        if self.interpolate_n_gaussian_camera > 0:
                            # # Process and crop the nvs mask image, seems will make inference worse
                            nvs_mask_img = model.get_outputs(camera)["rgb_mask"]  # (H, W, 3)
                            cropped_nvs_mask_image = nvs_mask_img[min_v:max_v, min_u:max_u].permute(2, 0, 1)  # (C, H, W)
                            masked_gaussian_outputs.append(cropped_nvs_mask_image)###########################################################################gaussian#####################################################################
                            cropped_nvs_mask_image = cropped_nvs_mask_image.cpu()
                            #nvs_mask_img_label_map = model.image_encoder.return_image_map(cropped_nvs_mask_image)#for wandb
                            nvs_mask_img_pil = transforms.ToPILImage()(cropped_nvs_mask_image)#for wandb
                            # Combine GT image and mask horizontally
                        combined_nvs_image = concat_images_vertically([nvs_img_pil, nvs_mask_img_pil])#for wandb
                        # combined_nvs_image_label_map = concat_images_vertically([nvs_img_label_map, nvs_mask_img_label_map])#for wandb
                        interpolated_images.append(combined_nvs_image)#for wandb
                        # interpolated_images_label_map.append(combined_nvs_image_label_map)#for wandb
                    else:
                        print(f"Invalid bounding box for image {interpolation_index}: "
                            f"min_u={min_u}, max_u={max_u}, min_v={min_v}, max_v={max_v}")
            #################################################################################################
            #gt camera pose
            if self.gt_camera_rgb or self.gt_camera_gaussian:
                gt_images = []
                # gt_images_label_map = []
                for index, pose_index in enumerate(best_camera_indices):
                    pose_index = pose_index.item()

                    single_camera = model.cameras[pose_index : pose_index + 1]
                    assert single_camera.shape[0] == 1, "Only one camera at a time"
                    # img = model.get_outputs(single_camera)["rgb_mask"]#["rgb"]#
                    with Image.open(model.image_file_names[pose_index]) as img:
                        img = transforms.ToTensor()(img).cuda()  # (C,H,W)

                    # nvs_img = model.get_outputs(single_camera)["rgb"]  # (H,W,3)
                    min_u, min_v, max_u, max_v = bounding_boxes[index]
                    min_u = 0 if min_u == float('-inf') else min_u
                    min_v = 0 if min_v == float('-inf') else min_v
                    max_u = W if max_u == float('inf') else max_u
                    max_v = H if max_v == float('inf') else max_v
                    
                    min_u, min_v, max_u, max_v = map(int, [min_u, min_v, max_u, max_v])
                    _, H, W = img.shape

                    # Clamp values to ensure they are within the valid range
                    min_u = max(0, min(min_u, W - 1))
                    min_v = max(0, min(min_v, H - 1))
                    max_u = max(0, min(max_u, W))
                    max_v = max(0, min(max_v, H))

                    # 检查裁剪区域是否有效（确保裁剪区域有正面积）
                    if min_u < max_u and min_v < max_v:
                        gt_img_pil = None
                        gt_mask_img_pil = None
                        # gt_mask_img_label_map = None  
                        # gt_img_pil_label_map = None

                        # 如果有效，则裁剪图像
                        if self.gt_camera_rgb:
                            cropped_image = img[:, min_v:max_v, min_u:max_u]
                            rgb_outputs.append(cropped_image)#######################################################################################rgb#####################
                            cropped_image = cropped_image.cpu()#for wandb
                            # gt_img_pil_label_map = model.image_encoder.return_image_map(cropped_image) #for wandb
                            gt_img_pil = transforms.ToPILImage()(cropped_image)#for wandb
                        if self.gt_camera_gaussian:
                            nvs_mask_img = model.get_outputs(single_camera)["rgb_mask"]  # ["rgb_mask"]  # (H,W,3)
                            cropped_nvs_mask_image = nvs_mask_img[min_v:max_v, min_u:max_u].permute(2, 0, 1)
                            masked_gaussian_outputs.append(cropped_nvs_mask_image)#############################################################################gaussian###################
                            cropped_nvs_mask_image = cropped_nvs_mask_image.cpu()#for wandb
                            # gt_mask_img_label_map = model.image_encoder.return_image_map(cropped_nvs_mask_image)#for wandb
                            gt_mask_img_pil = transforms.ToPILImage()(cropped_nvs_mask_image) #for wandb
                        # Combine GT image and mask horizontally
                        combined_gt_image = concat_images_vertically([gt_img_pil, gt_mask_img_pil])
                        # combined_gt_image_label_map = concat_images_vertically([gt_img_pil_label_map, gt_mask_img_label_map])
                        gt_images.append(combined_gt_image)
                        # gt_images_label_map.append(combined_gt_image_label_map)

                    else:# skip
                        print(f"Invalid bounding box for image {pose_index}: "
                            f"min_u={min_u}, max_u={max_u}, min_v={min_v}, max_v={max_v}")
                        #outputs.append(img)  # 添加未裁剪的图像
                        #cropped_nvs_mask_image = nvs_mask_img.permute(2, 0, 1)
                    ###################save rendered image#################
                    # from nvsmask3d.utils.utils import save_img

                    # save_img(
                    #     cropped_image.permute(1, 2, 0), f"tests/output_{i}_{pose_index}.png"
                    # )
                    ######################################################

            # Clear intermediate memory before encoding
            if 'img' in locals():
                del img
            if 'cropped_image' in locals():
                del cropped_image
            if 'cropped_nvs_mask_image' in locals():
                del cropped_nvs_mask_image
            if 'nvs_mask_img' in locals():
                del nvs_mask_img
            torch.cuda.empty_cache()

            if len(rgb_outputs)+len(masked_gaussian_outputs) == 0:
                print(f"Skipping inference for mask {i} due to no valid camera poses")
                continue
            # output is a list, which has tensors of the shape (C,H,W)
            
            
            with torch.no_grad():
                algorithm = 1
                T = 1.0# refer to temperature
                if len(rgb_outputs) > 0:
                    rgb_features = model.image_encoder.encode_batch_list_image(
                        rgb_outputs
                    )  
                    rgb_logits = torch.mm(
                        rgb_features, model.image_encoder.pos_embeds.T
                    )  
                    #pretrained text prompt
                    rgb_logits_pretrain_text = torch.mm(rgb_features, self.pretrain_embeddings.T)
                if len(masked_gaussian_outputs) > 0:
                    mask_features = model.image_encoder.encode_batch_list_image(
                        masked_gaussian_outputs
                    )  
                    mask_logits = torch.mm(
                        mask_features, model.image_encoder.pos_embeds.T
                    )  
                    mask_logits_pretrain_text = torch.mm(mask_features, self.pretrain_embeddings.T)
                # if self.run_name_for_wandb == "test":
                if algorithm == 1:
                    weights_mask = None
                    weights_rgb = None
                    # if self.run_name_for_wandb == "test":
                    if len(masked_gaussian_outputs) > 0:
                        correction_mask = mask_logits_pretrain_text.mean(dim=1, keepdim=True)
                        weights_mask = mask_logits.max(dim=1, keepdim=True).values # accriss text prompt Mx1
                        weights_mask = torch.softmax((weights_mask - correction_mask) / T, dim=0) 
                    
                    if len(rgb_outputs) > 0:
                        correction_rgb = rgb_logits_pretrain_text.mean(dim=1, keepdim=True)
                        weights_rgb = rgb_logits.max(dim=1, keepdim=True).values  # accriss text prompt
                        weights_rgb = torch.softmax((weights_rgb - correction_rgb) / T, dim=0)
                        
                    #concat weights
                    if weights_mask is not None and weights_rgb is not None:
                        weights = torch.cat([weights_mask, weights_rgb], dim=0) 
                        all_logits = torch.cat([mask_logits, rgb_logits], dim=0)
                    elif weights_mask is not None:
                        weights = weights_mask
                        all_logits = mask_logits
                    else:
                        weights = weights_rgb
                        all_logits = rgb_logits    

                    weighted_logits = all_logits * weights
                    scores = torch.sum(weighted_logits, dim=0)

                # if algorithm == 2:
                #     E_pretrain_text = torch.mean(similarity_scores_pretrain_text, dim=1)  # (B,)
                #     assert E_pretrain_text.shape == (B,)
                #     logit_normalized = similarity_scores - ( E_pretrain_text).unsqueeze(1)# (B,C)
                #     weights = torch.softmax(logit_normalized/T, dim=0)
                #     scores = torch.sum(similarity_scores * weights, dim=0)
                max_ind = torch.argmax(scores).item()
                
                # else:
                #     #aggregate similarity scores 你目前是将批次中的相似度分数进行求和（sum），这可能会导致信息丢失，尤其是在增强视图之间存在较大差异的情况下。
                #     scores = similarity_scores.sum(dim=0)  # Shape: (200,) for scannet200 
                #     max_ind = torch.argmax(scores).item()
                
                #mean scores
                # mean_scores = similarity_scores.mean(dim=0)  # Shape: (200,)
                # max_ind = torch.argmax(mean_scores).item()
                
                #max pooling
                # scores, _ = similarity_scores.max(dim=0)  # Shape: (200,)
                # max_ind = torch.argmax(scores).item()

                
                # max_ind_remapped = model.image_encoder.label_mapper[max_ind], replica no need remapping
                pred_classes[i] = max_ind  # max_ind_remapped
                
                
                # Log interpolated images
                if 'interpolated_images' in locals() and len(interpolated_images) > 0:
                    final_interpolated_image = concat_images_horizontally(interpolated_images)
                    wandb.log({f"Interpolated Scene: {scene_name}": wandb.Image(final_interpolated_image, caption=f"Interpolated Image for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})

                # Log GT images
                if 'gt_images' in locals() and len(gt_images) > 0:
                    final_gt_image = concat_images_horizontally(gt_images)
                    wandb.log({f"GT Scene: {scene_name}": wandb.Image(final_gt_image, caption=f"GT Camera Pose for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})

                # del rgb_outputs, masked_gaussian_outputs, rgb_logits, mask_logits, rgb_logits_pretrain_text, mask_logits_pretrain_text, weights, all_logits, scores, weights_mask, weights_rgb, correction_mask, correction_rgb, weighted_logits
                # if rgb_features in locals():
                #     del rgb_features
                # if mask_features in locals():
                #     del mask_features
                # torch.cuda.empty_cache()
                
                # if 'interpolated_images_label_map' in locals() and len(interpolated_images_label_map) > 0:
                #     final_interpolated_image_label_map = concat_images_horizontally(interpolated_images_label_map)
                #     wandb.log({f"Interpolated Scene: {scene_name}": wandb.Image(final_interpolated_image_label_map, caption=f"Interpolated Image Label Map for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})
                
                # if 'gt_images_label_map' in locals() and len(gt_images_label_map) > 0:
                #     final_gt_image_label_map = concat_images_horizontally(gt_images_label_map)
                #     wandb.log({f"GT Scene: {scene_name}": wandb.Image(final_gt_image_label_map, caption=f"GT Camera Pose Label Map for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})

        return pred_classes


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ComputePSNR, tyro.conf.subcommand(name="psnr")],
        Annotated[ComputeForAP, tyro.conf.subcommand(name="for_ap")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa