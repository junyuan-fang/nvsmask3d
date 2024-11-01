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
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import wandb
from PIL import Image, ImageFilter, ImageDraw
import torchvision.transforms as transforms
from nvsmask3d.utils.utils import save_img
from nvsmask3d.encoders.sam_encoder import SAMNetworkConfig, SamNetWork
import json
from nvsmask3d.utils.utils import plot_images_and_logits
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from typing import Literal, Optional, Tuple, Union, Callable
import torch
from nvsmask3d.utils.utils import blur_non_masked_areas, make_square_image, generate_txt_files_optimized, save_predictions, rle_encode, run_command_and_save_output
from nvsmask3d.utils.camera_utils import (
    get_camera_pose_in_opencv_convention,
    object_optimal_k_camera_poses_2D_mask,
    interpolate_camera_poses_with_camera_trajectory,
    make_cameras,
    compute_camera_pose_bounding_boxes,
    compute_camera_pose_2D_masks,
    object_optimal_k_camera_poses_bounding_box,
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
from nvsmask3d.eval.scannetpp.eval_semantic_instance import SCANNETPP_CLASSES
import tyro
import wandb
import threading
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nvsmask3d.utils.utils import concat_images_vertically, concat_images_horizontally,log_evaluation_results_to_wandb, select_low_entropy_logits
    

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
    path: Path = Path("nvsmask3d/data/replica")
    load_configs: Optional[List[str]] = None
    scene_names: Optional[ List[str]] = None
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
    kind: Literal["crop", "blur"] = "crop"
    prompt_threshold: float = 0.1
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    # inference
    inference_dataset: Literal["scannetpp", "replica"] = "replica"
    num_levels: int = 3
    #if run_name_for_wandb == "test":
    #pretrain_embeddings = torch.load("../../hanlin/pretrain_embeddings.pt", map_location="cuda")#20000,768


    def main(self) -> None:
        if self.run_name_for_wandb == "test":
            wandb.init(project=self.project_name, name=self.run_name_for_wandb)
        # 假设从配置或命令行参数中读取 project_name 和 run_name
        if self.inference_dataset == "replica":
            gt_dir = self.path / "ground_truth"
        if self.inference_dataset == "scannetpp":
            # scene_names = ['7b6477cb95', 'c50d2d1d42', 'cc5237fd77', 'acd95847c5', 'fb5a96b1a2', 'a24f64f7fb',
            # '1ada7a0617', '5eb31827b7', '3e8bba0176', '3f15a9266d', '21d970d8de', '5748ce6f01',
            # 'c4c04e6d6c', '7831862f02', 'bde1e479ad', '38d58a7a31', '5ee7c22ba0', 'f9f95681fd',
            # '3864514494', '40aec5fffa', '13c3e046d7', 'e398684d27', 'a8bf42d646', '45b0dac5e3',
            # '31a2c91c43', 'e7af285f7d', '286b55a2bf', '7bc286c1b6', 'f3685d06a9', 'b0a08200c9',
            # '825d228aec', 'a980334473', 'f2dc06b1d2', '5942004064', '25f3b7a318', 'bcd2436daf',
            # 'f3d64c30f8', '0d2ee665be', '3db0a1c8f3', 'ac48a9b736', 'c5439f4607', '578511c8a9',
            # 'd755b3d9d8', '99fa5c25e1', '09c1414f1b', '5f99900f09', '9071e139d9', '6115eddb86',
            # '27dd4da69e', 'c49a8c6cff']
            scene_names = self.scene_names
            test_mode = "all scannetpp"
            load_configs = self.load_configs
            # import pdb;pdb.set_trace()
            # gt_dir = "nvsmask3d/data/ScannetPP/sem_gt_val"
        if self.inference_dataset == "replica":
            scene_names = self.scene_names
            test_mode = "all replica"
            load_configs = self.load_configs
            # #save scene names, test_mode, load_configs to txt file for testing
            # with open("result.txt", "w") as f:
            #     f.write(f"scene_names: {scene_names}\n")
            #     f.write(f"test_mode: {test_mode}\n")
            #     f.write(f"load_configs: {load_configs}\n")
            # quit()
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
                self.model = pipeline.model
                # scene_id = scene_name[5:]
                #seed_points_0 = self.model.seed_points[0].cuda()  # shape (N, 3)
                pred_classes =  self.pred_classes_with_sam(
                                    scene_name=scene_name,
                                ) if self.sam  else self.pred_classes(
                                    model=self.model,
                                    class_agnostic_3d_mask=self.model.points3D_mask,
                                    seed_points_0=self.model.seed_points[0].cuda(),
                                    scene_name=scene_name,
                                )
                pred_masks = self.model.points3D_mask.cpu().numpy()  # move to cpu
                pred_scores = np.ones(pred_classes.shape[0])
                # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
                print(
                    f"pred_masks.shape, pred_scores.shape, pred_classes.shape {pred_masks.shape, pred_scores.shape, pred_classes.shape}"
                )
                preds[scene_name] = {
                    "pred_masks": pred_masks,#(num_points, num_cls) with 0 or 1 value
                    "pred_scores": pred_scores, # (num_cls,) with 1 value
                    "pred_classes": pred_classes, # (num_cls,) with value from dataset's class id
                }
                if self.inference_dataset == "scannetpp":#save pred one by one, later scannet_repo will do the evaluation in seperated script
                    VALID_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69, 70, 71, 72, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
                    output_dir = "/results/pred"### written?
                    save_predictions(preds, output_dir,VALID_CLASS_IDS)
                    
                    # # save preds to torch file
                    # output_folder = "results/segmentation"
                    # if not os.path.exists(output_folder):
                    #     os.makedirs(output_folder)
                    # # 确保路径中有分隔符
                    # torch.save(preds, os.path.join(output_folder, "preds.pth"))
                    # generate_txt_files_optimized(preds, "results/segmentation")
                    
            if self.inference_dataset == "replica":
                inst_AP = evaluate_replica(
                    preds, gt_dir, output_file="output.txt", dataset="replica"
                )
                # Create folder if it doesn't exist
                folder_name = "results"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # Add date and time to the filename
                timestamp = datetime.now().strftime("%m-%d_%H")
                file_name = os.path.join(folder_name, f"{timestamp}_{self.run_name_for_wandb}.json")

                # Write to the file
                with open(file_name, "w") as f:
                    json.dump(inst_AP, f, indent=4)
                #log_evaluation_results_to_wandb(inst_AP,self.run_name_for_wandb)
            if self.inference_dataset == "scannetpp":   
                # use scannetpp evaluation script
                command = "python -m scannetpp.semantic.eval.eval_instance eval_instance.yml"  # Example command
                output_file = self.run_name_for_wandb + ".txt"
                run_command_and_save_output(command, output_file)

    def pred_classes_with_sam(self, scene_name=""):
        """
        Args:
        class_agnostic_3d_mask (torch.Tensor): The class-agnostic 3D mask (N, num_cls)
        seed_points_0 (torch.Tensor): The seed points (N,3)
        k_poses (int): The number of poses to render
        """
        class_agnostic_3d_mask=self.model.points3D_mask
        seed_points_0=self.model.seed_points[0].cuda()

        sam_config = SAMNetworkConfig()
        sam_network = SamNetWork(sam_config)
        # Move camera transformations to the GPU
    
        camera_to_world_opengl = self.model.cameras.camera_to_worlds.to(
            "cuda"
        )  # shape (M, 4, 4)
        camera_to_world_opencv = get_camera_pose_in_opencv_convention(camera_to_world_opengl)

        # Move intrinsics to the GPU
        K = self.model.cameras.get_intrinsics_matrices().to("cuda")  # shape (M, 3, 3)
        W, H = int(self.model.cameras.width[0].item()), int(self.model.cameras.height[0].item())

        # Convert class-agnostic mask to a boolean tensor and move to GPU
        # boolean_masks = torch.from_numpy(class_agnostic_3d_mask).bool().to('cuda')  # shape (N, 166)
        cls_num = class_agnostic_3d_mask.shape[1]
        pred_classes = np.full(cls_num, 0)  # -1)
        
        # Loop through each mask/ object
        for i in tqdm(range(cls_num), 
              desc="Inferenceing objects", 
              total=cls_num):
            # set instance
            self.model.cls_index = i
            boolean_mask = class_agnostic_3d_mask[:, i]
            masked_seed_points = seed_points_0[boolean_mask]  # shape (N_masked, 3)
            best_camera_indices, valid_u, valid_v = object_optimal_k_camera_poses_2D_mask( 
                seed_points_0=seed_points_0,
                optimized_camera_to_world=camera_to_world_opencv,
                K=K,
                W=W,
                H=H,
                boolean_mask=boolean_mask,  # select i_th mask
                depth_filenames=self.model.metadata["depth_filenames"] if self.occlusion_aware else None,
                depth_scale=self.model.depth_scale,
                k_poses=self.top_k,
                score_fn=self.visibility_score,
                vis_depth_threshold=0.05 if self.inference_dataset != "replica" else 0.4
            )
            #sorted camera indices and its index
            # this is for smoother interpolation, keep the order of camera indices
            # Note! pose_sorted_index is not aligned with valid_u and valid_v's index anymore
            best_camera_indices, pose_sorted_index = torch.sort(best_camera_indices) 

            # Skip if no valid camera poses
            if best_camera_indices.shape[0] == 0 or len(valid_u) == 0 or len(valid_v) == 0:
                # print(
                #     f"Skipping inference for mask {i} due to no valid camera poses, assign",
                # )
                continue
            
            # kind = "crop"
            kind = self.kind
            blur_std_dev = 100.0

            rgb_outputs = []
            masked_gaussian_outputs = []
            # Interpolate camera poses and bounding boxes
            if self.interpolate_n_camera*self.interpolate_n_rgb_camera > 0 or self.interpolate_n_camera*self.interpolate_n_gaussian_camera > 0:
                #################################################################################################camera interpolation################################################################################################
                interpolated_poses = interpolate_camera_poses_with_camera_trajectory(
                    camera_to_world_opengl[best_camera_indices],
                    masked_seed_points,
                    self.interpolate_n_camera,
                )# ((pose-1) * step, 3, 4)
                interpolated_cameras = make_cameras(self.model.cameras[0:1], interpolated_poses) 

                # Project points to 2D image coordinates for all camera poses
                interpolated_poses_cv = get_camera_pose_in_opencv_convention(interpolated_poses)
                u, v, z = get_points_projected_uv_and_depth(masked_seed_points, interpolated_poses_cv, K = interpolated_cameras.get_intrinsics_matrices().to(device="cuda"))
                valid_points = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)  # shape (M, N_masked)

                interpolated_images = []
                #interpolated_images_label_map = []
                # Render NVS images, crop, save, and add to outputs
                for interpolation_index in range(interpolated_cameras.shape[0]):
                    camera = interpolated_cameras[interpolation_index:interpolation_index+1]
                    valid = valid_points[interpolation_index]  # 形状：(N,)
                    if valid_points[interpolation_index].any():
                        nvs_img = self.model.get_outputs(camera)["rgb"]  # (H, W, 3)
                        u_i = u[interpolation_index][valid]#(N,)
                        v_i = v[interpolation_index][valid]
                        
                        # 转换为整数索引
                        u_i = u_i.long()
                        v_i = v_i.long()
                        
                        # 确保索引在图像范围内
                        u_i = torch.clamp(u_i, 0, W - 1)
                        v_i = torch.clamp(v_i, 0, H - 1)
                        #3DMask logic 
                        min_u = u_i.min()
                        max_u = u_i.max()
                        min_v = v_i.min()
                        max_v = v_i.max()

                        # ##################SAM logic#####################
                        # proposal_points_coords_2d = torch.stack((v_i, u_i), dim=1)  # (N, 2)
                        # sam_network.set_image(nvs_img.permute(2,0,1))#3,H,W
                        # mask_i = sam_network.get_best_mask(proposal_points_coords_2d)
                        # if mask_i.sum() == 0:
                        #     # print(f"Skipping inference for object {i} pose {index} due to no valid camera poses, assign")
                        #     continue
                        
                        # #multilevel mask
                        # #for level in range(self.num_levels):
                        # level = 0
                        # min_u, min_v, max_u, max_v = sam_network.mask2box_multi_level(mask_i, level, expansion_ratio = 0.1)
                        # H, W, _ = nvs_img.shape # (H, W, 3)
                        # min_u = max(0, min_u)
                        # min_v = max(0, min_v)
                        # max_u = min(W, max_u)
                        # max_v = min(H, max_v)
                        
                        # Check if bounding box is valid
                        if min_u < max_u and min_v < max_v:
                            nvs_mask_img_pil = None
                            nvs_img_pil = None
                            # nvs_img_label_map = None
                            # nvs_mask_img_label_map = None
                            if self.interpolate_n_rgb_camera > 0:
                                if kind == "crop":
                                    #Get output dimensions to validate bounding box
                                    cropped_nvs_img = nvs_img[min_v:max_v, min_u:max_u]# (H, W, 3)
                                    cropped_nvs_img = cropped_nvs_img.permute(2, 0, 1) # (C, H, W)
                                    rgb_outputs.append(cropped_nvs_img) # (C, H, W) ------->add to rgb_outputs
                                    cropped_nvs_img = cropped_nvs_img.cpu()#for wandb
                                    # nvs_img_label_map = model.image_encoder.return_image_map(cropped_nvs_img)#for wandb
                                    nvs_img_pil = transforms.ToPILImage()(cropped_nvs_img)#for wandb
                                    #############debug################
                                    # try:
                                    #     sparse_map = torch.zeros((H, W, 3), dtype=torch.float32, device="cuda")
                                    #     # sparse_map[ v_i, u_i] = 1
                                    #     sparse_map[mask_i] = 1
                                    #     from nvsmask3d.utils.utils import save_img
                                    #     save_img(nvs_img, f"tests/interp_object_{i}_camera_{interpolation_index}.png")
                                    #     save_img(sparse_map, f"tests/interp_sparse_map_object_{i}_camera_{interpolation_index}.png")
                                    #     save_img(cropped_nvs_img.permute(1,2,0), f"tests/interp_object_{i}_cropped_camera_{interpolation_index}.png")
                                    # except Exception as e:
                                    #     print(f"Failed to save image {interpolation_index}: {e}")
                                    #     continue  
                                    ##################################
                                
                                elif kind == "blur":
                                    # temp = transforms.ToPILImage()(nvs_img.permute(2, 0, 1).cpu())
                                    
                                    # result = temp.copy()
                                    # result = result.filter(ImageFilter.GaussianBlur(blur_std_dev))
                                    
                                    # width, height = temp.size
                                    # mask = Image.new("L", (width, height), 0)
                                    # draw = ImageDraw.Draw(mask)
                                    # draw.rectangle((min_u, min_v, max_u, max_v), fill=255)
                                    
                                    # result.paste(temp, mask=mask)
                                    result_tensor = make_square_image(nvs_img, min_v, max_v, min_u, max_u)
                                    
                                    # result_tensor = transforms.ToTensor()(result)
                                    rgb_outputs.append(result_tensor.to(device="cuda"))
                                    
                                    nvs_img_pil = transforms.ToPILImage(result_tensor)
                                    
                            if self.interpolate_n_gaussian_camera > 0:
                                # # Process and crop the nvs mask image, seems will make inference worse
                                nvs_mask_img = self.model.get_outputs(camera)["rgb_mask"]  # (H, W, 3)
                                cropped_nvs_mask_image = nvs_mask_img[min_v:max_v, min_u:max_u].permute(2, 0, 1)  # (C, H, W)
                                masked_gaussian_outputs.append(cropped_nvs_mask_image) # (C, H, W)------->add to masked_gaussian_outputs
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
            #################################################################################################GT################################################################################################
            #gt camera pose
            if self.gt_camera_rgb or self.gt_camera_gaussian:
                gt_images = []
                # gt_images_label_map = []
                
                for pose_index, index in zip(best_camera_indices, pose_sorted_index):
                    if valid_u[index].shape[0] == 0 or valid_v[index].shape[0] == 0:
                        # print(f"Skipping inference for object {i} pose {index} due to no valid camera poses, assign")
                        continue
                    pose_index = pose_index.item()
                    single_camera = self.model.cameras[pose_index : pose_index + 1]
                    assert single_camera.shape[0] == 1, "Only one camera at a time"
                    with Image.open(self.model.image_file_names[pose_index]) as img:
                        img = transforms.ToTensor()(img).cuda()  # (C,H,W)

                    if self.sam==False:#no SAM
                        pass
                    else:#SAM logic
                        proposal_points_coords_2d = torch.stack((valid_v[index], valid_u[index]), dim=1)  # (N, 2)
                        assert(len(proposal_points_coords_2d.shape) == 2)
                        sam_network.set_image(img)#3,H,W
                        mask_i = sam_network.get_best_mask(proposal_points_coords_2d)# [shape: (H, W)]
                        if mask_i.sum() == 0:
                            # print(f"Skipping inference for object {i} pose {index} due to no valid camera poses, assign")
                            continue
                        
                        #multilevel mask
                        for level in range(self.num_levels):
                            #level = 0
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
                                    if kind == "crop":
                                        cropped_image = img[:, min_v:max_v, min_u:max_u]
                                        cropped_mask = mask_i[min_v:max_v, min_u:max_u]
                                        #blurred_image = blur_non_masked_areas(img, mask_i)
                                        #cropped_image = blur_non_masked_areas(cropped_image, cropped_mask)
                                        #############debug################
                                        # try:
                                        #     from nvsmask3d.utils.utils import save_img
                                        #     save_img(img.permute(1,2,0), f"tests/gt_object_{i}_camera_{index}.png")
                                        #     # save_img(cropped_image.permute(1,2,0), f"tests/gt_object_{i}_blurred_camera_{index}.png")
                                        #     save_img(cropped_image.permute(1,2,0), f"tests/gt_object_{i}_cropped_camera_{index}.png")
                                        #     import pdb;pdb.set_trace()
                                        # except Exception as e:
                                        #     import pdb;pdb.set_trace()
                                        #     print(f"Failed to save image {interpolation_index}: {e}")
                                        #     continue  
                                        # import pdb;pdb.set_trace()

                                        ##################################

                                        rgb_outputs.append(cropped_image)#######################################################################################rgb#####################
                                        cropped_image = cropped_image.cpu()#for wandb
                                        # gt_img_pil_label_map = model.image_encoder.return_image_map(cropped_image) #for wandb
                                        gt_img_pil = transforms.ToPILImage()(cropped_image)#for wandb
                                        
                                    elif kind == "blur":
                                        # temp = transforms.ToPILImage()(nvs_img.permute(2, 0, 1).cpu())
                                        
                                        # # test
                                        # # assert torch.allclose(nvs_img.permute(2, 0, 1).cpu(), transforms.ToTensor()(img))
                                        
                                        # result = temp.copy()
                                        # result = result.filter(ImageFilter.GaussianBlur(blur_std_dev))
                                        # # result.paste(temp, mask=transforms.ToPILImage()(mask_i.cpu()))
                                        
                                        # width, height = temp.size
                                        # mask = Image.new("L", (width, height), 0)
                                        # draw = ImageDraw.Draw(mask)
                                        # draw.rectangle((min_u, min_v, max_u, max_v), fill=255)
                                        
                                        # result.paste(temp, mask=mask)
                                        # result_tensor = transforms.ToTensor()(result)
                                        
                                        result_tensor = make_square_image(nvs_img, min_v, max_v, min_u, max_u)
                                        
                                        rgb_outputs.append(result_tensor.to(device="cuda"))
                                        
                                        gt_img_pil = transforms.ToPILImage(result_tensor)
                                        
                                if self.gt_camera_gaussian:
                                    nvs_mask_img = self.model.get_outputs(single_camera)["rgb_mask"]  # ["rgb_mask"]  # (H,W,3)
                                    cropped_nvs_mask_image = nvs_mask_img[min_v:max_v, min_u:max_u,:].permute(2, 0, 1)#(C,H,W)
                                    masked_gaussian_outputs.append(cropped_nvs_mask_image)#############################################################################gaussian###################
                                    cropped_nvs_mask_image = cropped_nvs_mask_image.cpu()#for wandb
                                    # gt_mask_img_label_map = model.image_encoder.return_image_map(cropped_nvs_mask_image)#for wandb
                                    gt_mask_img_pil = transforms.ToPILImage()(cropped_nvs_mask_image) #for wandb
                                        
                                # Combine GT image and mask horizontally
                                combined_gt_image = concat_images_vertically([gt_img_pil, gt_mask_img_pil])
                                # combined_gt_image_label_map = concat_images_vertically([gt_img_pil_label_map, gt_mask_img_label_map])
                                gt_images.append(combined_gt_image)
                                # gt_images_label_map.append(combined_gt_image_label_map)
                                #############debug################
                                # try:
                                #     sparse_map = torch.zeros((H, W, 3), dtype=torch.float32, device="cuda")
                                #     sparse_map[ valid_v[index], valid_u[index]] = 1
                                #     from nvsmask3d.utils.utils import save_img
                                #     save_img(img.permute(1,2,0), f"tests/gt_object_{i}_camera_{index}.png")
                                #     save_img(sparse_map, f"tests/gt_sparse_map_object_{i}_camera_{index}.png")
                                #     save_img(cropped_image.permute(1,2,0), f"tests/gt_object_{i}_cropped_camera_{index}.png")
                                # except Exception as e:
                                #     import pdb;pdb.set_trace()
                                #     print(f"Failed to save image {interpolation_index}: {e}")
                                #     continue  
                                ##################################

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
            
            
            #################################################################################################inference################################################################################################
            with torch.no_grad():
                T = 1.0# refer to temperature
                if len(rgb_outputs) > 0:
                    rgb_features = self.model.image_encoder.encode_batch_list_image(
                        rgb_outputs
                    )  
                    rgb_logits = torch.mm(
                        rgb_features, self.model.image_encoder.pos_embeds.T
                    )  
                    #pretrained text prompt
                    #rgb_logits_pretrain_text = torch.mm(rgb_features, self.pretrain_embeddings.T)
                if len(masked_gaussian_outputs) > 0:
                    mask_features = self.model.image_encoder.encode_batch_list_image(
                        masked_gaussian_outputs
                    )  
                    mask_logits = torch.mm(
                        mask_features, self.model.image_encoder.pos_embeds.T
                    )  
                    #mask_logits_pretrain_text = torch.mm(mask_features, self.pretrain_embeddings.T)
                    
                if self.algorithm == 0:
                #aggregate similarity scores 你目前是将批次中的相似度分数进行求和（sum），这可能会导致信息丢失，尤其是在增强视图之间存在较大差异的情况下。
                    if len(masked_gaussian_outputs) > 0:
                        if self.interpolate_n_camera > 1:
                            mask_logits[:-self.top_k]/=self.interpolate_n_camera
                        scores = mask_logits.sum(dim=0)
                    if len(rgb_outputs) > 0:
                        if self.interpolate_n_camera > 1:
                            rgb_logits[:-self.top_k]/=self.interpolate_n_camera
                        scores = rgb_logits.sum(dim=0)

                # if self.algorithm == 1:
                #     weights_mask = None
                #     weights_rgb = None
                #     # if self.run_name_for_wandb == "test":
                #     if len(masked_gaussian_outputs) > 0:
                #         correction_mask = mask_logits_pretrain_text.mean(dim=1, keepdim=True)
                #         weights_mask = mask_logits.max(dim=1, keepdim=True).values # accriss text prompt Mx1
                #         weights_mask = torch.softmax((weights_mask - correction_mask) / T, dim=0) 
                    
                #     if len(rgb_outputs) > 0:
                #         correction_rgb = rgb_logits_pretrain_text.mean(dim=1, keepdim=True)
                #         weights_rgb = rgb_logits.max(dim=1, keepdim=True).values  # accriss text prompt
                #         weights_rgb = torch.softmax((weights_rgb - correction_rgb) / T, dim=0)
                        
                #     #concat weights
                #     if weights_mask is not None and weights_rgb is not None:
                #         weights = torch.cat([weights_mask, weights_rgb], dim=0) 
                #         all_logits = torch.cat([mask_logits, rgb_logits], dim=0)
                #     elif weights_mask is not None:
                #         weights = weights_mask
                #         all_logits = mask_logits
                #     else:
                #         weights = weights_rgb
                #         all_logits = rgb_logits    

                #     weighted_logits = all_logits * weights
                #     scores = torch.sum(weighted_logits, dim=0)

                    
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
                if 'rgb_features' in locals():
                    del rgb_features
                if 'mask_features' in locals():
                    del mask_features
                torch.cuda.empty_cache()
                
            #     if 'interpolated_images_label_map' in locals() and len(interpolated_images_label_map) > 0:
            #         final_interpolated_image_label_map = concat_images_horizontally(interpolated_images_label_map)
            #         wandb.log({f"Interpolated Scene: {scene_name}": wandb.Image(final_interpolated_image_label_map, caption=f"Interpolated Image Label Map for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})
                
            #     if 'gt_images_label_map' in locals() and len(gt_images_label_map) > 0:
            #         final_gt_image_label_map = concat_images_horizontally(gt_images_label_map)
            #         wandb.log({f"GT Scene: {scene_name}": wandb.Image(final_gt_image_label_map, caption=f"GT Camera Pose Label Map for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})
            # # import pdb;pdb.set_trace()
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
        for i in tqdm(range(cls_num), 
              desc="Inferenceing objects", 
              total=cls_num):
            # set instance
            model.cls_index = i
            boolean_mask = class_agnostic_3d_mask[:, i]
            # try:
            #     result  = object_optimal_k_camera_poses_2D_mask(#object_optimal_k_camera_poses_bounding_box(
            #         seed_points_0=seed_points_0,
            #         optimized_camera_to_world=camera_to_world_opencv,
            #         K=K,
            #         W=W,
            #         H=H,
            #         boolean_mask=boolean_mask,  # select i_th mask
            #         depth_filenames=model.metadata["depth_filenames"] if self.occlusion_aware else None,
            #         depth_scale=model.depth_scale,
            #         k_poses=self.top_k,
            #         score_fn=self.visibility_score,
            #         vis_depth_threshold=0.05 if self.inference_dataset != "replica" else 0.4
            #         )
            #     best_camera_indices, valid_u, valid_v = result
            # except:
            #     import pdb;pdb.set_trace()
            #set time
            best_camera_indices, valid_u, valid_v  = object_optimal_k_camera_poses_2D_mask(#object_optimal_k_camera_poses_bounding_box(
                                                        seed_points_0=seed_points_0,
                                                        optimized_camera_to_world=camera_to_world_opencv,
                                                        K=K,
                                                        W=W,
                                                        H=H,
                                                        boolean_mask=boolean_mask,  # select i_th mask
                                                        depth_filenames=model.metadata["depth_filenames"] if self.occlusion_aware else None,
                                                        depth_scale=model.depth_scale,
                                                        k_poses=self.top_k,
                                                        score_fn=self.visibility_score,
                                                        vis_depth_threshold=0.05 if self.inference_dataset != "replica" else 0.4
                                                        )
            #sorted camera indices and its index
            # this is for smoother interpolation, keep the order of camera indices
            # Note! pose_sorted_index is not aligned with valid_u and valid_v's index anymore
            best_camera_indices, pose_sorted_index = torch.sort(best_camera_indices) 

            if best_camera_indices.shape[0] == 0 or len(valid_u) == 0 or len(valid_v) == 0:
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
                            if self.kind == "crop":
                                #Get output dimensions to validate bounding box
                                nvs_img = model.get_outputs(camera)["rgb"]  # (H, W, 3)
                                cropped_nvs_img = nvs_img[min_v:max_v, min_u:max_u]
                                cropped_nvs_img = cropped_nvs_img.permute(2, 0, 1) # (H, W, 3)
                                rgb_outputs.append(cropped_nvs_img)  # (C, H, W) ######################################################rgb##################################################################
                                cropped_nvs_img = cropped_nvs_img.cpu()#for wandb
                                # nvs_img_label_map = model.image_encoder.return_image_map(cropped_nvs_img)#for wandb
                                #############debug################
                                # Save the cropped image
                                # try:
                                #     save_img(nvs_img, f"tests/nvs_image_{interpolation_index}.png")
                                #     save_img(cropped_nvs_img.permute(1,2,0), f"tests/cropped_nvs_image_{interpolation_index}.png")
                                # except Exception as e:
                                #     print(f"Failed to save image {interpolation_index}: {e}")
                                #     continue  
                                ##################################
                            if self.kind == "blur":
                                pass
                        if self.interpolate_n_gaussian_camera > 0:
                            # # Process and crop the nvs mask image, seems will make inference worse
                            nvs_mask_img = model.get_outputs(camera)["rgb_mask"]  # (H, W, 3)
                            cropped_nvs_mask_image = nvs_mask_img[min_v:max_v, min_u:max_u].permute(2, 0, 1)  # (C, H, W)
                            masked_gaussian_outputs.append(cropped_nvs_mask_image)###########################################################################gaussian#####################################################################
                            cropped_nvs_mask_image = cropped_nvs_mask_image.cpu()
                            #nvs_mask_img_label_map = model.image_encoder.return_image_map(cropped_nvs_mask_image)#for wandb
                            # Combine GT image and mask horizontally
                        if self.wandb_mode != "disabled":
                            if self.interpolate_n_rgb_camera > 0:
                                nvs_img_pil = transforms.ToPILImage()(cropped_nvs_img)#for wandb
                            if self.interpolate_n_gaussian_camera > 0:
                                nvs_mask_img_pil = transforms.ToPILImage()(cropped_nvs_mask_image)#for wandb
                            combined_nvs_image = concat_images_vertically([nvs_img_pil, nvs_mask_img_pil])#for wandb
                            # combined_nvs_image_label_map = concat_images_vertically([nvs_img_label_map, nvs_mask_img_label_map])#for wandb
                            interpolated_images.append(combined_nvs_image)#for wandb
                            # interpolated_images_label_map.append(combined_nvs_image_label_map)#for wandb
                    else:
                        print(f"Invalid bounding box for image {interpolation_index}: "
                            f"min_u={min_u}, max_u={max_u}, min_v={min_v}, max_v={max_v}")
                # interpolated images's selection via entropy
            #################################################################################################
            #gt camera pose
            if self.gt_camera_rgb or self.gt_camera_gaussian:
                gt_images = []
                # gt_images_label_map = []
                for  pose_index, index in zip(best_camera_indices, pose_sorted_index):
                    if valid_u[index].shape[0] == 0 or valid_v[index].shape[0] == 0:
                        continue
                    pose_index = pose_index.item()
                    single_camera = model.cameras[pose_index : pose_index + 1]
                    assert single_camera.shape[0] == 1, "Only one camera at a time"
                    # img = model.get_outputs(single_camera)["rgb_mask"]#["rgb"]#
                    with Image.open(model.image_file_names[pose_index]) as img:
                        img = transforms.ToTensor()(img).cuda()  # (C,H,W)

                    min_u = min(torch.clamp(valid_u[index], 0, W-1))
                    min_v = min(torch.clamp(valid_v[index], 0, H-1))
                    max_u = max(torch.clamp(valid_u[index], 0, W-1))
                    max_v = max(torch.clamp(valid_v[index], 0, H-1))

                    # # nvs_img = model.get_outputs(single_camera)["rgb"]  # (H,W,3)
                    # min_u, min_v, max_u, max_v = bounding_boxes[index]
                    # min_u = 0 if min_u == float('-inf') else min_u
                    # min_v = 0 if min_v == float('-inf') else min_v
                    # max_u = W if max_u == float('inf') else max_u
                    # max_v = H if max_v == float('inf') else max_v
                    
                    # min_u, min_v, max_u, max_v = map(int, [min_u, min_v, max_u, max_v])
                    # _, H, W = img.shape

                    # # Clamp values to ensure they are within the valid range
                    # min_u = max(0, min(min_u, W - 1))
                    # min_v = max(0, min(min_v, H - 1))
                    # max_u = max(0, min(max_u, W))
                    # max_v = max(0, min(max_v, H))

                    # 检查裁剪区域是否有效（确保裁剪区域有正面积）
                    if min_u < max_u and min_v < max_v:
                        gt_img_pil = None
                        gt_mask_img_pil = None
                        # gt_mask_img_label_map = None  
                        # gt_img_pil_label_map = None

                        # 如果有效，则裁剪图像
                        if self.gt_camera_rgb:
                            if self.kind == "crop":
                                cropped_image = img[:, min_v:max_v, min_u:max_u]
                                rgb_outputs.append(cropped_image)#######################################################################################rgb#####################
                                cropped_image = cropped_image.cpu()#for wandb
                                # gt_img_pil_label_map = model.image_encoder.return_image_map(cropped_image) #for wandb
                                gt_img_pil = transforms.ToPILImage()(cropped_image)#for wandb
                            if self.kind == "blur":
                                pass
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
                # algorithm = 1
                # I AM CHANGING THIS
                algorithm = self.algorithm
                T = 1.0# refer to temperature
                if len(rgb_outputs) > 0:
                    rgb_features = model.image_encoder.encode_batch_list_image(
                        rgb_outputs
                    )  
                    rgb_logits = torch.mm(
                        rgb_features, model.image_encoder.pos_embeds.T
                    )  
                    #pretrained text prompt
                    #rgb_logits_pretrain_text = torch.mm(rgb_features, self.pretrain_embeddings.T)
                if len(masked_gaussian_outputs) > 0:
                    mask_features = model.image_encoder.encode_batch_list_image(
                        masked_gaussian_outputs
                    )  
                    mask_logits = torch.mm(
                        mask_features, model.image_encoder.pos_embeds.T
                    )  
                    #mask_logits_pretrain_text = torch.mm(mask_features, self.pretrain_embeddings.T)
                # if self.run_name_for_wandb == "test":
                if algorithm == 0:  
                #aggregate similarity scores 你目前是将批次中的相似度分数进行求和（sum），这可能会导致信息丢失，尤其是在增强视图之间存在较大差异的情况下。
                    if len(masked_gaussian_outputs) > 0:
                        if self.interpolate_n_camera > 1:
                            mask_logits[:-self.top_k]/=self.interpolate_n_camera
                        scores = mask_logits.sum(dim=0)  
                        #scores = select_low_entropy_logits(mask_logits, self.top_k, apply_softmax=True).sum(dim=0)
                    if len(rgb_outputs) > 0:
                        if self.interpolate_n_camera > 1:
                            rgb_logits[:-self.top_k]/=self.interpolate_n_camera
                        scores = rgb_logits.sum(dim=0) 
                            
                        #scores = select_low_entropy_logits(rgb_logits, self.top_k, apply_softmax=True).sum(dim=0)  

                        #weighted with entropy cant increase the performance
                        # # Step 1: Compute class probabilities for each view
                        # probs = torch.softmax(rgb_logits, dim=-1)  # Shape: [num_views, num_classes]
                        # # Step 2: Calculate entropy for each class across views
                        # entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=0)  # Shape: [num_classes]

                        # # Step 3: Use inverse entropy to compute weights (lower entropy = higher weight)
                        # weights = 1 / (entropy + 1e-9)  # Add small constant to avoid division by zero
                        # weights = weights / weights.sum()  # Normalize the weights

                        # # Step 4: Compute the weighted sum of the logits
                        # scores = torch.sum(rgb_logits * weights, dim=0)  # Shape: [num_classes]
                # if algorithm == 1:
                #     weights_mask = None
                #     weights_rgb = None
                #     # if self.run_name_for_wandb == "test":
                #     if len(masked_gaussian_outputs) > 0:
                #         correction_mask = mask_logits_pretrain_text.mean(dim=1, keepdim=True)
                #         weights_mask = mask_logits.max(dim=1, keepdim=True).values # accriss text prompt Mx1
                #         weights_mask = torch.softmax((weights_mask - correction_mask) / T, dim=0) 
                    
                #     if len(rgb_outputs) > 0:
                #         correction_rgb = rgb_logits_pretrain_text.mean(dim=1, keepdim=True)
                #         weights_rgb = rgb_logits.max(dim=1, keepdim=True).values  # accriss text prompt
                #         weights_rgb = torch.softmax((weights_rgb - correction_rgb) / T, dim=0)
                        
                #     #concat weights
                #     if weights_mask is not None and weights_rgb is not None:
                #         weights = torch.cat([weights_mask, weights_rgb], dim=0) 
                #         all_logits = torch.cat([mask_logits, rgb_logits], dim=0)
                #     elif weights_mask is not None:
                #         weights = weights_mask
                #         all_logits = mask_logits
                #     else:
                #         weights = weights_rgb
                #         all_logits = rgb_logits    

                #     weighted_logits = all_logits * weights
                #     scores = torch.sum(weighted_logits, dim=0)
                    
                # if self.algorithm == 2:
                    
                #     weights_mask = None
                #     weights_rgb = None
                #     # if self.run_name_for_wandb == "test":
                #     if len(masked_gaussian_outputs) > 0:
                #         correction_mask = mask_logits_pretrain_text.mean(dim=1, keepdim=True)
                #         weights_mask = mask_logits.max(dim=1, keepdim=True).values # accriss text prompt Mx1
                #         # weights_mask = torch.softmax((weights_mask - correction_mask) / T, dim=0) 
                    
                #     if len(rgb_outputs) > 0:
                #         correction_rgb = rgb_logits_pretrain_text.mean(dim=1, keepdim=True)
                #         weights_rgb = rgb_logits.max(dim=1, keepdim=True).values  # accriss text prompt
                #         # weights_rgb = torch.softmax((weights_rgb - correction_rgb) / T, dim=0)
                        
                #     #concat weights
                #     if weights_mask is not None and weights_rgb is not None:
                #         weights = torch.cat([weights_mask, weights_rgb], dim=0)
                #         correction = torch.cat([correction_mask, correction_rgb], dim=0)
                #         weights = torch.softmax((weights - correction) / T, dim=0)
                #         all_logits = torch.cat([mask_logits, rgb_logits], dim=0)
                #     elif weights_mask is not None:
                #         weights = torch.softmax((weights_mask - correction_mask) / T, dim=0)
                #         all_logits = mask_logits
                #     else:
                #         weights = torch.softmax((weights_rgb - correction_rgb) / T, dim=0)
                #         all_logits = rgb_logits    

                    # weighted_logits = all_logits * weights
                    # scores = torch.sum(weighted_logits, dim=0)
                    
                    # med_weights = torch.median(weights)
                    # med_diffs = torch.median(torch.abs(weights - med_weights))
                    # scores = (weights - med_weights) / med_diffs
                    # chosen = (scores > 0.5).squeeze()
                    # print(f"Choosen {torch.sum(chosen)} out of {len(weights)}")
                    # # print(all_logits.shape, weights.shape)
                    
                    # weighted_logits = all_logits[chosen, :] * weights[chosen, :]
                    # # print(weighted_logits.shape)
                    # scores = torch.sum(weighted_logits, dim=0)
                    

                # if algorithm == 2:
                #     E_pretrain_text = torch.mean(similarity_scores_pretrain_text, dim=1)  # (B,)
                #     assert E_pretrain_text.shape == (B,)
                #     logit_normalized = similarity_scores - ( E_pretrain_text).unsqueeze(1)# (B,C)
                #     weights = torch.softmax(logit_normalized/T, dim=0)
                #     scores = torch.sum(similarity_scores * weights, dim=0)
                
                # else:
                #     #aggregate similarity scores 你目前是将批次中的相似度分数进行求和（sum），这可能会导致信息丢失，尤其是在增强视图之间存在较大差异的情况下。
                #     scores = similarity_scores.sum(dim=0)  # Shape: (200,) for scannet200 
                #     max_ind = torch.argmax(scores).item()
                
                max_ind = torch.argmax(scores).item()
                pred_classes[i] = max_ind  
                # Log interpolated images
                if self.wandb_mode != "disabled":
                    if 'interpolated_images' in locals() and len(interpolated_images) > 0:
                        plot_images_and_logits(
                            i,
                            interpolated_images, rgb_logits[:-self.top_k] if len(rgb_outputs) > 0 else mask_logits[:-self.top_k], 
                            "Interpolated Scene", 'combined_image_with_logits_fixed_and_points.png', 
                            scene_name, max_ind, REPLICA_CLASSES if self.inference_dataset == "replica" else SCANNETPP_CLASSES
                        )
                        
                    # Log GT images
                    if 'gt_images' in locals() and len(gt_images) > 0:
                        # Use the helper function for GT images
                        plot_images_and_logits(
                            i,
                            gt_images, rgb_logits[-self.top_k:] if len(rgb_outputs) > 0 else mask_logits[-self.top_k:], 
                            "GT Scene", 'combined_image_with_logits_fixed_and_points.png', 
                            scene_name, max_ind, REPLICA_CLASSES if self.inference_dataset == "replica" else SCANNETPP_CLASSES
                        )
                        #wandb.log({f"GT Scene: {scene_name}": wandb.Image(final_gt_image, caption=f"GT Camera Pose for object {i} predicted class: {REPLICA_CLASSES[max_ind]}")})
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
get_parser_fn = lambda: tyro.extras.get_parser(Commands) 