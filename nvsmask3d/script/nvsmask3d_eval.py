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
"""

from __future__ import annotations
from PIL import Image
import torchvision.transforms as transforms

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Literal, Optional, Tuple, Union
import torch
from nvsmask3d.utils.camera_utils import (
    object_optimal_k_camera_poses,
    optimal_k_camera_poses_of_scene,
    object_optimal_k_camera_poses_clear,
    object_optimal_k_camera_poses_bounding_box_clear,
    object_optimal_k_camera_poses_bounding_box,
)
from nerfstudio.models.splatfacto import SplatfactoModel
from tqdm import tqdm
from typing_extensions import Annotated
import numpy as np
from nvsmask3d.eval.scannet200.eval_semantic_instance import evaluate as evaluate_scannet200
from nvsmask3d.eval.replica.eval_semantic_instance import evaluate as evaluate_replica

import tyro
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


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


class InstSegEvaluator:
    def __init__(self, dataset_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type

    def evaluate_full(
        self,
        preds,
        scene_gt_dir,
        dataset,
        output_file="temp_output.txt",
        pretrained_on_scannet200=True,
    ):
        if dataset == "scannet200":
            inst_AP = evaluate_scannet200(
                preds,
                scene_gt_dir,
                output_file=output_file,
                dataset=dataset,
                pretrained_on_scannet200=pretrained_on_scannet200,
            )
        else:
            print("DATASET NOT SUPPORTED!")
            exit()
        return inst_AP


@dataclass
class ComputeForAP:  # pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 177), (177,), (177,))
    """Load a checkpoint, compute some pred_scores and pred_classes for latter AP computation."""

    # Path to config YAML file.
    load_config: Path = Path("nvsmask3d/data/replica")
    # Name of the output file.
    output_path: Path = Path("")
    top_k: int = 15

    #inference
    inference_dataset: Literal["scannet200","replica"] = "replica"

    def main(self) -> None:
        gt_dir = self.load_config / "ground_truth"
        if self.inference_dataset == "replica":
            scene_names = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
            test_mode =  "all replica"
            
            load_configs = ["outputs/office0/nvsmask3d/2024-08-12_215616/config.yml",
                            "outputs/office1/nvsmask3d/2024-08-12_170536/config.yml",
                            "outputs/office2/nvsmask3d/2024-08-12_173744/config.yml",
                            "outputs/office3/nvsmask3d/2024-08-12_173744/config.yml",
                            "outputs/office4/nvsmask3d/2024-08-12_180405/config.yml",
                            "outputs/room0/nvsmask3d/2024-08-12_180418/config.yml",
                            "outputs/room1/nvsmask3d/2024-08-12_182825/config.yml",
                            "outputs/room2/nvsmask3d/2024-08-12_182844/config.yml"]
            

        preds = {}
        #scene_names = ["scene0011_00"]  # hard coded for now
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
                #scene_id = scene_name[5:]
                seed_points_0 = model.seed_points[0].half().cuda()  # shape (N, 3)
                
                pred_classes = self.pred_classes(
                    model = model,
                    class_agnostic_3d_mask=model.points3D_mask,
                    seed_points_0=seed_points_0,
                    k_poses=2,
                )
                pred_masks = model.points3D_mask.cpu().numpy()# move to cpu
                pred_scores = np.ones(pred_classes.shape[0])
                # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
                print(f"pred_masks.shape, pred_scores.shape, pred_classes.shape {pred_masks.shape, pred_scores.shape, pred_classes.shape}")
                preds[scene_name] = {
                    "pred_masks": pred_masks, 
                    "pred_scores": pred_scores,
                    "pred_classes": pred_classes,
                }       
            if self.inference_dataset == "replica":
                inst_AP = evaluate_replica(
                    preds, gt_dir, output_file="output.txt", dataset="replica"
            )

    def pred_classes(self, model, class_agnostic_3d_mask, seed_points_0, k_poses =2):
        """
        Args:
        model (NVSMask3DModel): The model to use for inference
        class_agnostic_3d_mask (torch.Tensor): The class-agnostic 3D mask (N, num_cls)
        seed_points_0 (torch.Tensor): The seed points (N,3)
        k_poses (int): The number of poses to render
        """
        # Move camera transformations to the GPU
        OPENGL_TO_OPENCV = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        camera = model.cameras
        optimized_camera_to_world = camera.camera_to_worlds.half().to("cuda")  # shape (M, 4, 4)
        opengl_to_opencv = torch.tensor(OPENGL_TO_OPENCV, device="cuda", dtype=optimized_camera_to_world.dtype)  # shape (4, 4)
        optimized_camera_to_world = torch.matmul(optimized_camera_to_world, opengl_to_opencv)  # shape (M, 4, 4)

        # Move intrinsics to the GPU
        K = camera.get_intrinsics_matrices().to("cuda")  # shape (M, 3, 3)
        W, H = int(camera.width[0].item()), int(camera.height[0].item())

        # Convert class-agnostic mask to a boolean tensor and move to GPU
        # boolean_masks = torch.from_numpy(class_agnostic_3d_mask).bool().to('cuda')  # shape (N, 166)

        ################for inference################
        N = class_agnostic_3d_mask.shape[0]
        cls_num = class_agnostic_3d_mask.shape[1]
        # pred_classes = torch.full(
        #     (N,), -1, dtype=torch.int64, device="cuda"
        # ) # shape (N,)
        #############################################
        # Loop through each mask
        pred_classes = np.full(cls_num, 0)#-1)

        for i in range(cls_num):
            boolean_mask = class_agnostic_3d_mask[:, i]
            best_poses_indices, bounding_boxes = object_optimal_k_camera_poses_bounding_box_clear(
                seed_points_0=seed_points_0,
                optimized_camera_to_world=optimized_camera_to_world,
                K=K,
                W=W,
                H=H,
                boolean_mask=boolean_mask,  # select i_th mask
                k_poses=self.top_k,
            )
            if best_poses_indices.shape[0] == 0:
                print(f"Skipping inference for mask {i} due to no valid camera poses, assign", )
                continue
            ########################inference######################
            outputs = []
            for index, pose_index in enumerate(best_poses_indices):
                pose_index = pose_index.item()
                
                single_camera = camera[pose_index : pose_index + 1]
                assert single_camera.shape[0] == 1, "Only one camera at a time"
                # set instance
                model.cls_index = i
                #img = model.get_outputs(single_camera)["rgb_mask"]#["rgb"]#
                with Image.open(model.image_file_names[pose_index]) as img:
                    img = transforms.ToTensor()(img).cuda()#(C,H,W)
                nvs_mask_img = model.get_outputs(single_camera)["rgb_mask"]#["rgb_mask"]  # (H,W,3)
                #nvs_img = model.get_outputs(single_camera)["rgb"]  # (H,W,3)
                min_u, min_v, max_u, max_v = bounding_boxes[index]
                #print(min_u, min_v, max_u, max_v)
                if any(map(lambda x: torch.isinf(x) or x < 0, [min_u, min_v, max_u, max_v])):
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
                    outputs.append(cropped_image)
                    cropped_nvs_mask_image = nvs_mask_img[min_v:max_v, min_u:max_u].permute(2, 0, 1)
                    outputs.append(cropped_nvs_mask_image)
                    # cropped_nvs_image = nvs_img[min_v:max_v, min_u:max_u].permute(2, 0, 1)
                    # outputs.append(cropped_nvs_image)
                    
                                ###################save rendered image#################
                # from  nvsmask3d.utils.utils import save_img
                # save_img(cropped_image.permute(1,2,0), f"tests/output_{i}{pose_index}.png")
                ######################################################
                

                
                
                
                #append nvs mask3d outputs
                
                
            #outputs = torch.stack(outputs)
            # (B,H,W,3)->(B,C,H,W)
            #  outputs = outputs.permute(0, 3, 1, 2)

            # output is a list, which has tensors of the shape (C,H,W)
            mask_features = model.image_encoder.encode_batch_list_image(outputs)  # (B,512)
            similarity_scores = torch.mm(
                mask_features, model.image_encoder.pos_embeds.T
            )  # (B,200)
            # Aggregate scores across all images
            aggregated_scores = similarity_scores.sum(dim=0)  # Shape: (200,)
            #normalized_scores = aggregated_scores / aggregated_scores.sum()
            # Find the text index with the maximum aggregated score
            max_ind = torch.argmax(aggregated_scores).item()  #
            

            #max_ind_remapped = model.image_encoder.label_mapper[max_ind], replica no need remapping
            pred_classes[i] = max_ind  #max_ind_remapped
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
