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
)
from nerfstudio.models.splatfacto import SplatfactoModel
from tqdm import tqdm
from typing_extensions import Annotated
import numpy as np
from nvsmask3d.eval.scannet200.eval_semantic_instance import evaluate


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
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("")

    def main(self) -> None:
        gt_dir = Path(
            "/home/wangs9/junyuan/nerfstudio-nvsmask3d/nvsmask3d/data/scene0011_00/instance_gt/validation"
        )
        config, pipeline, checkpoint_path, _ = eval_setup(
            self.load_config,
            test_mode="all",
        )
        global model
        model = pipeline.model
        preds = {}
        scene_names = ["scene0011_00"]  # hard coded for now
        with torch.no_grad():
            # for each scene
            for i, scene_name in tqdm(
                enumerate(scene_names), desc="Evaluating", total=len(scene_names)
            ):
                scene_id = scene_name[5:]
                # optimal_cameras = object_optimal_k_camera_poses(seed_points_0 = model.seed_points[i].cuda(),
                #                               class_agnostic_3d_mask=model.points3D_mask[:,model.cls_index],
                #                               camera=model.cameras,
                #                               k_poses = 2)
                # for i in range(optimal_cameras.camera_to_worlds.shape[0]):
                #     single_camera = optimal_cameras[i:i+1]
                #     assert single_camera.shape[0] == 1, "Only one camera at a time"
                #     img = model.get_outputs(single_camera)["rgb_mask"]#(H,W,3)
                #     ###################save rendered image#################
                #     from  nvsmask3d.utils.utils import save_img
                #     save_img(img, f"tests/output_{i}.png")
                #     ##########################################
                class_agnostic_3d_mask = (
                    torch.from_numpy(model.points3D_mask).bool().to("cuda")
                )  # shape (N, 166)
                seed_points_0 = model.seed_points[0].half().cuda()  # shape (N, 3)
                # optimal_camera_poses_of_scene = optimal_k_camera_poses_of_scene(seed_points_0=seed_points_0,
                #                                                                 class_agnostic_3d_mask=class_agnostic_3d_mask,
                #                                                                 camera=model.cameras)

                pred_classes = self.pred_classes(
                    class_agnostic_3d_mask=model.points3D_mask,
                    seed_points_0=seed_points_0,
                    k_poses=2,
                )
                pred_scores = np.ones(pred_classes.shape)
                pred_masks = model.points3D_mask
                pred_classes = pred_classes.cpu().numpy()
                preds["scene0011_00"] = {
                    "pred_masks": pred_masks,
                    "pred_scores": pred_scores,
                    "pred_classes": pred_classes,
                }
                inst_AP = evaluate(
                    preds, gt_dir, output_file="output.txt", dataset="scannet200"
                )

    def pred_classes(self, class_agnostic_3d_mask, seed_points_0, k_poses):
        """
        Args:

        """
        # Move camera transformations to the GPU
        OPENGL_TO_OPENCV = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        camera = model.cameras
        optimized_camera_to_world = camera.camera_to_worlds.half().to(
            "cuda"
        )  # shape (M, 4, 4)
        opengl_to_opencv = torch.tensor(
            OPENGL_TO_OPENCV, device="cuda", dtype=optimized_camera_to_world.dtype
        )  # shape (4, 4)
        optimized_camera_to_world = torch.matmul(
            optimized_camera_to_world, opengl_to_opencv
        )  # shape (M, 4, 4)

        # Move intrinsics to the GPU
        K = camera.get_intrinsics_matrices().to("cuda")  # shape (M, 3, 3)
        W, H = int(camera.width[0].item()), int(camera.height[0].item())

        # Convert class-agnostic mask to a boolean tensor and move to GPU
        # boolean_masks = torch.from_numpy(class_agnostic_3d_mask).bool().to('cuda')  # shape (N, 166)

        ################for inference################
        pred_classes = torch.zeros(
            class_agnostic_3d_mask.shape[1], dtype=torch.int64, device="cuda"
        )  # shape (N,)
        #############################################
        # Loop through each mask
        for i in range(class_agnostic_3d_mask.shape[1]):
            best_poses, boolean_mask = object_optimal_k_camera_poses_clear(
                seed_points_0=seed_points_0,
                optimized_camera_to_world=optimized_camera_to_world,
                K=K,
                W=W,
                H=H,
                class_agnostic_3d_mask=class_agnostic_3d_mask[:, i],  # select i_th mask
                camera=model.cameras,
            )

            ########################inference######################
            outputs = []
            for j in range(k_poses):
                single_camera = best_poses[j : j + 1]
                assert single_camera.shape[0] == 1, "Only one camera at a time"
                # set instance
                model.cls_index = i
                img = model.get_outputs(single_camera)["rgb_mask"]
                ###################save rendered image#################
                # from  nvsmask3d.utils.utils import save_img
                # save_img(img, f"tests/output_{i}{j}.png")
                ######################################################
                outputs.append(img)
            outputs = torch.stack(outputs)
            # (B,H,W,3)->(B,C,H,W)
            outputs = outputs.permute(0, 3, 1, 2)
            mask_features = model.image_encoder.encode_image(outputs)  # (B,512)
            similarity_scores = torch.mm(
                mask_features, model.image_encoder.pos_embeds.T
            )  # (B,200)
            # Aggregate scores across all images
            aggregated_scores = similarity_scores.sum(dim=0)  # Shape: (200,)
            # Find the text index with the maximum aggregated score
            max_ind = torch.argmax(aggregated_scores).item()  #

            max_ind_remapped = model.image_encoder.label_mapper[max_ind]
            pred_classes[i] = max_ind_remapped
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
