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
"""Data parser for ScanNet++ datasets."""

from __future__ import annotations
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type
import json

import numpy as np
import torch
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
)
from nerfstudio.data.scene_box import SceneBox


def camera_to_world_from_wc(world_to_camera):
    """
    Convert a world-to-camera matrix to camera-to-world transformation without matrix inversion.

    Args:
        world_to_camera (np.ndarray): A 4x4 world-to-camera matrix.

    Returns:
        camera_to_world (np.ndarray): The corresponding camera-to-world matrix.
    """
    # Extract the rotation part (R_wc) and translation part (T_wc)
    R_wc = world_to_camera[:3, :3]  # 3x3 rotation matrix
    T_wc = world_to_camera[:3, 3]  # 3x1 translation vector

    # Transpose the rotation matrix (this is equivalent to inverting it)
    R_cw = R_wc.T  # The transpose of the rotation matrix (equivalent to inverse for orthogonal matrices)

    # Compute the camera translation in world coordinates
    T_cw = -R_cw @ T_wc

    # Construct the camera-to-world transformation matrix
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = R_cw
    camera_to_world[:3, 3] = T_cw

    return camera_to_world


OPENGL_TO_OPENCV = np.float32(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
)


@dataclass
class ScanNetppDataParserConfig(DataParserConfig):
    """ScanNet++ dataset config.
    ScanNet++ dataset (https://kaldir.vc.in.tum.de/scannetpp/) is a real-world 3D indoor dataset for semantics understanding and novel view synthesis.
    This dataparser follow the file structure of the dataset.
    Expected structure of the directory:

    .. code-block:: text

        root/
        |── mask3d_processed/
            ├── {SCENE_ID}.pt/
            ...
        |── data/
            ├── SCENE_ID0
                ├── dslr
                    |── colmap
                        |── cameras.txt
                        |── images.txt
                    ├── undistorted_images
                    ├── nerfstudio/transforms.json
                ├── scans
                    ├── mesh_aligned_0.05.ply
            ├── SCENE_ID1/
        ...
    """

    _target: Type = field(default_factory=lambda: ScanNetpp)
    """target class to instantiate"""
    data: Path = Path("/data/scannetpp/ScannetPP/data")
    """Directory to the root of the data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    mode: Literal["dslr", "iphone", "dslr_colmap"] = "iphone"
    """Which camera to use"""
    scene_scale: float = 1.5
    """How much to scale the region of interest by. Default is 1.5 since the cameras are inside the rooms."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    images_dir: Path = Path("dslr/resized_images")
    """Relative path to the images directory (default: resized_images)"""
    masks_dir: Path = Path("dslr/resized_anon_masks")
    """Relative path to the masks directory (default: resized_anon_masks)"""
    transforms_path: Path = Path("dslr/nerfstudio/transforms.json")
    """Relative path to the transforms.json file"""
    load_3D_points: bool = True
    """Whether to load the 3D points from the .ply"""
    skip_every_for_val_split: int = 10
    """sub sampling validation images"""
    train_split_fraction: float = 1
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    point_cloud_color: bool = True
    # """read point cloud colors from .ply files or not """
    # ply_file_path: Path = data / (data.name + ".ply")
    """path to the .ply file containing the 3D points"""
    load_every: int = 5
    """load every n'th frame from the dense trajectory"""
    load_masks: bool = True
    # validation set of scannetpp
    sequence: Literal[
        "7b6477cb95",
        "c50d2d1d42",
        "cc5237fd77",
        "acd95847c5",
        "fb5a96b1a2",
        "a24f64f7fb",
        "1ada7a0617",
        "5eb31827b7",
        "3e8bba0176",
        "3f15a9266d",
        "21d970d8de",
        "5748ce6f01",
        "c4c04e6d6c",
        "7831862f02",
        "bde1e479ad",
        "38d58a7a31",
        "5ee7c22ba0",
        "f9f95681fd",
        "3864514494",
        "40aec5fffa",
        "13c3e046d7",
        "e398684d27",
        "a8bf42d646",
        "45b0dac5e3",
        "31a2c91c43",
        "e7af285f7d",
        "286b55a2bf",
        "7bc286c1b6",
        "f3685d06a9",
        "b0a08200c9",
        "825d228aec",
        "a980334473",
        "f2dc06b1d2",
        "5942004064",
        "25f3b7a318",
        "bcd2436daf",
        "f3d64c30f8",
        "0d2ee665be",
        "3db0a1c8f3",
        "ac48a9b736",
        "c5439f4607",
        "578511c8a9",
        "d755b3d9d8",
        "99fa5c25e1",
        "09c1414f1b",
        "5f99900f09",
        "9071e139d9",
        "6115eddb86",
        "27dd4da69e",
        "c49a8c6cff",
    ] = "7b6477cb95"


transform = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)


@dataclass
class ScanNetpp(ColmapDataParser):
    """ScanNet++ DatasetParser"""

    config: ScanNetppDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        print("split is: ", split)
        if self.config.mode == "dslr_colmap":
            self.input_folder = (
                self.config.data / "data" / self.config.sequence / "dslr"
            )
        else:
            self.input_folder = (
                self.config.data / "data" / self.config.sequence / self.config.mode
            )
        assert (
            self.input_folder.exists()
        ), f"Data directory {self.input_folder} does not exist."
        if self.config.mode == "iphone":
            image_dir = self.input_folder / "rgb"
            # mask_dir = self.input_folder / "rgb_masks"
            depth_dir = self.input_folder / "depth"
            pose_path = self.input_folder / "pose_intrinsic_imu.json"
        elif self.config.mode == "dslr":
            image_dir = self.input_folder / "undistorted_images"
            # mask_dir = self.input_folder / self.config.masks_dir
            depth_dir = self.input_folder / "depth"
            pose_path = self.input_folder / "nerfstudio" / "transforms_undistorted.json"
        elif self.config.mode == "dslr_colmap":
            image_dir = self.input_folder / "undistorted_images"
            depth_dir = self.input_folder / "render_depth"
            # mask_dir = self.input_folder / self.config.masks_dir
            intrinsics_path = (
                self.input_folder / "nerfstudio" / "transforms_undistorted.json"
            )  # self.input_folder / "colmap" / "cameras.txt"
            pose_path = self.input_folder / "colmap" / "images.txt"
        else:
            KeyError(f"Unknown mode {self.config.mode}, we don't support it yet.")

        self.ply_file_path = (
            self.config.data
            / "data"
            / self.config.sequence
            / "scans"
            / "mesh_aligned_0.05.ply"
        )

        poses = []

        if self.config.mode == "iphone":
            with open(pose_path) as f:
                data = json.load(f)
            intrinsics = []
            image_filenames = list(
                sorted(
                    image_dir.iterdir(),
                    key=lambda x: int(x.name.split("_")[1].split(".")[0]),
                )
            )
            depth_filenames = list(
                sorted(
                    depth_dir.iterdir(),
                    key=lambda x: int(x.name.split("_")[1].split(".")[0]),
                )
            )
            # mask_filenames = list(
            #     sorted(mask_dir.iterdir(), key=lambda x: int(x.name.split(".")[0]))
            # )
            for _, frame_data in data.items():
                intrinsics.append(np.array(frame_data["intrinsic"]))
                pose = np.array(frame_data["aligned_pose"])
                pose[:3, 1] *= -1
                pose[:3, 2] *= -1
                poses.append(pose)  # maybeaready opengl

        if self.config.mode == "dslr":
            with open(pose_path) as f:
                data = json.load(f)
            image_filenames = []
            depth_filenames = []
            if split == "train":
                for frame in data["frames"]:
                    is_bad = frame["is_bad"]
                    if is_bad:
                        continue
                    img_path = image_dir / frame["file_path"]
                    image_filenames.append(img_path)
                    # mask_path = frame['mask_path']
                    pose = np.array(frame["transform_matrix"])
                    pose = camera_to_world_from_wc(pose)
                    pose = pose @ OPENGL_TO_OPENCV
                    poses.append(pose)
            elif split in ["val", "test"]:
                for frame in data["test_frames"]:
                    is_bad = frame["is_bad"]
                    if is_bad:
                        continue
                    img_path = image_dir / frame["file_path"]
                    image_filenames.append(img_path)
                    # mask_path = frame['mask_path']
                    pose = np.array(frame["transform_matrix"])
                    pose = camera_to_world_from_wc(pose)
                    pose = pose @ OPENGL_TO_OPENCV
                    poses.append(pose)

        if self.config.mode == "dslr_colmap":
            image_filenames = []
            depth_filenames = []
            self._read_images_txt(
                image_dir, depth_dir, pose_path, image_filenames, poses, depth_filenames
            )

        # assert len(depth_filenames) == 0 or (
        #     len(depth_filenames) == len(image_filenames)
        # ), """
        # Different number of image and depth filenames.
        # You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        # """
        if self.config.mode == "iphone":
            num_imgs = len(image_filenames)
            indices = list(range(num_imgs))
            assert self.config.skip_every_for_val_split >= 1
            eval_indices = indices[:: self.config.skip_every_for_val_split]
            i_eval = [i for i in indices if i in eval_indices]
            i_train = [i for i in indices if i not in eval_indices]

            if split == "train":
                indices = i_train
                if self.config.load_every > 1:
                    indices = indices[:: self.config.load_every]
            elif split in ["val", "test"]:
                indices = i_eval
            elif split == "all":
                indices = indices
            else:
                raise ValueError(f"Unknown dataparser split {split}")
            print(indices)
            intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))

        orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses, method=orientation_method, center_method=self.config.center_method
        )
        self.orient_transform = transform_matrix
        # if self.config.mode == "dslr_colmap" or self.config.mode == "dslr":
        #     self.orient_transform = torch.from_numpy(transform)

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor
        # if self.config.mode == "dslr":
        # poses[:, :3, 2] *= -1
        # poses[:, :3, 1] *= -1
        # 对平移向量的 Y 和 Z 轴进行反转

        # if self.config.mode == "dslr_colmap":
        # poses[:, :3, 2] *= -1
        # poses[:, :3, 1] *= -1
        # 对平移向量的 Y 和 Z 轴进行反转

        if self.config.mode == "iphone":
            # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
            image_filenames = [image_filenames[i] for i in indices]
            # mask_filenames = (
            #     [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
            # )
            depth_filenames = (
                [depth_filenames[i] for i in indices]
                if len(depth_filenames) > 0
                else []
            )
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            poses = poses[idx_tensor]
            intrinsics = intrinsics[idx_tensor]
            first_img = cv2.imread(str(image_filenames[0].absolute()))  # type: ignore
            h, w, _ = first_img.shape
        self.poses = poses
        self.camera_to_worlds = poses[:, :3, :4]

        if self.config.mode == "iphone":
            if split == "train":
                self._write_json(
                    image_filenames,
                    depth_filenames,
                    intrinsics[0][0, 0],  # fx
                    intrinsics[0][1, 1],  # fy
                    intrinsics[0][0, 2],  # cx
                    intrinsics[0][1, 2],  # cy
                    w,
                    h,
                    poses[:, :3, :4],
                    "transforms.json",
                )
            else:
                self._write_json(
                    image_filenames,
                    depth_filenames,
                    intrinsics[0][0, 0],  # fx
                    intrinsics[0][1, 1],  # fy
                    intrinsics[0][0, 2],  # cx
                    intrinsics[0][1, 2],  # cy
                    w,
                    h,
                    poses[:, :3, :4],
                    "transforms_test.json",
                )

        metadata = {}

        # in x,y,z order
        # assumes that the scene is centered at the origin
        if not self.config.auto_scale_poses:
            # Set aabb_scale to scene_scale * the max of the absolute values of the poses
            aabb_scale = self.config.scene_scale * float(
                torch.max(torch.abs(poses[:, :3, 3]))
            )
        else:
            aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        camera_type = CameraType.PERSPECTIVE
        if self.config.mode == "iphone":
            cameras = Cameras(
                fx=intrinsics[:, 0, 0],
                fy=intrinsics[:, 1, 1],
                cx=intrinsics[:, 0, 2],
                cy=intrinsics[:, 1, 2],
                height=h,
                width=w,
                camera_to_worlds=poses[:, :3, :4],
                camera_type=camera_type,
            )
        elif self.config.mode == "dslr":
            cameras = Cameras(
                fx=data["fl_x"],
                fy=data["fl_y"],
                cx=data["cx"],
                cy=data["cy"],
                height=data["h"],
                width=data["w"],
                camera_to_worlds=poses[:, :3, :4],
                camera_type=camera_type,
            )

        elif self.config.mode == "dslr_colmap":
            # width, height, fx, fy, cx, cy = self._read_cameras(intrinsics_path)
            with open(intrinsics_path, "r") as f:
                data = json.load(f)
                fx = data["fl_x"]
                fy = data["fl_y"]
                cx = data["cx"]
                cy = data["cy"]
                height = data["h"]
                width = data["w"]

            cameras = Cameras(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                height=height,
                width=width,
                camera_to_worlds=poses[:, :3, :4],
                camera_type=camera_type,
            )

        metadata = {
            "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
            "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
        }

        if self.config.load_3D_points:
            point_color = self.config.point_cloud_color
            ply_file_path = self.ply_file_path
            point_cloud_data = self._load_3D_points(
                ply_file_path, transform_matrix, scale_factor, point_color
            )
            if point_cloud_data is not None:
                metadata.update(point_cloud_data)

        if self.config.load_masks:
            mask_path = (
                self.config.data / "mask3d_processed" / (self.config.sequence + ".pt")
            )
            # #load gt masks for amblation study
            # current_dir = os.getcwd()
            # mask_path = (
            #     Path(current_dir) / "nvsmask3d/data/Replica/replica_ground_truth_masks" / (self.config.sequence + ".pt")
            # )
            mask_data = self._load_mask(mask_path)
            if mask_data is not None:
                metadata.update(mask_data)
        ### test######################################################
        # from nvsmask3d.utils.camera_utils import project_pix
        # p = metadata["points3D_xyz"]#torch.Size([237360, 3])
        # colors = metadata["points3D_rgb"] / 255 #torch.Size([237360, 3])
        # fx=intrinsics[0, 0, 0].to(torch.device('cuda'))
        # fy=intrinsics[0, 1, 1].to(torch.device('cuda'))
        # cx=intrinsics[0, 0, 2].to(torch.device('cuda'))
        # cy=intrinsics[0, 1, 2].to(torch.device('cuda'))
        # c2w = poses[0, :3, :4].to(torch.device('cuda'))
        # device = torch.device('cuda')

        # colors = colors.to(device)
        # uv_coords = project_pix(p, fx, fy, cx, cy, c2w, device, return_z_depths=True) # returns uv -> (pix_x,pix_y,z_depth)
        # sparse_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
        # valid_points = (uv_coords[..., 0] >= 0) & (uv_coords[..., 0] < w) & (uv_coords[..., 1] >= 0) & (uv_coords[..., 1] < h ) &  (uv_coords[..., 2] > 0)
        # sparse_map[[uv_coords[valid_points,1].long(), uv_coords[valid_points,0].long()]] = colors[valid_points][None,:].float()

        # print("Projected UV coordinates's shape:", uv_coords.shape)#Projected UV coordinates's shape: torch.Size([237360, 3])
        # print(sparse_map.min(), sparse_map.max())
        # from  nvsmask3d.utils.utils import save_img, image_path_to_tensor
        # gt_img = image_path_to_tensor(image_filenames[0])
        # save_img(gt_img, "/home/wangs9/junyuan/nerfstudio-nvsmask3d/nvsmask3d/data/scene0000_00/gt_img.png")
        # save_img(sparse_map, "/home/wangs9/junyuan/nerfstudio-nvsmask3d/nvsmask3d/data/scene0000_00/rendered.png",)
        # quit()
        ###################################################################
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )
        return dataparser_outputs

    def _load_3D_points(
        self,
        ply_file_path: Path,
        transform_matrix: torch.Tensor,
        scale_factor: float,
        points_color: bool,
        sample_rate=1,
    ) -> dict:
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.
            points_color: Whether to load the point cloud colors or not

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
            or
            A dictionary of points: points3D_xyz if points_color is False
        """
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        pcd = o3d.io.read_point_cloud(str(ply_file_path))

        # if no points found don't read in an initial point cloud
        if len(pcd.points) == 0:
            return None

        num_points = len(pcd.points)
        sampled_indices = np.random.choice(
            num_points, int(num_points * sample_rate), replace=False
        )

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        # points3D = torch.from_numpy(np.asarray(pcd.points)[sampled_indices].astype(np.float32))

        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor
        out = {
            "points3D_xyz": points3D,
            "points3D_num": points3D.shape[0],
        }

        if points_color:
            # points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))
            points3D_rgb = torch.from_numpy(
                (np.asarray(pcd.colors)[sampled_indices] * 255).astype(np.uint8)
            )
            out["points3D_rgb"] = points3D_rgb

        return out

    def _load_mask(self, mask_path: Path):
        # mask[0] torch.Size([points_num, class_num]) mask
        # mask[1] torch.Size([36]) confidence of the mask
        masks = torch.load(mask_path)
        cls_num = masks[0].shape[1]
        out = {"points3D_mask": masks[0], "points3D_cls_num": cls_num}
        return out

    def _write_json(
        self,
        image_filenames,
        depth_filenames,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        camera_to_worlds,
        name,
    ):
        frames = []
        base_dir = Path(image_filenames[0]).parent.parent

        # 确保数值是Python原生类型
        fx = float(fx.cpu().item() if torch.is_tensor(fx) else fx)
        fy = float(fy.cpu().item() if torch.is_tensor(fy) else fy)
        cx = float(cx.cpu().item() if torch.is_tensor(cx) else cx)
        cy = float(cy.cpu().item() if torch.is_tensor(cy) else cy)
        width = int(width.cpu().item() if torch.is_tensor(width) else width)
        height = int(height.cpu().item() if torch.is_tensor(height) else height)

        for img, depth, c2w in zip(image_filenames, depth_filenames, camera_to_worlds):
            img = Path(img)
            file_path = img.relative_to(base_dir)
            depth_path = depth.relative_to(base_dir)

            # 确保c2w是Python list类型
            if torch.is_tensor(c2w):
                c2w = c2w.cpu().numpy().tolist()

            frame = {
                "file_path": file_path.as_posix(),
                "depth_file_path": depth_path.as_posix(),
                "transform_matrix": c2w,
            }
            frames.append(frame)

        out = {
            "fl_x": fx,
            "fl_y": fy,
            "k1": 0,
            "k2": 0,
            "p1": 0,
            "p2": 0,
            "cx": cx,
            "cy": cy,
            "w": width,
            "h": height,
            "frames": frames,
        }

        with open(base_dir / name, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

    def _read_cameras(self, file_path):
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                # params = list(map(float, parts[4:]))
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])
        return width, height, fx, fy, cx, cy

    def _read_images_txt(
        self, image_dir, depth_dir, pose_path, image_filenames, poses, depth_filenames
    ):
        with open(pose_path, "r") as file:
            lines = file.readlines()
            for i in range(4, len(lines), 2):  # 从第4行开始，每两行代表一张图像的数据
                line = lines[i].strip().split()

                image_id = int(line[0])
                qw, qx, qy, qz = map(float, line[1:5])  # 四元数
                tx, ty, tz = map(float, line[5:8])  # 平移向量
                camera_id = int(line[8])
                image_name = line[9]

                # 将四元数转换为旋转矩阵
                R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
                T = np.array([tx, ty, tz])

                # 构建位姿矩阵 P
                P = np.eye(4)
                P[:3, :3] = R
                P[:3, 3] = T

                image_filenames.append(image_dir / image_name)
                depth_filenames.append(depth_dir / (image_name.split(".")[0] + ".png"))
                P = camera_to_world_from_wc(P) @ OPENGL_TO_OPENCV  # important
                poses.append(P)
        return

    def _quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        # 使用四元数构造旋转矩阵
        R = np.array(
            [
                [
                    1 - 2 * qy**2 - 2 * qz**2,
                    2 * qx * qy - 2 * qz * qw,
                    2 * qx * qz + 2 * qy * qw,
                ],
                [
                    2 * qx * qy + 2 * qz * qw,
                    1 - 2 * qx**2 - 2 * qz**2,
                    2 * qy * qz - 2 * qx * qw,
                ],
                [
                    2 * qx * qz - 2 * qy * qw,
                    2 * qy * qz + 2 * qx * qw,
                    1 - 2 * qx**2 - 2 * qy**2,
                ],
            ]
        )
        return R


ScanNetppNvsmask3DParserSpecification = DataParserSpecification(
    config=ScanNetppDataParserConfig(load_3D_points=True),
    description="scannet dataparser",
)
