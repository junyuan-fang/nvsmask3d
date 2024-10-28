"""Slightly modified version of scannet dataparser adapted from: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannet_dataparser.py"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type
import os
import cv2
import numpy as np
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox


@dataclass
class ReplicaNvsmask3DParserConfig(DataParserConfig):
    """Replica dataset config.

    .. code-block:: text

        replica/
        # ├── ground_truth/
        #     ├── scene{num}.txt/
        |── replica_masks/
            ├── scene{num}.pt/
        |── scene{num}/
            ├── color/
            ├── depth/
            ├── intrinsics.txt
            ├── poses/
            |── scene{num}_mesh.ply/
    """

    _target: Type = field(default_factory=lambda: ReplicaNvsmask3D)
    """target class to instantiate"""
    data: Path = Path("data/replica/")
    """Path to Replica folder with densely extracted scenes."""
    sequence: Literal[
        "office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"
    ] = "room0"
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 1
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1/6553.5
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the .ply"""
    point_cloud_color: bool = True
    # """read point cloud colors from .ply files or not """
    # ply_file_path: Path = data / (data.name + ".ply")
    """path to the .ply file containing the 3D points"""
    load_every: int = 1
    """load every n'th frame from the dense trajectory"""
    load_masks: bool = True
    mask_path: Path = Path(
        "/home/wangs9/junyuan/openmask3d/output/2024-08-08-13-27-09-scene0000_00_/scene0011_00_vh_clean_2_masks.pt"
    )


@dataclass
class ReplicaNvsmask3D(DataParser):
    """ScanNet DatasetParser"""

    config: ReplicaNvsmask3DParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        self.input_folder = self.config.data / self.config.sequence
        image_dir = self.input_folder / "color"
        depth_dir = self.input_folder / "depth"
        pose_dir = self.input_folder / "poses"
        self.ply_file_path = self.input_folder / (self.config.sequence + "_mesh.ply")
        # self.config.mask_path =

        img_dir_sorted = list(
            sorted(image_dir.iterdir(), key=lambda x: int(x.name.split(".")[0]))
        )
        depth_dir_sorted = list(
            sorted(depth_dir.iterdir(), key=lambda x: int(x.name.split(".")[0]))
        )
        # pose_dir_sorted = list(sorted(pose_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))
        pose_dir_sorted = list(
            sorted(
                (f for f in pose_dir.iterdir() if f.name.split(".")[0].isdigit()),
                key=lambda x: int(x.name.split(".")[0]),
            )
        )

        first_img = cv2.imread(str(img_dir_sorted[0].absolute()))  # type: ignore
        h, w, _ = first_img.shape

        image_filenames, depth_filenames, intrinsics, poses = [], [], [], []

        K = np.loadtxt(self.input_folder / "intrinsics.txt")
        for img, depth, pose in zip(img_dir_sorted, depth_dir_sorted, pose_dir_sorted):
            pose = np.loadtxt(pose)
            pose = np.array(pose).reshape(4, 4)
            pose[:3, 1] *= -1
            pose[:3, 2] *= -1
            # pose[0:3, 1:3] *= -1
            # pose = pose[np.array([1, 0, 2, 3]), :]
            # pose[2, :] *= -1
            pose = torch.from_numpy(pose).float()
            # We cannot accept files directly, as some of the poses are invalid
            if np.isinf(pose).any():
                continue

            poses.append(pose)
            intrinsics.append(K)
            image_filenames.append(img)
            depth_filenames.append(depth)

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_fraction)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
            if self.config.load_every > 1:
                indices = indices[:: self.config.load_every]
        elif split in ["val", "test"]:
            indices = i_eval
        elif split == "all":
            indices = i_all
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.stack(poses).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method="up",
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = (
            [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        )
        intrinsics = intrinsics[indices.tolist()]
        poses = poses[indices.tolist()]

        # in x,y,z order
        # assumes that the scene is centered at the origin
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

        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=h,
            width=w,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
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
                self.config.data / "replica_masks" / (self.config.sequence + ".pt")
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
        masks = torch.load(mask_path, map_location="cuda")
        cls_num = masks[0].shape[1]
        out = {"points3D_mask": masks[0], "points3D_cls_num": cls_num}
        return out


ReplicaNvsmask3DParserSpecification = DataParserSpecification(
    config=ReplicaNvsmask3DParserConfig(load_3D_points=True),
    description="scannet dataparser",
)
