"""Utils for projection and camera coords with different conventions"""

import math
from typing import List, Optional, Tuple
from nerfstudio.cameras.cameras import Cameras

# from nerfstudio.models.splatfacto import get_viewmat
import numpy as np
import torch
from torch import Tensor

################debug################
# image_file_ = [image_file_names[int(i)] for i in best_poses_indices]
# print(image_file_)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# index = best_poses_indices[0]
# K = K[index]
# # take camera parameters
# fx = K[0, 0].to(device)
# fy = K[1, 1].to(device)
# cx = K[0, 2].to(device)
# cy = K[1, 2].to(device)
# c2w = optimized_camera_to_world[index].to(device)

# # 2D plane
# uv_coords = project_pix(seed_points_0[boolean_mask], fx, fy, cx, cy, c2w, device, return_z_depths=True)  # returns uv -> (pix_x, pix_y, z_depth)
# valid_points = (uv_coords[..., 0] >= 0) & (uv_coords[..., 0] < W) & (uv_coords[..., 1] >= 0) & (uv_coords[..., 1] < H) & (uv_coords[..., 2] > 0)
# sparse_map = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
# sparse_map[uv_coords[valid_points, 1].long(), uv_coords[valid_points, 0].long()] = 1
# # Apply mask to valid points
# from  nvsmask3d.utils.utils import save_img
# print(sparse_map.shape)
# save_img(sparse_map, "sparse_map.png")

################debug################

# opengl to opencv transformation matrix
OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# #give k optimal camera poses from pose data
# @torch.no_grad()
# def object_optimal_k_camera_poses(seed_points_0, class_agnostic_3d_mask, camera: Cameras,k_poses = 2):#,image_file_names = None):# after training ]
#     optimized_camera_to_world = camera.camera_to_worlds.cuda()
#     # multiply by opengl to opencv transformation matrix
#     optimized_camera_to_world = torch.matmul(
#         optimized_camera_to_world,
#         torch.tensor(OPENGL_TO_OPENCV, device=optimized_camera_to_world.device, dtype=optimized_camera_to_world.dtype)
#         )

#     K = camera.get_intrinsics_matrices().cuda()
#     W, H = int(camera.width[0].item()), int(camera.height[0].item())

#     visibility_scores = torch.tensor([])
#     boolean_mask = torch.from_numpy(class_agnostic_3d_mask).bool().cuda()
#     # calculate visibility score for each pose
#     for i, c2w in enumerate(optimized_camera_to_world):
#         score = compute_visibility_score(seed_points_0[boolean_mask], c2w, K[i], W, H)
#         visibility_scores = torch.cat((visibility_scores, torch.tensor([score])), dim=0)

#     # Step 4: Select top k scored poses
#     _, best_poses_indices = torch.topk(visibility_scores, k_poses)
#     best_poses = camera[best_poses_indices]


#     return best_poses#Cameras torch.Size([2])
@torch.no_grad()
def optimal_k_camera_poses_of_scene(
    seed_points_0, class_agnostic_3d_mask, camera, k_poses=2
):
    """
    Selects the top k optimal camera poses for each mask based on the visibility score of the 3D mask.
    Args:
        seed_points_0: torch.Tensor, size is (N, 3), point cloud
        class_agnostic_3d_mask: torch.Tensor, size is (N, 166), for 3D masks
        camera: Cameras, from this class can get camera poses, which size is (M, 4, 4)
        k_poses: int, for top k poses
    Returns:
        best_poses: torch.Tensor, size is (166,), top k poses (index) for each mask
    """
    # Move camera transformations to the GPU
    optimized_camera_to_world = camera.camera_to_worlds.to("cuda")  # shape (M, 4, 4)
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
    boolean_masks = (
        torch.from_numpy(class_agnostic_3d_mask).bool().to("cuda")
    )  # shape (N, 166)

    # Prepare a list to store best poses for each mask
    best_poses_per_mask = []

    # Loop through each mask
    for i in range(boolean_masks.shape[1]):
        # Get the mask for the current iteration
        boolean_mask = boolean_masks[:, i]  # shape (N,)
        masked_seed_points = seed_points_0[
            boolean_mask
        ]  # shape (P, 3) where P is number of valid points in the mask

        # Precompute necessary values
        visibility_scores = torch.zeros(
            len(optimized_camera_to_world), device="cuda"
        )  # shape (M,)

        # Vectorized computation for all camera poses
        points_cam = masked_seed_points.unsqueeze(0) - optimized_camera_to_world[
            :, :3, 3
        ].unsqueeze(
            1
        )  # shape (M, P, 3)
        points_cam = torch.bmm(
            points_cam, optimized_camera_to_world[:, :3, :3]
        )  # shape (M, P, 3)

        # Project to 2D image plane using vectorized operations
        u = points_cam[:, :, 0] * K[:, 0, 0].unsqueeze(1) / points_cam[:, :, 2] + K[
            :, 0, 2
        ].unsqueeze(1)
        v = points_cam[:, :, 1] * K[:, 1, 1].unsqueeze(1) / points_cam[:, :, 2] + K[
            :, 1, 2
        ].unsqueeze(1)

        # Check valid points within image boundaries and in front of the camera
        valid_points = (
            (u >= 0) & (u < W) & (v >= 0) & (v < H) & (points_cam[:, :, 2] > 0)
        )

        # Compute visibility scores for all poses
        visibility_scores = valid_points.float().mean(dim=1)

        # Select top k scored poses for current mask
        _, best_poses_indices = torch.topk(
            visibility_scores, k_poses
        )  # type of best_poses_indices is torch.Tensor

        # Ensure indices are on the CPU
        best_poses_indices = best_poses_indices.cpu()

        # Add best poses indices for this mask to the result list
        best_poses_per_mask.append(best_poses_indices)

    return best_poses_per_mask


@torch.no_grad()
def object_optimal_k_camera_poses(
    seed_points_0, boolean_mask, camera, k_poses=2
):  # tested with no problem, faster than the previous one
    """
    Selects the top k optimal camera poses based on the visibility score of the 3D mask.
    args:
        seed_points_0: torch.Tensor, size is (N, 3), point cloud
        boolean_mask: torch.Tenspr, size is (N,), for 3D mask #np.array, size is (N,), for 3D mask
        camera: Cameras, for camera poses, size is (M, 3)
        k_poses: int, for top k poses
    """
    # calculate time
    # start = time.time()

    # Move camera transformations to the GPU
    optimized_camera_to_world = camera.camera_to_worlds.to(
        "cuda"
    )  # torch.Size([M, 4, 4])
    opengl_to_opencv = torch.tensor(
        OPENGL_TO_OPENCV, device="cuda", dtype=optimized_camera_to_world.dtype
    )  # shape (4, 4)
    optimized_camera_to_world = torch.matmul(
        optimized_camera_to_world, opengl_to_opencv
    )  # shape (M, 4, 4)

    # Move intrinsics to the GPU
    K = camera.get_intrinsics_matrices().to("cuda")
    W, H = int(camera.width[0].item()), int(camera.height[0].item())

    # Prepare seed points and boolean mask
    # boolean_mask = (
    #     torch.from_numpy(class_agnostic_3d_mask).bool().to("cuda")
    # )  # shape (N,)
    masked_seed_points = seed_points_0[boolean_mask]  # shape (N, 3)

    # Precompute necessary values
    visibility_scores = torch.zeros(
        len(optimized_camera_to_world), device="cuda"
    )  # shape (N,)

    # Vectorized computation for all camera poses
    points_cam = masked_seed_points.unsqueeze(0) - optimized_camera_to_world[
        :, :3, 3
    ].unsqueeze(
        1
    )  # boardcast from (1,N,3) (M,1,3)to (M, N, 3)
    points_cam = torch.bmm(
        points_cam, optimized_camera_to_world[:, :3, :3]
    )  # matrix multiplication from (M, N, 3) (M, 3, 3) to (M, N, 3)
    # Project to 2D image plane using vectorized operations
    u = points_cam[:, :, 0] * K[:, 0, 0].unsqueeze(1) / points_cam[:, :, 2] + K[
        :, 0, 2
    ].unsqueeze(1)
    v = points_cam[:, :, 1] * K[:, 1, 1].unsqueeze(1) / points_cam[:, :, 2] + K[
        :, 1, 2
    ].unsqueeze(1)

    # Check valid points within image boundaries and in front of the camera
    valid_points = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (points_cam[:, :, 2] > 0)

    # Compute visibility scores for all poses
    visibility_scores = valid_points.float().mean(dim=1)

    # Select top k scored poses
    _, best_poses_indices = torch.topk(visibility_scores, k_poses)

    # Ensure indices are on the CPU
    best_poses_indices = best_poses_indices.cpu()

    best_poses = camera[best_poses_indices]
    # print time used
    # print("Time used: ", time.time() - start)

    return best_poses  # Cameras torch.Size([k_poses])


@torch.no_grad()
def object_optimal_k_camera_poses_clear(
    seed_points_0,
    optimized_camera_to_world,
    K,
    W,
    H,
    boolean_mask,
    camera,
    k_poses=2,
):  # tested with no problem

    # Prepare seed points and boolean mask
    # boolean_mask = (
    #     torch.from_numpy(class_agnostic_3d_mask).bool().to("cuda")
    # )  # shape (N,)
    masked_seed_points = seed_points_0[boolean_mask]  # shape (N, 3)

    # Precompute necessary values
    visibility_scores = torch.zeros(
        len(optimized_camera_to_world), device="cuda"
    )  # shape (N,)

    # Vectorized computation for all camera poses
    points_cam = masked_seed_points.unsqueeze(0) - optimized_camera_to_world[
        :, :3, 3
    ].unsqueeze(
        1
    )  # boardcast from (1,N,3) (M,1,3)to (M, N, 3)
    points_cam = torch.bmm(
        points_cam, optimized_camera_to_world[:, :3, :3]
    )  # matrix multiplication from (M, N, 3) (M, 3, 3) to (M, N, 3)
    # Project to 2D image plane using vectorized operations
    u = points_cam[:, :, 0] * K[:, 0, 0].unsqueeze(1) / points_cam[:, :, 2] + K[
        :, 0, 2
    ].unsqueeze(1)
    v = points_cam[:, :, 1] * K[:, 1, 1].unsqueeze(1) / points_cam[:, :, 2] + K[
        :, 1, 2
    ].unsqueeze(1)

    # Check valid points within image boundaries and in front of the camera
    valid_points = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (points_cam[:, :, 2] > 0)

    # Compute visibility scores for all poses
    visibility_scores = valid_points.float().mean(dim=1)

    # Select top k scored poses
    _, best_poses_indices = torch.topk(visibility_scores, k_poses)

    # Ensure indices are on the CPU
    best_poses_indices = best_poses_indices.cpu()

    best_poses = camera[best_poses_indices]
    # print time used
    # print("Time used: ", time.time() - start)

    return best_poses, boolean_mask  # Cameras torch.Size([k_poses])


def compute_visibility_score(p, camera_pose, K, W, H):
    """
    compute 3D mask visibility score

    :param p: torch.Tensor, size is (N, 3), for points(masked)
    :param camera_pose: torch.Tensor, size is (3, 4), for c2w poses
    :param K: torch.Tensor, size is (3, 3), for intrinsics
    :param W: int, imgage width
    :param H: int, image height
    :return: torch.Tensor, visibility score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # take camera parameters
    fx = K[0, 0].to(device)
    fy = K[1, 1].to(device)
    cx = K[0, 2].to(device)
    cy = K[1, 2].to(device)

    c2w = camera_pose.to(device)
    # 2D plane
    uv_coords = project_pix(
        p, fx, fy, cx, cy, c2w, device, return_z_depths=True
    )  # returns uv -> (pix_x, pix_y, z_depth)
    valid_points = (
        (uv_coords[..., 0] >= 0)
        & (uv_coords[..., 0] < W)
        & (uv_coords[..., 1] >= 0)
        & (uv_coords[..., 1] < H)
        & (uv_coords[..., 2] > 0)
    )

    visibility_score = valid_points.float().mean().item()
    return visibility_score


def project_pix(
    p: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    c2w: Tensor,
    device: torch.device,
    return_z_depths: bool = False,
) -> Tensor:
    """Projects a world 3D point to uv coordinates using intrinsics/extrinsics

    Returns:
        uv coords
    """
    if c2w is None:
        c2w = torch.eye((p.shape[0], 4, 4), device=device)  # type: ignore
    if c2w.device != device:
        c2w = c2w.to(device)

    points_cam = (p.to(device) - c2w[..., :3, 3]) @ c2w[..., :3, :3]
    # print(points_cam)
    u = points_cam[:, 0] * fx / points_cam[:, 2] + cx  # x
    v = points_cam[:, 1] * fy / points_cam[:, 2] + cy  # y
    if return_z_depths:
        return torch.stack([u, v, points_cam[:, 2]], dim=-1)
    return torch.stack([u, v], dim=-1)


def c2w_to_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """
    Converts a 3x4 camera-to-world matrix to a 4x4 world-to-camera matrix.

    Args:
        c2w: Tensor of shape (3, 4)

    Returns:
        w2c: Tensor of shape (4, 4)
    """
    # Convert 3x4 to 4x4 matrix
    c2w_hom = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    c2w_hom[:3, :4] = c2w

    # Compute the inverse to get the world-to-camera matrix
    w2c = torch.inverse(c2w_hom)

    return w2c


# ndc space is x to the right y up. uv space is x to the right, y down.
def pix2ndc_x(x, W):
    x = x.float()
    return (2 * x) / W - 1


def pix2ndc_y(y, H):
    y = y.float()
    return 1 - (2 * y) / H


# ndc is y up and x right. uv is y down and x right
def ndc2pix_x(x, W):
    return (x + 1) * 0.5 * W


def ndc2pix_y(y, H):
    return (1 - y) * 0.5 * H


def euclidean_to_z_depth(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    device: torch.device,
) -> Tensor:
    """Convert euclidean depths to z_depths given camera intrinsics"""
    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
    image_coords = get_camera_coords(img_size=img_size)
    image_coords = image_coords.to(device)

    z_depth = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    z_depth[:, 0] = (image_coords[:, 0] - cx) / fx  # x
    z_depth[:, 1] = (image_coords[:, 1] - cy) / fy  # y
    z_depth[:, 2] = 1  # z

    z_depth = z_depth / torch.norm(z_depth, dim=-1, keepdim=True)
    z_depth = (z_depth * depths)[:, 2]  # pick only z component

    z_depth = z_depth[..., None]
    z_depth = z_depth.view(img_size[1], img_size[0], 1)

    return z_depth


def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> Tensor:
    """Generates camera pixel coordinates [W,H]

    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """

    # img size is (w,h)
    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()

    return image_coords


def get_means3d_backproj(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    c2w: Tensor,
    device: torch.device,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, List]:
    """Backprojection using camera intrinsics and extrinsics

    image_coords -> (x,y,depth) -> (X, Y, depth)

    Returns:
        Tuple of (means: Tensor, image_coords: Tensor)
    """

    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w.device != device:
        c2w = c2w.to(device)

    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)  # note image_coords is (H,W)

    # TODO: account for skew / radial distortion
    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z

    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        means3d = means3d[mask]
        image_coords = image_coords[mask]

    if c2w is None:
        c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

    # to world coords
    means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords


def get_colored_points_from_depth(
    depths: Tensor,
    rgbs: Tensor,
    c2w: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return colored pointclouds from depth and rgb frame and c2w. Optional masking.

    Returns:
        Tuple of (points, colors)
    """
    points, _ = get_means3d_backproj(
        depths=depths.float(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_size=img_size,
        c2w=c2w.float(),
        device=depths.device,
    )
    points = points.squeeze(0)
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        colors = rgbs.view(-1, 3)[mask]
        points = points[mask]
    else:
        colors = rgbs.view(-1, 3)
        points = points
    return (points, colors)


def get_rays_x_y_1(H, W, focal, c2w):
    """Get ray origins and directions in world coordinates.

    Convention here is (x,y,-1) such that depth*rays_d give real z depth values in world coordinates.
    """
    assert c2w.shape == torch.Size([3, 4])
    image_coords = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="ij",
    )
    i, j = image_coords
    # dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], dim = -1)
    dirs = torch.stack(
        [(pix2ndc_x(i, W)) / focal, pix2ndc_y(j, H) / focal, -torch.ones_like(i)],
        dim=-1,
    )
    dirs = dirs.view(-1, 3)
    rays_d = dirs[..., :] @ c2w[:3, :3]
    rays_o = c2w[:3, -1].expand_as(rays_d)

    # return world coordinate rays_o and rays_d
    return rays_o, rays_d


def get_projection_matrix(znear=0.001, zfar=1000, fovx=None, fovy=None, **kwargs):
    """Opengl projection matrix

    Returns:
        projmat: Tensor
    """

    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        **kwargs,
    )
