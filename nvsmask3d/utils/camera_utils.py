"""Utils for projection and camera coords with different conventions"""

import math
from typing import List, Optional, Tuple
from nerfstudio.cameras.cameras import Cameras, CameraType
# from nerfstudio.cameras.camera_utils import get_interpolated_poses
from scipy.spatial.transform import Rotation as R

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
OPENCV_TO_OPENGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
def get_camera_pose_in_opencv_convention(optimized_camera_to_world: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of camera poses from OpenGL to OpenCV convention.

    Args:
        optimized_camera_to_world: Tensor of shape (M, 3, 4) or (3, 4) in OpenGL convention.

    Returns:
        optimized_camera_to_world: Tensor of shape (M, 3, 4) or (3, 4) in OpenCV convention.
    """
    opengl_to_opencv = torch.tensor(
        OPENGL_TO_OPENCV, device=optimized_camera_to_world.device, dtype=optimized_camera_to_world.dtype
    )  # shape (4, 4)

    # Expand `opengl_to_opencv` to match batch size if necessary
    if optimized_camera_to_world.dim() == 3:
        opengl_to_opencv = opengl_to_opencv.unsqueeze(0).expand(optimized_camera_to_world.shape[0], -1, -1)

    # Add a column to `optimized_camera_to_world` to make it (M, 4, 4) for matrix multiplication
    optimized_camera_to_world = torch.cat(
        [optimized_camera_to_world, torch.tensor([0, 0, 0, 1], device=optimized_camera_to_world.device, dtype=optimized_camera_to_world.dtype).view(1, 1, 4).expand(optimized_camera_to_world.shape[0], -1, -1)],
        dim=1
    )

    # Perform batch matrix multiplication
    optimized_camera_to_world = torch.matmul(optimized_camera_to_world, opengl_to_opencv)

    return optimized_camera_to_world[:, :3, :] # Remove the extra row and return to original shape (M, 3, 4)

def get_camera_pose_in_opengl_convention(optimized_camera_to_world) -> torch.Tensor:
    """
    Converts a camera pose from OpenCV to OpenGL convention.

    Args:
        optimized_camera_to_world: Tensor of shape (3, 4) in OpenCV convention

    Returns:
        optimized_camera_to_world: Tensor of shape (3, 4) in OpenGL convention
    """
    opencv_to_opengl = torch.tensor(
        OPENCV_TO_OPENGL, device="cuda", dtype=optimized_camera_to_world.dtype
    )  # shape (4, 4)
    optimized_camera_to_world = torch.matmul(
        optimized_camera_to_world, opencv_to_opengl
    )  # shape (M, 3, 4)
    return optimized_camera_to_world

@torch.no_grad()
def optimal_k_camera_poses_of_scene(
    seed_points_0, class_agnostic_3d_mask, camera, k_poses=2
):
    """
    Selects the top k optimal camera poses for each mask based on the visibility score of the 3D mask.
    Args:
        seed_points_0: torch.Tensor, size is (N, 3), point cloud
        class_agnostic_3d_mask: torch.Tensor, size is (N, 166), for 3D masks
        camera: Cameras, from this class can get camera poses, which size is (M, 3, 4)
        k_poses: int, for top k poses
    Returns:
        best_poses: torch.Tensor, size is (166,), top k poses (index) for each mask
    """
    # Move camera transformations to the GPU
    optimized_camera_to_world = get_camera_pose_in_opencv_convention(camera.camera_to_worlds.to("cuda"))  # shape (M, 3, 4)

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
    optimized_camera_to_world = get_camera_pose_in_opencv_convention(camera.camera_to_worlds.to("cuda"))#shape (N, 3,4) on cuda

    # Move intrinsics to the GPU
    K = camera.get_intrinsics_matrices().to("cuda")
    W, H = int(camera.width[0].item()), int(camera.height[0].item())

    # Prepare seed points and boolean mask
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
    valid_points = (
        (u >= 0) & (u < W) & (v >= 0) & (v < H) & (points_cam[:, :, 2] > 0)
    )  # shape  (M, N)

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
def object_optimal_k_camera_poses_bounding_box(
    seed_points_0,
    optimized_camera_to_world,
    K,
    W,
    H,
    boolean_mask,
    k_poses=2,
):
    """
    Selects the top k optimal camera poses based on the visibility score of the 3D mask.
    The visibility score is calculated as the product of the number of visible points
    and the bounding box area of the projected points.

    Args:
        seed_points_0 (torch.Tensor): (N,3) on cuda
        optimized_camera_to_world (torch.Tensor): (M,3,4) on cuda need to be in opencv convention
        K (torch.Tensor): (M,3,3) on cuda
        W (int): default is 640
        H (int): default is 360
        boolean_mask (torch.Tensor): (N,) on cuda
        k_poses (int, optional): Defaults to 2.

    Returns:
        best_poses_indices (torch.Tensor): (k_poses,) on cpu
        final_bounding_boxes (torch.Tensor): (k_poses, 4) on cpu
    """
    masked_seed_points = seed_points_0[boolean_mask]  # shape (N, 3)
    # Vectorized computation for all camera poses
    u, v, z = get_points_projected_uv_and_depth(masked_seed_points, optimized_camera_to_world, K)
    valid_points = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)

    if valid_points.any().item() != True:
        print("No valid points found")
        return torch.tensor([]), torch.tensor([])

    # Calculate min and max for u and v
    min_u, _ = u.masked_fill(~valid_points, float("inf")).min(dim=1)  # shape (M,)
    max_u, _ = u.masked_fill(~valid_points, float("-inf")).max(dim=1)
    min_v, _ = v.masked_fill(~valid_points, float("inf")).min(dim=1)
    max_v, _ = v.masked_fill(~valid_points, float("-inf")).max(dim=1)

    # check the shape of the min_u, max_u, min_v, max_v
    # print(u[valid_points].max())
    # Handle -inf and inf by finding the largest/smallest valid coordinates
    max_u = torch.where(max_u == float("-inf"), u[valid_points].max(), max_u)
    max_v = torch.where(max_v == float("-inf"), v[valid_points].max(), max_v)
    min_u = torch.where(min_u == float("inf"), u[valid_points].min(), min_u)
    min_v = torch.where(min_v == float("inf"), v[valid_points].min(), min_v)

    # Calculate bounding box area
    bounding_box_area = (max_u - min_u) * (max_v - min_v)

    # Compute visibility scores for all poses
    num_visible_points = valid_points.float().sum(dim=1)
    visibility_scores = num_visible_points * bounding_box_area

    # Select top k scored poses
    _, best_poses_indices = torch.topk(visibility_scores, k_poses)

    # Ensure indices are on the CPU
    best_poses_indices = best_poses_indices.cpu()
    # best_poses = camera[best_poses_indices]

    # Prepare final bounding boxes for visualization
    final_bounding_boxes = torch.stack(
        (
            min_u[best_poses_indices],
            min_v[best_poses_indices],
            max_u[best_poses_indices],
            max_v[best_poses_indices],
        ),
        dim=1,
    )  # (k_poses, 4)

    return (
        best_poses_indices,
        final_bounding_boxes,
    )  # Returns both the best camera poses and their bounding boxes
    
@torch.no_grad()
def compute_camera_pose_bounding_boxes(
    seed_points_0: torch.Tensor,
    optimized_camera_to_world: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    boolean_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the bounding box for each camera pose given a 3D mask.

    Args:
        seed_points_0 (torch.Tensor): (N,3) on cuda, the point cloud.
        optimized_camera_to_world (torch.Tensor): (M,3,4) on cuda, need to be in opencv convention.
        K (torch.Tensor): (M,3,3) on cuda, camera intrinsics.
        W (int): Image width (e.g., 640).
        H (int): Image height (e.g., 360).
        boolean_mask (torch.Tensor): (N,) on cuda, the boolean mask for filtering relevant 3D points.

    Returns:
        bounding_boxes (torch.Tensor): (M, 4) Bounding boxes for each camera pose, in the format [min_u, min_v, max_u, max_v].
    """
    masked_seed_points = seed_points_0[boolean_mask]  # shape (N, 3)
    # Project points to 2D image coordinates for all camera poses
    u, v, z = get_points_projected_uv_and_depth(masked_seed_points, optimized_camera_to_world, K)
    
    # Filter out invalid points (outside of the image boundaries or behind the camera)
    valid_points = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)

    # Handle case where no valid points are found
    if not valid_points.any():
        print("No valid points found")
        return torch.empty((optimized_camera_to_world.shape[0], 4), device="cuda")  # Return empty bounding boxes

    # Calculate min and max u and v for valid points
    min_u, _ = u.masked_fill(~valid_points, float("inf")).min(dim=1)  # shape (M,)
    max_u, _ = u.masked_fill(~valid_points, float("-inf")).max(dim=1)
    min_v, _ = v.masked_fill(~valid_points, float("inf")).min(dim=1)
    max_v, _ = v.masked_fill(~valid_points, float("-inf")).max(dim=1)

    # Handle cases where all points are invalid for a specific camera
    max_u = torch.where(max_u == float("-inf"), u[valid_points].max(), max_u)
    max_v = torch.where(max_v == float("-inf"), v[valid_points].max(), max_v)
    min_u = torch.where(min_u == float("inf"), u[valid_points].min(), min_u)
    min_v = torch.where(min_v == float("inf"), v[valid_points].min(), min_v)

    # Stack the bounding boxes in the format [min_u, min_v, max_u, max_v]
    bounding_boxes = torch.stack([min_u, min_v, max_u, max_v], dim=1)  # shape (M, 4)

    return bounding_boxes    

def make_cameras(camera: Cameras, poses):
    """
    Create a new Cameras object with the given camera poses.

    Args:
        camera: Cameras, for camera intrinsics
        poses: torch.Tensor, size is (M, 3, 4), for interpolated camera poses

    Returns:
        new_camera: Cameras, with the given camera poses
    """
    n = poses.shape[0]  # Number of cameras
    #poses = get_camera_pose_in_opengl_convention(poses)  # Convert to OpenGL convention
    new_cameras = Cameras(
            fx=camera.fx.squeeze(-1).repeat(n),
            fy=camera.fy.squeeze(-1).repeat(n),
            cx=camera.cx.squeeze(-1).repeat(n),
            cy=camera.cy.squeeze(-1).repeat(n),
            height=camera.height,
            width=camera.width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )
    return new_cameras

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

def get_points_projected_uv_and_depth(
    masked_seed_points: torch.Tensor, optimized_camera_to_world: torch.Tensor, K: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Projects 3D points to 2D pixel coordinates and depth for multiple camera poses.

    Args:
        masked_seed_points: Tensor of shape (N, 3) representing 3D points.
        optimized_camera_to_world: Tensor of shape (M, 3, 4) representing camera poses for M cameras.
        K: Tensor of shape (M, 3, 3) representing camera intrinsics for M cameras.

    Returns:
        u: Tensor of shape (M, N) representing x-coordinates in the image plane.
        v: Tensor of shape (M, N) representing y-coordinates in the image plane.
        depth: Tensor of shape (M, N) representing depth values.
    """
    # Transform points to camera coordinates
    points_cam = masked_seed_points.unsqueeze(0) - optimized_camera_to_world[:, :3, 3].unsqueeze(1)  # (M, N, 3)
    points_cam = torch.bmm(points_cam, optimized_camera_to_world[:, :3, :3])  # (M, N, 3)
    
    # Extract depth
    depth = points_cam[:, :, 2]  # (M, N)

    # Project to image plane
    u = (points_cam[:, :, 0] * K[:, 0, 0].unsqueeze(1) / depth) + K[:, 0, 2].unsqueeze(1)  # (M, N)
    v = (points_cam[:, :, 1] * K[:, 1, 1].unsqueeze(1) / depth) + K[:, 1, 2].unsqueeze(1)  # (M, N)

    return u, v, depth

def interpolate_camera_poses_with_camera_trajectory(poses, bounding_boxes,  K, W, H, steps_per_transition=10):
    """
    Interpolates camera poses between the given camera poses using the camera trajectory. Including start pose amd end pose rotation adjustment based on bounding box.

    Args:
        poses (torch.Tensor): (M, 3, 4) c2w
        K (torch.Tensor): (M, 3, 3) intrinsics
        W (int): image width
        H (int): image height
        bounding_boxes (torch.Tensor): (M, 4) bounding boxes
        steps_per_transition (int, optional): The number of steps to interpolate between the two cameras. Defaults to 10.

    Returns:
        interpolated_camera_poses (torch.Tensor): (M * step_per_transition, 3, 4) interpolated camera-to-world matrices
    """
    
    #poses = get_camera_pose_in_opencv_convention(poses)  # shape (M, 3, 4)
    num_poses = poses.shape[0]

    # Interpolate camera poses
    interpolated_camera_poses = []
    for i in range(num_poses - 1):
        # Get the start and end camera poses
        pose_a = poses[i]
        pose_b = poses[i + 1]
        # get bounding box
        min_u_a, min_v_a, max_u_a, max_v_a = bounding_boxes[i]
        min_u_b, min_v_b, max_u_b, max_v_b = bounding_boxes[i + 1]
        
        # Adjust the start and end camera poses based on the bounding box
        object_uv_a = torch.tensor([(min_u_a + max_u_a) / 2, (min_v_a + max_v_a) / 2], dtype=torch.float32)
        pose_a = compute_new_camera_pose_from_object_uv(camera_pose = pose_a, object_uv=object_uv_a, K = K[i], image_width=W, image_height=H)
        object_uv_b = torch.tensor([(min_u_b + max_u_b) / 2, (min_v_b + max_v_b) / 2], dtype=torch.float32)
        pose_b = compute_new_camera_pose_from_object_uv(camera_pose = pose_b, object_uv=object_uv_b, K = K[i], image_width=W, image_height=H)
        poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)# SLERP interpolation
        interpolated_camera_poses.append(poses_ab)
    interpolated_camera_poses.append(poses_ab)

    interpolated_camera_poses = torch.cat(interpolated_camera_poses, dim=0)
    # Final shape will be [3*n, 3, 4]
    return interpolated_camera_poses

def ideal_K_inverse(K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    K_inv = torch.tensor([
        [1 / fx, 0, -cx / fx],
        [0, 1 / fy, -cy / fy],
        [0, 0, 1]
    ], device=K.device)
    
    return K_inv

def compute_new_camera_pose_from_object_uv(camera_pose: torch.Tensor, object_uv: torch.Tensor, K: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """
    Get a new camera pose, which is centered on the view direction towards to the given object UV position using Rodrigues' rotation formula.
    
    Args:
    - camera_pose (torch.Tensor): Current 3x4 camera pose matrix (OpenGL convention).
    - object_uv (torch.Tensor): Object coordinates on UV plane, shape (2,). Assumes origin is top-left (OpenCV convention).
    - K (torch.Tensor): Camera intrinsics matrix, shape (3, 3).
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.
    
    Returns:
    - torch.Tensor: New 3x4 camera pose matrix.
    """

    # Ensure all tensors are on the same device
    device = camera_pose.device
    object_uv = object_uv.to(device)
    K = K.to(device)
    # Convert object_uv to normalized device coordinates (NDC) in OpenGL (-1 to 1 range)
    ndc_x = 2.0 * (object_uv[0] / image_width) - 1.0
    ndc_y = 2.0 * (object_uv[1] / image_height) - 1.0

    # Create the 3D direction vector in normalized device coordinates (in homogeneous coordinates)
    ndc_direction = torch.tensor([ndc_x, ndc_y, 1.0], device=device) #GPT use [ndc_x, -ndc_y, -1.0]， but our's work. So seems no need to change to OpenGL convention here

    # Convert the NDC direction to camera space using the inverse of the intrinsic matrix
    target_direction = ideal_K_inverse(K) @ ndc_direction

    # Normalize the target direction vector
    target_direction = target_direction / torch.norm(target_direction)

    # Extract the current forward direction (z-axis) from the camera pose
    current_forward = camera_pose[:, 2]

    # Compute the rotation axis (cross product of current and target directions)
    rotation_axis = torch.cross(current_forward, target_direction)
    rotation_axis = rotation_axis / torch.norm(rotation_axis)  # Normalize the axis
    
    # Compute the rotation angle (dot product gives the cosine of the angle)
    cos_theta = torch.dot(current_forward, target_direction)
    rotation_angle = torch.acos(cos_theta)

    # Compute Rodrigues' rotation matrix
    K_matrix = torch.tensor([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ], device=device)

    R = torch.eye(3, device=device) + torch.sin(rotation_angle) * K_matrix + (1 - torch.cos(rotation_angle)) * (K_matrix @ K_matrix)

    # The new rotation matrix
    new_rotation_matrix = R @ camera_pose[:, :3]

    # The translation remains the same
    new_translation = camera_pose[:, 3]

    # Construct the new 3x4 camera pose matrix
    new_camera_pose = torch.cat([new_rotation_matrix, new_translation.unsqueeze(1)], dim=1)

    return new_camera_pose


def quaternion_from_matrix(matrix: Tensor) -> Tensor:
    """Convert a rotation matrix to a quaternion."""
    M = matrix[:3, :3]
    trace = M.trace()

    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (M[2, 1] - M[1, 2]) / s
        qy = (M[0, 2] - M[2, 0]) / s
        qz = (M[1, 0] - M[0, 1]) / s
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = torch.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        qw = (M[2, 1] - M[1, 2]) / s
        qx = 0.25 * s
        qy = (M[0, 1] + M[1, 0]) / s
        qz = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = torch.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        qw = (M[0, 2] - M[2, 0]) / s
        qx = (M[0, 1] + M[1, 0]) / s
        qy = 0.25 * s
        qz = (M[1, 2] + M[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        qw = (M[1, 0] - M[0, 1]) / s
        qx = (M[0, 2] + M[2, 0]) / s
        qy = (M[1, 2] + M[2, 1]) / s
        qz = 0.25 * s

    return torch.tensor([qw, qx, qy, qz], device=matrix.device)


def quaternion_slerp(quat0: Tensor, quat1: Tensor, fraction: float, spin: int = 0, shortestpath: bool = True) -> Tensor:
    """Return spherical linear interpolation between two quaternions.

    Args:
        quat0: first quaternion.
        quat1: second quaternion.
        fraction: how much to interpolate between quat0 vs quat1.
        spin: how much of an additional spin to place on the interpolation.
        shortestpath: whether to return the short or long path to rotation.
    """
    q0 = quat0 / torch.linalg.norm(quat0)
    q1 = quat1 / torch.linalg.norm(quat1)
    d = torch.dot(q0, q1)
    if shortestpath and d < 0.0:
        d = -d
        q1 = -q1
    angle = torch.acos(d) + spin * math.pi
    if abs(angle) < 1e-6:
        return q0
    isin = 1.0 / torch.sin(angle)  # Ensure angle is a Tensor before using torch.sin
    return (torch.sin((1.0 - fraction) * angle) * q0 + torch.sin(fraction * angle) * q1) * isin


def quaternion_matrix(quaternion: Tensor) -> Tensor:
    """Return homogeneous rotation matrix from quaternion."""
    q = quaternion / torch.linalg.norm(quaternion)
    n = torch.dot(q, q)
    if n < 1e-6:
        return torch.eye(4, device=quaternion.device)
    q *= torch.sqrt(2.0 / n)
    q = torch.outer(q, q)
    return torch.tensor([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], device=quaternion.device)


def get_interpolated_poses(pose_a: Tensor, pose_b: Tensor, steps: int = 10) -> List[Tensor]:
    """Return interpolation of poses with specified number of steps.

    Args:
        pose_a: first pose.
        pose_b: second pose.
        steps: number of steps the interpolated pose path should contain.
    """
    quat_a = quaternion_from_matrix(pose_a[:3, :3])
    quat_b = quaternion_from_matrix(pose_b[:3, :3])
    ts = torch.linspace(0, 1, steps, device=pose_a.device)
    quats = [quaternion_slerp(quat_a, quat_b, t.item()) for t in ts]
    trans = [(1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3] for t in ts]

    poses_ab = []  # 使用列表来存储结果
    for quat, tran in zip(quats, trans):
        pose = torch.eye(4, device=pose_a.device)
        pose[:3, :3] = quaternion_matrix(quat)[:3, :3]
        pose[:3, 3] = tran
        poses_ab.append(pose[:3, :])  # 将每个pose添加到列表中
    # 最后将列表中的张量堆叠成一个Tensor
    return torch.stack(poses_ab, dim=0)


def get_interpolated_k(k_a: Tensor, k_b: Tensor, steps: int = 10) -> List[Tensor]:
    """Returns interpolated path between two camera intrinsics with specified number of steps.

    Args:
        k_a: camera matrix 1.
        k_b: camera matrix 2.
        steps: number of steps the interpolated pose path should contain.

    Returns:
        List of interpolated camera intrinsics.
    """
    ts = torch.linspace(0, 1, steps, device=k_a.device)
    return [(1 - t) * k_a + t * k_b for t in ts]


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
