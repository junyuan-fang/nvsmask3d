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
from PIL import Image


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
vis_depth_threshold = 0.4
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

def load_depth_maps(depth_maps_paths, depth_scale, device):
    depth_maps = []
    for depth_map_path in depth_maps_paths:
        depth_image = Image.open(depth_map_path)
        depth_array = np.array(depth_image) / depth_scale
        depth_maps.append(torch.from_numpy(depth_array).float().to(device))
    
    return torch.stack(depth_maps)

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
    optimized_camera_to_world = get_camera_pose_in_opencv_convention(camera.camera_to_worlds.to("cuda"))#shape (M, 3,4) on cuda

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
    ].unsqueeze(1)#shape (M, N)
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
def process_depth_maps_in_chunks(depth_maps, u_valid, v_valid, z_valid, chunk_size=100, vis_depth_threshold=0.4):
    """
    Processes depth maps in chunks and compares them with the provided z values to filter valid depths.

    Args:
        depth_maps (torch.Tensor): Tensor of shape (M, H, W), containing depth maps.
        u_valid (torch.Tensor): Tensor of valid u coordinates of shape (N,).
        v_valid (torch.Tensor): Tensor of valid v coordinates of shape (N,).
        z_valid (torch.Tensor): Tensor of valid z values of shape (N,).
        chunk_size (int): Number of depth maps to process per chunk.
        vis_depth_threshold (float): Visibility depth threshold for comparison.

    Returns:
        torch.Tensor: A flattened boolean tensor indicating valid depth points across all chunks.
    """
    valid_depths = torch.zeros_like(z_valid, dtype=torch.bool)  # Initialize to False

    # Iterate over chunks
    for start in range(0, depth_maps.shape[0], chunk_size):
        end = min(start + chunk_size, depth_maps.shape[0])
        depth_maps_chunk = depth_maps[start:end]

        # Efficiently index and process the chunk
        depth_at_valid_points = depth_maps_chunk[:, v_valid, u_valid]

        # Compare depth_at_valid_points with z_valid using the threshold
        valid_chunk = torch.abs(depth_at_valid_points - z_valid) <= vis_depth_threshold
        
        # Update the valid_depths tensor
        valid_depths |= valid_chunk.any(dim=0)

        # Cleanup after processing each chunk
        del depth_maps_chunk, depth_at_valid_points, valid_chunk
        torch.cuda.empty_cache()

    return valid_depths  # Return a flattened tensor

@torch.no_grad()
def object_optimal_k_camera_poses_bounding_box(
    seed_points_0,
    optimized_camera_to_world,
    K,
    W,
    H,
    boolean_mask,
    depth_filenames=None,
    depth_scale=None,
    k_poses=2,
    chunk_size=50,
    vis_depth_threshold=0.4,
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
        final_bounding_boxes (torch.Tensor): (k_poses, 4) on cpu, normalized to [0,1]
    """

    masked_seed_points = seed_points_0[boolean_mask]  # shape (N, 3)
    u, v, z = get_points_projected_uv_and_depth(masked_seed_points, optimized_camera_to_world, K)  # shape (M, N)
    valid_points = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)

    if depth_filenames:
        # Load depth image
        depth_maps = load_depth_maps(depth_filenames, depth_scale, device=seed_points_0.device).half()

        # Calculate valid point indices
        u_valid = u[valid_points].long()
        v_valid = v[valid_points].long()
        z_valid = z[valid_points]  # Keep as float for accuracy

        # Process depth maps in chunks
        valid_depths = process_depth_maps_in_chunks(depth_maps, u_valid, v_valid, z_valid, chunk_size=chunk_size, vis_depth_threshold=vis_depth_threshold)

        # Update valid_points directly using the valid_depths
        valid_points_clone = valid_points.clone()
        valid_points_clone[valid_points] = valid_depths

        valid_points = valid_points_clone

        # Cleanup
        del depth_maps, valid_depths, valid_points_clone
        torch.cuda.empty_cache()
        
    if not valid_points.any():
        print("No valid points found")
        return torch.tensor([]), torch.tensor([])

    # Calculate min and max for u and v
    min_u, _ = u.masked_fill(~valid_points, float("inf")).min(dim=1)  # shape (M,)
    max_u, _ = u.masked_fill(~valid_points, float("-inf")).max(dim=1)
    min_v, _ = v.masked_fill(~valid_points, float("inf")).min(dim=1)
    max_v, _ = v.masked_fill(~valid_points, float("-inf")).max(dim=1)

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

    return best_poses_indices, final_bounding_boxes

    
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
        optimized_camera_to_world (torch.Tensor): (M,3,4) on cuda, need to be in OpenCV convention.
        K (torch.Tensor): (M,3,3) on cuda, camera intrinsics for each camera.
        W (int): Image width (e.g., 640).
        H (int): Image height (e.g., 360).
        boolean_mask (torch.Tensor): (N,) on cuda, the boolean mask for filtering relevant 3D points.

    Returns:
        bounding_boxes (torch.Tensor): (M, 4) Bounding boxes for each camera pose, in the format [min_u, min_v, max_u, max_v].
    """
    # Filter out relevant 3D points using the boolean mask
    masked_seed_points = seed_points_0[boolean_mask]  # shape (N, 3)

    # Project points to 2D image coordinates for all camera poses
    u, v, z = get_points_projected_uv_and_depth(masked_seed_points, optimized_camera_to_world, K)  # shape (M, N)

    # Filter out invalid points (outside of the image boundaries or behind the camera)
    valid_points = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)

    # Handle cases where no valid points are found for any camera
    # if not valid_points.any():
    #     print("No valid points found for any camera pose")
    #     return torch.empty((optimized_camera_to_world.shape[0], 4), device="cuda")  # Return empty bounding boxes
    if not valid_points.any():
        print("No valid points found")
        return torch.tensor([]), torch.tensor([])
    
    # Calculate min and max u and v for valid points for each camera
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

def rotate_vector_to_vector(v1: Tensor, v2: Tensor):
    """
    Returns a rotation matrix that rotates v1 to align with v2.
    """
    assert v1.dim() == v2.dim()
    assert v1.shape[-1] == 3
    if v1.dim() == 1:
        v1 = v1[None, ...]
        v2 = v2[None, ...]
    N = v1.shape[0]

    u = v1 / torch.norm(v1, dim=-1, keepdim=True)
    Ru = v2 / torch.norm(v2, dim=-1, keepdim=True)
    I = torch.eye(3, 3, device=v1.device).unsqueeze(0).repeat(N, 1, 1)

    # the cos angle between the vectors
    c = torch.bmm(u.view(N, 1, 3), Ru.view(N, 3, 1)).squeeze(-1)

    eps = 1.0e-10
    # the cross product matrix of a vector to rotate around
    K = torch.bmm(Ru.unsqueeze(2), u.unsqueeze(1)) - torch.bmm(
        u.unsqueeze(2), Ru.unsqueeze(1)
    )
    # Rodrigues' formula
    ans = I + K + (K @ K) / (1 + c)[..., None]
    same_direction_mask = torch.abs(c - 1.0) < eps
    same_direction_mask = same_direction_mask.squeeze(-1)
    opposite_direction_mask = torch.abs(c + 1.0) < eps
    opposite_direction_mask = opposite_direction_mask.squeeze(-1)
    ans[same_direction_mask] = torch.eye(3, device=v1.device)
    ans[opposite_direction_mask] = -torch.eye(3, device=v1.device)
    return ans
  

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
        depth: Tensor of shape (M, N) representing depth values. Depth to the camera
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

def interpolate_camera_poses_with_camera_trajectory(poses, masked_seed_points0, steps_per_transition=10):
    """
    Interpolates camera poses between the given camera poses on opencv convention by using the possible camera trajectory. 
    Including start pose amd end pose rotation adjustment based on 3D object center point.

    Args:
        poses (torch.Tensor): (M, 3, 4) c2w opengl convention
        masked_seed_points0 (torch.Tensor): (N, 3) maked object point cloud opencv convention
        steps_per_transition (int, optional): The number of steps to interpolate between the two cameras. Defaults to 10.

    Returns:
        interpolated_camera_poses (torch.Tensor): (M * step_per_transition, 3, 4) interpolated camera-to-world matrices
    """
    
    # prepare object center point (3, ) vector
    v2 = masked_seed_points0.mean(dim=0)
    v2 = v2 / torch.norm(v2)
    
    # Interpolate camera poses
    num_poses = poses.shape[0]
    interpolated_camera_poses = []
    for i in range(num_poses - 1):
        # Get the start and end camera poses
        pose_a = poses[i]
        pose_b = poses[i + 1]
        
        #prepare camera (3,) vector.
        v1_a = -pose_a[:3, 2] # z-axis, filipped to opencv convention
        v1_a = v1_a / torch.norm(v1_a)
        v1_b = -pose_b[:3, 2]
        v1_b = v1_b / torch.norm(v1_b)
        
        #get rotation matrix
        R_a = rotate_vector_to_vector(v1_a, v2)
        R_a = R_a @ pose_a[:3, :3]
        pose_a[:3, :3] = R_a
        R_b = rotate_vector_to_vector(v1_b, v2)
        R_b = R_b @ pose_b[:3, :3]
        pose_b[:3, :3] = R_b
        
        # Interpolate between the two camera poses
        poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)# SLERP interpolation
        interpolated_camera_poses.append(poses_ab)

    interpolated_camera_poses = torch.cat(interpolated_camera_poses, dim=0)
    # Final shape will be [steps*num_poses, 3, 4]
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
        pose_a: first pose. (3, 4)
        pose_b: second pose. (3, 4)
        steps: number of steps the interpolated pose path should contain.
    returns:
        interpolated poses: (steps, 3, 4)
    """
    quat_a = quaternion_from_matrix(pose_a[:3, :3])
    quat_b = quaternion_from_matrix(pose_b[:3, :3])
    ts = torch.linspace(0, 1, steps, device=pose_a.device)
    quats = [quaternion_slerp(quat_a, quat_b, t.item()) for t in ts]
    trans = [(1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3] for t in ts]

    poses_ab = []  
    for quat, tran in zip(quats, trans):
        pose = torch.eye(4, device=pose_a.device)
        pose[:3, :3] = quaternion_matrix(quat)[:3, :3]
        pose[:3, 3] = tran
        poses_ab.append(pose[:3, :])  # 将每个pose添加到列表中

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
