"""Some useful util functions"""

import json
import math
import os
import random
from pathlib import Path
from typing import List, Literal, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import wandb

# import cm
from matplotlib import cm
from natsort import natsorted
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import resize
from tqdm import tqdm

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.base_model import Model
from nerfstudio.process_data.process_data_utils import (
    convert_video_to_images,
    get_num_frames_in_video,
)
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nvsmask3d.utils.camera_utils import OPENGL_TO_OPENCV, get_means3d_backproj

SCANNET200_CLASSES = [
    "wall",
    "chair",
    "floor",
    "table",
    "door",
    "couch",
    "cabinet",
    "shelf",
    "desk",
    "office chair",
    "bed",
    "pillow",
    "sink",
    "picture",
    "window",
    "toilet",
    "bookshelf",
    "monitor",
    "curtain",
    "book",
    "armchair",
    "coffee table",
    "box",
    "refrigerator",
    "lamp",
    "kitchen cabinet",
    "towel",
    "clothes",
    "tv",
    "nightstand",
    "counter",
    "dresser",
    "stool",
    "cushion",
    "plant",
    "ceiling",
    "bathtub",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "backpack",
    "toilet paper",
    "printer",
    "tv stand",
    "whiteboard",
    "blanket",
    "shower curtain",
    "trash can",
    "closet",
    "stairs",
    "microwave",
    "stove",
    "shoe",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "board",
    "washing machine",
    "mirror",
    "copier",
    "basket",
    "sofa chair",
    "file cabinet",
    "fan",
    "laptop",
    "shower",
    "paper",
    "person",
    "paper towel dispenser",
    "oven",
    "blinds",
    "rack",
    "plate",
    "blackboard",
    "piano",
    "suitcase",
    "rail",
    "radiator",
    "recycling bin",
    "container",
    "wardrobe",
    "soap dispenser",
    "telephone",
    "bucket",
    "clock",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "clothes dryer",
    "guitar",
    "toilet paper holder",
    "seat",
    "speaker",
    "column",
    "bicycle",
    "ladder",
    "bathroom stall",
    "shower wall",
    "cup",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "paper towel roll",
    "machine",
    "mat",
    "windowsill",
    "bar",
    "toaster",
    "bulletin board",
    "ironing board",
    "fireplace",
    "soap dish",
    "kitchen counter",
    "doorframe",
    "toilet paper dispenser",
    "mini fridge",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "water cooler",
    "paper cutter",
    "tray",
    "shower door",
    "pillar",
    "ledge",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "furniture",
    "cart",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "decoration",
    "sign",
    "projector",
    "closet door",
    "vacuum cleaner",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "dish rack",
    "broom",
    "guitar case",
    "range hood",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "mailbox",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "projector screen",
    "divider",
    "laundry detergent",
    "bathroom counter",
    "object",
    "bathroom vanity",
    "closet wall",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "dumbbell",
    "stair rail",
    "tube",
    "bathroom cabinet",
    "cd case",
    "closet rod",
    "coffee kettle",
    "structure",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "storage organizer",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "potted plant",
    "luggage",
    "mattress",
]

# Depth Scale Factor m to mm
SCALE_FACTOR = 0.001
# def select_low_entropy_logits(logits, top_k, apply_softmax=True):
#     """
#     选择给定logits中熵最小的top_k个视角，并返回这些视角的logits之和。

#     参数:
#         logits (torch.Tensor): 输入的logits，形状为 [num_views, num_classes]
#         top_k (int): 要选择的低熵视角数量
#         apply_softmax (bool): 是否对logits应用softmax进行熵计算，默认为True

#     返回:
#         torch.Tensor: 选定视角的logits之和，形状为 [num_classes]
#     """
#     if logits.size(0)<= (top_k):# skip camera interp = 1
#         return logits
#     # 计算 softmax 概率以便计算熵
#     if apply_softmax:
#         probs = torch.softmax(logits, dim=-1)  # [num_views, num_classes]
#     else:
#         probs = logits  # 若不需要 softmax，直接使用logits

#     # 计算每个视角的熵
#     entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)  # [num_views]

#     # 找出熵最小的前 top_k 个视角
#     _, top_k_indices = torch.topk(-entropy, k=top_k, largest=True)


#     # 取出这些低熵视角的 logits
#     top_k_logits = logits[top_k_indices[:top_k]]  # [top_k, num_classes]
#     return top_k_logits  # [num_classes]
def select_low_entropy_logits(logits, top_k, apply_softmax=True):
    """
    选择给定logits中熵最小的top_k个视角，并返回这些视角的logits之和。

    参数:
        logits (torch.Tensor): 输入的logits，形状为 [num_views, num_classes]
        top_k (int): 要选择的低熵视角数量
        apply_softmax (bool): 是否对logits应用softmax进行熵计算，默认为True

    返回:
        torch.Tensor: 选定视角的logits之和，形状为 [num_classes]
    """
    if logits.size(0) <= 2 * (top_k):  # skip camera interp = 1
        return logits
    interpolated_logits = logits[:-top_k]
    gt_rgb_logits = logits[-top_k:]  # default position of gt_rgb is the last top_k
    # 计算 softmax 概率以便计算熵
    if apply_softmax:
        probs = torch.softmax(interpolated_logits, dim=-1)  # [num_views, num_classes]
    else:
        probs = interpolated_logits  # 若不需要 softmax，直接使用logits

    # 计算每个视角的熵
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)  # [num_views]

    # 找出熵最小的前 top_k 个视角
    _, top_k_indices = torch.topk(-entropy, k=top_k, largest=True)

    # 取出这些低熵视角的 logits
    top_k_logits = logits[top_k_indices]  # [top_k, num_classes]

    # 将 gt_rgb_logits 与 top_k_logits 拼接并返回
    top_k_logits = torch.cat(
        [top_k_logits, gt_rgb_logits], dim=0
    )  # [top_k + num_gt_rgb, num_classes]
    return top_k_logits  # [num_classes]


def blur_non_masked_areas(img_tensor, mask, blur_kernel_size=15):
    """
    对输入图像的非掩码部分进行模糊处理。

    Parameters:
        img_tensor (torch.Tensor): 输入图像，形状为 (C, H, W)
        mask (torch.Tensor): 二值掩码，形状为 (H, W)
        blur_kernel_size (int): 高斯模糊核的大小

    Returns:
        torch.Tensor: 处理后的图像，形状为 (C, H, W)
    """
    # 将 mask 变为 (1, H, W) 以便与图像进行广播运算
    mask = mask.unsqueeze(0).float()  # Convert mask to float and add channel dimension

    # 生成模糊的图像
    blurred_img = F.conv2d(
        img_tensor.unsqueeze(0),  # 添加 batch 维度
        weight=torch.ones(
            (img_tensor.size(0), 1, blur_kernel_size, blur_kernel_size),
            device=img_tensor.device,
        )
        / (blur_kernel_size**2),
        stride=1,
        padding=blur_kernel_size // 2,
        groups=img_tensor.size(0),
    ).squeeze(0)  # 去除 batch 维度

    # 使用掩码合并模糊和原始图像
    combined_img = img_tensor * mask + blurred_img * (1 - mask)

    return combined_img  # 输出形状为 (C, H, W)


def video_to_frames(
    video_path: Path, image_dir: Path("./data/frames"), force: bool = False
):
    """Extract frames from video, requires nerfstudio install"""
    is_empty = False

    if not image_dir.exists():
        is_empty = True
    else:
        is_empty = not any(image_dir.iterdir())

    if is_empty or force:
        num_frames_target = get_num_frames_in_video(video=video_path)
        summary_log, num_extracted_frames = convert_video_to_images(
            video_path,
            image_dir=image_dir,
            num_frames_target=num_frames_target,
            num_downscales=0,
            verbose=True,
            image_prefix="frame_",
            keep_image_dir=False,
        )
        assert num_extracted_frames == num_frames_target


import subprocess


def run_command_and_save_output(cmd, output_file):
    # Run the command
    result = subprocess.run(
        cmd,
        shell=True,  # Use shell=True if the command is a shell command
        capture_output=True,  # Capture both stdout and stderr
        text=True,  # Return output as strings instead of bytes
    )

    # Save output to a file
    with open(output_file, "w") as f:
        f.write("Standard Output:\n")
        f.write(result.stdout)  # Write standard output
        f.write("\nStandard Error:\n")
        f.write(result.stderr)  # Write standard error

    # Optionally print the return code
    print(f"Command executed with return code: {result.returncode}")


def save_predictions(preds, output_dir, VALID_CLASS_IDS):
    # 创建存放预测掩码的目录
    masks_dir = os.path.join(output_dir, "predicted_masks")
    os.makedirs(masks_dir, exist_ok=True)

    for scene_name, scene_data in preds.items():
        pred_masks = scene_data[
            "pred_masks"
        ]  # 掩码数组，形状为 (num_points, num_instances)
        pred_classes_orig = scene_data[
            "pred_classes"
        ]  # 每个实例的类别ID，形状为 (num_instances,)
        # replace class id with valid class id
        pred_classes = [VALID_CLASS_IDS[c] for c in pred_classes_orig]
        pred_scores = scene_data[
            "pred_scores"
        ]  # 每个实例的置信度，形状为 (num_instances,)

        summary_lines = []

        # 遍历每个实例
        for i in range(pred_masks.shape[1]):
            mask = pred_masks[:, i]
            class_id = pred_classes[i]
            score = pred_scores[i]

            # 使用 RLE 编码掩码
            rle_mask_data = rle_encode(mask.astype(int))  # 返回带嵌套结构的字典
            rle_counts = rle_mask_data["counts"]  # 提取 counts 字符串

            # 生成掩码文件名
            mask_filename = f"{scene_name}_{i:03d}.json"
            mask_filepath = os.path.join(masks_dir, mask_filename)

            # 保存 RLE 编码到 JSON 文件中，确保 counts 是字符串
            rle_data = {
                "length": len(mask),
                "counts": rle_counts,
            }  # 确保这里是空格分隔字符串而非嵌套结构
            with open(mask_filepath, "w") as f:
                json.dump(rle_data, f)

            # 记录主 .txt 文件的内容，包括相对路径、类ID和置信度
            relative_path = os.path.join("predicted_masks", mask_filename)
            summary_lines.append(f"{relative_path} {class_id} {score:.4f}")

        # 将每个场景的摘要信息写入对应的 .txt 文件
        scene_txt_path = os.path.join(output_dir, f"{scene_name}.txt")
        with open(scene_txt_path, "w") as f:
            f.write("\n".join(summary_lines))


def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = " ".join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle


def get_filename_list(image_dir: Path, ends_with: Optional[str] = None) -> List:
    """List directory and save filenames

    Returns:
        image_filenames
    """
    image_filenames = os.listdir(image_dir)
    if ends_with is not None:
        image_filenames = [
            image_dir / name
            for name in image_filenames
            if name.lower().endswith(ends_with)
        ]
    else:
        image_filenames = [image_dir / name for name in image_filenames]
    image_filenames = natsorted(image_filenames)
    return image_filenames


def image_path_to_tensor(
    image_path: Path, size: Optional[tuple] = None, black_and_white=False
) -> Tensor:
    """Convert image from path to tensor

    Returns:
        image: Tensor
    """
    img = Image.open(image_path)
    if black_and_white:
        img = img.convert("1")
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    if size:
        img_tensor = resize(
            img_tensor.permute(2, 0, 1), size=size, antialias=None
        ).permute(1, 2, 0)
    return img_tensor


def depth_path_to_tensor(
    depth_path: Path, scale_factor: float = SCALE_FACTOR, return_color=False
) -> Tensor:
    """Load depth image in either .npy or .png format and return tensor

    Args:
        depth_path: Path
        scale_factor: float
        return_color: bool
    Returns:
        depth tensor and optionally colored depth tensor
    """
    if depth_path.suffix == ".png":
        depth = cv2.imread(str(depth_path.absolute()), cv2.IMREAD_ANYDEPTH)
    elif depth_path.suffix == ".npy":
        depth = np.load(depth_path, allow_pickle=True)
        if len(depth.shape) == 3:
            depth = depth[..., 0]
    else:
        raise Exception(f"Format is not supported {depth_path.suffix}")
    depth = depth * scale_factor
    depth = depth.astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(-1)
    if not return_color:
        return depth
    else:
        depth_color = colormaps.apply_depth_colormap(depth)
        return depth, depth_color  # type: ignore


def save_img(image, image_path, verbose=True) -> None:
    """helper to save images H x W x C

    Args:
        image: image to save (numpy, Tensor)
        image_path: path to save
        verbose: whether to print save path

    Returns:
        None
    """
    if image.shape[-1] == 1 and torch.is_tensor(image):
        image = image.repeat(1, 1, 3)
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy() * 255
        image = image.astype(np.uint8)
    if not Path(os.path.dirname(image_path)).exists():
        Path(os.path.dirname(image_path)).mkdir(parents=True)
    im = Image.fromarray(image)
    if verbose:
        print("saving to: ", image_path)
    im.save(image_path)


def save_depth(depth, depth_path, verbose=True, scale_factor=SCALE_FACTOR) -> None:
    """helper to save metric depths

    Args:
        depth: image to save (numpy, Tensor)
        depth_path: path to save
        verbose: whether to print save path
        scale_factor: depth metric scaling factor

    Returns:
        None
    """
    if torch.is_tensor(depth):
        depth = depth.float() / scale_factor
        depth = depth.detach().cpu().numpy()
    else:
        depth = depth / scale_factor
    if not Path(os.path.dirname(depth_path)).exists():
        Path(os.path.dirname(depth_path)).mkdir(parents=True)
    if verbose:
        print("saving to: ", depth_path)
    np.save(depth_path, depth)


def save_normal(
    normal: Union[np.array, Tensor],
    normal_path: Path,
    verbose: bool = True,
    format: Literal["png", "npy"] = "png",
) -> None:
    """helper to save normal

    Args:
        normal: image to save (numpy, Tensor)
        normal_path: path to save
        verbose: whether to print save path

    Returns:
        None
    """
    if torch.is_tensor(normal):
        normal = normal.float()
        normal = normal.detach().cpu().numpy()
    else:
        normal = normal
    if not Path(os.path.dirname(normal_path)).exists():
        Path(os.path.dirname(normal_path)).mkdir(parents=True)
    if verbose:
        print("saving to: ", normal_path)
    if format == "npy":
        np.save(normal_path, normal)
    elif format == "png":
        normal = normal * 255
        normal = normal.astype(np.uint8)
        nm = Image.fromarray(normal)
        nm.save(normal_path)


def gs_get_point_clouds(
    eval_data: Optional[InputDataset],
    train_data: Optional[InputDataset],
    model: Model,
    render_output_path: Path,
    num_points: int = 1_000_000,
) -> None:
    """Saves pointcloud rendered from a model using predicted eval/train depths

    Args:
        eval_data: eval input dataset
        train_data: train input dataset
        model: model object
        render_output_path: path to render results to
        num_points: number of points to extract in pd

    Returns:
        None
    """
    CONSOLE.print("[bold green] Generating pointcloud ...")
    H, W = (
        int(train_data.cameras[0].height.item()),
        int(train_data.cameras[0].width.item()),
    )
    pixels_per_frame = W * H
    samples_per_frame = (num_points + (len(train_data) + len(eval_data))) // (
        len(train_data) + len(eval_data)
    )
    points = []
    colors = []
    if len(train_data) > 0:
        for image_idx in tqdm(range(len(train_data)), leave=False):
            camera = train_data.cameras[image_idx : image_idx + 1].to(model.device)
            outputs = model.get_outputs(camera)
            rgb_out, depth_out = outputs["rgb"], outputs["depth"]

            c2w = torch.concatenate(
                [
                    camera.camera_to_worlds,
                    torch.tensor([[[0, 0, 0, 1]]]).to(model.device),
                ],
                dim=1,
            )
            # convert from opengl to opencv
            c2w = torch.matmul(
                c2w, torch.from_numpy(OPENGL_TO_OPENCV).float().to(model.device)
            )
            # backproject
            point, _ = get_means3d_backproj(
                depths=depth_out.float(),
                fx=camera.fx,
                fy=camera.fy,
                cx=camera.cx,
                cy=camera.cy,
                img_size=(W, H),
                c2w=c2w.float(),
                device=model.device,
            )
            point = point.squeeze(0)

            # sample pixels for this frame
            indices = random.sample(range(pixels_per_frame), samples_per_frame)
            mask = torch.tensor(indices, device=model.device)

            color = rgb_out.view(-1, 3)[mask].detach().cpu().numpy()
            point = point[mask].detach().cpu().numpy()
            points.append(point)
            colors.append(color)

    if len(eval_data) > 0:
        for image_idx in tqdm(range(len(eval_data)), leave=False):
            camera = eval_data.cameras[image_idx : image_idx + 1].to(model.device)
            outputs = model.get_outputs(camera)
            rgb_out, depth_out = outputs["rgb"], outputs["depth"]

            c2w = torch.concatenate(
                [
                    camera.camera_to_worlds,
                    torch.tensor([[[0, 0, 0, 1]]]).to(model.device),
                ],
                dim=1,
            )
            # convert from opengl to opencv
            c2w = torch.matmul(
                c2w, torch.from_numpy(OPENGL_TO_OPENCV).float().to(model.device)
            )
            # backproject
            point, _ = get_means3d_backproj(
                depths=depth_out.float(),
                fx=camera.fx,
                fy=camera.fy,
                cx=camera.cx,
                cy=camera.cy,
                img_size=(W, H),
                c2w=c2w.float(),
                device=model.device,
            )
            point = point.squeeze(0)

            # sample pixels for this frame
            indices = random.sample(range(pixels_per_frame), samples_per_frame)
            mask = torch.tensor(indices, device=model.device)

            color = rgb_out.view(-1, 3)[mask].detach().cpu().numpy()
            point = point[mask].detach().cpu().numpy()
            points.append(point)
            colors.append(color)

    points = np.vstack(points)
    colors = np.vstack(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    CONSOLE.print(
        f"[bold yellow]Saved pointcloud to {os.getcwd() + render_output_path}'/pointcloud.ply'"
    )
    o3d.io.write_point_cloud(os.getcwd() + f"{render_output_path}/pointcloud.ply", pcd)
    return (points, colors)


def gs_render_dataset_images(
    train_cache: List,
    eval_cache: List,
    train_dataset,
    eval_dataset,
    model,
    render_output_path: Path,
    mushroom=False,
    save_train_images=False,
) -> None:
    """Render and save all train/eval images of gs model to directory

    Args:
        train_cache: list of cached train images
        eval_cache: list of cached eval images
        eval_data: eval input dataset
        train_data: train input dataset
        model: model object
        render_output_path: path to render results to
        mushroom: if dataset is Mushroom dataset or not
        save_train_images: whether to save train images or not

    Returns:
        None
    """
    CONSOLE.print(f"[bold yellow]Saving results to {render_output_path}")
    if len(eval_cache) > 0:
        for i, _ in tqdm(enumerate(range(len(eval_cache))), leave=False):
            image_idx = i
            data = eval_cache[image_idx]
            # ground truth data
            gt_img = data["image"]
            if "sensor_depth" in data:
                depth_gt = data["sensor_depth"]
                depth_gt_color = colormaps.apply_depth_colormap(data["sensor_depth"])
            else:
                depth_gt = None
                depth_gt_color = None
            normal_gt = data["normal"] if "normal" in data else None
            camera = eval_dataset.cameras[image_idx : image_idx + 1].to(model.device)

            # save the image with its original name for easy comparison
            if mushroom:
                seq_name = Path(eval_dataset.image_filenames[image_idx])
                image_name = f"{seq_name.parts[-3]}_{seq_name.stem}"
            else:
                image_name = Path(eval_dataset.image_filenames[image_idx]).stem
            outputs = model.get_outputs(camera)
            rgb_out, depth_out, normal_out = (
                outputs["rgb"],
                outputs["depth"],
                outputs["normal"],
            )

            depth_color = colormaps.apply_depth_colormap(depth_out)
            depth = depth_out.detach().cpu().numpy()
            save_outputs_helper(
                rgb_out,
                gt_img,
                depth_color,
                depth_gt_color,
                depth_gt,
                depth,
                normal_gt if normal_gt is not None else None,
                normal_out if normal_out is not None else None,
                render_output_path,
                image_name,
            )

    if save_train_images and len(train_cache) > 0:
        for i, _ in tqdm(enumerate(range(len(train_cache))), leave=False):
            image_idx = i
            data = train_cache[image_idx]
            # ground truth data
            gt_img = data["image"]
            if "sensor_depth" in data:
                depth_gt = data["sensor_depth"]
                depth_gt_color = colormaps.apply_depth_colormap(data["sensor_depth"])
            else:
                depth_gt = None
                depth_gt_color = None
            normal_gt = data["normal"] if "normal" in data else None
            camera = train_dataset.cameras[image_idx : image_idx + 1].to(model.device)

            # save the image with its original name for easy comparison
            if mushroom:
                seq_name = Path(train_dataset.image_filenames[image_idx])
                image_name = f"{seq_name.parts[-3]}_{seq_name.stem}"
            else:
                image_name = Path(train_dataset.image_filenames[image_idx]).stem
            outputs = model.get_outputs(camera)
            rgb_out, depth_out, normal_out = (
                outputs["rgb"],
                outputs["depth"],
                outputs["normal"],
            )

            depth_color = colormaps.apply_depth_colormap(depth_out)
            depth = depth_out.detach().cpu().numpy()
            save_outputs_helper(
                rgb_out,
                gt_img,
                depth_color,
                depth_gt_color,
                depth_gt,
                depth,
                normal_gt if normal_gt is not None else None,
                normal_out if normal_out is not None else None,
                render_output_path,
                image_name,
            )


def ns_render_dataset_images(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    train_dataset: InputDataset,
    eval_dataset: InputDataset,
    model: Model,
    render_output_path: Path,
    mushroom=False,
    save_train_images=False,
) -> None:
    """render and save all train/eval images of nerfstudio model to directory

    Args:
        train_dataloader: train dataloader
        eval_dataloader: eval dataloader
        eval_data: eval input dataset
        train_data: train input dataset
        model: model object
        render_output_path: path to render results to
        mushroom: whether the dataset is Mushroom dataset or not
        save_train_images:  whether to save train images or not

    Returns:
        None
    """
    CONSOLE.print(f"[bold yellow]Saving results to {render_output_path}")
    if len(eval_dataloader) > 0:
        for image_idx, (camera, batch) in tqdm(enumerate(eval_dataloader)):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
            # ground truth data
            data = batch.copy()
            gt_img = data["image"]
            if "sensor_depth" in data:
                depth_gt = data["sensor_depth"]
                depth_gt_color = colormaps.apply_depth_colormap(data["sensor_depth"])
            else:
                depth_gt = None
                depth_gt_color = None
            normal_gt = data["normal"] if "normal" in data else None
            # save the image with its original name for easy comparison
            if mushroom:
                seq_name = Path(eval_dataset.image_filenames[image_idx])
                image_name = f"{seq_name.parts[-3]}_{seq_name.stem}"
            else:
                image_name = Path(eval_dataset.image_filenames[image_idx]).stem

            rgb_out, depth_out, normal_out = (
                outputs["rgb"],
                outputs["depth"],
                outputs["normal"] if "normal" in outputs else None,
            )
            depth_color = colormaps.apply_depth_colormap(depth_out)
            depth = depth_out.detach().cpu().numpy()
            save_outputs_helper(
                rgb_out,
                gt_img,
                depth_color,
                depth_gt_color,
                depth_gt,
                depth,
                normal_gt,
                normal_out,
                render_output_path,
                image_name,
            )

    if save_train_images and len(train_dataloader) > 0:
        for image_idx, (camera, batch) in tqdm(enumerate(train_dataloader)):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
            # ground truth data
            data = batch.copy()
            gt_img = data["image"]
            if "sensor_depth" in data:
                depth_gt = data["sensor_depth"]
                depth_gt_color = colormaps.apply_depth_colormap(data["sensor_depth"])
            else:
                depth_gt = None
                depth_gt_color = None
            normal_gt = data["normal"] if "normal" in data else None
            # save the image with its original name for easy comparison
            if mushroom:
                seq_name = Path(train_dataset.image_filenames[image_idx])
                image_name = f"{seq_name.parts[-3]}_{seq_name.stem}"
            else:
                image_name = Path(train_dataset.image_filenames[image_idx]).stem

            rgb_out, depth_out, normal_out = (
                outputs["rgb"],
                outputs["depth"],
                outputs["normal"] if "normal" in outputs else None,
            )
            depth_color = colormaps.apply_depth_colormap(depth_out)
            depth = depth_out.detach().cpu().numpy()
            save_outputs_helper(
                rgb_out,
                gt_img,
                depth_color,
                depth_gt_color,
                depth_gt,
                depth,
                normal_gt,
                normal_out,
                render_output_path,
                image_name,
            )


def save_outputs_helper(
    rgb_out: Optional[Tensor] = None,
    gt_img: Optional[Tensor] = None,
    depth_color: Optional[Tensor] = None,
    depth_gt_color: Optional[Tensor] = None,
    depth_gt: Optional[Tensor] = None,
    depth: Optional[Tensor] = None,
    normal_gt: Optional[Tensor] = None,
    normal: Optional[Tensor] = None,
    render_output_path: Optional[Path] = None,
    image_name: Optional[str] = None,
) -> None:
    """Helper to save model rgb/depth/gt outputs to disk

    Args:
        rgb_out: rgb image
        gt_img: gt rgb image
        depth_color: colored depth image
        depth_gt_color: gt colored depth image
        depth_gt: gt depth map
        depth: depth map
        render_output_path: save directory path
        image_name: stem of save name

    Returns:
        None
    """
    if image_name is None:
        image_name = ""

    if rgb_out is not None and gt_img is not None:
        save_img(
            rgb_out,
            os.getcwd() + f"/{render_output_path}/pred/rgb/{image_name}.png",
            False,
        )
        save_img(
            gt_img,
            os.getcwd() + f"/{render_output_path}/gt/rgb/{image_name}.png",
            False,
        )
    if depth_color is not None:
        save_img(
            depth_color,
            os.getcwd()
            + f"/{render_output_path}/pred/depth/colorised/{image_name}.png",
            False,
        )
    if depth_gt_color is not None:
        save_img(
            depth_gt_color,
            os.getcwd() + f"/{render_output_path}/gt/depth/colorised/{image_name}.png",
            False,
        )
    if depth_gt is not None:
        # save metric depths
        save_depth(
            depth_gt,
            os.getcwd() + f"/{render_output_path}/gt/depth/raw/{image_name}.npy",
            False,
        )
    if depth is not None:
        save_depth(
            depth,
            os.getcwd() + f"/{render_output_path}/pred/depth/raw/{image_name}.npy",
            False,
        )

    if normal is not None:
        save_normal(
            normal,
            os.getcwd() + f"/{render_output_path}/pred/normal/{image_name}.png",
            verbose=False,
        )

    if normal_gt is not None:
        save_normal(
            normal_gt,
            os.getcwd() + f"/{render_output_path}/gt/normal/{image_name}.png",
            verbose=False,
        )


from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


# Function to concatenate images horizontally and vertically
def concat_images_horizontally(imgs):
    # 过滤掉 None 的图像，确保只处理有效图像
    imgs = [img for img in imgs if img is not None]

    if len(imgs) == 0:  # 如果没有有效图像，抛出异常
        raise ValueError("No images to concatenate horizontally!")

    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)

    new_img = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in imgs:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_img


def plot_images_and_logits(
    i, image_set, rgb_logits, title, filename, scene_name, max_ind, replica_classes
):
    """
    Helper function to plot images and classification logits.
    Adjusts horizontal axis based on number of views.
    """
    if image_set and len(image_set) > 0:
        colormap = cm.get_cmap("viridis")
        final_image = concat_images_horizontally(image_set)

        # Adjust the figure width based on number of views
        num_views = len(rgb_logits)
        fig_width = 15 + num_views * 0.2  # Increase figure width with more views
        fig = plt.figure(
            figsize=(fig_width, 12)
        )  # Adjust width for more space between x-axis labels
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

        # 1. 上半部分：显示图片
        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(final_image)
        ax_img.axis("off")  # 隐藏坐标轴

        # 2. 下半部分：显示多视角的分类得分条形图
        ax_bar = fig.add_subplot(gs[1])

        rgb_logits = (100 * rgb_logits).softmax(dim=-1)
        data = rgb_logits.cpu().numpy().tolist()

        # Plot each view as a separate set of bars
        bar_width = 0.05  # Reduce bar width
        offset = bar_width * len(data)  # Adjust offset based on number of views

        for view_i, logits in enumerate(data):
            x = np.arange(len(logits))  # Class indices
            color = colormap(view_i / len(data))  # Normalize the view index
            ax_bar.bar(
                x + view_i * bar_width,
                logits,
                width=bar_width,
                alpha=0.7,
                label=f"View {view_i}",
                color=color,
            )

        # Sum the logits across all views and plot as scatter points
        scores = rgb_logits.sum(dim=0)  # Shape: (200,)
        scores_normalized = (
            torch.nn.functional.softmax(scores, dim=0).cpu().numpy()
        )  # Normalize and convert to numpy

        # 绘制散点图，显示归一化后的logits分数
        x_points = np.arange(len(scores_normalized))
        ax_bar.scatter(
            x_points + offset / 2,
            scores_normalized,
            color="red",
            label="Normalized Summed Logits",
            zorder=5,
        )

        # Set x-ticks and labels
        ax_bar.set_xticks(
            np.arange(len(replica_classes)) + offset / 2
        )  # Offset to center labels
        ax_bar.set_xticklabels(
            replica_classes, rotation=45, ha="right"
        )  # Rotate for better fit

        # Set labels and title
        ax_bar.set_xlabel("Class Index")
        ax_bar.set_ylabel("Logit Value")
        ax_bar.set_title(
            f"{title}. Scene: {scene_name}. Object {i} predicted class: {replica_classes[max_ind]}"
        )
        ax_bar.legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(filename)

        # Log the combined figure to WandB
        wandb.log({f"{title}: {scene_name}.": plt})

        # Close the plot to avoid memory issues
        plt.close()


def concat_images_vertically(imgs):
    # 过滤掉 None 的图像，确保只处理有效图像
    imgs = [img for img in imgs if img is not None]

    if len(imgs) == 0:  # 如果没有有效图像，抛出异常
        raise ValueError("No images to concatenate vertically!")

    widths, heights = zip(*(i.size for i in imgs))
    max_width = max(widths)
    total_height = sum(heights)

    new_img = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for img in imgs:
        new_img.paste(img, (0, y_offset))
        y_offset += img.size[1]
    return new_img


def format_to_percentage(value):
    """Safely format a value as a percentage with one decimal place."""
    try:
        return "{:.1f}%".format(value * 100)
    except (TypeError, ValueError):
        return "0.0%"


def generate_txt_files_optimized(preds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for scene_name, scene_data in preds.items():
        num_points, num_classes = scene_data["pred_masks"].shape
        pred_masks = scene_data["pred_masks"]
        pred_classes = scene_data["pred_classes"]

        # 初始化结果数组，所有元素设为 -100
        labels = np.full(num_points, -100, dtype=int)

        # 遍历每个类别，用 pred_classes[i] 替换 pred_masks 中为 1 的位置
        for i in range(num_classes):
            labels[pred_masks[:, i] == 1] = pred_classes[i]

        # 将所有标签转换为字符串形式，未分配的标签保持为 "0"
        result_lines = labels.astype(str)

        # 将结果写入文件
        output_path = os.path.join(output_dir, f"{scene_name}.txt")
        with open(output_path, "w") as f:
            f.write("\n".join(result_lines) + "\n")

    print(f"Files generated in {output_dir}")


def log_evaluation_results_to_wandb(avgs, run_name):
    # Define the metrics to log
    metrics = ["AP", "AP50", "AP25"]

    # Create a W&B Table with "Class" and the defined metrics as columns
    table = wandb.Table(columns=["Class"] + metrics)

    # Log overall average results in percentage format
    table.add_data(
        "Average",
        *[format_to_percentage(avgs.get(f"all_{m.lower()}", 0)) for m in metrics],
    )

    # Log per-class results (ensure values are retrieved safely)
    for class_name, class_metrics in avgs.get("classes", {}).items():
        table.add_data(
            class_name,
            *[format_to_percentage(class_metrics.get(m.lower(), 0)) for m in metrics],
        )

    # Log the table in W&B under a unique key
    wandb.log({f"{run_name}_evaluation_results": table})


def make_square_image(
    nvs_img,
    valid_u,
    valid_v,
    min_u,
    max_u,
    min_v,
    max_v,
    expand_factor=0.7,
    kernel_size=5,
):
    # 创建初始掩码
    mask = torch.zeros(
        (nvs_img.shape[1], nvs_img.shape[2]), dtype=torch.bool, device=nvs_img.device
    )
    mask[valid_v, valid_u] = True

    # 使用闭运算填充内部空洞
    # 先进行膨胀
    dilated_mask = F.max_pool2d(
        mask.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    dilated_mask = (dilated_mask.squeeze(0).squeeze(0) > 0).bool()
    # 再进行腐蚀
    eroded_mask = F.max_pool2d(
        (~dilated_mask).float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    eroded_mask = ~(eroded_mask.squeeze(0).squeeze(0) > 0).bool()

    # 应用高斯模糊
    gaussian_blur = GaussianBlur(kernel_size=(41, 41), sigma=5)
    blurred_img = gaussian_blur(nvs_img.unsqueeze(0)).squeeze(0)

    # 将背景像素替换为模糊后的图像
    output_img = nvs_img.clone()
    output_img[:, ~eroded_mask] = blurred_img[:, ~eroded_mask]

    # 计算裁剪区域
    crop_width = max_u - min_u
    crop_height = max_v - min_v
    max_dim = max(crop_width, crop_height)
    new_dim = int(max_dim / expand_factor)
    center_u = (min_u + max_u) // 2
    center_v = (min_v + max_v) // 2
    new_min_u = max(center_u - new_dim // 2, 0)
    new_max_u = min(center_u + new_dim // 2, nvs_img.shape[2])
    new_min_v = max(center_v - new_dim // 2, 0)
    new_max_v = min(center_v + new_dim // 2, nvs_img.shape[1])

    # 裁剪扩展区域
    output_tensor_cropped = output_img[:, new_min_v:new_max_v, new_min_u:new_max_u]

    return output_tensor_cropped

def make_blur_image(
    nvs_img,
    valid_u,
    valid_v,
    min_u,
    max_u,
    min_v,
    max_v,
    kernel_size=5,
):
    # 创建初始掩码
    mask = torch.zeros(
        (nvs_img.shape[1], nvs_img.shape[2]), dtype=torch.bool, device=nvs_img.device
    )
    mask[valid_v, valid_u] = True

    # 使用闭运算填充内部空洞
    # 先进行膨胀
    dilated_mask = F.max_pool2d(
        mask.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    dilated_mask = (dilated_mask.squeeze(0).squeeze(0) > 0).bool()
    # 再进行腐蚀
    eroded_mask = F.max_pool2d(
        (~dilated_mask).float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    eroded_mask = ~(eroded_mask.squeeze(0).squeeze(0) > 0).bool()

    # 应用高斯模糊
    gaussian_blur = GaussianBlur(kernel_size=(41, 41), sigma=5)
    blurred_img = gaussian_blur(nvs_img.unsqueeze(0)).squeeze(0)

    # 将背景像素替换为模糊后的图像
    output_img = nvs_img.clone()
    output_img[:, ~eroded_mask] = blurred_img[:, ~eroded_mask]

    # 按有效区域的边界进行裁剪
    new_min_u = max(min_u, 0)
    new_max_u = min(max_u, nvs_img.shape[2])
    new_min_v = max(min_v, 0)
    new_max_v = min(max_v, nvs_img.shape[1])

    # 裁剪指定区域
    output_tensor_cropped = output_img[:, new_min_v:new_max_v, new_min_u:new_max_u]

    return output_tensor_cropped


# def make_square_image(nvs_img, valid_u, valid_v, min_u, max_u, min_v, max_v, expand_factor=0.7, kernel_size=5):
#     # Create the initial mask from valid_u and valid_v
#     mask = torch.zeros((nvs_img.shape[1], nvs_img.shape[2]), dtype=torch.bool, device=nvs_img.device)
#     mask[valid_v, valid_u] = True

#     # Apply morphological closing to fill internal gaps without expanding the mask externally
#     mask_filled = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel_size, stride=1, padding=kernel_size // 2)
#     mask_filled = (mask_filled.squeeze(0).squeeze(0) > 0).bool()

#     # Apply blur to the entire image
#     gaussian_blur = GaussianBlur(kernel_size=(41, 41), sigma=5)
#     blurred_img = gaussian_blur(nvs_img.unsqueeze(0)).squeeze(0)

#     # Replace the background pixels in the original image with the blurred image
#     output_img = nvs_img.clone()
#     output_img[:, ~mask_filled] = blurred_img[:, ~mask_filled]

#     # Crop the expanded region (same logic as before)
#     crop_width = max_u - min_u
#     crop_height = max_v - min_v
#     max_dim = max(crop_width, crop_height)
#     new_dim = int(max_dim / expansion_factor)
#     center_u = (min_u + max_u) // 2
#     center_v = (min_v + max_v) // 2
#     new_min_u = max(center_u - new_dim // 2, 0)
#     new_max_u = min(center_u + new_dim // 2, nvs_img.shape[2])
#     new_min_v = max(center_v - new_dim // 2, 0)
#     new_max_v = min(center_v + new_dim // 2, nvs_img.shape[1])

#     # Crop the expanded region
#     output_tensor_cropped = output_img[:, new_min_v:new_max_v, new_min_u:new_max_u]

#     return output_tensor_cropped