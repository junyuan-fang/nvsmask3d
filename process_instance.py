import torch
import os

# 指定要处理的目录
input_dir = (
    "/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/ScannetPP/mask3d_processed/"
)

# 设置处理参数
th = 0.04  # 置信度阈值
nms_threshold = 0.6  # NMS阈值

# 获取目录下的所有.pt文件
file_list = [f for f in os.listdir(input_dir) if f.endswith(".pt")]

# 遍历并处理每个文件
for file_name in file_list:
    file_path = os.path.join(input_dir, file_name)
    print(f"Processing file: {file_name}")

    # 加载原始的 masks
    masks = torch.load(file_path)
    instance_masks = masks[0]  # 形状为 [num_points, num_instances]
    instance_scores = masks[1]  # 形状为 [num_instances]

    # 1. 过滤低置信度的实例
    high_conf_indices = instance_scores >= th
    instance_masks = instance_masks[:, high_conf_indices]
    instance_scores = instance_scores[high_conf_indices]

    # 更新实例数量
    num_instances = instance_masks.shape[1]
    num_points = instance_masks.shape[0]

    # 如果没有实例满足置信度阈值，跳过此文件
    if num_instances == 0:
        print(f"No instances with confidence >= {th} in file: {file_name}")
        continue

    # 2. 计算每个实例的点数（大小）
    instance_sizes = instance_masks.sum(dim=0)  # 形状为 [num_instances]

    # 3. 将实例掩码转换为布尔类型
    instance_masks_bool = instance_masks.bool()

    # 4. 初始化保留实例的标记
    keep_instances = torch.ones(num_instances, dtype=torch.bool)

    # 5. 应用非极大值抑制（NMS）
    for i in range(num_instances):
        if not keep_instances[i]:
            continue
        mask_i = instance_masks_bool[:, i]
        area_i = instance_sizes[i].float()
        for j in range(i + 1, num_instances):
            if not keep_instances[j]:
                continue
            mask_j = instance_masks_bool[:, j]
            area_j = instance_sizes[j].float()

            # 计算交集和并集
            intersection = (mask_i & mask_j).sum().float()
            union = area_i + area_j - intersection
            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union
            if iou > nms_threshold:
                # 保留点数较多的实例，抑制点数较少的实例
                if area_i >= area_j:
                    keep_instances[j] = False
                else:
                    keep_instances[i] = False
                    break  # 当前实例已被抑制，跳出循环

    # 6. 过滤被抑制的实例
    instance_masks = instance_masks[:, keep_instances]
    instance_scores = instance_scores[keep_instances]
    instance_sizes = instance_sizes[keep_instances]
    num_instances = instance_masks.shape[1]

    # 7. 保存处理后的结果，格式与原始的 masks 相同
    processed_masks = (instance_masks, instance_scores)
    save_path = os.path.join(input_dir, file_name)
    torch.save(processed_masks, save_path)

    print(f"Processed file saved: {file_name}")
