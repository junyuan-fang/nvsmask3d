# import torch

# def interpolate_poses(pose_a, pose_b, steps):
#     """
#     在两个 3D 姿态之间进行线性插值。
    
#     参数:
#     pose_a (torch.Tensor): 起始姿态的 4x4 变换矩阵
#     pose_b (torch.Tensor): 结束姿态的 4x4 变换矩阵
#     steps (int): 插值步数（不包括起始和结束姿态）
    
#     返回:
#     list of torch.Tensor: 包含插值结果的列表（仅包括中间值）
#     """
#     if steps == 0:
#         return []
    
#     # 创建一个从1到steps的等间隔序列
#     ts = torch.linspace(1, steps, steps)
    
#     # 使用线性插值公式，只计算中间值
#     trans = [pose_a[:3, 3] + (pose_b[:3, 3] - pose_a[:3, 3]) * t / (steps + 1) for t in ts]
    
#     return trans

# # 示例使用
# pose_a = torch.eye(4)  # 示例起始姿态
# pose_b = torch.eye(4)
# pose_b[:3, 3] = torch.tensor([1.0, 2.0, 3.0])  # 示例结束姿态，只改变平移部分

# print("pose_a:", pose_a)
# print("pose_b:", pose_b)

# # 步长为1的情况（一个中间点）
# result_1 = interpolate_poses(pose_a, pose_b, steps=1)
# print(result_1)
# print("\n插值结果 (steps=1):")
# for i, pose in enumerate(result_1, 1):
#     print(f"中间姿态 {i}:", pose)

# # 步长为2的情况（两个中间点）
# result_2 = interpolate_poses(pose_a, pose_b, steps=2)
# print("\n插值结果 (steps=2):")
# for i, pose in enumerate(result_2, 1):
#     print(f"中间姿态 {i}:", pose)
# import torch

# def geometric_median_pytorch_optimized_adaptive_eps(points, eps_ratio=1e-5, max_iter=1000, device=None):
#     """
#     计算3D点集的几何中位数，使用优化的Weiszfeld算法，并根据输入数据自适应选择eps。

#     参数:
#     - points (torch.Tensor): 形状为 (N, 3) 的点集，dtype为 float32 或 float16。
#     - eps_ratio (float): eps与数据尺度的比例。
#     - max_iter (int): 最大迭代次数。
#     - device (torch.device, optional): 计算设备。如果为None，则与points相同。

#     返回:
#     - torch.Tensor: 形状为 (3,) 的几何中位数。
#     """
#     if device is None:
#         device = points.device
#     points = points.to(device).float()  # 使用float32确保数值稳定性

#     N, dim = points.shape
#     if dim != 3:
#         raise ValueError("输入点集必须是形状为 (N, 3) 的张量")

#     # 计算数据的尺度（标准差）
#     data_std = torch.std(points, dim=0).mean().item()
#     eps = eps_ratio * data_std

#     # 初始化：使用所有点的均值
#     median = points.mean(dim=0)

#     for _ in range(max_iter):
#         # 计算每个点与当前中位数的差值
#         diff = points - median  # (N, 3)

#         # 计算欧式距离
#         distances = torch.norm(diff, dim=1)  # (N,)

#         # 创建掩码，避免距离为零的点
#         mask = distances > 1e-10
#         if not mask.any():
#             break  # 所有点都与中位数重合

#         # 计算权重：距离的倒数
#         weights = 1.0 / distances[mask]  # (M,)
#         weights = weights.unsqueeze(1)  # (M, 1)

#         # 计算加权和
#         weighted_sum = torch.sum(points[mask] * weights, dim=0)  # (3,)
#         weights_sum = weights.sum()  # scalar

#         # 更新中位数
#         new_median = weighted_sum / weights_sum  # (3,)

#         # 检查收敛
#         move_distance = torch.norm(new_median - median)
#         if move_distance < eps:
#             median = new_median
#             break

#         median = new_median

#     return median

# # 示例用法
# if __name__ == "__main__":
#     import time

#     # 选择设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")

#     # 生成大量随机3D点
#     N = 10**7  # 1000万点
#     points = torch.randn(N, 3, device=device) * 100  # 放大数据尺度

#     # 计算几何中位数
#     start_time = time.time()
#     median = geometric_median_pytorch_optimized_adaptive_eps(points, eps_ratio=1e-5, max_iter=1000, device=device)
#     # 如果使用GPU，确保所有计算完成
#     if device.type == 'cuda':
#         torch.cuda.synchronize()
#     end_time = time.time()

#     print(f"几何中位数: {median.cpu().numpy()}")
#     print(f"计算时间: {end_time - start_time:.4f} 秒")
def map_class_indices(top100, top100instance):
    # Create a list to store index mappings
    index_mapping = []

    # For each class in `top100instance`, find the index in `top100`
    for instance_class in top100instance:
        if instance_class in top100:
            index_mapping.append(top100.index(instance_class))  # Append index of matching class
        else:
            index_mapping.append(None)  # No direct match found, use None

    return index_mapping

def read_txt_to_array(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 去除每行的换行符并返回一个字符串数组
    return [line.strip() for line in lines]
# list = read_txt_to_array("/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/eval/scannetpp/top100.txt")
top100instance = read_txt_to_array("/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/eval/scannetpp/top100_instance.txt")
top100 = read_txt_to_array("/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/eval/scannetpp/top100.txt")
print(top100instance)
print(top100)

mapping = map_class_indices(top100, top100instance)
print(mapping)
