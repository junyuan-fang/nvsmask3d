import torch

def interpolate_poses(pose_a, pose_b, steps):
    """
    在两个 3D 姿态之间进行线性插值。
    
    参数:
    pose_a (torch.Tensor): 起始姿态的 4x4 变换矩阵
    pose_b (torch.Tensor): 结束姿态的 4x4 变换矩阵
    steps (int): 插值步数（不包括起始和结束姿态）
    
    返回:
    list of torch.Tensor: 包含插值结果的列表（仅包括中间值）
    """
    if steps == 0:
        return []
    
    # 创建一个从1到steps的等间隔序列
    ts = torch.linspace(1, steps, steps)
    
    # 使用线性插值公式，只计算中间值
    trans = [pose_a[:3, 3] + (pose_b[:3, 3] - pose_a[:3, 3]) * t / (steps + 1) for t in ts]
    
    return trans

# 示例使用
pose_a = torch.eye(4)  # 示例起始姿态
pose_b = torch.eye(4)
pose_b[:3, 3] = torch.tensor([1.0, 2.0, 3.0])  # 示例结束姿态，只改变平移部分

print("pose_a:", pose_a)
print("pose_b:", pose_b)

# 步长为1的情况（一个中间点）
result_1 = interpolate_poses(pose_a, pose_b, steps=1)
print(result_1)
print("\n插值结果 (steps=1):")
for i, pose in enumerate(result_1, 1):
    print(f"中间姿态 {i}:", pose)

# 步长为2的情况（两个中间点）
result_2 = interpolate_poses(pose_a, pose_b, steps=2)
print("\n插值结果 (steps=2):")
for i, pose in enumerate(result_2, 1):
    print(f"中间姿态 {i}:", pose)