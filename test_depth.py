import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取深度图的函数
def read_depth_map(file_path):
    # 使用 OpenCV 读取深度图
    depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return depth_map

# 可视化并保存深度图的函数
def visualize_and_save_depth_map(depth_map, output_path):
    # 归一化深度图以便可视化
    normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized_depth = np.uint8(normalized_depth)

    # 使用 matplotlib 可视化深度图
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_depth, cmap='viridis')  # 使用 'viridis' colormap 显示深度图
    plt.colorbar(label='Depth Value')  # 显示颜色条，标注深度值
    plt.title('Depth Map Visualization')
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_path, bbox_inches='tight')  # 保存图像
    plt.show()

# 主函数
if __name__ == "__main__":
    # 指定深度图的路径
    file_path = "/home/fangj1/Code/nerfstudio-nvsmask3d/DSC03874.png"  # 替换为实际的 PNG 文件路径
    output_path = "1.png"  # 保存可视化图像的路径

    # 读取深度图
    depth_map = read_depth_map(file_path)

    # 检查深度图是否读取成功
    if depth_map is None:
        print("Error: Could not read the depth map.")
    else:
        # 可视化并保存深度图
        visualize_and_save_depth_map(depth_map, output_path)
