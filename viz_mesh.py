import open3d as o3d
import torch
import numpy as np
from pathlib import Path

# Load the instance masks
masks = torch.load("/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/replica/replica_masks/room0.pt")
instance_labels = masks[0]  # Shape: [954492, 36]
unique_instances = masks[1]  # Shape: [36]

# Load the mesh file
mesh = o3d.io.read_triangle_mesh("/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/replica/room0/room0_mesh.ply")
o3d.visualization.draw_geometries([mesh])

# Check if the mesh is loaded correctly
if mesh.is_empty():
    print("Failed to load the mesh. Please check the file path.")
else:
    # Compute normals for proper lighting
    mesh.compute_vertex_normals()

    # Retrieve the original vertex colors
    original_colors = np.asarray(mesh.vertex_colors)

    # Prepare colors for each vertex based on instance labels
    vertex_colors = original_colors.copy()  # Start with original colors

    # Assign colors based on instance labels
    num_instances = unique_instances.shape[0]
    palette = np.loadtxt("/data/scannetpp/scannetpp_repo/semantic/configs/scannet200.txt", dtype=np.uint8)

    for i in range(num_instances):
        # Get vertices for the current instance
        current_instance_mask = instance_labels[:, i] > 0  # Boolean mask for the current instance
        vertex_colors[current_instance_mask] = palette[i % len(palette)] / 255.0  # Scale colors to [0, 1]

    # Set the vertex colors for the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Visualize the mesh with instance colors
    o3d.visualization.draw_geometries([mesh])



# # Get instance labels and unique instances
# instance_labels = masks[0]  # Shape: [954492, 36]
# unique_instances = masks[1]  # Shape: [36]

# # Prepare colors for visualization
# num_instances = len(unique_instances)
# colors = np.random.rand(num_instances, 3)  # Generate random colors for each instance

# # Create an array to hold the color for each vertex
# vertex_colors = np.zeros((instance_labels.shape[0], 3))

# # Assign colors based on instance labels
# for i in range(num_instances):
#     vertex_colors[instance_labels[:, i] > 0] = colors[i]

# # Set vertex colors for the mesh
# mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# # Visualize the mesh with instance colors
# o3d.visualization.draw_geometries([mesh])


mesh_path = "/data/scannetpp/semantics/sem_viz_val/0d2ee665be_gt_sem.ply"
mesh = o3d.io.read_triangle_mesh(mesh_path)

# Check if the mesh is loaded successfully
if mesh.is_empty():
    print("Failed to load the mesh. Please check the file path.")
else:
    # Optionally, compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    
    # Visualize the loaded mesh
    o3d.visualization.draw_geometries([mesh])