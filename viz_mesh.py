import open3d as o3d

# Load the .ply file
mesh = o3d.io.read_triangle_mesh("/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/ScannetPP/data/fe1733741f/scans/mesh_aligned_0.05.ply")

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])
