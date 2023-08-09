import matplotlib.pyplot as plt
import numpy as np
import requests
import open_clip
import torch
import types
import albumentations as A

from PIL import Image
from sklearn.decomposition import PCA
from torch_kmeans import KMeans, CosineSimilarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cmap = plt.get_cmap("tab20")
MEAN = np.array([123.675, 116.280, 103.530]) / 255
STD = np.array([58.395, 57.120, 57.375]) / 255

transforms = A.Compose(
    [
        A.Normalize(mean=list(MEAN), std=list(STD)),
    ]
)


import torch.nn.functional as F


def interpolate_positional_embedding(model, x):
    # Extract the original positional embedding from the model
    orig_pos_embed = (
        model.visual.positional_embedding
    )  # Shape: [num_patches + 1, dim] or [1, num_patches + 1, dim]

    # Ensure orig_pos_embed has a batch dimension
    if orig_pos_embed.dim() == 2:
        orig_pos_embed = orig_pos_embed.unsqueeze(
            0
        )  # Shape becomes [1, num_patches + 1, dim]

    # Separate class token and patch embeddings
    class_token = orig_pos_embed[:, :1, :]  # Class token, shape: [1, 1, dim]
    patch_pos_embed = orig_pos_embed[
        :, 1:, :
    ]  # Patch embeddings, shape: [1, num_patches, dim]

    # Calculate the original grid size (assume it's square)
    orig_num_patches = patch_pos_embed.shape[1]
    orig_grid_size = int(orig_num_patches**0.5)

    # Reshape patch embeddings to 2D grid
    orig_pos_embed_2d = patch_pos_embed.reshape(
        1, orig_grid_size, orig_grid_size, -1
    ).permute(0, 3, 1, 2)

    # Compute the new grid size based on the input image size
    new_grid_size = (
        x.shape[2] // model.visual.patch_size[0],
        x.shape[3] // model.visual.patch_size[0],
    )

    # Interpolate the positional embeddings to match the new grid size
    new_pos_embed_2d = F.interpolate(
        orig_pos_embed_2d, size=new_grid_size, mode="bilinear", align_corners=False
    )

    # Reshape back to [1, new_num_patches, dim]
    new_pos_embed = new_pos_embed_2d.permute(0, 2, 3, 1).reshape(
        1, -1, orig_pos_embed.shape[-1]
    )

    # Concatenate the class token back with the new positional embeddings
    new_pos_embed = torch.cat([class_token, new_pos_embed], dim=1)

    # Update the model's positional embedding
    model.visual.positional_embedding = torch.nn.Parameter(new_pos_embed)


def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    return_class_token: bool = False,
    norm: bool = True,
):
    outputs = []
    hooks = []
    # Interpolate positional embedding to handle dynamic input sizes
    interpolate_positional_embedding(self, x)

    # Define hook function to capture output
    def hook_fn(_, __, output):
        outputs.append(output)

    # Register hooks for specified layers
    for layer in n:
        hook = self.visual.transformer.resblocks[layer].register_forward_hook(hook_fn)
        hooks.append(hook)

    # Forward pass through the model to get the activations
    with torch.no_grad():
        self.encode_image(x)

    # Remove hooks after the forward pass
    for hook in hooks:
        hook.remove()

    # Apply normalization if required
    if norm:
        outputs = [self.visual.ln_post(out) for out in outputs]

    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, :1] for out in outputs]  # Assuming one prefix token
    outputs = [out[:, 1:] for out in outputs]  # Skip the class token

    # Reshape if necessary
    if reshape:
        B, C, H, W = x.shape
        patch_size = self.visual.patch_size
        grid_size = (
            (H - patch_size[0]) // patch_size[0] + 1,
            (W - patch_size[0]) // patch_size[0] + 1,
        )
        outputs = [
            out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in outputs
        ]

    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))

    return tuple(outputs)


def viz_feat(feat):
    _, _, h, w = feat.shape
    feat = feat.squeeze(0).permute((1, 2, 0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

    pca = PCA(n_components=3)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (
        pca_features.max() - pca_features.min()
    )
    pca_features = pca_features * 255
    res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))

    return res_pred


def plot_feats(image, ori_feats=None, ori_labels=None):
    ori_feats_map = viz_feat(ori_feats)

    if ori_labels is not None:
        fig, ax = plt.subplots(2, 3, figsize=(10, 5))
        ax[0][0].imshow(image)
        ax[0][0].set_title("Input image", fontsize=15)
        ax[0][1].imshow(ori_feats_map)
        ax[1][1].imshow(ori_labels)
        for xx in ax:
            for x in xx:
                x.xaxis.set_major_formatter(plt.NullFormatter())
                x.yaxis.set_major_formatter(plt.NullFormatter())
                x.set_xticks([])
                x.set_yticks([])
                x.axis("off")

    else:
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].imshow(image)
        ax[0].set_title("Input image", fontsize=15)
        ax[1].imshow(ori_feats_map)
        ax[2].set_title("Ours", fontsize=15)

        for x in ax:
            x.xaxis.set_major_formatter(plt.NullFormatter())
            x.yaxis.set_major_formatter(plt.NullFormatter())
            x.set_xticks([])
            x.set_yticks([])
            x.axis("off")

    plt.tight_layout()
    plt.close(fig)
    return fig


def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, "wb") as file:
        file.write(response.content)


def process_image(image, stride, transforms):
    transformed = transforms(image=np.array(image))
    image_tensor = torch.tensor(transformed["image"])
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0).to("cuda")

    h, w = image_tensor.shape[2:]

    height_int = (h // stride) * stride
    width_int = (w // stride) * stride

    image_resized = torch.nn.functional.interpolate(
        image_tensor, size=(height_int, width_int), mode="bilinear"
    )

    return image_resized


def kmeans_clustering(feats_map, n_clusters=20):
    B, D, h, w = feats_map.shape
    feats_map_flattened = feats_map.permute((0, 2, 3, 1)).reshape(B, -1, D)

    kmeans_engine = KMeans(n_clusters=n_clusters, distance=CosineSimilarity)
    kmeans_engine.fit(feats_map_flattened)
    labels = kmeans_engine.predict(feats_map_flattened)
    labels = labels.reshape(B, h, w).float()
    labels = labels[0].cpu().numpy()

    label_map = cmap(labels / n_clusters)[..., :3]
    label_map = np.uint8(label_map * 255)
    label_map = Image.fromarray(label_map)

    return label_map


def run_demo(model, image_path, kmeans=20):
    """
    Run the demo for a given model option and image
    image_path: path to the image
    kmeans: number of clusters for kmeans. Default is 20. -1 means no kmeans.
    """
    p = model.visual.patch_size
    stride = p if isinstance(p, int) else p[0]
    image = Image.open(image_path)
    image_resized = process_image(image, stride, transforms)
    with torch.no_grad():
        ori_feats = model.get_intermediate_layers(
            image_resized,
            n=[20, 21, 22, 23],
            reshape=True,
            return_prefix_tokens=False,
            return_class_token=False,
            norm=True,
        )

    ori_feats = ori_feats[-1]

    if kmeans != -1:
        ori_labels = kmeans_clustering(ori_feats, kmeans)
    else:
        ori_labels = None

    return plot_feats(image, ori_feats, ori_labels)


# Load OpenCLIP model (ViT-L/14)
model_name = "ViT-L-14"
pretrained_dataset = "openai"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained_dataset
)
model.to(device)
model.get_intermediate_layers = types.MethodType(get_intermediate_layers, model)

image_path = (
    "/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/replica/office0/color/0.jpg"
)
kmeans = 20
run_demo(model, image_path, kmeans)
