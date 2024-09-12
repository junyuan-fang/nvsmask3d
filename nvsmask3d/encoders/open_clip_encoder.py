from dataclasses import dataclass, field
from typing import Tuple, Type
import torch.nn.functional as F
import torch
import torchvision
from typing import Literal
from sklearn.decomposition import PCA
import albumentations as A
from torch_kmeans import KMeans, CosineSimilarity
import matplotlib.pyplot as plt
from PIL import Image


try:
    import open_clip
except ImportError:
    assert (
        False
    ), "open_clip is not installed, install it with `pip install open-clip-torch`"

from nvsmask3d.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig
from nerfstudio.viewer.viewer_elements import *
from nvsmask3d.utils.utils import SCANNET200_CLASSES
from nvsmask3d.eval.scannet200.scannet_constants import VALID_CLASS_IDS_200
from nvsmask3d.eval.replica.eval_semantic_instance import (
    VALID_CLASS_IDS as VALID_CLASS_IDS_REPLICA,
)
from nvsmask3d.eval.replica.eval_semantic_instance import (
    CLASS_LABELS as REPLICA_CLASSES,
)


@dataclass
class OpenCLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-L-14-336"  # "ViT-B-16"
    clip_model_pretrained: str = "openai"  # "laion2b_s34b_b88k"
    clip_n_dims: int = 768  # 对应ViT-L-14-336px的输出维度
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")


class OpenCLIPNetwork(BaseImageEncoder):
    def __init__(
        self,
        config: OpenCLIPNetworkConfig,
        test_mode: Literal[
            "test", "val", "inference", "train", "all_replica", "all_scannet"
        ] = "val",
    ):
        super().__init__()
        self.config = config
        self.testmode = test_mode
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((336, 336)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims
        self.positives = (
            SCANNET200_CLASSES if "scannet" in self.testmode else REPLICA_CLASSES
        )
        self.label_mapper = (
            torch.tensor(VALID_CLASS_IDS_200).cuda()
            if "scannet" in self.testmode
            else torch.tensor(VALID_CLASS_IDS_REPLICA).cuda()
        )
        print("the test mode is", test_mode)
        ############viewers############
        self.scannet_checkbox = ViewerCheckbox(
            name="Use ScanNet200",
            default_value=False,
            cb_hook=self._scannet_checkbox_update,
            visible=(
                True if self.testmode == "train" or "all" in self.testmode else False
            ),
        )

        self.replica_checkbox = ViewerCheckbox(
            name="Use Replica",
            default_value=True,
            cb_hook=self._replica_checkbox_update,
            visible=(
                True if self.testmode == "train" or "all" in self.testmode else False
            ),
        )

        self.positive_input = ViewerText(
            name="NVSMask3D Positives",
            default_value="object;things;stuff;texture",
            cb_hook=self._set_positives,
            hint="Seperate classes with ;",
            disabled=True,
            visible=(
                True if self.testmode == "train" or "all" in self.testmode else False
            ),
        )

        ##############################
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.negatives]
            ).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(
            self.config.clip_model_type, self.config.clip_model_pretrained
        )

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    @torch.no_grad()
    def get_intermediate_layers(self, x: torch.Tensor, layers=[19, 20, 21, 22], reshape: bool = False):
        outputs = []

        # Pass input through model's visual transformer which handles patch embedding
        x = self.model.visual(x)  # This includes the patch embedding process

        # Now pass through transformer blocks
        for i, blk in enumerate(self.model.visual.transformer.resblocks):
            x = blk(x)
            if i in layers:
                outputs.append(x)

        if reshape:
            # Reshape to desired dimensions
            B, N, C = x.shape
            patch_size = self.model.visual.patch_size[0]
            H = W = int(N ** 0.5)  # Assumes square patches
            outputs = [out.reshape(B, C, H, W) for out in outputs]

        return tuple(outputs)
    
    def kmeans_clustering(self, feats_map):
        n_clusters=len(self.positives)
        cmap = plt.get_cmap("tab20")

        B, D, h, w = feats_map.shape
        feats_map_flattened = feats_map.permute((0, 2, 3, 1)).reshape(B, -1, D)

        kmeans_engine = KMeans(n_clusters=n_clusters, distance=CosineSimilarity)
        kmeans_engine.fit(feats_map_flattened)
        labels = kmeans_engine.predict(
            feats_map_flattened
            )
        labels = labels.reshape(
            B, h, w
            ).float()
        labels = labels[0].cpu().numpy()

        label_map = cmap(labels / n_clusters)[..., :3]
        label_map = np.uint8(label_map * 255)
        label_map = Image.fromarray(label_map)

        return label_map
    def viz_feat(feat):

        _, _, h, w = feat.shape
        feat = feat.squeeze(0).permute((1,2,0))
        projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

        pca = PCA(n_components=3)
        pca.fit(projected_featmap)
        pca_features = pca.transform(projected_featmap)
        pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        pca_features = pca_features * 255
        res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))

        return res_pred
    @torch.no_grad()
    def return_image_map(self, img, layers=[20,21,22,23], kmeans=True):#(C,H,W)
        #
        # Load and preprocess the image
        image_resized = self.process_image_for_feature(img)

        # Get intermediate layers
        with torch.no_grad():
            feats = self.get_intermediate_layers(image_resized, layers=layers, reshape=True)
        
        # Use the last layer for K-means or raw feature visualization
        feats_map = feats[-1]

        if kmeans:
            label_map = self.kmeans_clustering(feats_map)
        else:
            label_map = self.viz_feat(feats_map)

        return label_map    
    
    def process_image_for_feature(self, image_tensor):  # (C, H, W) input
        stride = self.model.visual.patch_size[0]

        MEAN = np.array([123.675, 116.280, 103.530]) / 255
        STD = np.array([58.395, 57.120, 57.375]) / 255

        # Use PyTorch normalization instead of Albumentations
        transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((336, 336)),
            torchvision.transforms.Normalize(mean=MEAN, std=STD),
             ])

        # Apply normalization directly to the input tensor
        image_tensor = transforms(image_tensor).half()
        image_tensor = image_tensor.unsqueeze(0).to('cuda')  # Add batch dimension

        h, w = image_tensor.shape[2:]
        height_int = (h // stride) * stride
        width_int = (w // stride) * stride

        image_resized = torch.nn.functional.interpolate(image_tensor, size=(height_int, width_int), mode='bilinear')

        return image_resized


    def updata_text_embedding(self):
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def _scannet_checkbox_update(self, element):
        self.positive_input.set_disabled(element.value)
        if element.value:
            self.positives = SCANNET200_CLASSES
            self.positive_input.disable = False
            self.replica_checkbox.value = False
            self.updata_text_embedding()
        else:
            self.positive_input.disable = True
            self.positives = ""

    def _replica_checkbox_update(self, element):
        self.positive_input.set_disabled(element.value)
        if element.value:
            self.positives = REPLICA_CLASSES
            self.positive_input.disable = False
            self.scannet_checkbox.value = False
            self.updata_text_embedding()
        else:
            self.positive_input.disable = True
            self.positives = ""

    def _set_positives(self, element):
        self.positives = element.value.split(";")
        self.updata_text_embedding()

    def get_relevancy(self, image: torch.Tensor, positive_id: int) -> torch.Tensor:
        # Encode the image to get the embedding
        embed = self.encode_image(image)

        # Ensure phrases_embeds has the same dtype as embed
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512

        # Compute the relevancy
        output = torch.mm(embed, p.T)  # embedding x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # embedding x 1
        negative_vals = output[..., len(self.positives) :]  # embedding x N_phrase
        repeated_pos = positive_vals.repeat(
            1, len(self.negatives)
        )  # embedding x N_phrase

        sims = torch.stack(
            (repeated_pos, negative_vals), dim=-1
        )  # embedding x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # embedding x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # embedding x 2
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2),
        )[:, 0, :]

    def encode_image(self, input):
        """(B,C,H,W) -> (B,512)"""
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

    def encode_batch_list_image(self, input):
        """list shape B, which has element (C,H,W) -> (B,512)"""
        processed_images = [self.process(img) for img in input]
        batch_tensor = torch.stack(processed_images).half()  # Shape (B, C, H, W)
        return self.model.encode_image(batch_tensor)
    @torch.no_grad()
    def classify_images(self, images: torch.Tensor, batch_size = 10) -> str:
        """
        Args:
            images: a list [] of images (torch.Tensor). (C,W,H) in a list  # (B,C,H,W)

        Returns:
            str: inference object text
        """
        results = []
        phrases_embeds = self.pos_embeds.to(images[0].dtype).half()

        # Process images in batches
        for batch_start in range(0, len(images), batch_size):
            batch_images = images[batch_start:batch_start + batch_size]

            # Encode each image in the batch
            embeddings = [self.encode_image(img.unsqueeze(0)) for img in batch_images]
            embed = torch.cat(embeddings, dim=0)

            # Clear intermediate tensors to free memory
            del embeddings
            torch.cuda.empty_cache()

            output = torch.mm(embed, phrases_embeds.T)

            # Calculate softmax and store results
            for i in range(embed.shape[0]):
                probs = F.softmax(output[i], dim=-1)
                highest_score_index = probs.argmax(dim=-1).item()
                highest_score_value = probs[highest_score_index].item()
                results.append((self.positives[highest_score_index], highest_score_value))

            # Clear processed tensors to free memory
            del embed, output
            torch.cuda.empty_cache()

        # Return the class with the highest score across all images
        positive = max(results, key=lambda x: x[1])[0]

        return positive
    
    # @torch.no_grad()
    # def classify_images(self, images: torch.Tensor, batch_size = 10) -> str:
    #     """
    #     Args:
    #         images: a list [] of images (torch.Tensor). (C,W,H) in a list  # (B,C,H,W)

    #     Returns:
    #         str: inference object text
    #     """
        
    #     # Find the maximum width and height across all images
    #     max_width = max(img.shape[2] for img in images)
    #     max_height = max(img.shape[1] for img in images)

    #     def pad_image(img):
    #         # Pad the image to the max width and height
    #         padding = (0, max_width - img.shape[2], 0, max_height - img.shape[1])
    #         return F.pad(img, padding, mode='constant', value=0)

    #     # Pad all images to the same size
    #     padded_images = [pad_image(img) for img in images]

    #     all_logits = []

    #     # Process images in batches
    #     for batch_start in range(0, len(padded_images), batch_size):
    #         batch_images = padded_images[batch_start:batch_start + batch_size]

    #         # Stack images into a single batch for efficient encoding
    #         image_batch = torch.stack(batch_images).cuda()  # Shape: (B, C, max_height, max_width)

    #         # Encode all images in the batch in one go (batch processing)
    #         embed = self.encode_image(image_batch).to(self.pos_embeds.dtype)  # Shape: (B, D)

    #         # Prepare phrase embeddings
    #         phrases_embeds = self.pos_embeds.to(embed.device)  # Shape: (C, D)

    #         # Perform matrix multiplication for logits
    #         logits = torch.matmul(embed, phrases_embeds.T)  # Shape: (B, C)

    #         # Store logits for averaging later
    #         all_logits.append(logits)

    #         # Clear intermediate tensors to free memory
    #         del image_batch, embed, logits
    #         torch.cuda.empty_cache()

    #     # Concatenate logits from all batches
    #     all_logits = torch.cat(all_logits, dim=0)  # Shape: (P, C)

    #     # Compute the average logits across all images
    #     averaged_logits = all_logits.mean(dim=0)  # Shape: (C,)

    #     # Find the class with the highest score
    #     positive_index = torch.argmax(averaged_logits).item()
    #     positive = self.positives[positive_index]

    #     return positive