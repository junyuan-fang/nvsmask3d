from dataclasses import dataclass, field
from typing import Tuple, Type
import torch.nn.functional as F
import torch
import torchvision
from typing import Literal

import clip  # 使用原版CLIP库

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
class CLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: CLIPNetwork)
    clip_model_type: str = "ViT-L/14@336px"
    clip_n_dims: int = 768  # 对应ViT-L/14@336px的输出维度
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")


class CLIPNetwork(BaseImageEncoder):
    def __init__(
        self,
        config: CLIPNetworkConfig,
        test_mode: Literal[
            "test", "val", "inference", "train", "all_replica", "all_scannet"
        ] = "val",
    ):
        super().__init__()
        self.config = config
        self.testmode = test_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载CLIP模型
        self.model, self.preprocess = clip.load(
            self.config.clip_model_type, device=self.device
        )
        self.tokenizer = clip.tokenize
        self.clip_n_dims = self.config.clip_n_dims
        self.positives = (
            SCANNET200_CLASSES if "scannet" in self.testmode else REPLICA_CLASSES
        )
        self.label_mapper = (
            torch.tensor(VALID_CLASS_IDS_200).to(self.device)
            if "scannet" in self.testmode
            else torch.tensor(VALID_CLASS_IDS_REPLICA).to(self.device)
        )
        print("The test mode is", test_mode)

        # Viewer elements
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

        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = self.tokenizer(self.positives).to(self.device)
            self.pos_embeds = self.model.encode_text(tok_phrases).float()
            tok_phrases = self.tokenizer(self.negatives).to(self.device)
            self.neg_embeds = self.model.encode_text(tok_phrases).float()

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
        return f"clip_{self.config.clip_model_type}"

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def update_text_embedding(self):
        with torch.no_grad():
            tok_phrases = self.tokenizer(self.positives).to(self.device)
            self.pos_embeds = self.model.encode_text(tok_phrases).float()
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def _scannet_checkbox_update(self, element):
        self.positive_input.set_disabled(element.value)
        if element.value:
            self.positives = SCANNET200_CLASSES
            self.positive_input.disable = False
            self.replica_checkbox.value = False
            self.update_text_embedding()
        else:
            self.positive_input.disable = True
            self.positives = ""

    def _replica_checkbox_update(self, element):
        self.positive_input.set_disabled(element.value)
        if element.value:
            self.positives = REPLICA_CLASSES
            self.positive_input.disable = False
            self.scannet_checkbox.value = False
            self.update_text_embedding()
        else:
            self.positive_input.disable = True
            self.positives = ""

    def _set_positives(self, element):
        self.positives = element.value.split(";")
        self.update_text_embedding()

    def get_relevancy(self, image: torch.Tensor, positive_id: int) -> torch.Tensor:
        embed = self.encode_image(image)
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2),
        )[:, 0, :]

    # def encode_image(self, input):
    #     """(B,C,H,W) -> (B,768)"""
    #     processed_input = torch.stack([self.preprocess(img) for img in input]).to(self.device)
    #     return self.model.encode_image(processed_input)

    def encode_batch_list_image(self, input):
        """list shape B, which has element (C,H,W) -> (B,512)"""
        processed_images = [self.process(img) for img in input]
        batch_tensor = torch.stack(processed_images).half()  # Shape (B, C, H, W)
        return self.model.encode_image(batch_tensor)

    def classify_images(self, images: torch.Tensor) -> str:
        print("images shape", images[0].shape)
        embeddings = [self.model.encode_image(img.unsqueeze(0)) for img in images]
        embed = torch.cat(embeddings)
        results = []
        phrases_embeds = self.pos_embeds
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)

        for i in range(embed.shape[0]):
            probs = F.softmax(output[i], dim=-1)
            highest_score_index = probs.argmax(dim=-1).item()
            highest_score_value = probs[highest_score_index].item()
            results.append((self.positives[highest_score_index], highest_score_value))

        positive = max(results, key=lambda x: x[1])[0]
        return positive
