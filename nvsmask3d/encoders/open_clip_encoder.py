from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torchvision
from typing import Literal
try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

from nvsmask3d.encoders.image_encoder import (BaseImageEncoder,
                                         BaseImageEncoderConfig)
from nerfstudio.viewer.viewer_elements import *
from nvsmask3d.utils.utils import SCANNET200_CLASSES



@dataclass
class OpenCLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")


class OpenCLIPNetwork(BaseImageEncoder):
    def __init__(self, 
                 config: OpenCLIPNetworkConfig,
                test_mode: Literal["test", "val", "inference", "train"] = "val",
):
        super().__init__()
        self.config = config
        self.testmode = test_mode   
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
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

        ############viewers############
        self.scannet_checkbox= ViewerCheckbox(
            name="Use ScanNet200",
            default_value=False,
            cb_hook=self._scannet_checkbox_update,
            visible=True if self.testmode == "train" else False
        )
        
        self.positive_input = ViewerText(
            name = "NVSMask3D Positives", 
            default_value = "object;things;stuff;texture", 
            cb_hook=self._set_positives, 
            hint="Seperate classes with ;",
            visible=True if self.testmode == "train" else False)
        
        ##############################

        self.positives = self.positive_input.value.split(";")
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
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
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def _scannet_checkbox_update(self, element):
        self.positive_input.set_disabled(element.value)
        self.positives = SCANNET200_CLASSES
        
    def _set_positives(self, element):
        self.positives = element.value.split(";")

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
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # embedding x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # embedding x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # embedding x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # embedding x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)