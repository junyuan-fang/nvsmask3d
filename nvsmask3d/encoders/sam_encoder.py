from dataclasses import dataclass, field
from typing import Tuple, Type
import torch
from segment_anything import sam_model_registry  # Assuming SAM is wrapped as a class
from segment_anything import SamPredictor
from nvsmask3d.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig

@dataclass
class SAMNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: SamNetWork)
    sam_model_type: str = "default"  # "vit_h"
    sam_checkpoint: str = "nvsmask3d/checkpoints/sam_vit_h_4b8939.pth"
    num_random_rounds: int = 10  # Add parameters for randomness in mask prediction
    num_selected_points: int = 5  # Number of selected points for mask prediction



class SamNetWork(BaseImageEncoder):
    def __init__(
        self, 
        config: SAMNetworkConfig,
        device: str = "cuda",
    ):
        super().__init__()
        self.config = config
        self.model = sam_model_registry[self.config.sam_model_type](checkpoint=self.config.sam_checkpoint) # Load SAM model
        self.model.to(device)
        self.predictor = SamPredictor(self.model)
        self.num_random_rounds = config.num_random_rounds  # Access num_random_rounds from config
        self.num_selected_points = config.num_selected_points  # Access num_selected_points from config
        self.num_levels = 3  # Number of levels for bounding box expansion
    @property
    def name(self) -> str:
        return "SegmentAnything"

    @torch.no_grad()
    def set_image(self, image: torch.Tensor):
        """
        Set an image in tensor format for SAM to use.
        Arguments:
        image -- tensor of shape (1, 3, H, W)
        """
        _, _, H, W = image.shape
        self.original_size =  (H,W)
        transformed_image = self.predictor.transform.apply_image_torch(image)
        self.predictor.set_torch_image(transformed_image, original_image_size=self.original_size)

    @torch.no_grad()
    def get_best_mask(self, point_coords: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Returns the best mask and multi-level crops from SAM predictions using random sampling of points.
        
        Arguments:
        point_coords -- point coordinates to be used for mask generation (tensor) [shape: (N, 2), where N is the number of points]

        Returns:
        best_mask -- the mask with the highest confidence score [shape: (H, W), where H and W are the image dimensions]
        """
        best_score = 0
        best_mask = None
        image_size = (self.predictor.input_size[-2], self.predictor.input_size[-1])  # Image size [shape: (H, W)]

        point_coords_new = torch.clone(point_coords)  # [shape: (N, 2), where N is the number of points]
        #point_coords_new[:, [0, 1]] = point_coords_new[:, [1, 0]]  # [shape: (N, 2)] Swap X and Y 

        # Perform random rounds and select the best mask based on confidence scores
        for i in range(self.num_random_rounds):
            indices = torch.randperm(point_coords_new.size(0))[:self.num_selected_points]
            selected_coords = point_coords_new[indices]
            selected_coords = self.predictor.transform.apply_coords_torch(selected_coords, original_size=self.original_size).unsqueeze(0)

            masks, scores, _ = self.predictor.predict_torch(
                point_coords=selected_coords.to(self.model.device),
                point_labels=torch.ones((1,self.num_selected_points), device=self.model.device).int(),
                multimask_output=False
            )

            if scores[0] > best_score:
                best_score = scores[0]
                best_mask = masks[0]

        # Ensure we have a valid best mask
        if best_mask is None:
            best_mask = torch.zeros(image_size, dtype=torch.bool, device=self.model.device)

        return best_mask
    
    @staticmethod
    def mask2box(mask: torch.Tensor):
        """
        Get bounding box from a mask.
        
        Arguments:
        mask -- a 2D tensor mask

        Returns:
        Bounding box coordinates (x1, y1, x2, y2)
        """
        row = torch.nonzero(mask.sum(axis=0))[:, 0]
        if len(row) == 0:
            return None
        x1 = row.min().item()
        x2 = row.max().item()
        col = torch.nonzero(mask.sum(axis=1))[:, 0]
        y1 = col.min().item()
        y2 = col.max().item()
        return x1, y1, x2 + 1, y2 + 1

    @staticmethod
    def mask2box_multi_level(mask: torch.Tensor, level: int, expansion_ratio: float =1):
        """
        Generate a bounding box with expansion based on level.

        Arguments:
        mask -- a 2D tensor mask
        level -- expansion level
        expansion_ratio -- expansion factor for the bounding box
        
        Returns:
        Expanded bounding box (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = SamNetWork.mask2box(mask)
        if x1 is None or y1 is None:
            return None
        
        if level == 0:
            return x1, y1, x2, y2
        
        shape = mask.shape
        x_exp = int(abs(x2 - x1) * expansion_ratio) * level
        y_exp = int(abs(y2 - y1) * expansion_ratio) * level
        
        return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)



if __name__ == "__main__":
    config = SAMNetworkConfig()
    sam = SamNetWork(config)
    
    print(sam.name)
    # print(sam.embedding_dim)