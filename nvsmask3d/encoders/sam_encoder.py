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
            point_coords_new = point_coords_new[torch.randperm(point_coords_new.size(0))]
            # 选择前 self.num_selected_points 个点
            selected_coords = point_coords_new[:self.num_selected_points].unsqueeze(0)  # [shape: (1, num_selected_points, 2)]
            selected_coords = self.predictor.transform.apply_coords_torch(selected_coords, original_size=self.original_size)
            assert len(selected_coords.shape) == 3
            point_labels = torch.ones((1,self.num_selected_points if selected_coords.shape[1]> self.num_selected_points else selected_coords.shape[1]), device=self.model.device).int()
            assert len(point_labels.shape) == 2
            try:
                masks, scores, _ = self.predictor.predict_torch(
                    point_coords=selected_coords.to(self.model.device),
                    point_labels=point_labels.to(self.model.device),
                    multimask_output=False
                )
            except:
                import pdb; pdb.set_trace()

            if scores[0] > best_score:
                best_score = scores[0]
                best_mask = masks[0]

        # Ensure we have a valid best mask
        if best_mask is None:
            best_mask = torch.zeros(image_size, dtype=torch.bool, device=self.model.device)
        ## debug masks
        # from PIL import Image, ImageDraw, ImageOps
        # image = Image.open("/home/fangj1/Code/nerfstudio-nvsmask3d/tests/0.png").convert('RGB')  # Ensure 3-channel RGB

        # binary_mask = best_mask.squeeze() # [shape: (H, W)]
        # binary_mask_np = binary_mask.cpu().numpy().astype('uint8') * 255  # Convert to binary image (0 or 255)

        # binary_mask_img = Image.fromarray(binary_mask_np).convert("L")  # 转换为灰度图像 (L mode)

        # # 2. 将 mask 作为 alpha 通道叠加在原始图像上
        # # 将 mask 转换为透明度通道
        # alpha_mask = ImageOps.invert(binary_mask_img)  # 反转使得前景透明度较高
        # image_with_mask = image.copy()
        # image_with_mask.putalpha(alpha_mask)  # 将 mask 作为 alpha 通道叠加

        # # 3. 保存结果
        # image_with_mask.save("tests/1.png")
        # import pdb; pdb.set_trace()

        return best_mask
    
    @staticmethod
    def mask2box(mask: torch.Tensor):
        """
        Get bounding box from a mask.
        
        Arguments:
        mask -- a 3D tensor mask of shape [1, H, W]

        Returns:
        Bounding box coordinates (x1, y1, x2, y2) or None if mask is empty
        """
        # Ensure the mask is 3D and squeeze the first dimension
        assert mask.dim() == 3 and mask.shape[0] == 1, "Mask should be 3D with shape [1, H, W]"
        mask = mask.squeeze(0)  # Now mask shape is [H, W]

        # Find non-zero elements along both axes
        row = torch.any(mask, dim=0).nonzero(as_tuple=True)[0]
        col = torch.any(mask, dim=1).nonzero(as_tuple=True)[0]

        if len(row) == 0 or len(col) == 0:
            return None

        x1 = row.min().item()
        x2 = row.max().item()
        y1 = col.min().item()
        y2 = col.max().item()

        return x1, y1, x2 + 1, y2 + 1

    @staticmethod
    def mask2box_multi_level(mask_i: torch.Tensor, level: int, expansion_ratio: float =1):
        """
        Generate a bounding box with expansion based on level.

        Arguments:
        mask -- a 2D tensor mask
        level -- expansion level
        expansion_ratio -- expansion factor for the bounding box
        
        Returns:
        Expanded bounding box (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = SamNetWork.mask2box(mask_i)
        if x1 is None or y1 is None:
            return None
        
        if level == 0:
            return x1, y1, x2, y2
        
        shape = mask_i.shape
        x_exp = int(abs(x2 - x1) * expansion_ratio) * level
        y_exp = int(abs(y2 - y1) * expansion_ratio) * level
        
        return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)



if __name__ == "__main__":
    import torch
    from PIL import Image
    from torchvision import transforms
    from PIL import Image, ImageDraw, ImageOps

    config = SAMNetworkConfig()
    sam = SamNetWork(config)
    image = Image.open("/home/fangj1/Code/nerfstudio-nvsmask3d/tests/0.png").convert('RGB')  # Ensure 3-channel RGB
    
    # Define a transformation to convert the image to tensor and normalize it
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor with shape (3, H, W)
    ])
    
    # Apply the transformation to the image
    image_tensor = transform(image)
    
    # Add a batch dimension to the tensor (1, 3, H, W)
    image_tensor = image_tensor.unsqueeze(0).cuda()
    sam.set_image(image_tensor)
    selected_coords = torch.tensor([[[254, 196],
         [267, 212],
         [256, 215],
         [268, 204],
         [255, 197]]], device='cuda:0')
    # print(sam.embedding_dim)
    selected_coords = sam.predictor.transform.apply_coords_torch(selected_coords, original_size=sam.original_size)
    print(selected_coords.shape)
    image = Image.open("tests/0.png")
    
    # # 创建绘图对象
    # draw = ImageDraw.Draw(image)
    
    # # 将 PyTorch tensor 转换为 numpy 数组
    # coords = selected_coords.cpu().numpy().squeeze()
    # point_size = 5
    # # 在每个坐标上画红点
    # for x, y in coords:
    #     draw.ellipse([x-point_size, y-point_size, x+point_size, y+point_size], fill='red', outline='red')
    
    # # 保存图像
    # image.save("tests/1.png")
    # print(f"Marked image saved as tests/1.png")
    # import pdb; pdb.set_trace()
    point_labels = torch.ones((1,5), device='cuda:0').int()
    masks, scores, _ = sam.predictor.predict_torch(
    point_coords=selected_coords.to(sam.model.device),
    point_labels=point_labels.to(sam.model.device),
    multimask_output=False
    )
    binary_mask = masks.squeeze()[torch.argmax(scores)] # [shape: (H, W)]
    binary_mask_np = binary_mask.cpu().numpy().astype('uint8') * 255  # Convert to binary image (0 or 255)

    binary_mask_img = Image.fromarray(binary_mask_np).convert("L")  # 转换为灰度图像 (L mode)

    # 2. 将 mask 作为 alpha 通道叠加在原始图像上
    # 将 mask 转换为透明度通道
    alpha_mask = ImageOps.invert(binary_mask_img)  # 反转使得前景透明度较高
    image_with_mask = image.copy()
    image_with_mask.putalpha(alpha_mask)  # 将 mask 作为 alpha 通道叠加

    # 3. 保存结果
    image_with_mask.save("tests/1.png")
    print(f"Image with mask saved as tests/1.png"
    )
    import pdb; pdb.set_trace()
