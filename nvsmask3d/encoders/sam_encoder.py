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
        image -- tensor of shape ( 3, H, W)
        """
        image = (image * 255).unsqueeze(0) #(1,3,H,W)
        _, _, H, W = image.shape
        self.original_size =  (H,W)
        transformed_image = self.predictor.transform.apply_image_torch(image)
        self.predictor.set_torch_image(transformed_image, original_image_size=self.original_size)

    @torch.no_grad()
    def get_best_mask(self, point_coords: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Returns the best mask and multi-level crops from SAM predictions using random sampling of points.
        
        Arguments:
        point_coords -- point coordinates to be used for mask generation (tensor) [shape: (N, 2), where N is the number of points]  for v, u

        Returns:
        best_mask -- the mask with the highest confidence score [shape: (H, W), where H and W are the image dimensions]
        """
        best_score = 0
        best_mask = None
        H = self.original_size[0]
        W = self.original_size[1]
        #########debug ###########
        # sparse_map = torch.zeros((H, W, 3), dtype=torch.float32, device="cuda")
        # sparse_map[point_coords[:, 0], point_coords[:, 1]] = 1
        # from nvsmask3d.utils.utils import save_img
        # save_img(sparse_map, f"tests/sparse_map.png")
        ###########################
        point_coords[:, [0, 1]] = point_coords[:, [1, 0]] #convert to u,v

        # Perform random rounds and select the best mask based on confidence scores
        for i in range(self.num_random_rounds):
            torch.manual_seed(int(i))
            # 选择前 self.num_selected_points 个点
            selected_coords = point_coords[torch.randperm(point_coords.size(0))]
            selected_coords = selected_coords[:self.num_selected_points].unsqueeze(0)  # [shape: (1, num_selected_points, 2)]
            #########debug ###########
            # sparse_map = torch.zeros((H, W, 3), dtype=torch.float32, device="cuda")
            # sparse_map[selected_coords[0,:, 0], selected_coords[0,:, 1]] = 1
            # from nvsmask3d.utils.utils import save_img
            # save_img(sparse_map, f"tests/selected_sparse_map.png")
            # save_img(img.permute(1,2,0), f"tests/selected_img.png")#CHW
            #import pdb; pdb.set_trace()
            ###########################
            selected_coords = self.predictor.transform.apply_coords_torch(selected_coords, original_size=self.original_size)
            assert len(selected_coords.shape) == 3
            point_labels = torch.ones((1,self.num_selected_points if selected_coords.shape[1]> self.num_selected_points else selected_coords.shape[1]), device=self.model.device).int()
            assert len(point_labels.shape) == 2
            try:
                masks, scores, _ = self.predictor.predict_torch(#BxCxHxW format     scores torch.Size([1, 3])
                    point_coords=selected_coords.to(self.model.device),
                    point_labels=point_labels.to(self.model.device),
                    multimask_output=False
                )
                #########debug ###########
                # sparse_map = torch.zeros((H, W, 3), dtype=torch.float32, device="cuda")
                # sparse_map[masks.squeeze()] = 1
                # from nvsmask3d.utils.utils import save_img
                # save_img(sparse_map, f"tests/sam_sparse_map.png")
                # import pdb; pdb.set_trace()
                ###########################
            except:
                import pdb; pdb.set_trace()
            if scores.squeeze() > best_score:#multimask_output=False
                best_score = scores.squeeze().item()
                best_mask = masks.squeeze()  # [shape: (H, W)]
            # if scores.squeeze()[0] > best_score:multimask_output=False
            #     best_score = scores.squeeze()[0]
            #     best_mask = masks[:, 0, :, :]  # [shape: (1, H, W)]
        # Ensure we have a valid best mask
        if best_mask is None:
            best_mask = torch.zeros(self.original_size, dtype=torch.bool, device=self.model.device)
        return best_mask #[shape: (H, W)]
    
    @staticmethod
    def mask2box(mask: torch.Tensor):
        """
        Get bounding box from a mask.
        
        Arguments:
        mask -- a 3D tensor mask of shape [H, W]

        Returns:
        Bounding box coordinates (x1, y1, x2, y2) or None if mask is empty
        """
        # Ensure the mask is 3D and squeeze the first dimension
        assert mask.dim() == 2  #"Mask should be 3D with shape [H, W]"

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
    #validate the code
    config = SAMNetworkConfig()
    sam = SamNetWork(config)

    # Step 1: 加载图像
    image = Image.open("/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/replica/office0/color/7.jpg").convert('RGB')  # 确保是RGB三通道

    # Step 2: 将图像转换为Tensor格式
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为tensor，形状为(3, H, W)
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.cuda()

    # Step 3: 使用set_image设置图像
    sam.set_image(image_tensor)

    # Step 4: 定义坐标并进行预测
    point_coords = torch.tensor([[210, 238],
        [203, 252],
        [214, 257],
        [189, 245],
        [228, 250]], device='cuda:0')
    point_coords_new = torch.clone(point_coords)
    # 绘制红点
    image = Image.open("/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/replica/office0/color/7.jpg")
    draw = ImageDraw.Draw(image)
    coords = point_coords.cpu().numpy()
    point_size = 5
    for y, x in coords:
        draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], fill='red', outline='red')
    image.save("tests/1.png")
    print(f"Marked image saved as tests/1.png")
    # Step 5: 进行遮罩预测
    point_labels = torch.ones((5), device='cuda:0').int()
    masks, scores, _ = sam.predictor.predict(
        point_coords=point_coords.cpu().numpy(),  # `predict` 需要 numpy 格式的坐标
        point_labels=point_labels.cpu().numpy(),
        multimask_output=True
    )

    # Step 6: 处理输出遮罩
    binary_mask = masks.squeeze()[0]  # [shape: (H, W)]
    binary_mask_np = (binary_mask > 0).astype('uint8') * 255  # 将 True 转换为 255，False 转换为 0

    # 保存纯掩膜图像
    pure_mask_img = Image.fromarray(binary_mask_np, mode='L')  # 转换为灰度图像(L模式)
    pure_mask_img.save("tests/pure_mask.png")  # 保存为纯掩膜图像

    # Step 7: 将mask与原始图像叠加
    binary_mask_img = Image.fromarray(binary_mask_np, mode='L')  # 转换为灰度图像(L模式)
    alpha_mask = ImageOps.invert(binary_mask_img)  # 反转使得前景透明度较高
    image_with_mask = image.copy()
    image_with_mask.putalpha(alpha_mask)  # 将mask作为alpha通道叠加

    # Step 8: 保存结果图像
    image_with_mask.save("tests/2.png")
    binary_mask_img.save("tests/3.png")  # 如果需要保存灰度图像
    print(f"Image with mask saved as tests/2.png")

    import pdb; pdb.set_trace()
