import sys
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# 1. Chỉ định đường dẫn tới thư mục fast-reid
current_dir = os.path.dirname(os.path.abspath(__file__))
fast_reid_path = os.path.join(current_dir, 'fast-reid')

# Đưa fast-reid vào hệ thống đường dẫn của Python
if fast_reid_path not in sys.path:
    sys.path.insert(0, fast_reid_path)

# 2. Import module từ Fast-ReID (Bắt buộc phải nằm dưới phần sys.path ở trên)
try:
    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model
except ImportError:
    print("Lỗi: Không tìm thấy thư viện fastreid. Hãy kiểm tra lại thư mục fast-reid.")


class PyTorchFeatureExtractor:
    def __init__(self, model_name, model_path, device='cuda'):
        """
        Khởi tạo bộ trích xuất đặc trưng ngoại hình (Re-ID).
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 1. Thiết lập cấu hình chuẩn của sbs_R50-ibn
        self.cfg = get_cfg()
        config_file = os.path.join(fast_reid_path, "configs", "VeRi", "sbs_R50-ibn.yml")
        
        if os.path.exists(config_file):
            self.cfg.merge_from_file(config_file)
        else:
            print(f"Cảnh báo: Không tìm thấy {config_file}. Đang chạy với cấu hình mặc định.")
            
        self.cfg.MODEL.DEVICE = self.device
        
        # 2. Khởi tạo kiến trúc mạng (Tự động tạo các lớp IBN, Non-Local...)
        self.model = build_model(self.cfg)
        self.model.eval()
        
        # 3. Nạp trọng số thủ công (Tránh lỗi thư viện)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số Re-ID tại: {model_path}")
            
        weights = torch.load(model_path, map_location=self.device)
        if 'model' in weights:
            weights = weights['model']
        elif 'state_dict' in weights:
            weights = weights['state_dict']
            
        # Sửa lỗi key nếu model được train bằng DataParallel (có tiền tố 'module.')
        new_state_dict = {k.replace('module.', ''): v for k, v in weights.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        
        # 4. Transform chuẩn ImageNet + Kích thước cho VeRi (256x256)
        self.transform = T.Compose([
            T.Resize((256, 256)), # Bắt buộc phải resize để các crop đưa vào chung 1 batch
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    def extract(self, img_crops):
        """
        Trích xuất đặc trưng cho một danh sách các ảnh xe.
        Input: img_crops (List[numpy.ndarray]) - Danh sách ảnh cắt ra (crop) định dạng BGR của OpenCV.
        Output: numpy.ndarray kích thước (N, D) - Với N là số ảnh, D là số chiều đặc trưng.
        """
        if not img_crops:
            return np.array([])

        batch_tensors = []
        for img in img_crops:
            # Chuyển hệ màu từ BGR (OpenCV) sang RGB
            img_rgb = img[:, :, ::-1]
            img_pil = Image.fromarray(img_rgb)
            
            # Biến đổi thành tensor
            img_tensor = self.transform(img_pil)
            batch_tensors.append(img_tensor)

        # Gom N ảnh lại thành 1 tensor duy nhất (Batching) có shape: (N, C, H, W)
        batch_tensors = torch.stack(batch_tensors).to(self.device)

        # Đẩy qua mạng Neural Network
        with torch.no_grad():
            features = self.model(batch_tensors)
        
        # L2 Normalize theo chuẩn để tính Cosine Distance cực kỳ quan trọng cho DeepSORT
        feature_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # Trả về ma trận numpy (N, D)
        return feature_norm.cpu().numpy()