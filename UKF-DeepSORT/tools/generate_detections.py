import os
import argparse
import time
import logging
import numpy as np
import cv2
import torch
import sys

# Thêm đường dẫn thư mục gốc dự án vào hệ thống để Python tìm được thư mục modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modules.reid.feature_extractor import PyTorchFeatureExtractor

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ReID_Generator")

# =============================================================================
# DATASET LOADER (KITTI / MOT Format)
# =============================================================================

class MOTDatasetLoader:
    """Class đọc dữ liệu chuẩn MOT/KITTI"""
    def __init__(self, sequence_dir):
        self.sequence_dir = sequence_dir
        
        # Đọc danh sách ảnh
        self.image_dir = os.path.join(sequence_dir, "img1")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.image_dir}")
            
        self.image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png'))
        }
        
        # Ưu tiên đọc từ file detection (det.txt), nếu không thì dùng Ground Truth (gt.txt)
        det_path = os.path.join(sequence_dir, "det/det.txt")
        if not os.path.exists(det_path):
            det_path = os.path.join(sequence_dir, "gt/gt.txt")
            
        if not os.path.exists(det_path):
            raise FileNotFoundError(f"Không tìm thấy file tọa độ (det/gt.txt) trong: {sequence_dir}")
            
        self.detections = np.loadtxt(det_path, delimiter=',')
        logger.info(f"Đã load {os.path.basename(sequence_dir)}. Tìm thấy {len(self.image_filenames)} ảnh và {len(self.detections)} boxes.")

    def get_loader(self):
        """Generator sinh ra (frame_id, img_path, bboxes)"""
        min_frame = int(self.detections[:, 0].min())
        max_frame = int(self.detections[:, 0].max())
        
        for frame_idx in range(min_frame, max_frame + 1):
            if frame_idx not in self.image_filenames: 
                continue
            
            # Lọc tọa độ của frame hiện tại
            mask = self.detections[:, 0].astype(np.int32) == frame_idx
            rows = self.detections[mask]
            
            if len(rows) == 0: 
                continue
            
            yield frame_idx, self.image_filenames[frame_idx], rows

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def extract_image_patches(image, bboxes):
    """Cắt ảnh từ Bounding Box [x, y, w, h]"""
    patches = []
    height, width = image.shape[:2]
    
    for box in bboxes:
        x, y, w, h = box
        
        # Chuyển x, y, w, h thành tọa độ cắt
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Chặn biên tránh cắt ra ngoài lề
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        patch = image[y1:y2, x1:x2]
        if patch.size > 0:
            patches.append(patch)
        else:
            # Nếu box lỗi, tạo 1 nhiễu rỗng để mô hình không crash
            patches.append(np.zeros((64, 64, 3), dtype=np.uint8))
            
    return patches

def process_sequence(extractor, sequence_dir):
    """Hàm xử lý cho 1 sequence độc lập"""
    try:
        dataset = MOTDatasetLoader(sequence_dir)
        loader = dataset.get_loader()
    except Exception as e:
        logger.error(f"Bỏ qua {os.path.basename(sequence_dir)}: {e}")
        return

    detections_out = []
    start_time = time.time()
    total_boxes = 0
    seq_name = os.path.basename(sequence_dir)

    # Chạy vòng lặp trích xuất
    for frame_idx, img_path, rows in loader:
        print(f"[{seq_name}] Đang trích xuất frame {frame_idx:05d}...", end='\r')
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
            
        # Tọa độ gốc chuẩn MOT là x, y, w, h nằm ở cột index 2 đến 5
        bboxes = rows[:, 2:6].copy()
        
        # 1. Cắt các mảnh ảnh nhỏ
        patches = extract_image_patches(image, bboxes)
        
        # 2. Đẩy qua mô hình GPU (Batching)
        features = extractor.extract(patches) if len(patches) > 0 else []
        total_boxes += len(features)
        
        # 3. Nối vector đặc trưng vào sau mảng tọa độ gốc
        for row, feature in zip(rows, features):
            detections_out.append(np.r_[row, feature])

    print() # Xuống dòng cho sequence tiếp theo
    
    # Lưu file npy
    if len(detections_out) > 0:
        output_file = os.path.join(sequence_dir, "detections.npy")
        np.save(output_file, np.asarray(detections_out), allow_pickle=False)
        
        elapsed = time.time() - start_time
        logger.info(f"HOÀN TẤT {seq_name}! Đã xử lý {total_boxes} boxes trong {elapsed:.2f} giây.")
    else:
        logger.warning(f"Không tìm thấy dữ liệu để lưu cho {seq_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo file detections.npy hàng loạt bằng Fast-ReID")
    # Thay đổi tham số: Chỉ cần trỏ đến thư mục gốc chứa TẤT CẢ các KITTI
    parser.add_argument("--dataset_root", default="datasets/KITTI_MOT", help="Thư mục gốc chứa các sequence (VD: datasets/KITTI_MOT)")
    parser.add_argument("--model", default="modules/reid/weights/veri_sbs_R50-ibn.pt", help="Đường dẫn file trọng số Re-ID (.pt)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"Không tìm thấy file trọng số tại: {args.model}")
        exit(1)
        
    if not os.path.exists(args.dataset_root):
        logger.error(f"Không tìm thấy thư mục dataset gốc tại: {args.dataset_root}")
        exit(1)

    # 1. Lọc ra danh sách các sequence
    all_sequences = [d for d in os.listdir(args.dataset_root) 
                     if os.path.isdir(os.path.join(args.dataset_root, d)) and d.startswith("KITTI-")]
    all_sequences.sort()

    if len(all_sequences) == 0:
        logger.error(f"Không tìm thấy thư mục KITTI nào trong '{args.dataset_root}'.")
        exit(1)

    logger.info(f"Đã tìm thấy {len(all_sequences)} sequences. Chuẩn bị khởi tạo mô hình AI...")

    # 2. Khởi tạo mô hình Fast-ReID CHỈ MỘT LẦN DUY NHẤT
    extractor = PyTorchFeatureExtractor(
        model_name='sbs_R50-ibn', 
        model_path=args.model, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 3. Lặp qua tất cả các thư mục
    for seq_name in all_sequences:
        seq_dir = os.path.join(args.dataset_root, seq_name)
        process_sequence(extractor, seq_dir)
        
    logger.info("============== TOÀN BỘ QUÁ TRÌNH HOÀN TẤT ==============")