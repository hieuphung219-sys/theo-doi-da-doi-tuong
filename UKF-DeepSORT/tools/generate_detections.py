import os
import argparse
import time
import logging
import numpy as np
import cv2
import tensorflow as tf

# =============================================================================
# 1. LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ReID_Pipeline")

# =============================================================================
# 2. UTILS & PROFILING MODULE
# =============================================================================

class Profiler:
    """Class hỗ trợ đo thời gian thực thi."""
    def __init__(self):
        self.stats = {
            'io_time': 0.0,       # Thời gian đọc ảnh từ ổ cứng
            'preproc_time': 0.0,  # Thời gian cắt và resize bboxes
            'inference_time': 0.0,# Thời gian chạy model TFLite
            'total_frames': 0     # Tổng số lượng crop đã xử lý
        }
        self.start_t = 0

    def tic(self):
        """Bắt đầu đếm thời gian."""
        self.start_t = time.perf_counter()

    def toc(self, key):
        """Kết thúc đếm thời gian và cộng dồn vào thống kê."""
        duration = time.perf_counter() - self.start_t
        self.stats[key] += duration
        return duration

    def increment_frame(self, count=1):
        self.stats['total_frames'] += count

    def report(self):
        """In báo cáo hiệu năng ra màn hình."""
        total = self.stats['total_frames']
        if total == 0: 
            logger.warning("No frames processed to report.")
            return
        
        io_avg = (self.stats['io_time'] / total) * 1000
        pre_avg = (self.stats['preproc_time'] / total) * 1000
        inf_avg = (self.stats['inference_time'] / total) * 1000
        total_time = self.stats['io_time'] + self.stats['preproc_time'] + self.stats['inference_time']
        fps = total / total_time if total_time > 0 else 0

        logger.info("-" * 40)
        logger.info(f"PERFORMANCE REPORT (Processed {total} crops)")
        logger.info(f" > I/O Time       : {io_avg:.2f} ms/crop")
        logger.info(f" > Pre-proc Time  : {pre_avg:.2f} ms/crop")
        logger.info(f" > Inference Time : {inf_avg:.2f} ms/crop")
        logger.info(f" > Throughput     : {fps:.2f} crops/sec")
        logger.info("-" * 40)

def extract_image_patch(image, bbox, patch_shape):
    """
    Trích xuất và resize vùng ảnh chứa object.
    Đảm bảo bounding box nằm gọn trong ảnh và giữ nguyên aspect ratio.
    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # Cân chỉnh aspect ratio theo model yêu cầu
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # Chuyển (x, y, w, h) thành (x1, y1, x2, y2)
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int32)

    # Cắt (clip) box nếu bị tràn viền ảnh
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    
    # Kiểm tra box hợp lệ
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    # Resize về kích thước đầu vào của model
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image

# =============================================================================
# 3. REID MODULE (Feature Extractor)
# =============================================================================

class BaseEncoder:
    def __init__(self):
        self.image_shape = None
        self.feature_dim = None

    def normalize(self, features):
        """
        L2 Normalization: Bước CỰC KỲ quan trọng cho DeepSORT.
        Đưa các vector đặc trưng về không gian độ dài đơn vị (unit vector).
        Điều này đảm bảo phép tính Cosine Distance sau đó (trong Tracker) hoạt động đúng.
        """
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norm + 1e-10) # Thêm epsilon để tránh lỗi ZeroDivision

class TFLiteImageEncoder(BaseEncoder):
    """
    Engine chuyên biệt sử dụng TensorFlow Lite.
    """
    def __init__(self, checkpoint_filename):
        super().__init__()
        try:
            self.interpreter = tf.lite.Interpreter(model_path=checkpoint_filename)
        except Exception as e:
            logger.critical(f"Failed to load TFLite model at {checkpoint_filename}: {e}")
            raise e
            
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Parse cấu trúc model từ tflite
        self.image_shape = self.input_details[0]['shape'][1:] # (H, W, C)
        self.feature_dim = self.output_details[0]['shape'][1] # 128
        logger.info(f"Initialized TFLite Engine. Input: {self.image_shape}, Output: {self.feature_dim}")

    def __call__(self, data_x):
        """Thực hiện Inference trên list các ảnh."""
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        input_dtype = self.input_details[0]['dtype']

        # Xử lý tuần tự từng crop (TFLite thường config batch_size = 1)
        for i, img in enumerate(data_x):
            input_data = np.expand_dims(img, axis=0) # Thêm batch dimension -> (1, H, W, C)

            # Ép kiểu cho khớp với dtype của Tensor Input
            if input_data.dtype != input_dtype:
                input_data = input_data.astype(input_dtype)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            result = self.interpreter.get_tensor(self.output_details[0]['index'])
            out[i] = result[0]

        return self.normalize(out)

def create_box_encoder(model_filename):
    """
    Khởi tạo hàm encoder đóng gói toàn bộ quy trình: Extract box -> Inference.
    """
    encoder_engine = TFLiteImageEncoder(model_filename)
    image_shape = encoder_engine.image_shape

    def encoder(image, boxes, profiler=None):
        image_patches = []
        
        # 1. Pre-processing Loop
        if profiler: profiler.tic()
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                logger.debug(f"Invalid bounding box: {box}. Using noise instead.")
                # Nếu box lỗi/nằm ngoài ảnh, đổ nhiễu ngẫu nhiên để model không crash
                patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
            
        image_patches = np.asarray(image_patches)
        if profiler: profiler.toc('preproc_time')

        # 2. Inference Loop
        if profiler: profiler.tic()
        features = encoder_engine(image_patches)
        if profiler: profiler.toc('inference_time')
        
        return features

    return encoder

# =============================================================================
# 4. DATASET MODULE (MOT Format Loader)
# =============================================================================

class MOTDatasetLoader:
    """Class hỗ trợ đọc các sequence ảnh chuẩn MOTChallenge format."""
    def __init__(self, mot_dir, detection_mode='gt'):
        self.mot_dir = mot_dir
        self.mode = detection_mode
        self.sequences = [s for s in os.listdir(mot_dir) if os.path.isdir(os.path.join(mot_dir, s))]
        logger.info(f"Dataset loaded. Found {len(self.sequences)} sequences.")

    def get_loader(self, sequence):
        """Generator sinh ra dữ liệu (ảnh, array tọa độ) của từng frame."""
        sequence_dir = os.path.join(self.mot_dir, sequence)
        image_dir = os.path.join(sequence_dir, "img1")
        
        # Trỏ đến file Ground Truth hoặc file Detections
        det_file = os.path.join(sequence_dir, "gt/gt.txt") if self.mode == 'gt' else os.path.join(sequence_dir, "det/det.txt")
        
        if not os.path.exists(det_file):
            logger.warning(f"Detection file not found: {det_file}. Skipping sequence.")
            return None

        detections = np.loadtxt(det_file, delimiter=',')
        
        # Map Frame ID với Đường dẫn ảnh
        img_files = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))
        }

        min_frame = int(detections[:, 0].min())
        max_frame = int(detections[:, 0].max())
        
        logger.info(f"Sequence: {sequence} | Frames: {min_frame}-{max_frame} | Mode: {self.mode.upper()}")

        for frame_idx in range(min_frame, max_frame + 1):
            if frame_idx not in img_files: 
                continue
            
            mask = detections[:, 0].astype(np.int32) == frame_idx
            rows = detections[mask]
            
            if len(rows) == 0: 
                continue
            
            yield img_files[frame_idx], rows

# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def run_pipeline(args):
    """
    Quy trình chạy chính:
    Đọc dữ liệu -> Crop ảnh -> Trích xuất feature -> Lưu lại dưới định dạng .npy
    """
    profiler = Profiler()
    dataset = MOTDatasetLoader(args.mot_dir, args.detection_mode)
    
    # Khởi tạo encoder bằng TFLite
    encoder = create_box_encoder(args.model)
    
    output_dir = args.output_dir if args.output_dir else args.mot_dir
    os.makedirs(output_dir, exist_ok=True)

    for seq in dataset.sequences:
        loader = dataset.get_loader(seq)
        if loader is None: 
            continue
        
        detections_out = []
        
        # Process từng frame trong chuỗi
        for img_path, rows in loader:
            # Đo đạc thời gian I/O
            profiler.tic()
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            profiler.toc('io_time')
            
            if image is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            # Extract Features: rows[:, 2:6] chứa tọa độ x_min, y_min, width, height
            features = encoder(image, rows[:, 2:6].copy(), profiler)
            
            profiler.increment_frame(len(rows))

            # Ghép ma trận Detections gốc (MOT format) với Feature (128d) vừa tạo
            # Output array shape sẽ là (N, 10 + 128)
            detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]

        # Lưu lại kết quả
        if len(detections_out) > 0:
            out_file = os.path.join(output_dir, seq, "detections.npy")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            np.save(out_file, np.asarray(detections_out), allow_pickle=False)
            logger.info(f"Saved {len(detections_out)} detections to {out_file}")
        else:
            logger.warning(f"No detections generated for {seq}")

    # Xuất báo cáo thời gian chạy
    profiler.report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSORT Feature Generation Pipeline (TFLite Edition)")
    
    # Đã đổi default model sang .tflite
    parser.add_argument("--model", default="resources/networks/mars-small128.tflite", help="Path to TFLite ReID model (.tflite)")
    parser.add_argument("--mot_dir", required=True, help="Path to MOT dataset root folder")
    parser.add_argument("--output_dir", default=None, help="Output directory for generated .npy files")
    parser.add_argument("--detection_mode", default="gt", choices=["gt", "det"], help="Source of boxes: 'gt' or 'det'")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"TFLite Model file not found: {args.model}")
        exit(1)
        
    if not args.model.endswith('.tflite'):
        logger.error(f"This script now strictly supports only .tflite models. Received: {args.model}")
        exit(1)
        
    run_pipeline(args)