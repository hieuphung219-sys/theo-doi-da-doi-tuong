import os
import cv2
import glob
from tqdm import tqdm
from vehicle_detector import VehicleDetector

def generate_mot_det_file(detector, seq_name, input_img_dir, output_det_dir):
    """
    Hàm xử lý cho 1 sequence. Nhận mô hình YOLO (detector) đã được load sẵn từ ngoài.
    """
    os.makedirs(output_det_dir, exist_ok=True)
    det_file_path = os.path.join(output_det_dir, "det.txt")
    
    image_paths = glob.glob(os.path.join(input_img_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_img_dir, "*.png"))
    image_paths.sort()
    
    if len(image_paths) == 0:
        print(f"[-] Bỏ qua {seq_name}: Không tìm thấy ảnh trong {input_img_dir}")
        return

    print(f"[*] Đang xử lý {seq_name} ({len(image_paths)} frames)...")
    
    with open(det_file_path, 'w') as f:
        # Dùng tqdm để hiển thị thanh tiến trình cho từng sequence
        for frame_id, img_path in enumerate(tqdm(image_paths, desc=seq_name, leave=False), start=1):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            # Chạy YOLO detection
            detections = detector.detect(frame)
            
            # Ghi từng bounding box vào file
            for det in detections:
                x_min, y_min, w, h, conf, cls_id = det
                line = f"{frame_id},-1,{x_min:.2f},{y_min:.2f},{w:.2f},{h:.2f},{conf:.4f},{int(cls_id)},-1,-1\n"
                f.write(line)

if __name__ == "__main__":
    # --- ĐƯỜNG DẪN THƯ MỤC GỐC CHỨA TOÀN BỘ DATASET ---
    # Giả sử cấu trúc của bạn là: datasets/KITTI-0000, datasets/KITTI-0001,...
    DATASET_ROOT = "datasets/KITTI_MOT"
    
    # 1. Lấy danh sách tất cả các thư mục có chữ "KITTI-" bên trong DATASET_ROOT
    all_sequences = [d for d in os.listdir(DATASET_ROOT) 
                     if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith("KITTI-")]
    all_sequences.sort()
    
    if len(all_sequences) == 0:
        print(f"Không tìm thấy thư mục KITTI nào trong '{DATASET_ROOT}'. Vui lòng kiểm tra lại đường dẫn!")
        exit(1)
        
    print(f"Tìm thấy {len(all_sequences)} sequences. Đang tải mô hình YOLOv8m...")
    
    # 2. Khởi tạo YOLO CHỈ MỘT LẦN DUY NHẤT ở đây để tối ưu hiệu năng
    detector = VehicleDetector(model_path="yolov8m.pt")
    
    # 3. Lặp qua từng sequence và xử lý
    for seq_name in all_sequences:
        seq_dir = os.path.join(DATASET_ROOT, seq_name)
        
        input_img_dir = os.path.join(seq_dir, "img1")
        output_det_dir = os.path.join(seq_dir, "det")
        
        generate_mot_det_file(
            detector=detector, 
            seq_name=seq_name,
            input_img_dir=input_img_dir, 
            output_det_dir=output_det_dir
        )
        
    print("\n[+] HOÀN THÀNH!")