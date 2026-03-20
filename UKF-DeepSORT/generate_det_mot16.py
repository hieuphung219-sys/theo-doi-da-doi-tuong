import os
import cv2
import glob
from tqdm import tqdm
from vehicle_detector import VehicleDetector

def generate_mot_det_file(input_img_dir, output_det_dir):
    """
    Hàm đọc ảnh từ thư mục đầu vào, chạy YOLO và xuất ra file det.txt tại thư mục đầu ra.
    
    Args:
        input_img_dir: Đường dẫn tới thư mục chứa các frame ảnh (VD: 'dataset/KITTI-0000/image_02')
        output_det_dir: Đường dẫn tới thư mục muốn lưu file det.txt (VD: 'data/mot_format/KITTI-0000/det')
    """
    # 1. Tự động tạo cây thư mục đầu ra nếu nó chưa tồn tại
    os.makedirs(output_det_dir, exist_ok=True)
    det_file_path = os.path.join(output_det_dir, "det.txt")
    
    # 2. Lấy danh sách tất cả các ảnh và sắp xếp theo thứ tự bảng chữ cái
    image_paths = glob.glob(os.path.join(input_img_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_img_dir, "*.png"))
    image_paths.sort()
    
    if len(image_paths) == 0:
        print(f"Không tìm thấy ảnh nào trong thư mục: {input_img_dir}")
        return

    # 3. Khởi tạo module detector (từ file vehicle_detector.py)
    print("Đang tải mô hình YOLO...")
    detector = VehicleDetector()
    
    # 4. Bắt đầu duyệt qua từng frame và ghi file
    print(f"Bắt đầu xử lý {len(image_paths)} frame...")
    print(f"Kết quả sẽ được lưu tại: {det_file_path}")
    
    with open(det_file_path, 'w') as f:
        for frame_id, img_path in enumerate(tqdm(image_paths), start=1):
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
                
    print("Hoàn thành xuất file detection!")

if __name__ == "__main__":
    # --- BẠN THAY ĐỔI ĐƯỜNG DẪN Ở ĐÂY ---
    
    # 1. Đường dẫn đến thư mục chứa ảnh gốc của sequence
    # (Tùy thuộc vào việc ảnh của bạn đuôi jpg/png nằm ở đâu)
    INPUT_IMAGE_DIR = "data/mot_format/KITTI-0000/img1" 
    
    # 2. Đường dẫn đích mà bạn muốn lưu file det.txt
    OUTPUT_DET_DIR = "data/mot_format/KITTI-0000/det"
    
    generate_mot_det_file(
        input_img_dir=INPUT_IMAGE_DIR, 
        output_det_dir=OUTPUT_DET_DIR
    )