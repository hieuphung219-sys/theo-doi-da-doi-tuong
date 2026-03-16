import cv2
import numpy as np
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8m.pt", conf_thresh=0.5, iou_thresh=0.4):
        """
        Khởi tạo module Detector.
        """
        # Load mô hình YOLO
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Chỉ giữ lại các class liên quan đến xe cộ (dựa trên cấu hình trước đó của bạn)
        # 0: person (tuỳ chọn), 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [0, 1, 2, 3, 5, 7]

    def detect(self, frame):
        """
        Hàm nhận vào 1 frame ảnh gốc, chạy YOLO và trả về tọa độ Bounding Box.
        
        Args:
            frame: Ảnh numpy array (đọc từ cv2)
            
        Returns:
            detections: Danh sách các bounding box theo định dạng [x_min, y_min, w, h, conf, class_id] 
                        (hoặc định dạng bạn cần để xuất ra file MOT/đưa vào DeepSORT).
        """
        # Chạy model trên frame hiện tại
        results = self.model.predict(
            source=frame, 
            classes=self.target_classes, 
            conf=self.conf_thresh, 
            iou=self.iou_thresh, 
            verbose=False # Tắt log in ra console liên tục cho từng frame
        )
        
        detections = []
        
        # Xử lý kết quả trả về từ YOLOv8
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Trích xuất tọa độ [x_min, y_min, x_max, y_max]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Chuyển đổi sang [x_min, y_min, width, height] nếu cần cho DeepSORT
                w = x2 - x1
                h = y2 - y1
                
                detections.append([x1, y1, w, h, conf, cls_id])
                
        return np.array(detections)

# --- Đoạn code test nhanh (chỉ chạy khi gọi trực tiếp file này) ---
if __name__ == "__main__":
    detector = VehicleDetector()
    # Test thử với 1 frame ngẫu nhiên
    # test_frame = cv2.imread("test_image.jpg")
    # bboxes = detector.detect(test_frame)
    # print(bboxes)