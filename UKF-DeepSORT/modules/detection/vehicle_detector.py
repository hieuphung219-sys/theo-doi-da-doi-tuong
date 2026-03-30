import cv2
import numpy as np
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8m.pt", conf_thresh=0.5, iou_thresh=0.4):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_classes = [0, 1, 2, 3, 5, 7] 

        # --- BẢNG DỊCH MÃ (COCO -> MOT/KITTI) ---
        self.coco_to_mot_map = {
            0: 1,  # YOLO (Person) -> MOT (Pedestrian = 1)
            1: 4,  # YOLO (Bicycle) -> MOT (Cyclist = 4)
            2: 3,  # YOLO (Car) -> MOT (Car = 3)
            3: 12, # YOLO (Motorcycle) -> MOT (Misc = 12)
            5: 3,  # YOLO (Bus) -> MOT (Car = 3)
            7: 3   # YOLO (Truck) -> MOT (Car = 3)
        }

    def _merge_person_and_bicycle(self, persons, bicycles):
        merged_results = []
        used_persons = set()
        used_bicycles = set()

        for p_idx, p in enumerate(persons):
            px1, py1, px2, py2, pconf, pcls = p
            p_area = (px2 - px1) * (py2 - py1)
            p_bottom_center = ((px1 + px2) / 2, py2)
            
            best_match_idx = -1
            best_match_dist = float('inf')

            for b_idx, b in enumerate(bicycles):
                if b_idx in used_bicycles:
                    continue
                    
                bx1, by1, bx2, by2, bconf, bcls = b
                
                inter_x1 = max(px1, bx1)
                inter_y1 = max(py1, by1)
                inter_x2 = min(px2, bx2)
                inter_y2 = min(py2, by2)
                
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                ioa_person = inter_area / p_area if p_area > 0 else 0
                
                is_intersecting = ioa_person > 0.3
                is_horizontally_aligned = (bx1 - 20) <= p_bottom_center[0] <= (bx2 + 20)
                
                if is_intersecting and is_horizontally_aligned:
                    b_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                    dist = (p_bottom_center[0] - b_center[0])**2 + (p_bottom_center[1] - b_center[1])**2
                    
                    if dist < best_match_dist:
                        best_match_dist = dist
                        best_match_idx = b_idx

            if best_match_idx != -1:
                b = bicycles[best_match_idx]
                bx1, by1, bx2, by2, bconf, bcls = b
                
                new_x1 = min(px1, bx1)
                new_y1 = min(py1, by1)
                new_x2 = max(px2, bx2)
                new_y2 = max(py2, by2)
                new_conf = (pconf + bconf) / 2.0 
                # Gán mã 1 (Bicycle COCO) để lát nữa phía dưới tự động dịch thành 4 (Cyclist MOT)
                merged_results.append([new_x1, new_y1, new_x2, new_y2, new_conf, 1])
                
                used_persons.add(p_idx)
                used_bicycles.add(best_match_idx)

        for p_idx, p in enumerate(persons):
            if p_idx not in used_persons:
                merged_results.append(p)
                
        for b_idx, b in enumerate(bicycles):
            if b_idx not in used_bicycles:
                merged_results.append(b)

        return merged_results

    def detect(self, frame):
        results = self.model.predict(
            source=frame, 
            classes=self.target_classes, 
            conf=self.conf_thresh, 
            iou=self.iou_thresh, 
            verbose=False 
        )
        
        persons = []
        bicycles = []
        others = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id_coco = int(box.cls[0].cpu().numpy())
                det_array = [x1, y1, x2, y2, conf, cls_id_coco]
                
                if cls_id_coco == 0:
                    persons.append(det_array)
                elif cls_id_coco == 1:
                    bicycles.append(det_array)
                else:
                    others.append(det_array)
                    
        # Chạy logic gộp (tọa độ xyxy)
        if persons and bicycles:
            merged_cyclists_and_leftovers = self._merge_person_and_bicycle(persons, bicycles)
            final_raw_boxes = merged_cyclists_and_leftovers + others
        else:
            final_raw_boxes = persons + bicycles + others
            
        detections = []
        for box in final_raw_boxes:
            x1, y1, x2, y2, conf, cls_id_coco = box
            w = x2 - x1
            h = y2 - y1
            
            # --- THỰC HIỆN DỊCH MÃ ---
            cls_id_mot = self.coco_to_mot_map.get(int(cls_id_coco), 12)
            
            detections.append([x1, y1, w, h, conf, cls_id_mot])
                
        return np.array(detections)