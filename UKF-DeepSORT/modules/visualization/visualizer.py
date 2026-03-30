import cv2
import time
import numpy as np
from collections import defaultdict

class Visualizer:
    def __init__(self, output_path: str = "output_test.mp4", fps: int = 30):
        self.output_path = output_path
        self.target_fps = fps
        self.writer = None
        self.last_time = time.time()
        
        self.color_cache = {}
        
        # [Task 75] Thêm dictionary để lưu trữ lịch sử tâm BBox (dải đuôi)
        # Giới hạn độ dài đuôi là 30 frames để tránh rối mắt và tràn RAM
        self.track_history = defaultdict(lambda: [])
        self.max_history_len = 30

    def _get_color(self, track_id: int) -> tuple:
        if track_id not in self.color_cache:
            idx = track_id * 3
            self.color_cache[track_id] = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return self.color_cache[track_id]

    def draw_and_save(self, frame: np.ndarray, tracks: list, frame_idx: int) -> np.ndarray:
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        fps_sys = 1.0 / delta_time if delta_time > 0 else 0
        
        # In log ra terminal để theo dõi tiến độ
        print(f"Processing frame {frame_idx:05d} | System FPS: {fps_sys:.2f}")

        if self.writer is None:
            h, w = frame.shape[:2]
            # Sử dụng mp4v cho định dạng .mp4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.target_fps, (w, h))

        for track in tracks:
            # Bỏ qua các track chưa confirmed hoặc vừa bị mất dấu (missed)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            x, y, w_box, h_box = map(int, track.to_tlwh())
            color = self._get_color(track.track_id)
            label = f"ID: {track.track_id}"

            # 2. Vẽ Bounding Box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            
            # 3. Vẽ Label (ID)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x, y - 20), (x + text_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Ghi frame vào file video output
        self.writer.write(frame)
        return frame

    def release(self):
        if self.writer is not None:
            self.writer.release()
            print(f"\n[HOÀN TẤT] Video kết quả đã được lưu tại: {self.output_path}")