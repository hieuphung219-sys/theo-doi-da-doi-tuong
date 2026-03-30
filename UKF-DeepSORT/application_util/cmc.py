import cv2
import numpy as np

class CameraMotionCompensator:
    """
    Class xử lý bù trừ chuyển động của camera (CMC) sử dụng Optical Flow.
    """
    def __init__(self, max_corners=200, quality_level=0.01, min_distance=30):
        self.prev_gray = None
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

    def compute_affine_matrix(self, frame):
        """
        Tính toán ma trận dịch chuyển Affine H (2x3) giữa frame trước và frame hiện tại.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = None
        
        if self.prev_gray is not None:
            # 1. Tìm điểm đặc trưng trên bối cảnh
            prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray, 
                maxCorners=self.max_corners, 
                qualityLevel=self.quality_level, 
                minDistance=self.min_distance
            )
            
            if prev_pts is not None and len(prev_pts) > 0:
                # 2. Theo dõi sự di chuyển của các điểm
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_frame, prev_pts, None)
                
                good_old = prev_pts[status == 1]
                good_new = curr_pts[status == 1]
                
                if len(good_new) >= 4: # Cần ít nhất 4 điểm để ước lượng Affine 2D
                    # 3. Tính ma trận biến đổi
                    H, _ = cv2.estimateAffinePartial2D(good_old, good_new)
                    
        # Cập nhật trạng thái cho frame tiếp theo
        self.prev_gray = gray_frame.copy()
        
        return H