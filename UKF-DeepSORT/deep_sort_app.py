# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import cv2
import numpy as np
import time
import torch
import sys

from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# --- IMPORT CÁC MODULE OOP ĐÃ TÁCH ---
from modules.detection.vehicle_detector import VehicleDetector
from modules.reid.feature_extractor import PyTorchFeatureExtractor
from application_util.cmc import CameraMotionCompensator
from modules.visualization.visualizer import Visualizer
# ------------------------------------

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information."""
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())), cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = 0
        max_frame_idx = 0

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)
        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": None, 
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": 0,
        "update_ms": update_ms
    }
    return seq_info


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    # --- KHỞI TẠO CÁC MODULE AI & CÔNG CỤ ---
    print("1. Đang tải model YOLOv8...")
    detector = VehicleDetector(model_path='yolov8n.pt') 
    
    print("2. Đang tải model Re-ID...")
    # Lưu ý: Chỉnh lại tên file .pth nếu máy bạn đang dùng tên khác
    extractor = PyTorchFeatureExtractor(
        model_name='sbs_R50-ibn', 
        model_path='modules/reid/weights/veri_sbs_R50-ibn.pt', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("3. Đang khởi tạo Camera Motion Compensator (CMC)...")
    cmc = CameraMotionCompensator()

    print("4. Đang thiết lập Visualizer...")
    fps = 1000 / seq_info["update_ms"] if seq_info["update_ms"] else 30
    output_video_path = "results/demo_result.avi"
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    visualizer_oop = Visualizer(output_path=output_video_path, fps=int(fps))
    
    print("\n[THÀNH CÔNG] Hệ thống sẵn sàng! Bắt đầu Tracking...\n")
    # ----------------------------------------

    last_time = time.time()

    def frame_callback(vis, frame_idx):
        nonlocal last_time
        
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time 
        fps_sys = 1.0 / delta_time if delta_time > 0 else 0
            
        print(f"Processing frame {frame_idx:05d} | System FPS: {fps_sys:.2f}", end='\r')
        
        # Đọc ảnh gốc
        frame = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        # --- BƯỚC 1: BÙ TRỪ CHUYỂN ĐỘNG CAMERA (CMC) ---
        H = cmc.compute_affine_matrix(frame)
        if H is not None:
            tracker.camera_update(H)

        # --- BƯỚC 2: PHÁT HIỆN XE (YOLO) ---
        detections_array = detector.detect(frame)
        crops, valid_bboxes, valid_confs = [], [], []
        
        for det in detections_array:
            x1, y1, w, h, conf, cls_id = det
            x1_int, y1_int, w_int, h_int = int(x1), int(y1), int(w), int(h)
            x2_int, y2_int = x1_int + w_int, y1_int + h_int
            
            x1_int, y1_int = max(0, x1_int), max(0, y1_int)
            x2_int, y2_int = min(frame.shape[1], x2_int), min(frame.shape[0], y2_int)
            
            if h >= min_detection_height and conf >= min_confidence:
                crop = frame[y1_int:y2_int, x1_int:x2_int]
                if crop.size > 0:
                    crops.append(crop)
                    valid_bboxes.append([int(x1), int(y1), int(w), int(h)]) 
                    valid_confs.append(conf)

        # --- BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG (RE-ID) ---
        features = extractor.extract(crops) if len(crops) > 0 else []

        detections = []
        for bbox, conf, feat in zip(valid_bboxes, valid_confs, features):
            detections.append(Detection(np.array(bbox), conf, feat))

        # --- BƯỚC 4: CẬP NHẬT UKF TRACKER ---
        tracker.predict() # Không cần truyền frame vào nữa vì đã tính H ở trên
        tracker.update(detections)

        # Lưu kết quả text
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        # --- BƯỚC 5: VẼ HÌNH & LƯU VIDEO ---
        vis_frame = visualizer_oop.draw_and_save(frame, tracker.tracks, frame_idx)

        if display:
            cv2.imshow("Tracking Preview", vis_frame) 
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("\n\n[INFO] Đã nhận lệnh ngắt từ người dùng. Đang lưu kết quả...")
                visualizer_oop.release()
                cv2.destroyAllWindows()
                
                output_dir = os.path.dirname(output_file)
                if output_dir: os.makedirs(output_dir, exist_ok=True)
                with open(output_file, 'w') as f:
                    for row in results:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
                sys.exit(0)

    # Chạy vòng lặp ngầm của DeepSORT
    visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)
    
    # Dọn dẹp bộ nhớ khi chạy hết ảnh
    visualizer_oop.release()
    print(f"\nXong! Video da duoc luu tai: {output_video_path}")
        
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file, 'w') as f:
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]), file=f)

def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid True/False choice")
    else:
        return input_string == "True"

def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT OOP Architecture")
    parser.add_argument("--sequence_dir", help="Path to sequence directory", required=True)
    parser.add_argument("--detection_file", help="Not used anymore", default=None, required=False)
    parser.add_argument("--output_file", help="Path to the tracking output file.", default="results/ket_qua.txt")
    parser.add_argument("--min_confidence", help="Detection confidence threshold.", default=0.5, type=float)
    parser.add_argument("--min_detection_height", help="Threshold on the detection bounding box height.", default=0, type=int)
    parser.add_argument("--nms_max_overlap",  help="Non-maxima suppression threshold.", default=1.0, type=float)
    parser.add_argument("--max_cosine_distance", help="Gating threshold for cosine distance metric.", type=float, default=0.2)
    parser.add_argument("--nn_budget", help="Maximum size of the appearance descriptors gallery.", type=int, default=None)
    parser.add_argument("--display", help="Show intermediate tracking results", default=True, type=bool_string)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)