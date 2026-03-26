# vim: expandtab:ts=4:sw=4
import os
import argparse
import motmetrics as mm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Chấm điểm tự động thuật toán Tracking (MOTA, IDF1...)")
    parser.add_argument(
        "--gt_dir", required=True, 
        help="Đường dẫn đến thư mục gốc chứa Ground Truth (Ví dụ: dataset/MOT16/train)")
    parser.add_argument(
        "--res_dir", required=True, 
        help="Đường dẫn đến thư mục chứa kết quả của Tracker (Ví dụ: results)")
    return parser.parse_args()

def evaluate_tracking(gt_dir, res_dir):
    # Khởi tạo bộ chứa điểm số cho nhiều video
    accs = []
    seqs = []

    print("Đang tiến hành đối chiếu và chấm điểm...")
    
    # Quét tất cả các file kết quả (.txt) trong thư mục res_dir
    for res_file in sorted(os.listdir(res_dir)):
        if not res_file.endswith('.txt'):
            continue
        
        # Lấy tên sequence (Ví dụ: KITTI-0000)
        seq_name = res_file.replace('.txt', '')
        res_path = os.path.join(res_dir, res_file)
        
        # Đường dẫn file GT tương ứng
        gt_path = os.path.join(gt_dir, seq_name, 'gt', 'gt.txt')

        if not os.path.exists(gt_path):
            print(f"[Cảnh báo] Bỏ qua {seq_name}: Không tìm thấy file GT tại {gt_path}")
            continue

        print(f" -> Đang xử lý: {seq_name}")

        # Đọc file GT và file Result theo chuẩn MOT16 (chỉ lấy x, y, w, h)
        gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(res_path, fmt="mot15-2D")

        # So sánh 2 file dựa trên chỉ số Intersection over Union (IoU), ngưỡng 0.5
        acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
        
        accs.append(acc)
        seqs.append(seq_name)

    if len(accs) == 0:
        print("Lỗi: Không có dữ liệu để đánh giá. Hãy kiểm tra lại đường dẫn.")
        return

    # Tính toán tổng hợp các chỉ số
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, 
        metrics=mm.metrics.motchallenge_metrics, 
        names=seqs,
        generate_overall=True # Tạo thêm một dòng OVERALL tổng kết tất cả video
    )

    # Hiển thị bảng kết quả ra màn hình với định dạng đẹp mắt
    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names
    )
    
    print("\n" + "="*80)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ THUẬT TOÁN UKF-DEEPSORT".center(80))
    print("="*80)
    print(str_summary)
    
    # (Tùy chọn) Lưu kết quả ra file Excel/CSV để dễ dàng copy vào Báo cáo
    summary.to_csv("evaluation_results.csv")
    print("\n[!] Đã xuất kết quả chi tiết ra file: evaluation_results.csv")

if __name__ == "__main__":
    # Fix lỗi hiển thị cột của Pandas trên Terminal để bảng không bị ngắt dòng
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    args = parse_args()
    evaluate_tracking(args.gt_dir, args.res_dir)