import cv2
import os

# 1. Cấu hình đường dẫn
# Thay đổi đường dẫn này trỏ tới thư mục chứa hình ảnh của một sequence trong KITTI (ví dụ: sequence 0000)
image_folder = 'data/kitti_tracking/training/image_02/0000'
video_name = 'kitti_test_set_10s.mp4'

# Lấy danh sách các file ảnh và sắp xếp theo thứ tự bảng chữ cái/số
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()

# Giới hạn lấy 300 frames đầu tiên (tương đương 10 giây ở tốc độ 30 FPS)
num_frames = 300
images = images[:num_frames]

if not images:
    print("Không tìm thấy ảnh nào trong thư mục. Vui lòng kiểm tra lại đường dẫn!")
else:
    # 2. Đọc ảnh đầu tiên để lấy kích thước (width, height)
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 3. Khởi tạo đối tượng VideoWriter
    # Sử dụng codec mp4v cho định dạng .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10 # Tốc độ khung hình (KITTI thường dao động quanh 10-30fps, đặt 30fps cho mượt)
    
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # 4. Vòng lặp ghi từng ảnh vào video
    print(f"Đang tiến hành ghép {len(images)} frames thành video...")
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Giải phóng bộ nhớ sau khi hoàn tất
    cv2.destroyAllWindows()
    video.release()
    print(f"Hoàn tất! Video đã được lưu tại: {video_name}")