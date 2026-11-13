"""
src/geometry.py - DÀNH CHO THÀNH VIÊN 1
=======================================
Vai trò: Hình học 3D, Ước lượng Pose & Leveling (Cân bằng chân trời).

Nhiệm vụ cốt lõi:
1. Ước lượng FOV chuẩn xác (72 - 77 độ).
2. Tối ưu hóa toàn cục ma trận quay R cho 24 camera (Bundle Adjustment / Rotation Averaging).
3. Thuật toán Leveling triệt tiêu góc nghiêng: Bắt buộc Tilt < 5 deg (mục tiêu < 1 deg).
"""

from typing import Dict, Tuple
import numpy as np


def estimate_fov(matches_dict: Dict, image_size: Tuple[int, int] = (864, 648)) -> float:
    """
    [BƯỚC 1]: Ước lượng góc nhìn (FOV) chuẩn xác của bộ ảnh trong dải [72.0°, 77.0°].

    Giải thích bài toán:
        Ban tổ chức giấu FOV nhưng quy định: cả 24 ảnh trong 1 set có cùng 1 giá trị FOV,
        nằm ngẫu nhiên trong khoảng 72° đến 77°.
        Nếu tính sai FOV, mép ảnh chiếu lên hình cầu sẽ bị phồng/lõm, gây lệch khớp mép ảnh.

    Tham số đầu vào:
        - matches_dict (Dict): Bộ điểm tương ứng giữa các ảnh từ Thành viên 2.
        - image_size (Tuple[int, int]): Kích thước ảnh gốc (W=864, H=648).

    Giá trị trả về:
        - fov_deg (float): Giá trị góc nhìn ước lượng (ví dụ: 74.25).

    Gợi ý thuật toán cho Thành viên 1:
        - Grid Search: Quét thử fov từ 72.0 đến 77.0 với bước nhảy 0.25 độ.
        - Với mỗi fov, tính tiêu cự f = (W / 2) / tan(radians(fov) / 2).
        - Tính sai số chiếu lại (Reprojection Error) trên các cặp điểm inliers của tầng Horizon.
        - Giá trị fov nào cho sai số nhỏ nhất chính là kết quả cần tìm.
    """
    # TODO: Thành viên 1 cài đặt thuật toán tìm FOV tối ưu
    raise NotImplementedError("Thành viên 1 cần cài đặt hàm estimate_fov()")


def compute_relative_rotation(pts0: np.ndarray, pts1: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    [BƯỚC 2A]: Tính ma trận quay tương đối R_ij (3x3) giữa 2 camera cùng tâm.

    Giải thích bài toán:
        Vì người chụp xoay tại chỗ (không tịnh tiến, t=0), quan hệ giữa 2 ảnh là phép quay thuần:
        x1 ~ K @ R_ij @ inv(K) @ x0

    Tham số đầu vào:
        - pts0 (np.ndarray): Tọa độ điểm trên ảnh 0, shape (N, 2).
        - pts1 (np.ndarray): Tọa độ điểm trên ảnh 1, shape (N, 2).
        - K (np.ndarray): Ma trận nội số camera intrinsics (3x3).

    Giá trị trả về:
        - R_ij (np.ndarray): Ma trận quay 3x3 sao cho ray1 ~ R_ij @ ray0.

    Gợi ý thuật toán:
        - Đổi pts sang ray: d0 = inv(K) @ [u, v, 1]^T, chuẩn hóa ||d|| = 1.
        - Dùng thuật toán Kabsch / SVD kết hợp RANSAC để loại bỏ điểm nhiễu ngoại lai (outliers).
    """
    # TODO: Thành viên 1 cài đặt hàm tính R tương đối giữa 2 ảnh
    raise NotImplementedError("Thành viên 1 cần cài đặt hàm compute_relative_rotation()")


def solve_global_rotations(matches_dict: Dict, rings_dict: Dict, fov: float, 
                           image_size: Tuple[int, int] = (864, 648)) -> Dict[str, np.ndarray]:
    """
    [BƯỚC 2B]: Tối ưu hóa toàn cục ma trận quay R cho cả 24 camera (Bundle Adjustment).

    Giải thích bài toán:
        Từ các góc quay tương đối R_ij giữa các cặp ảnh liền kề, ta cần tìm 24 ma trận quay
        tuyệt đối {R_1, R_2, ..., R_24} sao cho sai số trên toàn bộ quả cầu là nhỏ nhất,
        tránh bị sai số tích lũy làm lệch vòng tròn khép kín 360 độ.

    Tham số đầu vào:
        - matches_dict (Dict): Điểm khớp từ Thành viên 2.
        - rings_dict (Dict): Cấu trúc 5 tầng từ Thành viên 2.
        - fov (float): Góc nhìn ước lượng từ Bước 1.
        - image_size (Tuple[int, int]): (864, 648).

    Giá trị trả về:
        - poses (Dict[str, np.ndarray]): Dictionary chứa 24 ma trận xoay (3x3):
          { "1380.jpg": R1, "21c5.jpg": R2, ..., "9a12.jpg": R24 }

    Gợi ý thuật toán:
        - Khởi tạo góc ban đầu (Initial Guess): Dùng góc danh định (Nominal angles) trong capture_plan()
          của common.py cho từng tầng (rất gần với góc thật, chỉ lệch rung tay +- 5 độ).
        - Dùng scipy.optimize.least_squares tối ưu hóa góc quay dưới dạng Lie Algebra so(3) hoặc Quaternion.
    """
    # TODO: Thành viên 1 cài đặt thuật toán Bundle Adjustment / Rotation Averaging
    raise NotImplementedError("Thành viên 1 cần cài đặt hàm solve_global_rotations()")


def apply_leveling(poses: Dict[str, np.ndarray], rings_dict: Dict) -> Dict[str, np.ndarray]:
    """
    [BƯỚC 3 - SỐNG CÒN]: Cân bằng đường chân trời (Leveling) để triệt tiêu độ nghiêng Tilt.

    Quy định của ban tổ chức:
        Tilt >= 5.0 độ -> 0 ĐIỂM NGAY LẬP TỨC!
        Mục tiêu: Đưa độ nghiêng Tilt về < 1.0 độ.

    Giải thích bài toán:
        Scorer cho phép tự do hướng nhìn ngang (Yaw), nhưng trục thẳng đứng (Up vector)
        của ảnh phải trùng khớp với phương trọng lực thực tế (Trục Y thế giới [0, 1, 0]).

    Tham số đầu vào:
        - poses (Dict[str, np.ndarray]): 24 ma trận quay trước khi cân bằng.
        - rings_dict (Dict): Danh sách ảnh 5 tầng từ TV2.

    Giá trị trả về:
        - leveled_poses (Dict[str, np.ndarray]): 24 ma trận quay mới đã được xoay phẳng:
          R_new = R_level @ R_old.

    Gợi ý thuật toán:
        - 8 ảnh ở tầng Horizon có trục quang (optical axis) tạo thành mặt phẳng song song chân trời.
        - Dùng SVD/PCA tìm vector pháp tuyến của mặt phẳng này -> Đó chính là vector Up thực tế v_up.
        - Tìm ma trận quay R_level đưa v_up về [0, 1, 0].
    """
    # TODO: Thành viên 1 cài đặt thuật toán Leveling triệt tiêu Tilt
    raise NotImplementedError("Thành viên 1 cần cài đặt hàm apply_leveling()")
