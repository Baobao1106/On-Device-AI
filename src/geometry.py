"""
src/geometry.py - THÀNH VIÊN 1 (Geometry, Pose Estimation & Leveling)
====================================================================
Nhiệm vụ:
1. Ước lượng FOV chuẩn xác (72 - 77 độ).
2. Tối ưu hóa toàn cục ma trận quay R cho 24 camera (Bundle Adjustment / Rotation Averaging).
3. Thuật toán Leveling triệt tiêu góc nghiêng: Bắt buộc Tilt < 5 deg (mục tiêu < 1 deg).
"""

import numpy as np
import cv2
from common import capture_plan, pose_to_R, intrinsics


def estimate_fov(matches_dict, w=864, h=648):
    """
    Quét dải FOV từ 72 đến 77 độ để tìm giá trị tối ưu với sai số chiếu lại nhỏ nhất.
    Output:
        best_fov: float (ví dụ 74.5)
    """
    # TODO (Thành viên 1): Grid search FOV dải 72.0 - 77.0 deg
    default_fov = 74.5
    print(f"[TV1] FOV ước tính: {default_fov:.2f} deg")
    return default_fov


def solve_camera_rotations(matches_dict, rings_dict, fov, w=864, h=648):
    """
    Tính toán 24 ma trận xoay R (3x3) từ các điểm khớp và vị trí các tầng.
    Output:
        poses: dict { 'image_name': R_matrix_3x3, ... }
    """
    poses = {}
    plan = capture_plan()
    
    # Gán góc ban đầu dựa trên capture plan lý thuyết (Nominal angles)
    # TODO (Thành viên 1): Tinh chỉnh bằng Rotation Averaging / Bundle Adjustment
    
    return poses


def apply_leveling(poses, rings_dict):
    """
    Cân bằng đường chân trời (Leveling):
    Tìm Up-vector thực tế và xoay toàn bộ 24 camera sao cho Up-vector trùng với trục Y thế giới [0, 1, 0].
    Đảm bảo: Tilt < 1.0 deg (Quy chế: Tilt >= 5.0 deg -> 0 điểm).
    """
    # TODO (Thành viên 1): Tìm pháp tuyến của mặt phẳng 8 ảnh Horizon
    # Sau đó nhân ma trận R_level vào tất cả các poses
    return poses
