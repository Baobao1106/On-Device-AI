"""
src/blender.py - DÀNH CHO THÀNH VIÊN 3
======================================
Vai trò: Warping, Gain Compensation & Seam Blending (Ghép ảnh & Xuất kết quả).

Nhiệm vụ cốt lõi:
1. Chiếu ngược (Inverse Spherical Warping) 24 ảnh lên canvas ERP kích thước chuẩn 2048x1024.
2. Cân bằng phơi sáng & cân bằng trắng (Gain Compensation) để triệt tiêu loang màu giữa các ảnh.
3. Multi-band Blending (Laplacian Pyramid) giữ độ sắc nét tối đa cho LPIPS.
4. Xử lý mép nối 360 độ (Circular Boundary Wrap) để vượt qua bài kiểm tra seam_hint_ok.
"""

from typing import Dict, Tuple
import numpy as np


def compute_gain_compensation(images_dict: Dict[str, np.ndarray], 
                              overlaps: Dict[Tuple[str, str], np.ndarray]) -> Dict[str, np.ndarray]:
    """
    [BƯỚC 1]: Tính toán hệ số bù sáng và cân bằng màu (Gain Compensation) cho từng ảnh.

    Giải thích bài toán:
        24 ảnh bị lệch độ sáng +- 15% và sai lệch cân bằng trắng (RGB).
        Nếu không bù sáng, khi ghép lại ảnh sẽ bị loang lổ các mảng sáng tối cộp cạch,
        làm giảm mạnh điểm PSNR và LPIPS.

    Tham số đầu vào:
        - images_dict (Dict[str, np.ndarray]): 24 ảnh gốc dạng mảng numpy.
        - overlaps: Thông tin các vùng ảnh chồng lấn lên nhau giữa các cặp ảnh kề nhau.

    Giá trị trả về:
        - gains (Dict[str, np.ndarray]): Mỗi ảnh k có một vector hệ số bù 3 kênh màu:
          g_k = [gain_B, gain_G, gain_R].

    Gợi ý thuật toán:
        - Thiết lập hệ phương trình bình phương tối thiểu (Least Squares):
          min sum_{(i, j)} || g_i * I_i - g_j * I_j ||^2
        - Khóa 1 ảnh làm mốc (gain = 1.0) và giải tìm gains cho 23 ảnh còn lại.
    """
    # TODO: Thành viên 3 cài đặt thuật toán Gain Compensation
    raise NotImplementedError("Thành viên 3 cần cài đặt hàm compute_gain_compensation()")


def inverse_spherical_warp(image: np.ndarray, R: np.ndarray, fov: float, 
                           out_size: Tuple[int, int] = (2048, 1024)) -> Tuple[np.ndarray, np.ndarray]:
    """
    [BƯỚC 2]: Chiếu ngược (Inverse Remapping) 1 bức ảnh phối cảnh lên tấm canvas mặt cầu ERP.

    Giải thích bài toán:
        Với mỗi pixel (u, v) trên canvas ERP 2048x1024, ta bắn một tia sáng 3D d_world ra ngoài,
        xoay tia sáng về hệ trục camera qua ma trận R, sau đó tìm tọa độ (x, y) trên ảnh gốc 864x648.

    Tham số đầu vào:
        - image (np.ndarray): Ảnh phối cảnh gốc (648, 864, 3).
        - R (np.ndarray): Ma trận xoay 3x3 của camera này (từ TV1).
        - fov (float): Góc nhìn ước lượng (từ TV1).
        - out_size (Tuple[int, int]): Kích thước ảnh ERP chuẩn (W=2048, H=1024).

    Giá trị trả về (Tuple):
        - warped_img: Ảnh đã được uốn cong lên canvas (1024, 2048, 3).
        - mask: Mặt nạ nhị phân (1024, 2048) đánh dấu vùng pixel hợp lệ của ảnh này.

    Lưu ý quan trọng:
        - Dùng phép nội suy cv2.INTER_CUBIC hoặc cv2.INTER_LINEAR để giữ ảnh sắc nét.
        - Tránh dùng Nearest-Neighbor vì sẽ làm vỡ nét răng cưa.
    """
    # TODO: Thành viên 3 cài đặt hàm Inverse Warping
    raise NotImplementedError("Thành viên 3 cần cài đặt hàm inverse_spherical_warp()")


def multiband_blend(warped_images: list, masks: list, num_bands: int = 3) -> np.ndarray:
    """
    [BƯỚC 3]: Hòa trộn các ảnh bằng Kim tự tháp đa dải tần (Multi-Band / Laplacian Pyramid Blending).

    Giải thích bài toán:
        Nếu lấy trung bình cộng đơn giản, các mép ảnh chồng lên nhau sẽ bị nhòe mờ (ghosting),
        làm điểm LPIPS tụt rất nặng.
        Multi-Band Blending tách ảnh thành các dải tần số:
        - Tần số thấp (màu sắc, độ sáng): trộn dải rộng để chuyển tiếp mượt mà.
        - Tần số cao (đường nét, gờ tường): trộn dải cực hẹp để giữ nguyên độ sắc nét tối đa.

    Tham số đầu vào:
        - warped_images (list): Danh sách các ảnh đã được uốn cong lên canvas 2048x1024.
        - masks (list): Danh sách mặt nạ tương ứng của các ảnh.
        - num_bands (int): Số tầng kim tự tháp (thường từ 2 đến 4 tầng).

    Giá trị trả về:
        - blended_erp (np.ndarray): Ảnh kết quả sau khi trộn (1024, 2048, 3).
    """
    # TODO: Thành viên 3 cài đặt thuật toán Multi-Band Blending
    raise NotImplementedError("Thành viên 3 cần cài đặt hàm multiband_blend()")


def warp_and_blend(images_dir: str, poses: Dict[str, np.ndarray], fov: float, 
                   output_path: str, out_size: Tuple[int, int] = (2048, 1024)) -> str:
    """
    [BƯỚC 4 - HÀM TỔNG HỢP]: Điều phối toàn bộ quy trình từ 24 ảnh gốc -> File PNG 2048x1024.

    Quy trình:
        1. Đọc 24 ảnh từ images_dir.
        2. Áp dụng Gain Compensation để cân bằng màu.
        3. Chiếu 24 ảnh lên canvas ERP bằng inverse_spherical_warp().
        4. Xử lý mép nối vòng tròn 360 độ (Padding biên trái/phải với cv2.BORDER_WRAP).
        5. Trộn ảnh bằng multiband_blend().
        6. Lưu ảnh đầu ra dạng PNG đúng kích thước 2048x1024 tại output_path.

    Tham số đầu vào:
        - images_dir (str): Thư mục chứa 24 ảnh gốc.
        - poses (Dict[str, np.ndarray]): 24 ma trận quay chuẩn từ TV1.
        - fov (float): Góc nhìn FOV từ TV1.
        - output_path (str): Đường dẫn lưu file ảnh (ví dụ: 'runs/myteam/t01.png').
        - out_size (Tuple[int, int]): Bắt buộc (2048, 1024).

    Giá trị trả về:
        - output_path (str): Đường dẫn file ảnh đã được lưu thành công.
    """
    # TODO: Thành viên 3 kết nối các bước trên và lưu file PNG
    raise NotImplementedError("Thành viên 3 cần cài đặt hàm warp_and_blend()")
