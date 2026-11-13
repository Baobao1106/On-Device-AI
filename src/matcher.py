"""
src/matcher.py - DÀNH CHO THÀNH VIÊN 2
=====================================
Vai trò: Trí tuệ Nhân tạo (AI Feature Matching) & Phân loại 5 tầng ảnh.

Nhiệm vụ cốt lõi:
1. Phân loại 24 bức ảnh vào đúng 5 tầng của Capture Plan (Zenith, Up, Horizon, Down, Nadir).
2. Dùng SuperPoint + LightGlue trích xuất và bắt cặp đặc trưng cho các ảnh tiếp giáp.
3. Giải quyết bài toán tường trắng trơn, trần nhà nhẵn bóng (textureless).
"""

from typing import Dict, List, Tuple
import numpy as np


def classify_rings(images_dir: str, image_names: List[str]) -> Dict[str, any]:
    """
    [BƯỚC 1]: Phân loại 24 bức ảnh bị xáo trộn tên vào đúng 5 tầng theo Capture Plan.

    Giải thích bài toán:
        24 ảnh được chụp theo quy luật cố định nhưng bị xáo trộn tên ngẫu nhiên:
        - 1 ảnh đỉnh trần (Zenith, pitch ~ +90°)
        - 7 ảnh tầng ngửa (Up, pitch ~ +45°)
        - 8 ảnh tầng ngang tầm mắt (Horizon, pitch ~ 0°)
        - 7 ảnh tầng úp (Down, pitch ~ -45°)
        - 1 ảnh đáy sàn (Nadir, pitch ~ -90°)

    Tham số đầu vào:
        - images_dir (str): Đường dẫn đến thư mục chứa 24 ảnh (ví dụ: 'sets/t01/public/images').
        - image_names (List[str]): Danh sách 24 tên file ảnh (ví dụ: ['1380.jpg', '21c5.jpg', ...]).

    Giá trị trả về (Dict):
        {
            "zenith": "ten_anh_tran.jpg",           # 1 tên file
            "up": ["anh1.jpg", ..., "anh7.jpg"],    # list 7 tên file
            "horizon": ["anh1.jpg", ..., "anh8.jpg"],# list 8 tên file
            "down": ["anh1.jpg", ..., "anh7.jpg"],  # list 7 tên file
            "nadir": "ten_anh_san.jpg"              # 1 tên file
        }

    Gợi ý thuật toán cho Thành viên 2:
        - Cách 1 (Topology/Đồ thị): Ảnh Zenith kết nối dạng nan hoa với 7 ảnh Up; 
          ảnh Nadir kết nối nan hoa với 7 ảnh Down; 8 ảnh Horizon nối thành 1 chu trình khép kín.
        - Cách 2 (Vanishing Lines): Dùng cv2.HoughLinesP() đo độ dốc các đường thẳng đứng. 
          Ảnh Up các đường thẳng tụ về phía dưới ảnh; ảnh Down tụ về phía trên; ảnh Horizon thẳng đứng song song.
    """
    # TODO: Thành viên 2 viết code thuật toán phân loại tại đây
    raise NotImplementedError("Thành viên 2 cần cài đặt hàm classify_rings()")


def extract_features(image_path: str) -> Dict[str, any]:
    """
    [BƯỚC 2A]: Trích xuất điểm đặc trưng (Keypoints) và Descriptors bằng mô hình AI.

    Giải thích bài toán:
        Để ghép được tường trắng trơn và trần nhà nhẵn bóng (điểm yếu chết người của SIFT),
        cần dùng mô hình SuperPoint để tìm được các điểm đặc trưng ngay cả ở chi tiết mờ nhất.

    Tham số đầu vào:
        - image_path (str): Đường dẫn tới 1 bức ảnh JPG (kích thước 864x648).

    Giá trị trả về (Dict):
        - Chứa tensor tọa độ keypoints (N, 2), descriptors (N, D), và điểm tin cậy scores (N,).
    """
    # TODO: Thành viên 2 gọi mô hình SuperPoint từ thư viện LightGlue
    raise NotImplementedError("Thành viên 2 cần cài đặt hàm extract_features()")


def match_pair(img_path0: str, img_path1: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    [BƯỚC 2B]: So sánh và bắt cặp điểm tương ứng giữa 2 bức ảnh tiếp giáp.

    Giải thích bài toán:
        Dùng mô hình LightGlue (Transformer-based matcher) để tìm xem điểm nào ở ảnh 0 
        chính là vật thể đó ở ảnh 1.

    Tham số đầu vào:
        - img_path0 (str): Đường dẫn ảnh thứ nhất.
        - img_path1 (str): Đường dẫn ảnh thứ hai.

    Giá trị trả về (Tuple):
        - pts0: Mảng numpy shape (M, 2) lưu tọa độ [x, y] trên ảnh 0.
        - pts1: Mảng numpy shape (M, 2) lưu tọa độ [x, y] trên ảnh 1.
          (với M là số điểm khớp hợp lệ).
    """
    # TODO: Thành viên 2 dùng LightGlue để match 2 ảnh
    raise NotImplementedError("Thành viên 2 cần cài đặt hàm match_pair()")


def build_match_graph(images_dir: str, rings_dict: Dict[str, any]) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """
    [BƯỚC 2C]: Xây dựng đồ thị kết nối điểm cho toàn bộ các cặp ảnh kề nhau.

    Giải thích bài toán:
        Thay vì so sánh mù quáng 276 cặp (rất chậm), ta tận dụng rings_dict từ Bước 1
        để chỉ match các cặp ảnh thực sự tiếp giáp nhau (~45 cặp):
        - Giữa các ảnh kề nhau trong cùng tầng Horizon (8 cặp).
        - Giữa tầng Horizon và tầng Up / Down liền kề.
        - Giữa ảnh Zenith với 7 ảnh Up; giữa ảnh Nadir với 7 ảnh Down.

    Tham số đầu vào:
        - images_dir (str): Thư mục chứa 24 ảnh.
        - rings_dict (Dict): Kết quả phân tầng từ hàm classify_rings().

    Giá trị trả về (Dict):
        {
            ("anh_A.jpg", "anh_B.jpg"): {
                "pts0": np.ndarray shape (M, 2),
                "pts1": np.ndarray shape (M, 2),
                "num_matches": M
            },
            ...
        }
        (Dữ liệu này sẽ được bàn giao trực tiếp cho THÀNH VIÊN 1).
    """
    # TODO: Thành viên 2 duyệt qua danh sách các cặp lân cận và gọi match_pair()
    raise NotImplementedError("Thành viên 2 cần cài đặt hàm build_match_graph()")
