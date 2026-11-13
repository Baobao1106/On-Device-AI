"""
src/matcher.py - THÀNH VIÊN 2 (AI Feature Matching & Ring Classification)
========================================================================
Nhiệm vụ:
1. Phân loại 24 ảnh vào 5 tầng (Zenith, Up, Horizon, Down, Nadir).
2. Dùng SuperPoint + LightGlue trích xuất và bắt cặp đặc trưng cho các ảnh lân cận.
3. Giải quyết vùng textureless (tường trắng, trần nhà nhẵn).
"""

import os
import glob
import numpy as np
import cv2

# Khởi tạo mô hình LightGlue khi cần
_extractor = None
_matcher = None
_device = None

def get_models():
    global _extractor, _matcher, _device
    if _matcher is None:
        import torch
        from lightglue import LightGlue, SuperPoint
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _extractor = SuperPoint(max_num_keypoints=1024).eval().to(_device)
        _matcher = LightGlue(features="superpoint").eval().to(_device)
    return _extractor, _matcher, _device


def classify_rings(images_dir, image_names):
    """
    Phân loại 24 bức ảnh vào 5 tầng dựa theo Capture Pattern:
    - zenith: 1 ảnh (pitch ~ +90)
    - up: 7 ảnh (pitch ~ +45)
    - horizon: 8 ảnh (pitch ~ 0)
    - down: 7 ảnh (pitch ~ -45)
    - nadir: 1 ảnh (pitch ~ -90)
    
    Output:
        rings_dict = {
            'zenith': '...',
            'up': ['...', ...],
            'horizon': ['...', ...],
            'down': ['...', ...],
            'nadir': '...'
        }
    """
    # TODO (Thành viên 2): Hoàn thiện thuật toán phân tầng bằng Vanishing Lines hoặc Graph Topology
    # Tạm thời trả về danh sách placeholder nếu chưa phân loại xong
    return {
        "zenith": image_names[0],
        "up": image_names[1:8],
        "horizon": image_names[8:16],
        "down": image_names[16:23],
        "nadir": image_names[23] if len(image_names) > 23 else image_names[-1]
    }


def match_image_pair(img_path1, img_path2):
    """
    Tìm điểm tương ứng giữa 2 ảnh bằng LightGlue.
    Output:
        pts0: (N, 2) tọa độ pixel trên img1
        pts1: (N, 2) tọa độ pixel trên img2
    """
    import torch
    from lightglue.utils import load_image, rbd
    extractor, matcher, device = get_models()

    image0 = load_image(img_path1).to(device)
    image1 = load_image(img_path2).to(device)

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01["matches"]
    pts0 = feats0["keypoints"][matches[..., 0]].cpu().numpy()
    pts1 = feats1["keypoints"][matches[..., 1]].cpu().numpy()

    return pts0, pts1


def build_match_graph(images_dir, rings_dict=None):
    """
    Tạo đồ thị kết nối điểm giữa các cặp ảnh tiếp giáp.
    Output:
        matches_dict: { (img_name_A, img_name_B): {'pts0': ..., 'pts1': ...}, ... }
    """
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    image_names = [os.path.basename(p) for p in image_paths]
    
    matches_dict = {}
    
    # TODO (Thành viên 2): Chỉ matching các cặp lân cận theo rings_dict để tăng tốc
    # Ví dụ minh họa:
    print(f"[TV2] Đang trích xuất đặc trưng cho {len(image_names)} ảnh...")
    
    return matches_dict
