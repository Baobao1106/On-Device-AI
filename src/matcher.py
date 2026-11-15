"""
src/matcher.py - DÀNH CHO THÀNH VIÊN 2
=====================================
Vai trò: Trí tuệ Nhân tạo (AI Feature Matching) & Phân loại 5 tầng ảnh.

Nhiệm vụ:
1. classify_rings(): Phân loại 24 ảnh vào đúng 5 tầng (Zenith, Up, Horizon, Down, Nadir). [ĐÃ HOÀN THÀNH]
2. extract_features(): Trích xuất đặc trưng bằng SuperPoint.
3. match_pair(): Ghép điểm đặc trưng bằng LightGlue.
4. build_match_graph(): Xây dựng đồ thị liên kết giữa các cặp ảnh tiếp giáp.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["zenith", "up", "horizon", "down", "nadir"]

# Biến toàn cục lưu trữ mô hình phân loại để tránh nạp lại nhiều lần
_classifier_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_classifier(weights_path: str = "ring_classifier.pth") -> nn.Module:
    """Tải mô hình MobileNetV3-Small đã được huấn luyện với weights_path."""
    global _classifier_model
    if _classifier_model is not None:
        return _classifier_model

    # Nếu weights_path tương đối, tìm ở thư mục hiện tại hoặc thư mục gốc dự án
    if not os.path.exists(weights_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(project_root, weights_path)
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số '{weights_path}'. "
                "Vui lòng chạy 'python train_classifier.py' trước để tạo file này!"
            )

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 5)
    model.load_state_dict(torch.load(weights_path, map_location=_device))
    model = model.to(_device).eval()
    _classifier_model = model
    return _classifier_model


def classify_rings(images_dir: str, image_names: List[str], weights_path: str = "ring_classifier.pth") -> Dict[str, any]:
    """
    [BƯỚC 1 - ĐÃ HOÀN THIỆN]: Phân loại 24 bức ảnh vào đúng 5 tầng của Capture Plan.

    Thuật toán:
        1. Chạy mạng MobileNetV3-Small để dự đoán ma trận xác suất 5 tầng cho 24 ảnh.
        2. Áp dụng thuật toán ràng buộc số lượng cấu trúc (Constraint-aware Assignment):
           - Chọn Top 1 ảnh có xác suất Zenith cao nhất -> Gán cho Zenith (1 ảnh)
           - Chọn Top 1 ảnh có xác suất Nadir cao nhất -> Gán cho Nadir (1 ảnh)
           - Chọn Top 7 ảnh có xác suất Up cao nhất trong các ảnh còn lại -> Tầng Up (7 ảnh)
           - Chọn Top 7 ảnh có xác suất Down cao nhất trong các ảnh còn lại -> Tầng Down (7 ảnh)
           - 8 ảnh còn lại chính là Tầng Horizon (8 ảnh)

    Giá trị trả về (Dict):
        {
            "zenith": "ten_anh_tran.jpg",
            "up": ["anh1.jpg", ..., "anh7.jpg"],
            "horizon": ["anh1.jpg", ..., "anh8.jpg"],
            "down": ["anh1.jpg", ..., "anh7.jpg"],
            "nadir": "ten_anh_san.jpg"
        }
    """
    model = get_classifier(weights_path)
    probs = []

    # 1. Dự đoán xác suất cho cả 24 ảnh
    with torch.no_grad():
        for name in image_names:
            img_path = os.path.join(images_dir, name)
            img = _transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(_device)
            p = torch.softmax(model(img), dim=1).cpu().numpy()[0]
            probs.append(p)
    probs = np.array(probs)  # Shape (24, 5)

    assigned = {}
    remain = set(range(len(image_names)))

    # 2. Top 1 Zenith (Trần nhà, Class 0)
    zenith_idx = max(remain, key=lambda i: probs[i, 0])
    assigned["zenith"] = image_names[zenith_idx]
    remain.remove(zenith_idx)

    # 3. Top 1 Nadir (Sàn nhà, Class 4)
    nadir_idx = max(remain, key=lambda i: probs[i, 4])
    assigned["nadir"] = image_names[nadir_idx]
    remain.remove(nadir_idx)

    # 4. Top 7 Up (Tầng ngửa, Class 1)
    sorted_up = sorted(remain, key=lambda i: probs[i, 1], reverse=True)
    up_idx = sorted_up[:7]
    assigned["up"] = [image_names[i] for i in up_idx]
    remain -= set(up_idx)

    # 5. Top 7 Down (Tầng úp, Class 3)
    sorted_down = sorted(remain, key=lambda i: probs[i, 3], reverse=True)
    down_idx = sorted_down[:7]
    assigned["down"] = [image_names[i] for i in down_idx]
    remain -= set(down_idx)

    # 6. 8 ảnh còn lại chính xác là Tầng Horizon (Class 2)
    assigned["horizon"] = [image_names[i] for i in remain]

    print(f"[TV2] Phân tầng hoàn tất: Zenith=1, Up={len(assigned['up'])}, "
          f"Horizon={len(assigned['horizon'])}, Down={len(assigned['down'])}, Nadir=1")
    return assigned


def extract_features(image_path: str) -> Dict[str, any]:
    """
    [BƯỚC 2A]: Trích xuất điểm đặc trưng (Keypoints) và Descriptors bằng SuperPoint.
    """
    # TODO: Thành viên 2 hoàn thiện hàm này
    raise NotImplementedError("Thành viên 2 cần cài đặt hàm extract_features()")


def match_pair(img_path0: str, img_path1: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    [BƯỚC 2B]: So sánh và bắt cặp điểm tương ứng giữa 2 bức ảnh tiếp giáp bằng LightGlue.
    """
    # TODO: Thành viên 2 hoàn thiện hàm này
    raise NotImplementedError("Thành viên 2 cần cài đặt hàm match_pair()")


def build_match_graph(images_dir: str, rings_dict: Dict[str, any]) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """
    [BƯỚC 2C]: Xây dựng đồ thị kết nối điểm cho toàn bộ các cặp ảnh kề nhau.
    """
    # TODO: Thành viên 2 hoàn thiện hàm này
    raise NotImplementedError("Thành viên 2 cần cài đặt hàm build_match_graph()")


# =====================================================================
# KHỐI KIỂM THỬ TRỰC TIẾP TỪ TERMINAL (CLI TEST HARNESS)
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử phân loại 5 tầng ảnh (Thành viên 2)")
    parser.add_argument("--set", default="sets/t01", help="Đường dẫn đến thư mục set (default: sets/t01)")
    parser.add_argument("--weights", default="ring_classifier.pth", help="Đường dẫn file trọng số (default: ring_classifier.pth)")
    args = parser.parse_args()

    images_dir = os.path.join(args.set, "public", "images")
    if not os.path.isdir(images_dir):
        images_dir = args.set  # Nếu truyền thẳng thư mục images

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not image_paths:
        print(f"Lỗi: Không tìm thấy ảnh .jpg nào trong '{images_dir}'!")
        exit(1)

    image_names = [os.path.basename(p) for p in image_paths]
    print(f"\n--- BẮT ĐẦU KIỂM THỬ PHÂN LOẠI 5 TẦNG TRÊN: {args.set} ({len(image_names)} ảnh) ---")

    # 1. Chạy hàm phân loại
    rings = classify_rings(images_dir, image_names, weights_path=args.weights)

    # 2. Đọc nhãn Ground Truth (nếu có thư mục hidden) để so sánh
    true_poses_path = os.path.join(args.set, "hidden", "true_poses.json")
    ground_truth = {}
    if os.path.exists(true_poses_path):
        with open(true_poses_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            for shot in gt_data.get("shots", []):
                p = shot["pitch_deg"]
                if p > 65.0:
                    lbl = "zenith"
                elif p > 20.0:
                    lbl = "up"
                elif p >= -20.0:
                    lbl = "horizon"
                elif p >= -65.0:
                    lbl = "down"
                else:
                    lbl = "nadir"
                ground_truth[shot["image"]] = (lbl, p)

    # 3. Tạo ánh xạ dự đoán cho từng ảnh
    pred_map = {}
    pred_map[rings["zenith"]] = "zenith"
    pred_map[rings["nadir"]] = "nadir"
    for name in rings["up"]:
        pred_map[name] = "up"
    for name in rings["horizon"]:
        pred_map[name] = "horizon"
    for name in rings["down"]:
        pred_map[name] = "down"

    # 4. In bảng so sánh trực quan
    print("\n" + "=" * 65)
    print(f"{'Tên ảnh':<12} | {'Tầng dự đoán':<14} | {'Tầng Ground Truth':<18} | {'Kết quả'}")
    print("-" * 65)

    correct_count = 0
    for name in sorted(image_names):
        pred_ring = pred_map.get(name, "unknown")
        if name in ground_truth:
            true_ring, pitch = ground_truth[name]
            is_ok = (pred_ring == true_ring)
            if is_ok:
                correct_count += 1
            status = "✓ Khớp" if is_ok else "✗ Sai"
            print(f"{name:<12} | {pred_ring:<14} | {true_ring + f' ({pitch:4.1f}°)' :<18} | {status}")
        else:
            print(f"{name:<12} | {pred_ring:<14} | {'(Không có GT)':<18} | -")

    print("-" * 65)
    if ground_truth:
        print(f"Tổng kết độ chính xác: {correct_count}/{len(image_names)} "
              f"({correct_count / len(image_names) * 100:.1f}%)")
    print("=" * 65 + "\n")
