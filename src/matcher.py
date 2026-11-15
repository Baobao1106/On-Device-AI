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


# Biến toàn cục lưu trữ mô hình Feature Matcher
_extractor = None
_matcher = None


def get_lightglue_models():
    """Tải và lưu trữ mô hình SuperPoint + LightGlue."""
    global _extractor, _matcher
    if _matcher is None:
        from lightglue import LightGlue, SuperPoint
        _extractor = SuperPoint(max_num_keypoints=1024).eval().to(_device)
        _matcher = LightGlue(features="superpoint").eval().to(_device)
    return _extractor, _matcher


def extract_features(image_path: str) -> Dict[str, any]:
    """
    [BƯỚC 2A]: Trích xuất điểm đặc trưng (Keypoints) và Descriptors bằng SuperPoint.
    """
    from lightglue.utils import load_image
    extractor, _ = get_lightglue_models()
    image = load_image(image_path).to(_device)
    with torch.no_grad():
        feats = extractor.extract(image)
    return feats


def match_pair(img_path0: str, img_path1: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    [BƯỚC 2B]: So sánh và bắt cặp điểm tương ứng giữa 2 bức ảnh tiếp giáp bằng LightGlue.
    
    Giá trị trả về:
        - pts0: Tọa độ pixel (M, 2) trên ảnh 0
        - pts1: Tọa độ pixel (M, 2) trên ảnh 1
        - num_matches: Số lượng điểm khớp hợp lệ (M)
    """
    from lightglue.utils import load_image, rbd
    extractor, matcher = get_lightglue_models()

    im0 = load_image(img_path0).to(_device)
    im1 = load_image(img_path1).to(_device)

    with torch.no_grad():
        f0 = extractor.extract(im0)
        f1 = extractor.extract(im1)
        res = matcher({"image0": f0, "image1": f1})

    f0, f1, res = [rbd(x) for x in [f0, f1, res]]
    matches = res["matches"]
    pts0 = f0["keypoints"][matches[..., 0]].cpu().numpy()
    pts1 = f1["keypoints"][matches[..., 1]].cpu().numpy()

    return pts0, pts1, len(matches)


def build_match_graph(images_dir: str, rings_dict: Dict[str, any]) -> Dict[Tuple[str, str], Dict[str, any]]:
    """
    [BƯỚC 2C]: Xây dựng đồ thị kết nối điểm cho toàn bộ các cặp ảnh kề nhau.
    
    Quy trình:
        1. Trích xuất đặc trưng SuperPoint cho 24 ảnh một lần duy nhất (cache).
        2. Sắp xếp 8 ảnh tầng Horizon thành vòng tròn 360 độ khép kín (Hamilton cycle).
        3. Kết nối tầng Up với Zenith và Horizon; tầng Down với Nadir và Horizon.
        4. Trả về dictionary chứa tọa độ điểm khớp giữa các cặp ảnh tiếp giáp.
    """
    from lightglue.utils import rbd
    _, matcher = get_lightglue_models()

    all_names = [rings_dict["zenith"]] + rings_dict["up"] + rings_dict["horizon"] + rings_dict["down"] + [rings_dict["nadir"]]
    print(f"[TV2] Trích xuất đặc trưng SuperPoint cho {len(all_names)} ảnh...")

    feats_cache = {}
    for name in all_names:
        p = os.path.join(images_dir, name)
        feats_cache[name] = extract_features(p)

    def match_cached(name_a, name_b):
        f0, f1 = feats_cache[name_a], feats_cache[name_b]
        with torch.no_grad():
            res = matcher({"image0": f0, "image1": f1})
        f0_r, f1_r, res_r = [rbd(x) for x in [f0, f1, res]]
        m = res_r["matches"]
        p0 = f0_r["keypoints"][m[..., 0]].cpu().numpy()
        p1 = f1_r["keypoints"][m[..., 1]].cpu().numpy()
        return p0, p1, len(m)

    matches_dict = {}

    # 1. XẾP THỨ TỰ VÒNG TRÒN 8 ẢNH HORIZON (PITCH ~ 0°)
    h_imgs = rings_dict["horizon"]
    n_h = len(h_imgs)
    h_adj = np.zeros((n_h, n_h), dtype=int)
    h_cache = {}

    print(f"[TV2] Đang xác định thứ tự xoay 360 độ cho 8 ảnh tầng Horizon...")
    for i in range(n_h):
        for j in range(i + 1, n_h):
            p0, p1, n_m = match_cached(h_imgs[i], h_imgs[j])
            h_adj[i, j] = h_adj[j, i] = n_m
            h_cache[(i, j)] = (p0, p1, n_m)
            h_cache[(j, i)] = (p1, p0, n_m)

    # Tìm chu trình vòng tròn nối 8 ảnh có trọng số match lớn nhất
    curr = 0
    visited = [curr]
    while len(visited) < n_h:
        unvisited = [k for k in range(n_h) if k not in visited]
        next_node = max(unvisited, key=lambda k: h_adj[curr, k])
        visited.append(next_node)
        curr = next_node

    ordered_horizon = [h_imgs[i] for i in visited]
    rings_dict["horizon_ordered"] = ordered_horizon

    # Lưu matches giữa các cặp kề nhau trong vòng tròn Horizon
    for k in range(n_h):
        i_curr, i_next = visited[k], visited[(k + 1) % n_h]
        p0, p1, n_m = h_cache.get((i_curr, i_next), match_cached(h_imgs[i_curr], h_imgs[i_next]))
        matches_dict[(h_imgs[i_curr], h_imgs[i_next])] = {
            "pts0": p0, "pts1": p1, "num_matches": n_m
        }

    # 2. XẾP THỨ TỰ VÒNG TRÒN 7 ẢNH DOWN (PITCH ~ -45°)
    d_imgs = rings_dict["down"]
    n_d = len(d_imgs)
    d_adj = np.zeros((n_d, n_d), dtype=int)
    d_cache = {}

    print(f"[TV2] Đang xác định thứ tự xoay 360 độ cho 7 ảnh tầng Down...")
    for i in range(n_d):
        for j in range(i + 1, n_d):
            p0, p1, n_m = match_cached(d_imgs[i], d_imgs[j])
            d_adj[i, j] = d_adj[j, i] = n_m
            d_cache[(i, j)] = (p0, p1, n_m)
            d_cache[(j, i)] = (p1, p0, n_m)

    curr_d = 0
    visited_d = [curr_d]
    while len(visited_d) < n_d:
        unvisited = [k for k in range(n_d) if k not in visited_d]
        next_node = max(unvisited, key=lambda k: d_adj[curr_d, k])
        visited_d.append(next_node)
        curr_d = next_node

    ordered_down = [d_imgs[i] for i in visited_d]
    rings_dict["down_ordered"] = ordered_down

    for k in range(n_d):
        i_curr, i_next = visited_d[k], visited_d[(k + 1) % n_d]
        p0, p1, n_m = d_cache.get((i_curr, i_next), match_cached(d_imgs[i_curr], d_imgs[i_next]))
        if n_m > 0:
            matches_dict[(d_imgs[i_curr], d_imgs[i_next])] = {
                "pts0": p0, "pts1": p1, "num_matches": n_m
            }

    # 3. KẾT NỐI NỘI BỘ TẦNG UP (PITCH ~ +45°)
    u_imgs = rings_dict["up"]
    n_u = len(u_imgs)
    for i in range(n_u):
        for j in range(i + 1, n_u):
            p0, p1, n_m = match_cached(u_imgs[i], u_imgs[j])
            if n_m >= 15:
                matches_dict[(u_imgs[i], u_imgs[j])] = {
                    "pts0": p0, "pts1": p1, "num_matches": n_m
                }

    # 4. KẾT NỐI ẢNH TRẦN (ZENITH) VỚI 7 ẢNH TẦNG UP
    zenith = rings_dict["zenith"]
    for up_name in rings_dict["up"]:
        p0, p1, n_m = match_cached(zenith, up_name)
        if n_m > 5:
            matches_dict[(zenith, up_name)] = {"pts0": p0, "pts1": p1, "num_matches": n_m}

    # 5. KẾT NỐI ẢNH SÀN (NADIR) VỚI 7 ẢNH TẦNG DOWN
    nadir = rings_dict["nadir"]
    for down_name in rings_dict["down"]:
        p0, p1, n_m = match_cached(nadir, down_name)
        if n_m > 5:
            matches_dict[(nadir, down_name)] = {"pts0": p0, "pts1": p1, "num_matches": n_m}

    # 6. KẾT NỐI TẦNG UP VỚI HORIZON & TẦNG DOWN VỚI HORIZON
    for up_name in rings_dict["up"]:
        best_h = max(h_imgs, key=lambda h: match_cached(up_name, h)[2])
        p0, p1, n_m = match_cached(up_name, best_h)
        if n_m > 5:
            matches_dict[(up_name, best_h)] = {"pts0": p0, "pts1": p1, "num_matches": n_m}

    for down_name in rings_dict["down"]:
        best_h = max(h_imgs, key=lambda h: match_cached(down_name, h)[2])
        p0, p1, n_m = match_cached(down_name, best_h)
        if n_m > 5:
            matches_dict[(down_name, best_h)] = {"pts0": p0, "pts1": p1, "num_matches": n_m}

    print(f"[TV2] Hoàn tất xây dựng đồ thị ghép điểm! Tổng số cặp liên kết: {len(matches_dict)}")
    return matches_dict


# =====================================================================
# KHỐI KIỂM THỬ TRỰC TIẾP TỪ TERMINAL (CLI TEST HARNESS)
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử phân loại 5 tầng ảnh (Thành viên 2)")
    parser.add_argument("--set", default="sets/t01", help="Đường dẫn đến thư mục set (default: sets/t01)")
    parser.add_argument("--weights", default="ring_classifier.pth", help="Đường dẫn file trọng số (default: ring_classifier.pth)")
    parser.add_argument("--build_graph", action="store_true", help="Chạy tiếp tục bước xây dựng đồ thị ghép điểm LightGlue")
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

    if args.build_graph:
        print("\n--- BẮT ĐẦU XÂY DỰNG ĐỒ THỊ GHÉP ĐIỂM (LIGHTGLUE) ---")
        graph = build_match_graph(images_dir, rings)
        print(f"\nThứ tự xoay vòng tròn 8 ảnh Horizon (360 độ):")
        print(" -> ".join(rings.get("horizon_ordered", [])) + f" -> {rings.get('horizon_ordered', [''])[0]}")
        total_pts = sum(v["num_matches"] for v in graph.values())
        print(f"Tổng số cặp tiếp giáp được kết nối: {len(graph)}")
        print(f"Tổng số điểm matches thu được: {total_pts} (trung bình {total_pts / max(len(graph), 1):.1f} điểm/cặp)\n")
