"""
src/matcher.py - DÀNH CHO THÀNH VIÊN 2 (BẢN TỐI ƯU HOÀN THIỆN)
==============================================================
Vai trò: Trí tuệ Nhân tạo (AI Feature Matching) & Phân loại 5 tầng ảnh.

Nhiệm vụ:
1. classify_rings(): Phân loại 24 ảnh vào đúng 5 tầng (Zenith, Up, Horizon, Down, Nadir).
2. extract_features(): Trích xuất đặc trưng bằng SuperPoint.
3. match_pair(): Ghép điểm đặc trưng bằng LightGlue.
4. build_match_graph(): Xây dựng đồ thị liên kết ~45 cặp ảnh tiếp giáp theo capture plan.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MIN_REL_GAIN = 0.001  # ngưỡng cải thiện tối thiểu của refine_rings_by_geometry
CLASS_NAMES = ["zenith", "up", "horizon", "down", "nadir"]

# Biến toàn cục lưu trữ mô hình để tránh nạp lại nhiều lần gây tốn RAM/VRAM
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
    [BƯỚC 1]: Phân loại 24 bức ảnh vào đúng 5 tầng của Capture Plan:
        - 1 Zenith  (Class 0, pitch ~ +90°)
        - 7 Up      (Class 1, pitch ~ +45°)
        - 8 Horizon (Class 2, pitch ~ 0°)
        - 7 Down    (Class 3, pitch ~ -45°)
        - 1 Nadir   (Class 4, pitch ~ -90°)
    """
    model = get_classifier(weights_path)
    probs = []

    # 1. Dự đoán ma trận xác suất Softmax cho cả 24 ảnh
    with torch.no_grad():
        for name in image_names:
            img_path = os.path.join(images_dir, name)
            img = _transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(_device)
            p = torch.softmax(model(img), dim=1).cpu().numpy()[0]
            probs.append(p)
    probs = np.array(probs)  # Shape (24, 5)

    assigned = {"zenith": None, "up": [], "horizon": [], "down": [], "nadir": None}

    # 2. Sử dụng Hungarian Algorithm để gán tối ưu toàn cục (tránh lỗi dây chuyền)
    try:
        from scipy.optimize import linear_sum_assignment
        # 24 vị trí cần gán: 1 Zenith, 7 Up, 8 Horizon, 7 Down, 1 Nadir
        slots = [0] + [1] * 7 + [2] * 8 + [3] * 7 + [4]
        cost_matrix = -np.log(probs[:, slots] + 1e-7)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            cls_id = slots[c]
            img_name = image_names[r]
            if cls_id == 0:
                assigned["zenith"] = img_name
            elif cls_id == 1:
                assigned["up"].append(img_name)
            elif cls_id == 2:
                assigned["horizon"].append(img_name)
            elif cls_id == 3:
                assigned["down"].append(img_name)
            elif cls_id == 4:
                assigned["nadir"] = img_name

    except ImportError:
        # Fallback an toàn nếu môi trường chưa cài đặt scipy
        remain = set(range(len(image_names)))

        zenith_idx = max(remain, key=lambda i: probs[i, 0])
        assigned["zenith"] = image_names[zenith_idx]
        remain.remove(zenith_idx)

        nadir_idx = max(remain, key=lambda i: probs[i, 4])
        assigned["nadir"] = image_names[nadir_idx]
        remain.remove(nadir_idx)

        sorted_up = sorted(remain, key=lambda i: probs[i, 1], reverse=True)
        up_idx = sorted_up[:7]
        assigned["up"] = [image_names[i] for i in up_idx]
        remain -= set(up_idx)

        sorted_down = sorted(remain, key=lambda i: probs[i, 3], reverse=True)
        down_idx = sorted_down[:7]
        assigned["down"] = [image_names[i] for i in down_idx]
        remain -= set(down_idx)

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
    """Trích xuất điểm đặc trưng (Keypoints) và Descriptors bằng SuperPoint."""
    from lightglue.utils import load_image
    extractor, _ = get_lightglue_models()
    image = load_image(image_path).to(_device)
    with torch.no_grad():
        feats = extractor.extract(image)
    return feats


def match_pair(img_path0: str, img_path1: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """So sánh và ghép điểm tương ứng giữa 2 bức ảnh bằng LightGlue."""
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
    [BƯỚC 2]: Xây dựng đồ thị kết nối ~45 cặp ảnh tiếp giáp theo chuẩn capture plan:
        1. Trích xuất đặc trưng SuperPoint cho 24 ảnh một lần duy nhất (caching).
        2. Sắp xếp thứ tự xoay 360 độ cho Horizon (8 ảnh), Up (7 ảnh), Down (7 ảnh).
        3. Kết nối Zenith với 7 ảnh Up; Nadir với 7 ảnh Down.
        4. Kết nối tầng Up và Down với các ảnh Horizon tiếp giáp gần nhất.
    """
    from lightglue.utils import rbd
    _, matcher = get_lightglue_models()

    all_names = [rings_dict["zenith"]] + rings_dict["up"] + rings_dict["horizon"] + rings_dict["down"] + [rings_dict["nadir"]]
    print(f"[TV2] Trích xuất đặc trưng SuperPoint cho {len(all_names)} ảnh...")

    feats_cache = {}
    for name in all_names:
        p = os.path.join(images_dir, name)
        feats_cache[name] = extract_features(p)

    # Bộ nhớ đệm kết quả ghép điểm tránh gọi lại LightGlue nhiều lần
    match_cache = {}

    def match_cached(name_a: str, name_b: str):
        pair_key = (name_a, name_b)
        rev_key = (name_b, name_a)
        if pair_key in match_cache:
            return match_cache[pair_key]
        if rev_key in match_cache:
            p0, p1, n = match_cache[rev_key]
            return p1, p0, n

        f0, f1 = feats_cache[name_a], feats_cache[name_b]
        with torch.no_grad():
            res = matcher({"image0": f0, "image1": f1})
        f0_r, f1_r, res_r = [rbd(x) for x in [f0, f1, res]]
        m = res_r["matches"]
        p0 = f0_r["keypoints"][m[..., 0]].cpu().numpy()
        p1 = f1_r["keypoints"][m[..., 1]].cpu().numpy()
        res_tuple = (p0, p1, len(m))
        match_cache[pair_key] = res_tuple
        return res_tuple

    def order_ring_circle(imgs: List[str]) -> List[str]:
        """Sắp xếp chu trình vòng tròn khép kín tối ưu (Greedy NN + 2-opt Refinement)."""
        n = len(imgs)
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                _, _, n_m = match_cached(imgs[i], imgs[j])
                adj[i, j] = adj[j, i] = n_m

        # Khởi tạo chu trình bằng Greedy Nearest Neighbor
        curr = 0
        cycle = [curr]
        unvisited = set(range(1, n))
        while unvisited:
            next_node = max(unvisited, key=lambda k: adj[cycle[-1], k])
            cycle.append(next_node)
            unvisited.remove(next_node)

        # Tối ưu hóa chu trình bằng 2-opt (khử xoắn chéo vòng tròn)
        def cycle_weight(c):
            return sum(adj[c[k], c[(k + 1) % len(c)]] for k in range(len(c)))

        improved = True
        while improved:
            improved = False
            best_w = cycle_weight(cycle)
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    new_c = cycle[:i + 1] + cycle[i + 1:j + 1][::-1] + cycle[j + 1:]
                    w = cycle_weight(new_c)
                    if w > best_w:
                        cycle = new_c
                        best_w = w
                        improved = True
                        break
                if improved:
                    break

        return [imgs[i] for i in cycle]

    def refine_rings_by_geometry(rings: Dict[str, any]) -> None:
        """
        Tinh chỉnh phân tầng bằng đồ thị match (tổng quát cho mọi cặp tầng, không chỉ Nadir<->Down).
        Điểm của một cách gán = tổng (top-2 match tới ảnh cùng tầng) cho các vòng Horizon/Up/Down
        + tổng match Zenith->Up + tổng match Nadir->Down. Hill-climbing hoán đổi 2 ảnh khác tầng.
        """
        names = [rings["zenith"]] + rings["up"] + rings["horizon"] + rings["down"] + [rings["nadir"]]
        tier = {rings["zenith"]: "zenith", rings["nadir"]: "nadir"}
        for key in ("up", "horizon", "down"):
            for n in rings[key]:
                tier[n] = key

        adj = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                adj[(a, b)] = adj[(b, a)] = match_cached(a, b)[2]

        def score(t):
            total = 0
            for key in ("up", "horizon", "down"):
                members = [n for n in names if t[n] == key]
                for n in members:
                    vals = sorted((adj[(n, m)] for m in members if m != n), reverse=True)
                    total += sum(vals[:2])
            z = next(n for n in names if t[n] == "zenith")
            nd = next(n for n in names if t[n] == "nadir")
            total += sum(adj[(z, n)] for n in names if t[n] == "up")
            total += sum(adj[(nd, n)] for n in names if t[n] == "down")
            return total

        best = score(tier)
        print(f"[TV2] Điểm đồ thị phân tầng ban đầu: {best}")
        changed = False
        while True:
            cand = None
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if tier[a] == tier[b]:
                        continue
                    t2 = dict(tier)
                    t2[a], t2[b] = tier[b], tier[a]
                    sc = score(t2)
                    # Chỉ nhận hoán đổi nếu cải thiện đáng kể (>=0.1%) để tránh nhiễu (vd. t01 Up<->Horizon)
                    if sc > best * (1 + MIN_REL_GAIN) and (cand is None or sc > cand[3]):
                        cand = (a, b, t2, sc)
            if cand is None:
                break
            a, b, tier, best = cand
            changed = True
            print(f"[TV2] Tinh chỉnh phân tầng qua hình học: hoán đổi {a} <-> {b} (điểm đồ thị -> {best})")

        if changed:
            rings["zenith"] = next(n for n in names if tier[n] == "zenith")
            rings["nadir"] = next(n for n in names if tier[n] == "nadir")
            for key in ("up", "horizon", "down"):
                rings[key] = [n for n in names if tier[n] == key]

    refine_rings_by_geometry(rings_dict)

    # 1. XẾP THỨ TỰ VÒNG TRÒN CHO CẢ 3 TẦNG
    print(f"[TV2] Đang xác định thứ tự xoay 360 độ cho 8 ảnh tầng Horizon...")
    ordered_horizon = order_ring_circle(rings_dict["horizon"])
    rings_dict["horizon_ordered"] = ordered_horizon

    print(f"[TV2] Đang xác định thứ tự xoay 360 độ cho 7 ảnh tầng Up...")
    ordered_up = order_ring_circle(rings_dict["up"])
    rings_dict["up_ordered"] = ordered_up

    print(f"[TV2] Đang xác định thứ tự xoay 360 độ cho 7 ảnh tầng Down...")
    ordered_down = order_ring_circle(rings_dict["down"])

    # KIỂM CHỨNG HÌNH HỌC (GEOMETRIC SANITY CHECK):
    # Một vòng tròn Down chuẩn 7 ảnh luôn có các cạnh kề liền mạch (min_edge >= 25).
    # Nếu min_edge < 25, vòng tròn có thể đang bị thiếu 1 ảnh Down (bị tráo nhầm sang Nadir).
    def get_cycle_min_edge(cycle_imgs):
        n = len(cycle_imgs)
        return min(match_cached(cycle_imgs[k], cycle_imgs[(k + 1) % n])[2] for k in range(n))

    down_min_edge = get_cycle_min_edge(ordered_down)
    if down_min_edge < 25:
        lower_8 = [rings_dict["nadir"]] + rings_dict["down"]
        best_cand = None
        best_min = down_min_edge
        best_ordered = ordered_down

        for cand in rings_dict["down"]:
            test_down = [img for img in lower_8 if img != cand]
            test_ordered = order_ring_circle(test_down)
            test_min = get_cycle_min_edge(test_ordered)
            if test_min > best_min and test_min >= 25:
                best_min = test_min
                best_cand = cand
                best_ordered = test_ordered

        if best_cand:
            print(f"[TV2] Tự động sửa lỗi phân tầng qua hình học: Hoán đổi Nadir -> {best_cand} (Độ khớp vòng nhảy từ {down_min_edge} lên {best_min})")
            rings_dict["nadir"] = best_cand
            rings_dict["down"] = [img for img in lower_8 if img != best_cand]
            ordered_down = best_ordered

    rings_dict["down_ordered"] = ordered_down

    matches_dict = {}

    # 2. LƯU MATCHES VÒNG TRÒN TIẾP GIÁP (8 + 7 + 7 = 22 CẶP)
    for k in range(len(ordered_horizon)):
        a, b = ordered_horizon[k], ordered_horizon[(k + 1) % len(ordered_horizon)]
        p0, p1, n = match_cached(a, b)
        matches_dict[(a, b)] = {"pts0": p0, "pts1": p1, "num_matches": n}

    for k in range(len(ordered_up)):
        a, b = ordered_up[k], ordered_up[(k + 1) % len(ordered_up)]
        p0, p1, n = match_cached(a, b)
        if n >= 5:
            matches_dict[(a, b)] = {"pts0": p0, "pts1": p1, "num_matches": n}

    for k in range(len(ordered_down)):
        a, b = ordered_down[k], ordered_down[(k + 1) % len(ordered_down)]
        p0, p1, n = match_cached(a, b)
        if n >= 5:
            matches_dict[(a, b)] = {"pts0": p0, "pts1": p1, "num_matches": n}

    # 3. KẾT NỐI ẢNH TRẦN (ZENITH) VỚI 7 ẢNH TẦNG UP (7 CẶP)
    zenith = rings_dict["zenith"]
    for up_name in ordered_up:
        p0, p1, n = match_cached(zenith, up_name)
        if n >= 5:
            matches_dict[(zenith, up_name)] = {"pts0": p0, "pts1": p1, "num_matches": n}

    # 4. KẾT NỐI ẢNH SÀN (NADIR) VỚI 7 ẢNH TẦNG DOWN (7 CẶP)
    nadir = rings_dict["nadir"]
    for down_name in ordered_down:
        p0, p1, n = match_cached(nadir, down_name)
        if n >= 5:
            matches_dict[(nadir, down_name)] = {"pts0": p0, "pts1": p1, "num_matches": n}

    # 5. KẾT NỐI TẦNG UP VỚI HORIZON (LỌC THÔNG MINH THEO GÓC DANH ĐỊNH, ~7 CẶP)
    best_h_idx = max(range(len(ordered_horizon)), key=lambda idx: match_cached(ordered_up[0], ordered_horizon[idx])[2])
    for i_u in range(len(ordered_up)):
        approx_h_idx = int(round(best_h_idx + i_u * (8.0 / 7.0))) % len(ordered_horizon)
        candidates = [(approx_h_idx - 1) % len(ordered_horizon), approx_h_idx, (approx_h_idx + 1) % len(ordered_horizon)]
        best_cand = max(candidates, key=lambda idx: match_cached(ordered_up[i_u], ordered_horizon[idx])[2])
        p0, p1, n = match_cached(ordered_up[i_u], ordered_horizon[best_cand])
        if n >= 5:
            matches_dict[(ordered_up[i_u], ordered_horizon[best_cand])] = {"pts0": p0, "pts1": p1, "num_matches": n}

    # 6. KẾT NỐI TẦNG DOWN VỚI HORIZON (LỌC THÔNG MINH THEO GÓC DANH ĐỊNH, ~7 CẶP)
    best_hd_idx = max(range(len(ordered_horizon)), key=lambda idx: match_cached(ordered_down[0], ordered_horizon[idx])[2])
    for i_d in range(len(ordered_down)):
        approx_h_idx = int(round(best_hd_idx + i_d * (8.0 / 7.0))) % len(ordered_horizon)
        candidates = [(approx_h_idx - 1) % len(ordered_horizon), approx_h_idx, (approx_h_idx + 1) % len(ordered_horizon)]
        best_cand = max(candidates, key=lambda idx: match_cached(ordered_down[i_d], ordered_horizon[idx])[2])
        p0, p1, n = match_cached(ordered_down[i_d], ordered_horizon[best_cand])
        if n >= 5:
            matches_dict[(ordered_down[i_d], ordered_horizon[best_cand])] = {"pts0": p0, "pts1": p1, "num_matches": n}

    print(f"[TV2] Hoàn tất xây dựng đồ thị ghép điểm! Tổng số cặp liên kết: {len(matches_dict)}")
    return matches_dict
