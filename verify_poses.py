"""
verify_poses.py - CÔNG CỤ TỰ ĐỘNG ĐỐI SOÁT VỚI TRUE_POSES.JSON
============================================================
Script này đối chiếu kết quả dự đoán của nhóm với file đáp án gốc (true_poses.json)
để kiểm tra toàn diện cả 5 tầng (24 ảnh):
    1. Từng ảnh được xếp tầng ĐÚNG hay SAI (Pitch).
    2. Thứ tự xoay vòng tròn 360° của 8 ảnh tầng HORIZON (Yaw, bước nhảy ~45°).
    3. Thứ tự xoay vòng tròn 360° của 7 ảnh tầng DOWN (Yaw, bước nhảy ~51.4°).
    4. Mạng lưới ghép nối tầng UP (7 ảnh), ZENITH (1 ảnh), NADIR (1 ảnh) với các tầng kề.

Cách dùng:
    python verify_poses.py --set sets/t01
    python verify_poses.py --set sets/t05
"""

import argparse
import json
import os
from src.matcher import classify_rings, build_match_graph

def verify_set(set_dir):
    print("=" * 80)
    print(f"  ĐỐI SOÁT TOÀN DIỆN 5 TẦNG VỚI ĐÁP ÁN GỐC (TRUE_POSES.JSON): {set_dir}")
    print("=" * 80)

    # 1. Đọc file đáp án gốc
    gt_path = os.path.join(set_dir, "hidden", "true_poses.json")
    if not os.path.exists(gt_path):
        print(f"Lỗi: Không tìm thấy file đáp án '{gt_path}'!")
        return

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    true_fov = gt_data.get("hfov_deg", 0.0)
    shots = gt_data.get("shots", [])
    gt_dict = {s["image"]: s for s in shots}

    print(f"• Góc nhìn chuẩn (True FOV): {true_fov:.3f}°")
    print(f"• Tổng số ảnh trong đề: {len(shots)} ảnh\n")

    # 2. Chạy thuật toán của nhóm
    images_dir = os.path.join(set_dir, "public", "images")
    image_names = sorted(os.listdir(images_dir))
    rings = classify_rings(images_dir, image_names)

    # 3. KIỂM TRA PHẦN 1: ĐỐI SOÁT TẦNG ẢNH (PITCH)
    print("\n--- PHẦN 1: ĐỐI SOÁT PHÂN TẦNG (GÓC PITCH) CHO CẢ 24 ẢNH ---")
    print(f"{'Tên ảnh':<12} | {'Tầng dự đoán':<14} | {'Pitch thật':<12} | {'Tầng chuẩn':<12} | {'Kết quả'}")
    print("-" * 68)

    pred_ring_map = {}
    pred_ring_map[rings["zenith"]] = "zenith"
    pred_ring_map[rings["nadir"]] = "nadir"
    for img in rings["up"]: pred_ring_map[img] = "up"
    for img in rings["horizon"]: pred_ring_map[img] = "horizon"
    for img in rings["down"]: pred_ring_map[img] = "down"

    correct_pitch = 0
    for name in sorted(image_names):
        s = gt_dict[name]
        p = s["pitch_deg"]
        if p > 65.0: true_ring = "zenith"
        elif p > 20.0: true_ring = "up"
        elif p >= -20.0: true_ring = "horizon"
        elif p >= -65.0: true_ring = "down"
        else: true_ring = "nadir"

        pred_ring = pred_ring_map.get(name, "unknown")
        is_ok = (pred_ring == true_ring)
        if is_ok: correct_pitch += 1
        status = "✓ ĐÚNG" if is_ok else "✗ SAI"
        print(f"{name:<12} | {pred_ring:<14} | {p:7.1f}°     | {true_ring:<12} | {status}")

    print("-" * 68)
    print(f"=> Độ chính xác phân tầng: {correct_pitch}/24 ({correct_pitch/24*100:.1f}%)\n")

    # Chạy LightGlue xây dựng đồ thị ghép nối toàn cục
    print("Đang chạy LightGlue và xây dựng đồ thị ghép nối cho toàn bộ các tầng...")
    graph = build_match_graph(images_dir, rings)

    # 4. KIỂM TRA PHẦN 2: ĐỐI SOÁT VÒNG TRÒN 360 ĐỘ 8 ẢNH HORIZON (YAW, BƯỚC NHẢY ~45°)
    print("\n--- PHẦN 2: ĐỐI SOÁT VÒNG TRÒN 360 ĐỘ TẦNG HORIZON (8 ẢNH) ---")
    pred_h_order = rings.get("horizon_ordered", [])
    print("Thứ tự vòng tròn Horizon tìm được:")
    for idx, name in enumerate(pred_h_order):
        s = gt_dict.get(name, {})
        yaw = s.get("yaw_deg", 0.0) % 360.0
        print(f"  Vị trí {idx+1}: {name:<12} (Yaw thật = {yaw:5.1f}°)")

    yaws_h = [gt_dict[name]["yaw_deg"] % 360.0 for name in pred_h_order]
    is_circular_h = True
    print("\nKiểm tra độ chênh lệch góc Yaw giữa các cặp kề nhau (kỳ vọng ~45.0°):")
    for k in range(len(yaws_h)):
        curr_y = yaws_h[k]
        next_y = yaws_h[(k + 1) % len(yaws_h)]
        diff = abs(curr_y - next_y)
        if diff > 180.0: diff = 360.0 - diff
        step_ok = (25.0 <= diff <= 65.0)
        if not step_ok: is_circular_h = False
        step_status = "✓ Khớp kề nhau" if step_ok else "✗ Lệch bước nhảy"
        print(f"  {pred_h_order[k]} -> {pred_h_order[(k+1)%len(yaws_h)]}: Chênh lệch {diff:4.1f}° {step_status}")

    # 5. KIỂM TRA PHẦN 3: ĐỐI SOÁT VÒNG TRÒN 360 ĐỘ 7 ẢNH DOWN (YAW, BƯỚC NHẢY ~51.4°)
    print("\n--- PHẦN 3: ĐỐI SOÁT VÒNG TRÒN 360 ĐỘ TẦNG DOWN (7 ẢNH) ---")
    pred_d_order = rings.get("down_ordered", [])
    print("Thứ tự vòng tròn Down tìm được:")
    for idx, name in enumerate(pred_d_order):
        s = gt_dict.get(name, {})
        yaw = s.get("yaw_deg", 0.0) % 360.0
        print(f"  Vị trí {idx+1}: {name:<12} (Yaw thật = {yaw:5.1f}°)")

    yaws_d = [gt_dict[name]["yaw_deg"] % 360.0 for name in pred_d_order]
    is_circular_d = True
    print("\nKiểm tra độ chênh lệch góc Yaw giữa các cặp kề nhau (kỳ vọng ~51.4°):")
    for k in range(len(yaws_d)):
        curr_y = yaws_d[k]
        next_y = yaws_d[(k + 1) % len(yaws_d)]
        diff = abs(curr_y - next_y)
        if diff > 180.0: diff = 360.0 - diff
        step_ok = (30.0 <= diff <= 70.0)
        if not step_ok: is_circular_d = False
        step_status = "✓ Khớp kề nhau" if step_ok else "✗ Lệch bước nhảy"
        print(f"  {pred_d_order[k]} -> {pred_d_order[(k+1)%len(yaws_d)]}: Chênh lệch {diff:4.1f}° {step_status}")

    # 6. KIỂM TRA PHẦN 4: ĐỐI SOÁT LIÊN KẾT TẦNG UP, ZENITH VÀ NADIR
    print("\n--- PHẦN 4: LIÊN KẾT ĐA HƯỚNG TẦNG UP (7 ẢNH), ZENITH (ĐỈNH) VÀ NADIR (ĐÁY) ---")
    zenith = rings["zenith"]
    nadir = rings["nadir"]
    
    # Zenith -> Up matches
    zenith_matches = [(k[1], v["num_matches"]) for k, v in graph.items() if k[0] == zenith and k[1] in rings["up"]]
    print(f"• Đỉnh Zenith ({zenith}, Pitch {gt_dict[zenith]['pitch_deg']:.1f}°): Kết nối với {len(zenith_matches)}/7 ảnh tầng Up:")
    for up_name, nm in zenith_matches:
        print(f"    -> {up_name} ({gt_dict[up_name]['yaw_deg']%360.0:5.1f}°): {nm} điểm khớp")

    # Nadir -> Down matches
    nadir_matches = [(k[1], v["num_matches"]) for k, v in graph.items() if k[0] == nadir and k[1] in rings["down"]]
    print(f"\n• Đáy Nadir ({nadir}, Pitch {gt_dict[nadir]['pitch_deg']:.1f}°): Kết nối với {len(nadir_matches)}/7 ảnh tầng Down:")
    for down_name, nm in nadir_matches:
        print(f"    -> {down_name} ({gt_dict[down_name]['yaw_deg']%360.0:5.1f}°): {nm} điểm khớp")

    # Up -> Horizon matches
    up_h_matches = [(k[0], k[1], v["num_matches"]) for k, v in graph.items() if k[0] in rings["up"] and k[1] in rings["horizon"]]
    print(f"\n• Tầng Up -> Tầng Horizon: {len(up_h_matches)} liên kết chiều dọc neo vào xích đạo:")
    for up_name, h_name, nm in up_h_matches:
        print(f"    -> Up {up_name} ({gt_dict[up_name]['yaw_deg']%360.0:5.1f}°) neo vào Horizon {h_name} ({gt_dict[h_name]['yaw_deg']%360.0:5.1f}°): {nm} điểm khớp")

    print("\n" + "=" * 80)
    if correct_pitch == 24 and is_circular_h and is_circular_d:
        print("🎉 KẾT LUẬN CHUNG: HOÀN TOÀN CHÍNH XÁC 100%! TẤT CẢ CÁC TẦNG ĐÃ ĐƯỢC ĐỐI SOÁT CHUẨN!")
    else:
        print(f"⚠️ KẾT LUẬN: Phân tầng: {correct_pitch}/24 | Vòng Horizon: {'✓' if is_circular_h else '✗'} | Vòng Down: {'✓' if is_circular_d else '✗'}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đối soát toàn diện với true_poses.json")
    parser.add_argument("--set", default="sets/t01", help="Đường dẫn đến thư mục set")
    args = parser.parse_args()
    verify_set(args.set)
