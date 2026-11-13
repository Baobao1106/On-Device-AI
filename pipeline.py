"""
pipeline.py - QUẢN LÝ TIẾN TRÌNH TỔNG HỢP (THÀNH VIÊN 4 & CẢ NHÓM)
===================================================================
Script này kết nối các hàm của TV1, TV2, TV3 thành một pipeline thống nhất.
Được gọi tự động bởi run_eval.py trong quá trình đánh giá:
    python pipeline.py --images {images} --out {out}
"""

import argparse
import os
import glob

from src.matcher import classify_rings, build_match_graph
from src.geometry import estimate_fov, solve_global_rotations, apply_leveling
from src.blender import warp_and_blend


def main():
    parser = argparse.ArgumentParser(description="Jetson ERP 360 Panorama Stitching Pipeline")
    parser.add_argument("--images", required=True, help="Thư mục chứa 24 ảnh đầu vào")
    parser.add_argument("--out", required=True, help="Đường dẫn lưu file ảnh ERP 2048x1024 PNG đầu ra")
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    if len(image_paths) != 24:
        print(f"[Cảnh báo] Tìm thấy {len(image_paths)} ảnh trong {args.images} (chuẩn đề bài là 24 ảnh).")

    image_names = [os.path.basename(p) for p in image_paths]

    # BƯỚC 1: THÀNH VIÊN 2 (AI Feature Matching & Phân loại 5 tầng)
    print("\n--- BƯỚC 1: Phân loại tầng ảnh & Ghép điểm đặc trưng (TV2) ---")
    rings_dict = classify_rings(args.images, image_names)
    matches_dict = build_match_graph(args.images, rings_dict)

    # BƯỚC 2: THÀNH VIÊN 1 (Ước lượng FOV & Tối ưu hóa Pose 3D & Leveling)
    print("\n--- BƯỚC 2: Ước lượng FOV, giải góc quay R & Cân bằng chân trời (TV1) ---")
    fov = estimate_fov(matches_dict)
    poses = solve_global_rotations(matches_dict, rings_dict, fov)
    poses = apply_leveling(poses, rings_dict)

    # BƯỚC 3: THÀNH VIÊN 3 (Chiếu ảnh, Cân bằng sáng & Ghép mép 360 độ)
    print("\n--- BƯỚC 3: Chiếu ảnh cầu & Trộn màu xuất file PNG 2048x1024 (TV3) ---")
    warp_and_blend(args.images, poses, fov, args.out)

    print(f"\n[Hoàn tất] File kết quả đã tạo tại: {args.out}")


if __name__ == "__main__":
    main()
