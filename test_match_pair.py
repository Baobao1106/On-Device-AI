"""
test_match_pair.py - TRỰC QUAN HÓA KẾT NỐI ĐIỂM GIỮA 2 ẢNH BẰNG LIGHTGLUE
========================================================================
Script này chạy SuperPoint + LightGlue trên 2 bức ảnh tiếp giáp,
vẽ các đường nối điểm màu sắc giữa 2 ảnh và lưu ra file ảnh match_demo.jpg
để bạn có thể mở xem trực tiếp bằng mắt!

Cách chạy:
    python test_match_pair.py
    python test_match_pair.py --img0 sets/t01/public/images/21c5.jpg --img1 sets/t01/public/images/db8d.jpg
"""

import argparse
import os
import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa ghép điểm LightGlue")
    parser.add_argument("--img0", default="sets/t01/public/images/21c5.jpg", help="Đường dẫn ảnh 1")
    parser.add_argument("--img1", default="sets/t01/public/images/db8d.jpg", help="Đường dẫn ảnh 2")
    parser.add_argument("--out", default="match_demo.jpg", help="File ảnh kết quả (default: match_demo.jpg)")
    parser.add_argument("--max_draw", type=int, default=100, help="Số đường nối tối đa hiển thị (default: 100)")
    args = parser.parse_args()

    print(f"--- ĐANG SO KHỚP ĐIỂM GIỮA 2 ẢNH BẰNG SUPERPOINT + LIGHTGLUE ---")
    print(f"Ảnh 1: {args.img0}")
    print(f"Ảnh 2: {args.img1}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device.type.upper()}")

    # 1. Khởi tạo mô hình SuperPoint và LightGlue
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # 2. Đọc ảnh vào PyTorch
    image0 = load_image(args.img0).to(device)
    image1 = load_image(args.img1).to(device)

    # 3. Trích xuất và so khớp
    with torch.no_grad():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01["matches"]  # Shape (M, 2)
    pts0 = feats0["keypoints"][matches[..., 0]].cpu().numpy()
    pts1 = feats1["keypoints"][matches[..., 1]].cpu().numpy()

    num_matches = len(pts0)
    print(f"\n=> TÌM THẤY {num_matches} ĐIỂM TRÙNG KHỚP HỢP LỆ GIỮA 2 ẢNH!")

    # 4. Vẽ hình trực quan hóa bằng OpenCV
    im0_bgr = cv2.imread(args.img0)
    im1_bgr = cv2.imread(args.img1)

    h0, w0 = im0_bgr.shape[:2]
    h1, w1 = im1_bgr.shape[:2]
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1

    # Tạo canvas ghép 2 ảnh cạnh nhau
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = im0_bgr
    canvas[:h1, w0:w0+w1] = im1_bgr

    # Chọn ngẫu nhiên hoặc lấy đều tối đa max_draw điểm để vẽ không bị rối mắt
    indices = np.arange(num_matches)
    if num_matches > args.max_draw:
        indices = np.linspace(0, num_matches - 1, args.max_draw, dtype=int)

    # Bảng màu ngẫu nhiên cho các đường nối
    rng = np.random.default_rng(42)
    colors = [tuple(int(c) for c in rng.integers(50, 255, 3)) for _ in range(len(indices))]

    for i, idx in enumerate(indices):
        p0 = tuple(np.round(pts0[idx]).astype(int))
        p1 = tuple(np.round(pts1[idx] + np.array([w0, 0])).astype(int))
        color = colors[i]

        cv2.circle(canvas, p0, 4, color, -1)
        cv2.circle(canvas, p1, 4, color, -1)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

    # Thêm dòng chữ thông báo lên ảnh
    label_text = f"SuperPoint + LightGlue: {num_matches} matches"
    cv2.putText(canvas, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(args.out, canvas)
    print(f"Đã lưu ảnh trực quan hóa tại: {args.out}")
    print(f"Bạn có thể mở file '{args.out}' lên để xem các đường nối điểm giữa 2 ảnh!")

if __name__ == "__main__":
    main()
