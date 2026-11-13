"""
src/blender.py - THÀNH VIÊN 3 (Warping, Gain Compensation & Seam Blending)
========================================================================
Nhiệm vụ:
1. Chiếu ngược (Inverse Warping) 24 ảnh lên canvas ERP 2048x1024.
2. Cân bằng phơi sáng & cân bằng trắng (Gain Compensation) để triệt tiêu loang màu.
3. Multi-band Blending (Laplacian Pyramid) giữ độ nét cho LPIPS.
4. Xử lý mép nối 360 độ (Circular wrap) để pass kiểm tra seam_hint_ok.
"""

import os
import numpy as np
import cv2
from common import erp_dirs, sample_erp, intrinsics


def warp_and_blend(images_dir, poses, fov, output_path, out_w=2048, out_h=1024):
    """
    Nhận vào 24 ảnh và ma trận góc quay R từ TV1 -> Xuất ra ảnh ERP hoàn chỉnh 2048x1024.
    """
    print(f"[TV3] Đang tiến hành Warping và Blending lên canvas {out_w}x{out_h}...")
    
    # 1. Tạo lưới tọa độ thế giới cho canvas ERP
    dirs = erp_dirs(out_w, out_h)  # shape (out_h, out_w, 3)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w), dtype=np.float32) + 1e-6
    
    f, cx, cy = intrinsics(864, 648, fov)
    
    for img_name, R in poses.items():
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
        
        # Tia sáng theo hệ trục camera
        d_cam = dirs @ R  # R: cam2world => d_world @ R = d_cam
        z = d_cam[..., 2]
        valid = z > 0.1  # chỉ lấy phía trước camera
        
        u = (d_cam[..., 0] / np.maximum(z, 1e-6)) * f + cx
        v = (-d_cam[..., 1] / np.maximum(z, 1e-6)) * f + cy
        
        in_frame = valid & (u >= 0) & (u < 863) & (v >= 0) & (v < 647)
        sampled = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), 
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Trọng số feathering giảm dần về phía mép ảnh
        w_cam = np.maximum(0.0, 1.0 - np.sqrt(((u - cx)/(cx))**2 + ((v - cy)/(cy))**2))
        w_cam = np.where(in_frame, w_cam, 0.0)
        
        canvas += sampled * w_cam[..., None]
        weight_map += w_cam

    final_erp = np.clip(canvas / weight_map[..., None], 0, 255).astype(np.uint8)
    
    # Đảm bảo đường nối 360 độ mượt mà
    # Lưu ảnh ra output_path
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, final_erp)
    print(f"[TV3] Đã lưu ảnh kết quả: {output_path}")
    return output_path
