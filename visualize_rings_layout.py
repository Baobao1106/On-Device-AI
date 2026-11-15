"""
visualize_rings_layout.py - GHÉP VÀ TRỰC QUAN HÓA 24 ẢNH THEO 5 TẦNG
===================================================================
Tạo ra một bức ảnh tổng hợp sắp xếp 24 ảnh theo thứ tự không gian 3 chiều:
  - Tầng 1 (Trên cùng): 1 ảnh Zenith (Trần nhà, Pitch +90°)
  - Tầng 2: 7 ảnh Up (Ngửa lên, Pitch +45°) sắp theo vòng xoay 360°
  - Tầng 3 (Giữa): 8 ảnh Horizon (Xích đạo, Pitch 0°) sắp theo vòng xoay 360°
  - Tầng 4: 7 ảnh Down (Úp xuống, Pitch -45°) sắp theo vòng xoay 360°
  - Tầng 5 (Dưới cùng): 1 ảnh Nadir (Sàn nhà, Pitch -90°)
"""

import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
from src.matcher import classify_rings

def create_rings_visualization(set_dir: str, output_path: str):
    images_dir = os.path.join(set_dir, "public", "images")
    image_names = sorted(os.listdir(images_dir))
    
    # 1. Đọc đáp án thật nếu có để lấy góc Yaw/Pitch chuẩn xác
    gt_path = os.path.join(set_dir, "hidden", "true_poses.json")
    gt_dict = {}
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            gt_dict = {s["image"]: s for s in gt_data.get("shots", [])}

    # 2. Phân tầng bằng mô hình của nhóm
    rings = classify_rings(images_dir, image_names)

    # 3. Sắp xếp thứ tự góc Yaw từ trái qua phải (0° -> 360°)
    def sort_by_yaw(img_list):
        if gt_dict:
            return sorted(img_list, key=lambda x: gt_dict[x]["yaw_deg"] % 360.0)
        return img_list

    zenith_img = rings["zenith"]
    up_imgs = sort_by_yaw(rings["up"])
    horizon_imgs = sort_by_yaw(rings["horizon"])
    down_imgs = sort_by_yaw(rings["down"])
    nadir_img = rings["nadir"]

    # 4. Thiết lập kích thước Canvas
    tile_w, tile_h = 240, 180
    gap_x, gap_y = 12, 16
    n_cols = 8  # Tầng Horizon rộng nhất có 8 ảnh
    content_w = n_cols * tile_w + (n_cols - 1) * gap_x
    margin_x = 30
    canvas_w = content_w + margin_x * 2

    header_h = 70
    tier_header_h = 32
    label_h = 36
    row_h = tier_header_h + tile_h + label_h + gap_y
    footer_h = 40
    canvas_h = header_h + row_h * 5 + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(24, 24, 27)) # Nền xám đậm hiện đại #18181b
    draw = ImageDraw.Draw(canvas)

    # Fonts
    font_title = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 24)
    font_subtitle = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
    font_tier = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    font_name = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    font_deg = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 12)

    # Tiêu đề chính
    title_text = f"SƠ ĐỒ TRỰC QUAN 5 TẦNG KHÔNG GIAN (24 ẢNH) - BỘ DỮ LIỆU: {os.path.basename(set_dir.rstrip('/\\\\'))}"
    draw.text((canvas_w // 2, 22), title_text, font=font_title, fill=(255, 255, 255), anchor="mt")
    sub_text = "Zenith (Đỉnh)  →  Up (Ngửa +45°)  →  Horizon (Xích đạo 0°)  →  Down (Úp -45°)  →  Nadir (Đáy) | Xoay 360° từ trái sang phải"
    draw.text((canvas_w // 2, 50), sub_text, font=font_subtitle, fill=(161, 161, 170), anchor="mt")

    # Cấu hình từng tầng: (tên hiển thị, danh sách ảnh, màu viền)
    tiers_config = [
        ("TẦNG 1: ZENITH (ĐỈNH TRẦN • PITCH +90° • 1 ẢNH)", [zenith_img], (245, 158, 11)),       # Vàng Amber
        ("TẦNG 2: UP (NGỬA LÊN • PITCH +45° • 7 ẢNH)", up_imgs, (56, 189, 248)),                  # Xanh Da Trời
        ("TẦNG 3: HORIZON (XÍCH ĐẠO 360° • PITCH 0° • 8 ẢNH)", horizon_imgs, (16, 185, 129)),     # Xanh Lục Emerald
        ("TẦNG 4: DOWN (ÚP XUỐNG • PITCH -45° • 7 ẢNH)", down_imgs, (251, 146, 60)),             # Cam Coral
        ("TẦNG 5: NADIR (ĐÁY SÀN • PITCH -90° • 1 ẢNH)", [nadir_img], (192, 132, 252)),         # Tím Violet
    ]

    curr_y = header_h

    for tier_idx, (tier_title, img_list, color) in enumerate(tiers_config):
        # Tiêu đề tầng
        draw.rectangle([(margin_x, curr_y), (margin_x + 8, curr_y + 20)], fill=color)
        draw.text((margin_x + 16, curr_y + 1), tier_title, font=font_tier, fill=color)
        curr_y += tier_header_h

        # Tính tọa độ X căn giữa cho danh sách ảnh của tầng
        n_imgs = len(img_list)
        tier_total_w = n_imgs * tile_w + (n_imgs - 1) * gap_x
        start_x = (canvas_w - tier_total_w) // 2

        for i, name in enumerate(img_list):
            tile_x = start_x + i * (tile_w + gap_x)
            tile_y = curr_y

            # Đọc và resize ảnh thumbnail
            img_path = os.path.join(images_dir, name)
            thumb = Image.open(img_path).convert("RGB")
            thumb = thumb.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (tile_x, tile_y))

            # Vẽ khung viền màu quanh thumbnail
            draw.rectangle([(tile_x, tile_y), (tile_x + tile_w - 1, tile_y + tile_h - 1)], outline=color, width=2)

            # Vẽ thanh thông tin dưới ảnh
            label_box_y = tile_y + tile_h
            draw.rectangle([(tile_x, label_box_y), (tile_x + tile_w - 1, label_box_y + label_h - 1)], fill=(39, 39, 42))

            # Tên file
            draw.text((tile_x + tile_w // 2, label_box_y + 4), name, font=font_name, fill=(255, 255, 255), anchor="mt")

            # Thông số góc Yaw & Pitch
            if name in gt_dict:
                s = gt_dict[name]
                yaw = s["yaw_deg"] % 360.0
                pitch = s["pitch_deg"]
                angle_str = f"Yaw {yaw:5.1f}° | Pitch {pitch:+4.1f}°"
            else:
                angle_str = f"Vị trí #{i+1}"
            draw.text((tile_x + tile_w // 2, label_box_y + 20), angle_str, font=font_deg, fill=color, anchor="mt")

        curr_y += tile_h + label_h + gap_y

    # Lưu kết quả
    canvas.save(output_path, quality=95)
    print(f"[THÀNH CÔNG] Đã tạo ảnh sơ đồ trực quan tại: {output_path} ({canvas_w}x{canvas_h}px)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo ảnh trực quan 5 tầng")
    parser.add_argument("--set", default="sets/t01", help="Thư mục set cần trực quan hóa (default: sets/t01)")
    parser.add_argument("--out", default=None, help="Đường dẫn file ảnh xuất ra (mặc định: rings_layout_<tên_set>.jpg)")
    args = parser.parse_args()

    set_name = os.path.basename(args.set.rstrip('/\\'))
    out_path = args.out if args.out else f"rings_layout_{set_name}.jpg"
    create_rings_visualization(args.set, out_path)
