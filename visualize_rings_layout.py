"""
visualize_rings_layout.py - GHÉP VÀ TRỰC QUAN HÓA 24 ẢNH THEO 5 TẦNG
===================================================================
Tạo ra một bức ảnh tổng hợp sắp xếp 24 ảnh theo thứ tự không gian 3 chiều:
  - Tầng 1 (Trên cùng): 1 ảnh Zenith (Trần nhà, Pitch +90°)
  - Tầng 2: 7 ảnh Up (Ngửa lên, Pitch +45°) sắp theo vòng xoay 360°
  - Tầng 3 (Giữa): 8 ảnh Horizon (Xích đạo, Pitch 0°) sắp theo vòng xoay 360°
  - Tầng 4: 7 ảnh Down (Úp xuống, Pitch -45°) sắp theo vòng xoay 360°
  - Tầng 5 (Dưới cùng): 1 ảnh Nadir (Sàn nhà, Pitch -90°)

Tính năng mới:
  - Tự động áp dụng kiểm chứng hình học (Geometric Sanity Check) bằng LightGlue
    để sửa lỗi tráo tầng Nadir / Down (mặc định bật).
  - Hỗ trợ cờ `--fast` / `--no-refine` để xem nhanh dự đoán thô của MobileNetV3 (~1s).
  - Tự động phát hiện và cảnh báo viền đỏ với ảnh bị xếp lệch tầng (nếu có ground truth).
"""

import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
from src.matcher import classify_rings, build_match_graph


def create_rings_visualization(set_dir: str, output_path: str, refine: bool = True):
    images_dir = os.path.join(set_dir, "public", "images")
    if not os.path.exists(images_dir):
        print(f"Lỗi: Không tìm thấy thư mục ảnh '{images_dir}'!")
        return

    image_names = sorted(os.listdir(images_dir))
    
    # 1. Đọc đáp án thật nếu có để lấy góc Yaw/Pitch chuẩn xác
    gt_path = os.path.join(set_dir, "hidden", "true_poses.json")
    gt_dict = {}
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            gt_dict = {s["image"]: s for s in gt_data.get("shots", [])}

    # 2. Phân tầng bằng mô hình AI của nhóm
    print(f"\n[visualize] Bước 1: Phân tầng 24 ảnh bằng mô hình MobileNetV3...")
    rings = classify_rings(images_dir, image_names)

    # 3. Kiểm chứng hình học & sắp xếp vòng tròn (nếu bật refine)
    if refine:
        print("[visualize] Bước 2: Đang chạy kiểm chứng hình học & sắp xếp vòng tròn (build_match_graph)...")
        print("            (Mẹo: Dùng thêm cờ '--fast' nếu bạn muốn xem nhanh kết quả thô của AI)")
        build_match_graph(images_dir, rings)
    else:
        print("[visualize] Chạy chế độ --fast: Bỏ qua kiểm chứng hình học LightGlue (dùng kết quả thô của AI).")

    # 4. Sắp xếp thứ tự góc Yaw từ trái qua phải (0° -> 360°)
    if gt_dict:
        def sort_by_yaw(img_list):
            return sorted(img_list, key=lambda x: gt_dict[x]["yaw_deg"] % 360.0)
        zenith_img = rings["zenith"]
        up_imgs = sort_by_yaw(rings["up"])
        horizon_imgs = sort_by_yaw(rings["horizon"])
        down_imgs = sort_by_yaw(rings["down"])
        nadir_img = rings["nadir"]
    else:
        zenith_img = rings["zenith"]
        up_imgs = rings.get("up_ordered", rings["up"])
        horizon_imgs = rings.get("horizon_ordered", rings["horizon"])
        down_imgs = rings.get("down_ordered", rings["down"])
        nadir_img = rings["nadir"]

    # 5. Thiết lập kích thước Canvas
    tile_w, tile_h = 240, 180
    gap_x, gap_y = 12, 16
    n_cols = 8  # Tầng Horizon rộng nhất có 8 ảnh
    content_w = n_cols * tile_w + (n_cols - 1) * gap_x
    margin_x = 30
    canvas_w = content_w + margin_x * 2

    header_h = 70
    tier_header_h = 32
    label_h = 38
    row_h = tier_header_h + tile_h + label_h + gap_y
    footer_h = 40
    canvas_h = header_h + row_h * 5 + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(24, 24, 27))  # Nền tối hiện đại #18181b
    draw = ImageDraw.Draw(canvas)

    # Fonts
    font_path = "C:/Windows/Fonts/segoeui.ttf"
    try:
        font_title = ImageFont.truetype(font_path, 24)
        font_subtitle = ImageFont.truetype(font_path, 14)
        font_tier = ImageFont.truetype(font_path, 16)
        font_name = ImageFont.truetype(font_path, 13)
        font_deg = ImageFont.truetype(font_path, 12)
    except IOError:
        font_title = font_subtitle = font_tier = font_name = font_deg = ImageFont.load_default()

    # Tiêu đề chính
    mode_tag = " (Đã kiểm chứng hình học)" if refine else " (Kết quả AI thô --fast)"
    set_name = os.path.basename(set_dir.rstrip('/\\\\'))
    title_text = f"SƠ ĐỒ TRỰC QUAN 5 TẦNG KHÔNG GIAN (24 ẢNH) - BỘ DỮ LIỆU: {set_name}{mode_tag}"
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

            # Kiểm tra xem ảnh có đúng tầng theo ground truth hay không
            border_color = color
            is_mismatched = False
            if name in gt_dict:
                p = gt_dict[name]["pitch_deg"]
                if tier_idx == 0:
                    is_correct_tier = (p > 65.0)
                elif tier_idx == 1:
                    is_correct_tier = (20.0 < p <= 65.0)
                elif tier_idx == 2:
                    is_correct_tier = (-20.0 <= p <= 20.0)
                elif tier_idx == 3:
                    is_correct_tier = (-65.0 <= p < -20.0)
                elif tier_idx == 4:
                    is_correct_tier = (p < -65.0)
                else:
                    is_correct_tier = True
                
                if not is_correct_tier:
                    border_color = (239, 68, 68)  # Viền đỏ cảnh báo lệch tầng
                    is_mismatched = True

            # Vẽ khung viền màu quanh thumbnail
            draw.rectangle([(tile_x, tile_y), (tile_x + tile_w - 1, tile_y + tile_h - 1)], outline=border_color, width=3 if is_mismatched else 2)

            # Vẽ thanh thông tin dưới ảnh
            label_box_y = tile_y + tile_h
            box_bg = (69, 10, 10) if is_mismatched else (39, 39, 42)
            draw.rectangle([(tile_x, label_box_y), (tile_x + tile_w - 1, label_box_y + label_h - 1)], fill=box_bg)

            # Tên file
            draw.text((tile_x + tile_w // 2, label_box_y + 4), name, font=font_name, fill=(255, 255, 255), anchor="mt")

            # Thông số góc Yaw & Pitch
            if name in gt_dict:
                s = gt_dict[name]
                yaw = s["yaw_deg"] % 360.0
                pitch = s["pitch_deg"]
                status_str = " [SAI TẦNG!]" if is_mismatched else ""
                angle_str = f"Yaw {yaw:5.1f}° | Pitch {pitch:+4.1f}°{status_str}"
                font_c = (239, 68, 68) if is_mismatched else color
            else:
                angle_str = f"Vị trí #{i+1}"
                font_c = color
            draw.text((tile_x + tile_w // 2, label_box_y + 20), angle_str, font=font_deg, fill=font_c, anchor="mt")

        curr_y += tile_h + label_h + gap_y

    # Lưu kết quả
    canvas.save(output_path, quality=95)
    print(f"\n[THÀNH CÔNG] Đã tạo ảnh sơ đồ trực quan tại: {output_path} ({canvas_w}x{canvas_h}px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo ảnh trực quan 5 tầng không gian")
    parser.add_argument("--set", default="sets/t01", help="Thư mục set cần trực quan hóa (default: sets/t01)")
    parser.add_argument("--out", default=None, help="Đường dẫn file ảnh xuất ra (mặc định: rings_layout_<tên_set>.jpg)")
    parser.add_argument("--fast", action="store_true", help="Chế độ xem nhanh: chỉ dùng MobileNetV3 (~1s), bỏ qua kiểm chứng hình học")
    parser.add_argument("--no-refine", action="store_true", help="Không chạy build_match_graph để sửa tầng")
    args = parser.parse_args()

    set_name = os.path.basename(args.set.rstrip('/\\\\'))
    out_path = args.out if args.out else f"rings_layout_{set_name}.jpg"
    do_refine = not (args.fast or args.no_refine)

    create_rings_visualization(args.set, out_path, refine=do_refine)
