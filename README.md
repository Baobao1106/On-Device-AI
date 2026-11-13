# 🌌 Jetson ERP Challenge - Team 2
> **From 24 Perspective Images to 360° ERP Panorama on NVIDIA Jetson Orin Nano Super 8GB**

Dự án nghiên cứu và phát triển hệ thống ghép ảnh toàn cảnh $360^\circ$ Equirectangular Projection (ERP) từ 24 bức ảnh góc rộng rời rạc chạy trực tiếp trên thiết bị nhúng On-Device AI.

---

## 📌 1. Tổng Quan Bài Toán & Thách Thức Kỹ Thuật

* **Đầu vào:** 24 bức ảnh phối cảnh kích thước $864 \times 648$ (JPG), bị xáo trộn tên ngẫu nhiên, không biết thứ tự chụp hay góc xoay.
* **Góc nhìn (FOV):** Nằm trong dải $72^\circ$ – $77^\circ$ (giống nhau cho cả 24 ảnh trong 1 set) $\rightarrow$ *Cần thuật toán tự ước lượng*.
* **Mẫu chụp (Capture Pattern):**
  - **Tầng 1 (Zenith):** 1 ảnh chụp thẳng lên trần ($+90^\circ$).
  - **Tầng 2 (Up):** 7 ảnh ngửa lên ($+45^\circ$, cách nhau $\approx 51.4^\circ$).
  - **Tầng 3 (Horizon):** 8 ảnh ngang tầm mắt ($0^\circ$, cách nhau $45^\circ$).
  - **Tầng 4 (Down):** 7 ảnh chúc xuống ($-45^\circ$, cách nhau $\approx 51.4^\circ$).
  - **Tầng 5 (Nadir):** 1 ảnh chụp thẳng xuống sàn ($-90^\circ$).
  - *Nhiễu thực tế:* Rung tay $\pm 5^\circ$, phơi sáng lệch $\pm 15\%$, sai màu (white balance), tối góc (vignetting), nhiễu hạt.
* **Đầu ra:** 1 ảnh toàn cảnh dạng **Equirectangular Projection (ERP) kích thước chuẩn $2048 \times 1024$ (PNG)**.
* **Quy chế chấm điểm sống còn:**
  - $\text{Tilt} \ge 5^\circ$ hoặc sai kích thước $\rightarrow$ **0 ĐIỂM NGAY LẬP TỨC**.
  - **Chất lượng (80%):** $50 \times \frac{\text{PSNR} - 15}{20} + 50 \times (1 - \frac{\text{LPIPS}}{0.5})$. Khu vực sàn dưới $-60^\circ$ không tính điểm.
  - **Tốc độ (20%):** Giới hạn tối đa 300 giây/set. Đội nhanh nhất được trọn 20 điểm.
  - **Phần cứng thi đấu:** NVIDIA Jetson Orin Nano Super 8GB (chế độ MAXN SUPER).

---

## 👥 2. Phân Công Vai Trò Nhóm 4 Thành Viên

```
                 ┌──────────────────────────────────────────────┐
                 │          THÀNH VIÊN 4: DATA & DEPLOY         │
                 │   - Vận hành run_eval.py & score.py          │
                 │   - Mở rộng dataset qua make_set.py          │
                 │   - Chuẩn bị Jetson, ONNX / TensorRT FP16    │
                 └──────────────────────┬───────────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│    THÀNH VIÊN 2      │     │    THÀNH VIÊN 1      │     │    THÀNH VIÊN 3      │
│  (AI & Phân tầng)    │────►│ (Hình học & Tilt)    │────►│  (Warping & Blend)   │
│ - Phân loại 5 tầng   │     │ - Ước lượng FOV      │     │ - Chiếu canvas 2048  │
│ - SuperPoint +       │     │ - Tối ưu Bundle Adj. │     │ - Bù sáng (Gain)     │
│   LightGlue          │     │ - Leveling (Tilt<1°) │     │ - Khử seam 360 độ    │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### 👤 Thành viên 1: Geometry & Optimization (`src/geometry.py`)
* **Nhiệm vụ:**
  1. Quét tìm FOV tối ưu trong dải $[72^\circ, 77^\circ]$.
  2. Xây dựng bộ giải ma trận xoay toàn cục (Rotation Averaging / Bundle Adjustment) cho 24 camera.
  3. Thuật toán **Leveling (Chống Tilt)**: Tìm Up-vector qua mặt phẳng 8 ảnh Horizon để đưa độ nghiêng Tilt về $< 1^\circ$ (Tránh bị 0 điểm).
* **Đầu ra bàn giao:** FOV chuẩn + Dictionary 24 ma trận xoay $3\times3$ ($R$) đã cân bằng thăng bằng.

### 👤 Thành viên 2: AI Deep Matcher & Ring Classification (`src/matcher.py`)
* **Nhiệm vụ:**
  1. **Phân loại 24 ảnh vào 5 tầng** theo cấu trúc Capture Pattern (1 Zenith, 7 Up, 8 Horizon, 7 Down, 1 Nadir).
  2. Tích hợp **SuperPoint + LightGlue** để tìm điểm tương ứng, giải quyết triệt để điểm yếu ở các bức ảnh **tường trắng trơn, trần nhà nhẵn bóng** (khắc phục lỗi set `t05`).
  3. Lọc cặp ảnh lân cận thông minh (~45 cặp thay vì 276 cặp) để tối ưu thời gian chạy.
* **Đầu ra bàn giao:** Danh sách phân tầng + Tọa độ các điểm khớp (matches) giữa các ảnh kề nhau.

### 👤 Thành viên 3: Warping, Color Alignment & Seam Blending (`src/blender.py`)
* **Nhiệm vụ:**
  1. Chiếu ngược (Inverse Spherical Remapping) 24 ảnh lên canvas $2048 \times 1024$.
  2. Thuật toán **Gain Compensation**: Bù sáng và cân bằng trắng để triệt tiêu hiện tượng loang lổ sáng/tối.
  3. **Multi-band Blending (Laplacian Pyramid)**: Trộn màu mượt mà, giữ độ sắc nét tối đa để đạt điểm LPIPS cao.
  4. Xử lý mép nối vòng tròn $360^\circ$ (`BORDER_WRAP`) để đạt chuẩn kiểm tra `seam_hint_ok`.
* **Đầu ra bàn giao:** Ảnh toàn cảnh `output.png` kích thước $2048 \times 1024$.

### 👤 Thành viên 4: Data Engineering, Benchmark & Jetson Deployment
* **Nhiệm vụ:**
  1. **Tuần 1 (Offline PC):** Thiết lập script tự động chạy đo điểm trên cả 10 set qua `run_eval.py`. Dùng `make_set.py` tạo thêm 15–20 set thi thử mới để kiểm tra chống Overfit.
  2. **Tuần 2 (Jetson):** Quản lý slot SSH Tailscale, setup môi trường mamba, kích hoạt chế độ MAXN SUPER, chuyển đổi mô hình sang **ONNX / TensorRT FP16**, kiểm soát RAM $< 6\text{GB}$ (chống OOM) và tối ưu thời gian $< 30\text{s}$/set.

---

## ⚡ 3. Cơ Chế Làm Việc Song Song (Parallel Workflow)

**Không ai phải chờ ai!** Tất cả 4 thành viên đều có thể code và tự kiểm thử ngay hôm nay:
* **TV3** lấy ngay góc xoay chuẩn từ file `sets/t01/hidden/true_poses.json` làm dữ liệu giả lập để hoàn thiện module Warping & Blending. Chạy `score.py` kiểm tra xem điểm có đạt $> 85$ không.
* **TV1** lấy `true_poses.json` làm chuẩn đối sánh để kiểm tra thuật toán ước lượng FOV và ma trận $R$ của mình có lệch $< 0.5^\circ$ hay không.
* **TV2** lấy các ảnh trong set khó `sets/t05/public/images/` để kiểm tra xem LightGlue có bắt được điểm tường trắng hay không.
* **TV4** chạy `run_eval.py` và `make_set.py` để tạo hạ tầng benchmark.

---

## 🛠️ 4. Cài Đặt Môi Trường (Installation)

Yêu cầu: Python 3.10 hoặc 3.11.

```bash
# 1. Clone repository
git clone <URL_REPO_CUA_BAN>
cd "Jetson ERP-20261113T040917Z-1-001"

# 2. Tạo môi trường ảo (khuyên dùng)
python -m venv venv
# Trên Windows:
.\venv\Scripts\activate
# Trên Linux / macOS:
source venv/bin/activate

# 3. Cài đặt các thư viện bắt buộc
pip install -r requirements.txt
```

---

## 🚀 5. Hướng Dẫn Sử Dụng & Đánh Giá

### Chạy pipeline trên 1 set ảnh:
```bash
python pipeline.py --images sets/t01/public/images --out runs/pred_t01.png
```

### Chấm điểm 1 kết quả đầu ra:
```bash
python score.py --set sets/t01 --pred runs/pred_t01.png
```

### Chạy đánh giá tự động trên toàn bộ 10 set (Đo điểm & Thời gian):
```bash
python run_eval.py --sets sets --out runs/my_run --cmd "python pipeline.py --images {images} --out {out}" --score
```

### Tự sinh thêm set kiểm thử mới từ ảnh 360° bất kỳ:
```bash
python make_set.py --erp panos/t01.jpg --out sets/scene_moi --seed 42
```

---

## 📁 6. Cấu Trúc Thư Mục Dự Án

```
├── .gitignore              # Danh sách file loại trừ không đẩy lên Git
├── README.md               # Tài liệu hướng dẫn & phân công công việc
├── requirements.txt        # Danh mục thư viện Python chuẩn
├── pipeline.py             # Script chạy chính kết nối TV1, TV2, TV3
├── common.py               # Thư viện hình học cầu chuẩn của BTC
├── score.py                # Script chấm điểm PSNR, LPIPS, Tilt
├── run_eval.py             # Script tự động đo thời gian và điểm 10 set
├── make_set.py             # Script sinh thêm dataset 24 ảnh từ ảnh 360
├── 26 On-device Team.xlsx  # Danh sách phân công nhóm thi đấu
├── jetson-erp-en.pptx      # Slide quy chế cuộc thi (Tiếng Anh)
│
├── src/                    # Mã nguồn các module của nhóm
│   ├── __init__.py
│   ├── matcher.py          # [TV2] Trích xuất đặc trưng & Phân 5 tầng
│   ├── geometry.py         # [TV1] Ước tính FOV, Pose 3D & Leveling
│   └── blender.py          # [TV3] Chiếu ảnh cầu, Bù sáng & Ghép mép
│
├── panos/                  # Các ảnh panorama gốc mẫu (t01 -> t10)
└── sets/                   # 10 tập dữ liệu chuẩn (t01 -> t10)
    ├── t01/
    │   ├── public/images/  # 24 ảnh đầu vào (tên hash ngẫu nhiên)
    │   └── hidden/         # Ground truth, mask, true_poses.json
    └── ...
```

---

## 🌿 7. Quy Ước Nhánh Git (Git Branching)

* `main`: Nhánh ổn định, chỉ merge khi pipeline đã chạy thông suốt cả 10 set.
* `dev`: Nhánh tích hợp chung của nhóm.
* `feature/tv1-geometry`: Nhánh phát triển của Thành viên 1.
* `feature/tv2-matcher`: Nhánh phát triển của Thành viên 2.
* `feature/tv3-blender`: Nhánh phát triển của Thành viên 3.
* `feature/tv4-eval-opt`: Nhánh phát triển của Thành viên 4.
