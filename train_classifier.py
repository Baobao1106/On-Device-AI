"""
train_classifier.py - HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI 5 TẦNG ẢNH (THÀNH VIÊN 2)
=============================================================================
Mô tả:
    Huấn luyện mạng nơ-ron siêu nhẹ MobileNetV3-Small để tự động nhận diện
    24 bức ảnh phối cảnh thuộc về tầng nào trong 5 tầng của Capture Pattern:
        - Class 0: zenith  (Trần nhà, pitch ~ +90°)
        - Class 1: up      (Tầng ngửa, pitch ~ +45°)
        - Class 2: horizon (Tầng ngang, pitch ~ 0°)
        - Class 3: down    (Tầng úp, pitch ~ -45°)
        - Class 4: nadir   (Sàn nhà, pitch ~ -90°)

Cách chạy:
    python train_classifier.py
    python train_classifier.py --epochs 15 --lr 0.0005 --out ring_classifier.pth
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["zenith", "up", "horizon", "down", "nadir"]


def pitch_to_label(pitch_deg: float) -> int:
    """Chuyển đổi góc pitch thành nhãn từ 0 đến 4."""
    if pitch_deg > 65.0:
        return 0  # zenith (+90 deg)
    elif pitch_deg > 20.0:
        return 1  # up (+45 deg)
    elif pitch_deg >= -20.0:
        return 2  # horizon (0 deg)
    elif pitch_deg >= -65.0:
        return 3  # down (-45 deg)
    else:
        return 4  # nadir (-90 deg)


class ERPDataset(Dataset):
    """Dataset đọc ảnh và nhãn góc pitch từ các set dữ liệu."""

    def __init__(self, sets_dir: str, set_ids: List[str], transform=None):
        self.samples: List[Tuple[str, int, float, str]] = []
        self.transform = transform

        for sid in set_ids:
            json_path = os.path.join(sets_dir, sid, "hidden", "true_poses.json")
            images_dir = os.path.join(sets_dir, sid, "public", "images")
            if not os.path.exists(json_path) or not os.path.exists(images_dir):
                print(f"[Cảnh báo] Bỏ qua {sid}: không tìm thấy file json hoặc images.")
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for shot in data.get("shots", []):
                img_name = shot["image"]
                img_path = os.path.join(images_dir, img_name)
                if not os.path.exists(img_path):
                    continue
                pitch = float(shot["pitch_deg"])
                label = pitch_to_label(pitch)
                self.samples.append((img_path, label, pitch, img_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, pitch, img_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, pitch, img_name


def get_transforms():
    """Hàm tiền xử lý và augmentation ảnh."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),           
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_model(num_classes: int = 5, device: str = "cpu") -> nn.Module:
    """Tải MobileNetV3-Small pre-trained và thay thế classifier cuối cùng."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Thay tầng Linear cuối cùng thành 5 classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(device)


def evaluate(model, dataloader, criterion, device):
    """Đánh giá loss và accuracy trên một dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = (correct / max(total, 1)) * 100.0
    return avg_loss, acc


def test_inference_on_set(model, sets_dir: str, test_set_id: str, transform, device):
    """Chạy thử nghiệm phân loại thực tế trên 24 ảnh của một set mẫu."""
    print(f"\n=======================================================")
    print(f"  CHẠY THỬ NGHIỆM PHÂN LOẠI TRÊN SET {test_set_id}")
    print(f"=======================================================")
    model.eval()
    dataset = ERPDataset(sets_dir, [test_set_id], transform=transform)
    if len(dataset) == 0:
        return

    correct_count = 0
    print(f"{'Ảnh':<12} | {'Pitch thật':<10} | {'Tầng thật':<10} | {'Dự đoán':<10} | {'Kết quả'}")
    print("-" * 60)

    with torch.no_grad():
        for i in range(len(dataset)):
            image, true_label, pitch, img_name = dataset[i]
            img_tensor = image.unsqueeze(0).to(device)
            outputs = model(img_tensor)
            pred_label = torch.argmax(outputs, dim=1).item()

            is_correct = pred_label == true_label
            if is_correct:
                correct_count += 1
            status = "✓ Đúng" if is_correct else "✗ Sai"
            print(f"{img_name:<12} | {pitch:7.1f}°   | {CLASS_NAMES[true_label]:<10} | {CLASS_NAMES[pred_label]:<10} | {status}")

    print("-" * 60)
    print(f"Độ chính xác trên {test_set_id}: {correct_count}/{len(dataset)} ({correct_count/len(dataset)*100:.1f}%)\n")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại 5 tầng ERP")
    parser.add_argument("--sets", default="sets", help="Thư mục chứa các set (default: sets)")
    parser.add_argument("--train_sets", nargs="+", default=[f"t{i:02d}" for i in range(1, 21)],
                        help="Danh sách set dùng để huấn luyện")
    parser.add_argument("--val_sets", nargs="+", default=[f"t{i:02d}" for i in range(21, 26)],
                        help="Danh sách set dùng để kiểm thử (validation)")
    parser.add_argument("--epochs", type=int, default=10, help="Số lượt huấn luyện (default: 10)")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch (default: 16)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--out", default="ring_classifier.pth", help="Tên file lưu trọng số (default: ring_classifier.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị sử dụng: {device.type.upper()}")

    train_tf, val_tf = get_transforms()

    print(f"Đang chuẩn bị dữ liệu...")
    train_dataset = ERPDataset(args.sets, args.train_sets, transform=train_tf)
    val_dataset = ERPDataset(args.sets, args.val_sets, transform=val_tf)

    print(f"- Tập Train ({len(args.train_sets)} sets): {len(train_dataset)} ảnh")
    print(f"- Tập Val   ({len(args.val_sets)} sets): {len(val_dataset)} ảnh")

    if len(train_dataset) == 0:
        sys.exit("Lỗi: Không tìm thấy ảnh huấn luyện nào trong thư mục sets!")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nKhởi tạo mô hình MobileNetV3-Small Pre-trained...")
    model = build_model(num_classes=5, device=device)

    class_weights = torch.tensor([5.0, 1.0, 1.0, 1.0, 5.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Tách tham số thành 2 nhóm với Learning Rate khác nhau
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},    # Backbone học chậm để giữ đặc trưng tốt
        {"params": model.classifier.parameters(), "lr": 1e-3},  # Head phân loại học nhanh
    ], weight_decay=0.01)                                       # Chống Overfitting

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    start_time = time.time()

    print("\nBắt đầu huấn luyện...")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<12}")
    print("-" * 62)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"{epoch:<8} | {train_loss:<12.4f} | {train_acc:<10.1f}% | {val_loss:<12.4f} | {val_acc:<10.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
        scheduler.step()

    elapsed = time.time() - start_time
    print("-" * 62)
    print(f"Huấn luyện hoàn tất trong {elapsed:.1f} giây!")
    print(f"Mô hình tốt nhất với Val Acc = {best_val_acc:.1f}% đã được lưu tại: {args.out}")

    # Tải lại trọng số tốt nhất để chạy thử nghiệm trên set t09
    model.load_state_dict(torch.load(args.out, map_location=device))
    test_inference_on_set(model, args.sets, "t25", val_tf, device)


if __name__ == "__main__":
    main()
