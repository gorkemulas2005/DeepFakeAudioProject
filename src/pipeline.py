import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import gc

from dataloader import get_augmented_dataloaders
from models.rawnet2 import RawNet2_Lite
from models.senet import SENet_Mel
from losses import FocalLoss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- GRID SEARCH PARAMETRE UZAYI ---
GRID_CONFIG = [
    {"mode": "1d", "loss_type": "ce", "gamma": 0.0, "name": "rawnet2_baseline.pth"},
    {"mode": "1d", "loss_type": "focal", "gamma": 2.0, "name": "rawnet2_focal_g2.pth"},
    {"mode": "1d", "loss_type": "focal", "gamma": 5.0, "name": "rawnet2_focal_g5.pth"},
    {"mode": "2d", "loss_type": "ce", "gamma": 0.0, "name": "senet_baseline.pth"},
    {"mode": "2d", "loss_type": "focal", "gamma": 2.0, "name": "senet_focal_g2.pth"},
    {"mode": "1d", "loss_type": "ce_ls", "gamma": 0.0, "name": "rawnet2_robust.pth"},
    {"mode": "2d", "loss_type": "ce_ls", "gamma": 0.0, "name": "senet_robust.pth"}
]

BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
EPOCHS = 15  # Hızlı otonom tarama için epoch kısıtlandı (Erken durdurma entegre edilebilir)


def run_grid_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"=== OTONOM MLOps GRID SEARCH PIPELINE BAŞLATILDI ===")
    print(f"Cihaz: {device.type.upper()} | AMP: {'AKTİF' if use_amp else 'KAPALI'}")
    if use_amp:
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print()

    for config in GRID_CONFIG:
        model_path = os.path.join(CHECKPOINT_DIR, config["name"])

        # KORUMA KALKANI: Eğer bu model daha önce eğitildiyse, eğitimi atla, zamanı boşa harcama!
        if os.path.exists(model_path):
            print(f"[PAS GEÇİLDİ] {config['name']} zaten diskte mevcut. Üzerine yazılmıyor.")
            continue

        print(
            f"\n{'='*60}\n[Ateşleniyor] Mimari: {config['mode'].upper()} | Loss: {config['loss_type'].upper()} (Gamma: {config['gamma']})\n{'='*60}")

        # VRAM Temizliği — önceki modelden kalan fragmantasyonu sıfırla
        if use_amp:
            torch.cuda.empty_cache()
            gc.collect()

        try:
            # Dataloader ve Model Seçimi
            train_loader, val_loader = get_augmented_dataloaders(mode=config["mode"], batch_size=BATCH_SIZE)

            if config["mode"] == "1d":
                model = RawNet2_Lite(num_classes=2).to(device)
            else:
                model = SENet_Mel(num_classes=2).to(device)

            # Kayıp Fonksiyonu Seçimi
            if config["loss_type"] == "ce":
                criterion = nn.CrossEntropyLoss()
            elif config["loss_type"] == "ce_ls":
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            else:
                criterion = FocalLoss(gamma=config["gamma"])

            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            # device_type keyword kullan — positional 'cuda' argümanı bazı PyTorch/driver
            # kombinasyonlarında STATUS_STACK_BUFFER_OVERRUN (0xC0000409) hatasına yol açıyor.
            scaler = torch.amp.GradScaler(enabled=use_amp)

            best_val_loss = float('inf')
            best_val_acc = 0.0

            # --- EĞİTİM DÖNGÜSÜ ---
            for epoch in range(EPOCHS):
                model.train()
                train_loss, train_correct, train_total = 0.0, 0, 0
                optimizer.zero_grad()

                for i, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels) / ACCUMULATION_STEPS

                    scaler.scale(loss).backward()

                    if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    train_loss += loss.item() * ACCUMULATION_STEPS
                    _, preds = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += preds.eq(labels).sum().item()

                avg_train_loss = train_loss / len(train_loader)
                train_acc = 100. * train_correct / train_total

                # --- DOĞRULAMA FAZI ---
                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, preds = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += preds.eq(labels).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100. * val_correct / val_total

                print(f"  [Epoch {epoch + 1:02d}/{EPOCHS}] "
                      f"Train Loss: {avg_train_loss:.4f} Acc: %{train_acc:.2f} | "
                      f"Val Loss: {avg_val_loss:.4f} Acc: %{val_acc:.2f}")

                # En iyi ağırlığı koru
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), model_path)

                scheduler.step()

            print(
                f"-> [BAŞARILI] {config['name']} optimize edildi. En İyi Val Loss: {best_val_loss:.4f} | Acc: %{best_val_acc:.2f}")

        except RuntimeError as e:
            print(f"\n[HATA] {config['name']} eğitimi başarısız oldu: {e}")
            if "out of memory" in str(e).lower():
                print("[BİLGİ] VRAM yetersiz. Bellek temizlenip sonraki modele geçiliyor...")
            continue

        finally:
            # Bellek sızıntısını önle — modeli, optimizer'ı ve scaler'ı RAM/VRAM'den sil
            for obj_name in ['model', 'optimizer', 'scaler', 'scheduler']:
                if obj_name in locals():
                    del locals()[obj_name]
            if use_amp:
                torch.cuda.empty_cache()
            gc.collect()

    print("\n=== PIPELINE TAMAMLANDI ===")


if __name__ == "__main__":
    run_grid_pipeline()