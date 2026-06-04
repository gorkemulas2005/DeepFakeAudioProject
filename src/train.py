import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_augmented_dataloaders
from models.senet import SENet_Mel
from models.rawnet2 import RawNet2_Lite

# --- AGRESİF HİPERPARAMETRELER ---
BATCH_SIZE = 8  # VRAM dostu
ACCUMULATION_STEPS = 4  # 8 x 4 = 32 (Sanal Yığın Boyutu)
EPOCHS = 40  # Erken durdurma ile kesilecek
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3  # Eskiye göre 10 kat artırıldı (Overfit'i ezer geçer)
PATIENCE = 6  # Erken durdurma sabrı

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class EarlyStopping:
    def __init__(self, patience=5, path='checkpoint.pth'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            print(f"Uyarı: Erken Durdurma Sayacı: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_sota(mode='1d'):
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== SOTA EĞİTİM MOTORU | HEDEF: {mode.upper()} ===")
    print(f"Çekirdek: {device.type.upper()} | AMP: AKTİF | ACCUMULATION: {ACCUMULATION_STEPS}x\n")

    train_loader, val_loader = get_augmented_dataloaders(mode=mode, batch_size=BATCH_SIZE)

    if mode == '1d':
        model = RawNet2_Lite(num_classes=2).to(device)
        model_name = "rawnet2_sota.pth"
    else:
        model = SENet_Mel(num_classes=2).to(device)
        model_name = "senet_sota.pth"

    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name)
    early_stopping = EarlyStopping(patience=PATIENCE, path=checkpoint_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        # --- EĞİTİM FAZI ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad()  # Döngüden önce sıfırla

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                # Loss değerini birikim sayısına bölüyoruz ki gradyanlar patlamasın
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            # Sadece belirlenen adımda bir ağırlıkları güncelle (Gradient Accumulation)
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # İstatistikleri toplarken orijinal loss'u geri çarp (doğru ortalama için)
            train_loss += loss.item() * ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # --- DOĞRULAMA FAZI ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        print(f"[Epoch {epoch + 1:02d}] "
              f"Train Loss: {avg_train_loss:.4f} | Acc: %{train_acc:.2f} || "
              f"Val Loss: {avg_val_loss:.4f} | Acc: %{val_acc:.2f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"\n[!] Model optimum noktayı geçti. Eğitim {epoch + 1}. Epoch'ta durduruldu!")
            break

    print(f"\nOperasyon Tamamlandı! Zırhlı ağırlıklar kaydedildi: {checkpoint_path}")


if __name__ == "__main__":
    # İlk kapışma: Sinyal canavarı RawNet2 ile başlıyoruz.
    train_sota(mode='2d')