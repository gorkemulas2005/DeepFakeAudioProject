import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    1 Boyutlu Kalıntı Bloğu (Residual Block).
    Derin ağlarda zaman serisi bilgisinin (gradyanların) kaybolmasını engeller.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Eğer giriş ve çıkış boyutları uyuşmuyorsa, kestirme yolu (shortcut) eşitlemek için 1x1 evrişim kullan
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # LeakyReLU kullanıyoruz çünkü ses sinyallerinde negatif genlikler (amplitude) kritik bilgi taşır
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.3)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=0.3)
        return out


class RawNet2_Lite(nn.Module):
    """
    64.000 örneklemlik ham ses (waveform) verisini doğrudan okuyan
    ve RTX 4050 (6GB VRAM) için optimize edilmiş 1D Evrişimsel Sinir Ağı.
    """

    def __init__(self, num_classes=2):
        super(RawNet2_Lite, self).__init__()

        # 1. Dev Adım Katmanı: 64.000 uzunluğundaki diziyi hızla küçült (Memory Optimization)
        # 64000 -> 2000 zaman adımına düşer
        self.first_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=128, stride=32, padding=64, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.3)
        )

        # 2. Kalıntı (Residual) Blokları
        self.layer1 = ResidualBlock1D(32, 64, stride=2)  # 2000 -> 1000
        self.layer2 = ResidualBlock1D(64, 64, stride=2)  # 1000 -> 500
        self.layer3 = ResidualBlock1D(64, 128, stride=2)  # 500 -> 250

        # 3. Global Sıkıştırma ve Sınıflandırma
        self.pool = nn.AdaptiveMaxPool1d(1)  # Tüm zaman eksenini tek bir en yüksek özelliğe ezer

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(0.5),  # Aşırı ezberlemeyi (overfitting) önler
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x).squeeze(-1)  # Boyutu (Batch, 128, 1) -> (Batch, 128) yapar
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # --- MİMARİ RAW TESTİ (Sanity Check) ---
    print("RawNet2 (1D) Mimarisi Test Ediliyor...")
    model = RawNet2_Lite(num_classes=2)

    # Dataloader'dan gelen o devasa ham ses boyutunu taklit edelim
    dummy_input = torch.randn(16, 1, 64000)
    print(f"Giriş Boyutu: {dummy_input.shape}")

    # Ham sesi ağdan geçir
    output = model(dummy_input)
    print(f"Çıkış Boyutu: {output.shape} -> (Batch, Sınıf Sayısı)")

    if output.shape == (16, 2):
        print("-> BAŞARILI: RawNet2'nin ileri yönlü geçişi kusursuz çalışıyor!")