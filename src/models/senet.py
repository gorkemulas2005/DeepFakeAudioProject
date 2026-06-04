import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Bloğu.
    Kanallar (Frekans filtreleri) arasındaki ilişkileri öğrenip, önemli kanalları vurgular.
    """

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: Uzamsal boyutları (H x W) 1x1'e ezen Global Average Pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation: Kanalların önem derecelerini öğrenen 2 katmanlı MLP
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()  # 0 ile 1 arasında ağırlık üretir
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        # Squeeze işlemi
        out = self.squeeze(x).view(batch, channels)
        # Excitation (Ağırlıkları hesapla)
        out = self.excitation(out).view(batch, channels, 1, 1)
        # Orijinal giriş matrisini bu ağırlıklarla çarp (Scale)
        return x * out.expand_as(x)


class SENet_Mel(nn.Module):
    """
    Mel-Spektrogramlar (2D Görüntüler) üzerinden çalışan hafifletilmiş SENet Mimarisi.
    6 GB VRAM sınırına özel optimize edilmiştir.
    """

    def __init__(self, num_classes=2):
        super(SENet_Mel, self).__init__()

        # 1. Blok (Giriş Kanalı: 1 -> 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 2. Blok (Kanal: 32 -> 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 3. Blok (Kanal: 64 -> 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Sınıflandırıcı Kafa (Classifier Head)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # Aşırı öğrenmeyi (Overfitting) engeller
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # --- MİMARİ MATRİS TESTİ (Sanity Check) ---
    print("SENet Mimarisi Test Ediliyor...")
    model = SENet_Mel(num_classes=2)

    # Dataloader'dan gelen o mükemmel tensör boyutunu taklit edelim
    dummy_input = torch.randn(16, 1, 128, 126)
    print(f"Giriş Boyutu: {dummy_input.shape}")

    # Matrisi ağdan geçir
    output = model(dummy_input)
    print(f"Çıkış Boyutu: {output.shape} -> (Batch, Sınıf Sayısı)")

    if output.shape == (16, 2):
        print("-> BAŞARILI: Ağın ileri yönlü geçişi (Forward Pass) kusursuz çalışıyor!")