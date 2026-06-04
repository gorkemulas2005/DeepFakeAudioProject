import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Gereksiz uyarıları sustur
warnings.filterwarnings('ignore')

# --- DOSYA YOLLARI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_FILE_PATH = os.path.join(BASE_DIR, "data", "keys", "trial_metadata.txt")
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw")


def run_advanced_eda():
    print("=== ASVspoof 2021 İleri Düzey Veri Analizi (EDA) Başlatıldı ===\n")

    # ---------------------------------------------------------
    # 1. METADATA OKUMA VE NULL (KAYIP VERİ) KONTROLÜ
    # ---------------------------------------------------------
    print("[1/4] Cevap Anahtarı (Metadata) Okunuyor ve Temizleniyor...")
    try:
        # Sadece Audio_ID (1) ve Label (5) kolonlarını çekiyoruz (RAM optimizasyonu)
        df_keys = pd.read_csv(KEYS_FILE_PATH, sep='\s+', header=None, usecols=[1, 5], names=['Audio_ID', 'Label'])
    except FileNotFoundError:
        print(f"HATA: {KEYS_FILE_PATH} bulunamadı!")
        return

    null_counts = df_keys.isnull().sum()
    if null_counts.sum() == 0:
        print(" -> HARİKA: Veri setinde hiçbir eksik/bozuk (Null) satır yok. %100 Temiz.")
    else:
        print(f" -> DİKKAT: Veri setinde {null_counts.sum()} adet Null değer bulundu. Satırlar siliniyor...")
        df_keys = df_keys.dropna()

    print(f" -> Toplam Kayıtlı Etiket Sayısı: {len(df_keys):,}\n")

    # ---------------------------------------------------------
    # 2. FİZİKSEL DOSYA TARAMASI (I/O)
    # ---------------------------------------------------------
    print("[2/4] Fiziksel Disk Taranıyor... (Yüz binlerce dosya okunduğu için biraz sürebilir)")
    try:
        # 'set' kullanıyoruz ki arama (lookup) hızı O(1) olsun
        physical_files = set(f.replace('.flac', '') for f in os.listdir(RAW_AUDIO_DIR) if f.endswith('.flac'))
        print(f" -> 'data/raw' klasöründe okunan fiziksel dosya sayısı: {len(physical_files):,}\n")
    except FileNotFoundError:
        print(f"HATA: {RAW_AUDIO_DIR} klasörü bulunamadı!")
        return

    # ---------------------------------------------------------
    # 3. TOPOLOJİK KESİŞİM (INNER JOIN)
    # ---------------------------------------------------------
    print("[3/4] Topolojik Kesişim ve Mantık Testi (Sanity Check) Yapılıyor...")

    # Klasördeki dosyalar ile metadata eşleşiyor mu?
    df_valid = df_keys[df_keys['Audio_ID'].isin(physical_files)]
    missing_in_folder = len(df_keys) - len(df_valid)

    print(f" -> Eğitime Hazır Net Dosya Sayısı: {len(df_valid):,}")
    if missing_in_folder > 0:
        print(f" -> BİLGİ: İndirmediğin (sadece listede olan) dosya sayısı: {missing_in_folder:,}")

    # ---------------------------------------------------------
    # 4. GÖRSELLEŞTİRME VE DAĞILIM (DATA VISUALIZATION)
    # ---------------------------------------------------------
    print("\n[4/4] Analitik Grafikler Çizdiriliyor... (Lütfen açılan pencereyi inceleyin)")

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ASVspoof 2021 Veri Topolojisi Analizi", fontsize=16, fontweight='bold')

    # Sol Grafik: Tüm Veri Seti (İndirilmeyenler Dahil)
    sns.countplot(data=df_keys, x='Label', ax=axes[0], palette="muted", order=['bonafide', 'spoof'])
    axes[0].set_title(f"Genel Kayıt Dağılımı\nToplam: {len(df_keys):,}")
    axes[0].set_xlabel("Sınıflar (Gerçek vs Sahte)")
    axes[0].set_ylabel("Dosya Sayısı")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Sağ Grafik: Fiziksel Olarak Elimizde Olanlar
    sns.countplot(data=df_valid, x='Label', ax=axes[1], palette="pastel", order=['bonafide', 'spoof'])
    axes[1].set_title(f"Disk Üzerindeki (Kullanılabilir) Dağılım\nToplam: {len(df_valid):,}")
    axes[1].set_xlabel("Sınıflar (Gerçek vs Sahte)")
    axes[1].set_ylabel("Dosya Sayısı")
    for p in axes[1].patches:
        axes[1].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_advanced_eda()