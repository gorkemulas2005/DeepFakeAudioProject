import os
import librosa
import numpy as np
import pandas as pd
import warnings

# Librosa'nın gereksiz STFT uyarılarını sustur
warnings.filterwarnings('ignore')

# --- TEMİZ MİMARİ DOSYA YOLLARI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_FILE_PATH = os.path.join(BASE_DIR, "data", "keys", "trial_metadata.txt")
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw")

# Çıktı klasörleri (Tek merkezden yönetiliyor)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DIR_1D = os.path.join(PROCESSED_DIR, "1d")
DIR_2D = os.path.join(PROCESSED_DIR, "2d")

# Klasörleri oluştur (Yoksa yaratır)
os.makedirs(DIR_1D, exist_ok=True)
os.makedirs(DIR_2D, exist_ok=True)

# --- SİNYAL İŞLEME PARAMETRELERİ ---
SR = 16000
DURATION = 4
MAX_SAMPLES = SR * DURATION  # 64.000 (Zaman serisi vektörü)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def process_audio(file_path):
    """
    Sesi diskten tek seferde okur, boyutu sabitler ve çatallayarak (Bifurcation)
    hem 1D (Raw) hem de 2D (Mel-Spec) matrislerini döndürür.
    """
    try:
        y, sr = librosa.load(file_path, sr=SR)

        # Boyut Sabitleme (Zero-Padding veya Cropping)
        if len(y) > MAX_SAMPLES:
            y_fixed = y[:MAX_SAMPLES]
        else:
            padding = MAX_SAMPLES - len(y)
            y_fixed = np.pad(y, (0, padding), 'constant')

        # [DAL 1]: 1D Zaman Serisi (RawNet2 için)
        matrix_1d = np.array(y_fixed, dtype=np.float32)

        # [DAL 2]: 2D Uzamsal Matris (SENet için)
        mel_spec = librosa.feature.melspectrogram(
            y=y_fixed, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        matrix_2d = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

        return matrix_1d, matrix_2d
    except Exception as e:
        return None, None


def build_unified_dataset(samples_per_class=20000):
    print("=== Tek Merkezli (Unified) Üretim Hattı Başlatıldı ===")
    print(
        f"Hedef: Her sınıftan {samples_per_class:,} adet olmak üzere toplam {samples_per_class * 2:,} dosya işlenecek.\n")

    # 1. Metadata'yı Oku
    df_keys = pd.read_csv(KEYS_FILE_PATH, sep='\s+', header=None, usecols=[1, 5], names=['Audio_ID', 'Label'])

    # 2. Sınıf Dengeleme (Undersampling)
    bonafide_df = df_keys[df_keys['Label'] == 'bonafide'].sample(n=samples_per_class, random_state=42)
    spoof_df = df_keys[df_keys['Label'] == 'spoof'].sample(n=samples_per_class, random_state=42)

    # 3. Stokastik Karıştırma (Sequential Bias'ı Yok Etmek İçin)
    final_df = pd.concat([bonafide_df, spoof_df]).sample(frac=1, random_state=1337).reset_index(drop=True)

    print("Veri matrisi stokastik olarak karıştırıldı. Dönüşüm başlıyor...")
    print("NOT: Bu işlem işlemci (CPU) gücüne bağlı olarak yaklaşık 1-2 saat sürebilir. Fan seslerine hazırlıklı ol!\n")

    success_count = 0
    total = len(final_df)

    for idx, row in final_df.iterrows():
        audio_id = row['Audio_ID']
        file_path = os.path.join(RAW_AUDIO_DIR, f"{audio_id}.flac")

        # Sesi çatallı fonksiyona yolla
        mat_1d, mat_2d = process_audio(file_path)

        if mat_1d is not None and mat_2d is not None:
            # 1D ve 2D matrisleri diske yaz
            np.save(os.path.join(DIR_1D, f"{audio_id}.npy"), mat_1d)
            np.save(os.path.join(DIR_2D, f"{audio_id}.npy"), mat_2d)
            success_count += 1

        # İlerleme Raporu (Her 500 dosyada bir)
        if (idx + 1) % 500 == 0:
            print(f"İlerleyiş: {idx + 1:,} / {total:,} | Başarıyla Kaydedilen: {success_count:,}")

    print(f"\n=== Operasyon Tamamlandı! ===")
    print(f"Toplam {success_count:,} dosya hem 1D hem 2D formatında 'data/processed' klasörüne yazıldı.")


if __name__ == "__main__":
    # Toplam 40.000 veri için her sınıftan 20.000 çekiyoruz.
    build_unified_dataset(samples_per_class=20000)