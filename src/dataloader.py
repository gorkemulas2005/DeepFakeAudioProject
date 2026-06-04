import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- KLASÖR YOLLARI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_FILE_PATH = os.path.join(BASE_DIR, "data", "keys", "trial_metadata.txt")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DIR_1D = os.path.join(PROCESSED_DIR, "1d")
DIR_2D = os.path.join(PROCESSED_DIR, "2d")


class AugmentedAudioDataset(Dataset):
    def __init__(self, dataframe, mode='1d', is_train=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.mode = mode
        self.is_train = is_train  # Sadece eğitimde (Train) gürültü ekleyeceğiz, testte (Val) pürüzsüz olacak.

    def __len__(self):
        return len(self.dataframe)

    def add_1d_noise(self, matrix):
        """1D Sinyale rastgele statik (Gaussian) beyaz gürültü ekler."""
        noise_amp = 0.005 * np.random.uniform() * np.amax(matrix)
        noise = noise_amp * np.random.normal(size=matrix.shape)
        return matrix + noise

    def add_2d_mask(self, matrix):
        """2D Spektrogramın rastgele frekans ve zaman bantlarını siyaha (0) boyar (SpecAugment)."""
        masked = matrix.copy()
        # Rastgele zaman maskesi
        t_width = np.random.randint(1, 10)
        t_start = np.random.randint(0, masked.shape[1] - t_width)
        masked[:, t_start:t_start + t_width] = 0
        # Rastgele frekans maskesi (SpecAugment genişletmesi)
        f_width = np.random.randint(1, 15)
        f_start = np.random.randint(0, masked.shape[0] - f_width)
        masked[f_start:f_start + f_width, :] = 0
        return masked

    def __getitem__(self, idx):
        audio_id = self.dataframe.iloc[idx]['Audio_ID']
        target = self.dataframe.iloc[idx]['Target']

        if self.mode == '1d':
            path = os.path.join(DIR_1D, f"{audio_id}.npy")
            matrix = np.load(path)
            if self.is_train and np.random.rand() > 0.5:  # %50 ihtimalle gürültü ekle
                matrix = self.add_1d_noise(matrix)
            tensor_data = torch.FloatTensor(matrix).unsqueeze(0)

        elif self.mode == '2d':
            path = os.path.join(DIR_2D, f"{audio_id}.npy")
            matrix = np.load(path)
            if self.is_train and np.random.rand() > 0.5:
                matrix = self.add_2d_mask(matrix)
            tensor_data = torch.FloatTensor(matrix).unsqueeze(0)

        return tensor_data, torch.tensor(target, dtype=torch.long)


def get_augmented_dataloaders(mode='1d', batch_size=8, val_split=0.2):
    df_keys = pd.read_csv(KEYS_FILE_PATH, sep='\s+', header=None, usecols=[1, 5], names=['Audio_ID', 'Label'])

    # Processed klasöründeki dosyaları oku
    target_dir = DIR_1D if mode == '1d' else DIR_2D
    processed_files = [f.replace('.npy', '') for f in os.listdir(target_dir) if f.endswith('.npy')]

    data_df = df_keys[df_keys['Audio_ID'].isin(processed_files)].copy()
    data_df['Target'] = data_df['Label'].map({'bonafide': 0, 'spoof': 1})

    # 40.000 veriyi 32.000 Train, 8.000 Val olarak jilet gibi (stratify) böl
    train_df, val_df = train_test_split(
        data_df, test_size=val_split, random_state=42, stratify=data_df['Target']
    )

    print(f"--- Gürültü Enjektörlü Dataloader ({mode.upper()}) ---")
    print(f"Train (Eğitim): {len(train_df):,} | Validation (Doğrulama): {len(val_df):,}")

    train_dataset = AugmentedAudioDataset(train_df, mode=mode, is_train=True)
    val_dataset = AugmentedAudioDataset(val_df, mode=mode, is_train=False)  # Validation saf kalır

    # Num_workers=0 (Windows'ta multi-processing çökmesini engellemek için)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


if __name__ == "__main__":
    # --- KÜÇÜK BİR MANTIK TESTİ (SANITY CHECK) ---
    print("Dataloader Testi Başlatılıyor...")

    # 1D (RawNet2) için DataLoader'ı çağır
    train_loader, val_loader = get_augmented_dataloaders(mode='1d', batch_size=8)

    # Sadece 1 adet Batch (8'li yığın) çek
    data, label = next(iter(train_loader))

    print("\n--- TEST BAŞARILI ---")
    print(f"Mermi (Veri) Boyutu: {data.shape} -> (Batch, Channel, Time Serisi)")
    print(f"Etiketler (0=Gerçek, 1=Sahte): {label}")
    print("Veri hattı kusursuz, gürültü enjektörü aktif!")