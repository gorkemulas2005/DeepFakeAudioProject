import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             f1_score, precision_score, recall_score, accuracy_score)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

from dataloader import get_augmented_dataloaders
from models.rawnet2 import RawNet2_Lite
from models.senet import SENet_Mel

# --- DOSYA YOLLARI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(DATA_DIR, exist_ok=True)

# Test Edilecek Tüm MLOps Varyantları
MODELS = [
    {"mode": "1d", "name": "rawnet2_baseline.pth", "title": "1D RawNet2 (Baseline)"},
    {"mode": "1d", "name": "rawnet2_focal_g2.pth", "title": "1D RawNet2 (Focal G=2)"},
    {"mode": "1d", "name": "rawnet2_focal_g5.pth", "title": "1D RawNet2 (Focal G=5)"},
    {"mode": "2d", "name": "senet_baseline.pth", "title": "2D SENet (Baseline)"},
    {"mode": "2d", "name": "senet_focal_g2.pth", "title": "2D SENet (Focal G=2)"},
    {"mode": "1d", "name": "rawnet2_robust.pth", "title": "1D RawNet2 (Robust LS)"},
    {"mode": "2d", "name": "senet_robust.pth", "title": "2D SENet (Robust LS)"}
]

BATCH_SIZE = 16
THRESHOLD = 0.50


# --- EER HESAPLAMA (ASVspoof Standart Metriği) ---
def calculate_eer(labels, probs):
    """Equal Error Rate: FPR == FNR noktasını bulur. ASVspoof yarışmasının temel metriği."""
    fpr, tpr, _ = roc_curve(labels, probs)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except ValueError:
        eer = 0.5  # Hesaplanamadığında en kötü senaryo
    return eer * 100  # Yüzde olarak


def get_probabilities(model, dataloader, device):
    all_labels, all_probs = [], []
    use_amp = device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


def run_ultimate_visual_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== KAPSAMLI PERFORMANS DEĞERLENDİRME MOTORU ===\n")

    print("[1/5] Test Mermileri Yükleniyor...")
    _, val_loader_1d = get_augmented_dataloaders(mode='1d', batch_size=BATCH_SIZE)
    _, val_loader_2d = get_augmented_dataloaders(mode='2d', batch_size=BATCH_SIZE)

    results = {}

    print("[2/5] Modeller Sırayla Çıkarım Yapıyor (Inference)...")
    for config in MODELS:
        model_path = os.path.join(CHECKPOINT_DIR, config["name"])
        if not os.path.exists(model_path):
            print(f"[HATA] {config['name']} bulunamadı, atlanıyor.")
            continue

        print(f" -> İşleniyor: {config['title']}")

        if config["mode"] == "1d":
            model = RawNet2_Lite(num_classes=2).to(device)
            loader = val_loader_1d
        else:
            model = SENet_Mel(num_classes=2).to(device)
            loader = val_loader_2d

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        labels, probs = get_probabilities(model, loader, device)
        preds = (probs >= THRESHOLD).astype(int)

        # Tüm metrikleri hesapla
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        eer = calculate_eer(labels, probs)
        acc = accuracy_score(labels, preds) * 100
        f1 = f1_score(labels, preds) * 100
        prec = precision_score(labels, preds, zero_division=0) * 100
        rec = recall_score(labels, preds) * 100

        results[config["title"]] = {
            "labels": labels,
            "probs": probs,
            "preds": preds,
            "cm": confusion_matrix(labels, preds),
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc,
            "eer": eer,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec
        }

        # Model temizliği
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(results) == 0:
        print("[UYARI] Hiçbir model bulunamadı! Önce pipeline.py ile eğitim yapın.")
        return

    # --- TERMİNAL ÖZET TABLOSU ---
    print("\n" + "=" * 100)
    print(f"{'MODEL':<35} {'ACC':>7} {'F1':>7} {'PREC':>7} {'REC':>7} {'AUC':>7} {'EER':>7}")
    print("-" * 100)
    for title, data in results.items():
        print(f"{title:<35} {data['accuracy']:>6.2f}% {data['f1']:>6.2f}% {data['precision']:>6.2f}% "
              f"{data['recall']:>6.2f}% {data['auc']:>6.4f} {data['eer']:>6.2f}%")
    print("=" * 100)

    print("\n[3/5] Karmaşıklık Matrisleri Çiziliyor...")

    # --- GÖRSEL 1: KARMAŞIKLIK MATRİSLERİ ---
    sns.set_theme(style="whitegrid")
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig1.suptitle(f"Hiperparametre ve Mimari Etkisi - Karmaşıklık Matrisleri (Eşik: {THRESHOLD})", fontsize=16,
                  fontweight='bold')
    axes = axes.flatten() if n_models > 1 else [axes]

    for i, (title, data) in enumerate(results.items()):
        cmap = 'Blues' if '1D' in title else 'Oranges'
        sns.heatmap(data['cm'], annot=True, fmt='d', cmap=cmap, ax=axes[i],
                    xticklabels=['Gerçek', 'Sahte'], yticklabels=['Gerçek', 'Sahte'],
                    annot_kws={"size": 12, "weight": "bold"})
        axes[i].set_title(f"{title}\nAcc: %{data['accuracy']:.1f} | F1: %{data['f1']:.1f}",
                          fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Gerçek Sınıf')
        axes[i].set_xlabel('Model Tahmini')

    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig1.savefig(os.path.join(DATA_DIR, "all_variants_confusion_matrices.png"), dpi=300)

    print("[4/5] ROC Eğrileri ve EER Noktaları Çiziliyor...")

    # --- GÖRSEL 2: ROC EĞRİLERİ + EER NOKTASI ---
    fig2, ax_roc = plt.subplots(figsize=(10, 8))
    colors = ['#1f77b4', '#17becf', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']

    for idx, (title, data) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        ax_roc.plot(data['fpr'], data['tpr'], color=color, lw=2,
                    label=f'{title} (AUC={data["auc"]:.4f}, EER=%{data["eer"]:.2f})')

        # EER noktasını ROC üzerine işaretle
        eer_fpr = data['eer'] / 100
        eer_tpr = 1 - eer_fpr
        ax_roc.scatter(eer_fpr, eer_tpr, color=color, s=100, zorder=5,
                       edgecolors='black', linewidths=1.5)

    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    ax_roc.plot([0, 1], [1, 0], color='red', linestyle=':', alpha=0.3, label='EER Referans Çizgisi')
    ax_roc.set_title('Hiperparametrelerin Hata Toleransına Etkisi (ROC/AUC + EER)', fontsize=14, fontweight='bold')
    ax_roc.set_xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=12)
    ax_roc.set_ylabel('Doğru Pozitif Oranı (TPR)', fontsize=12)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(DATA_DIR, "all_variants_roc_curves.png"), dpi=300)

    print("[5/5] Metrik Karşılaştırma Grafikleri Çiziliyor...")

    # --- GÖRSEL 3: METRİK KARŞILAŞTIRMA BAR CHART ---
    metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC×100', '100−EER']
    model_titles = list(results.keys())
    short_titles = [t.replace("1D RawNet2 ", "R2-").replace("2D SENet ", "SE-")
                    .replace("(", "").replace(")", "") for t in model_titles]

    metric_data = []
    for title in model_titles:
        d = results[title]
        metric_data.append([
            d['accuracy'], d['f1'], d['precision'], d['recall'],
            d['auc'] * 100, 100 - d['eer']
        ])
    metric_data = np.array(metric_data)

    fig3, ax_bar = plt.subplots(figsize=(16, 8))
    x = np.arange(len(metric_names))
    width = 0.8 / max(n_models, 1)

    for i, (short_title, full_title) in enumerate(zip(short_titles, model_titles)):
        color = colors[i % len(colors)]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax_bar.bar(x + offset, metric_data[i], width * 0.9, label=short_title,
                          color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        # Değerleri barların üstüne yaz
        for bar, val in zip(bars, metric_data[i]):
            if val > 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax_bar.set_xlabel('Performans Metrikleri', fontsize=12)
    ax_bar.set_ylabel('Skor (%)', fontsize=12)
    ax_bar.set_title('Tüm Model Varyantlarının Kapsamlı Performans Karşılaştırması', fontsize=14, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, fontsize=11)
    ax_bar.legend(loc='lower right', fontsize=8, ncol=2)
    ax_bar.set_ylim(0, 110)
    ax_bar.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig3.savefig(os.path.join(DATA_DIR, "all_variants_metrics_comparison.png"), dpi=300)

    # --- GÖRSEL 4: METRİK ÖZET TABLOSU (GÖRSEL) ---
    fig4, ax_table = plt.subplots(figsize=(16, max(3, n_models * 0.7 + 2)))
    ax_table.axis('off')

    table_data = []
    for title in model_titles:
        d = results[title]
        table_data.append([
            title,
            f"%{d['accuracy']:.2f}",
            f"%{d['f1']:.2f}",
            f"%{d['precision']:.2f}",
            f"%{d['recall']:.2f}",
            f"{d['auc']:.4f}",
            f"%{d['eer']:.2f}"
        ])

    col_labels = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC', 'EER']
    table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center',
                           cellLoc='center', colColours=['#2d3436'] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Başlık hücrelerini beyaz yazı yap
    for j in range(len(col_labels)):
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Satır renklerini mimari tipine göre ayarla
    for i, title in enumerate(model_titles):
        row_color = '#dfe6e9' if '1D' in title else '#ffeaa7'
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(row_color)

    ax_table.set_title('Tüm Model Varyantları — Performans Metrik Özet Tablosu',
                        fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig4.savefig(os.path.join(DATA_DIR, "all_variants_metrics_table.png"), dpi=300)

    print(f"\n-> Operasyon Tamam! Tüm görseller {DATA_DIR} klasörüne kaydedildi:")
    print(f"   1. all_variants_confusion_matrices.png")
    print(f"   2. all_variants_roc_curves.png")
    print(f"   3. all_variants_metrics_comparison.png")
    print(f"   4. all_variants_metrics_table.png")
    plt.show()


if __name__ == "__main__":
    run_ultimate_visual_evaluation()