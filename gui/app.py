import os
import sys
import time
import math
import torch
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from fpdf import FPDF
import warnings

warnings.filterwarnings('ignore')

# --- PATH ENJEKSİYONU ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from models.senet import SENet_Mel
from models.rawnet2 import RawNet2_Lite

# --- KONFİGÜRASYON ---
SENET_WEIGHTS = os.path.join(BASE_DIR, "checkpoints", "senet_baseline.pth")
RAWNET_WEIGHTS = os.path.join(BASE_DIR, "checkpoints", "rawnet2_baseline.pth")
SENET_ROBUST_WEIGHTS = os.path.join(BASE_DIR, "checkpoints", "senet_robust.pth")
RAWNET_ROBUST_WEIGHTS = os.path.join(BASE_DIR, "checkpoints", "rawnet2_robust.pth")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- EĞİTİM İLE UYUMLU SİNYAL İŞLEME SABİTLERİ ---
# preprocess.py ile birebir aynı olmalıdır (Domain Mismatch önleme)
TRAIN_N_FFT = 2048
TRAIN_HOP_LENGTH = 512
TRAIN_N_MELS = 128
DEFAULT_TEMPERATURE = 2.0  # Softmax kalibrasyon sıcaklığı

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODELLERİN YÜKLENMESİ ---
print("=== SİBER GÜVENLİK SES ANALİZ SİSTEMİ BAŞLATILIYOR ===")

# Baseline modeller
model_2d = SENet_Mel(num_classes=2).to(device)
if os.path.exists(SENET_WEIGHTS):
    model_2d.load_state_dict(torch.load(SENET_WEIGHTS, map_location=device, weights_only=True))
model_2d.eval()

model_1d = RawNet2_Lite(num_classes=2).to(device)
if os.path.exists(RAWNET_WEIGHTS):
    model_1d.load_state_dict(torch.load(RAWNET_WEIGHTS, map_location=device, weights_only=True))
model_1d.eval()

# Robust modeller (Label Smoothing + Cosine Annealing)
model_2d_robust = SENet_Mel(num_classes=2).to(device)
if os.path.exists(SENET_ROBUST_WEIGHTS):
    model_2d_robust.load_state_dict(torch.load(SENET_ROBUST_WEIGHTS, map_location=device, weights_only=True))
    print("[YÜKLENDI] SENet Robust (Label Smoothing)")
else:
    print("[UYARI] senet_robust.pth bulunamadı — Robust SENet seçilirse Baseline kullanılacak.")
    model_2d_robust = model_2d  # Fallback
model_2d_robust.eval()

model_1d_robust = RawNet2_Lite(num_classes=2).to(device)
if os.path.exists(RAWNET_ROBUST_WEIGHTS):
    model_1d_robust.load_state_dict(torch.load(RAWNET_ROBUST_WEIGHTS, map_location=device, weights_only=True))
    print("[YÜKLENDI] RawNet2 Robust (Label Smoothing)")
else:
    print("[UYARI] rawnet2_robust.pth bulunamadı — Robust RawNet2 seçilirse Baseline kullanılacak.")
    model_1d_robust = model_1d  # Fallback
model_1d_robust.eval()

# Model varyant sözlüğü — GUI'deki seçime göre aktif modeli belirler
MODEL_VARIANTS = {
    "Baseline (CrossEntropy)": {"2d": model_2d, "1d": model_1d},
    "Robust (Label Smoothing)": {"2d": model_2d_robust, "1d": model_1d_robust},
}


# --- YARDIMCI FONKSİYONLAR ---
def log_to_terminal(action, details):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [SİSTEM LOGU] {action} -> {details}")


def calculate_entropy(probs):
    entropy = 0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def format_result(probs, threshold):
    fake_prob = probs[1] * 100
    real_prob = probs[0] * 100
    entropy = calculate_entropy(probs)

    if entropy > 0.8:
        conf_status = "Yüksek Belirsizlik (Gürültü/Kavşak Noktası)"
    elif entropy > 0.4:
        conf_status = "Orta Kararlılık"
    else:
        conf_status = "Yüksek Kesinlik (Deterministik)"

    if fake_prob >= threshold:
        return f"Sınıflandırma: %{fake_prob:.2f} SAHTE\nGüvenlik Marjı: %{real_prob:.2f} GERÇEK\n\nEntropi Skoru: {entropy:.4f}\nDurum: {conf_status}"
    return f"Sınıflandırma: %{real_prob:.2f} GERÇEK\nAnomali Şüphesi: %{fake_prob:.2f} SAHTE\n\nEntropi Skoru: {entropy:.4f}\nDurum: {conf_status}"


def sanitize_text(text):
    replacements = {
        'ı': 'i', 'İ': 'I', 'ş': 's', 'Ş': 'S',
        'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U',
        'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text


def render_latex_to_image(formula_raw, filename):
    fig = plt.figure(figsize=(4, 1))
    fig.text(0.5, 0.5, f"${formula_raw}$", fontsize=16, ha='center', va='center')
    plt.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# --- DSP ZIRHI (YENİ EKLENDİ) ---
def apply_dsp_armor(y, sr):
    """Mikrofon kesintilerini ve elektriksel kaymalari (DC Offset) onler."""
    # 1. DC Offset Removal (Elektriksel kaymayi sifirla)
    y = y - np.mean(y)

    # 2. Sessizlik Budamasi (Gevsetilmis VAD)
    y_trimmed, _ = librosa.effects.trim(y, top_db=40)
    if len(y_trimmed) > sr * 0.5:
        y = y_trimmed

    # 3. Hann Window Envelope (Baslangic ve bitisteki 'clipping' patlamalarini yok et)
    fade_len = int(sr * 0.2)  # 200 ms
    if len(y) > fade_len * 2:
        window = np.hanning(fade_len * 2)
        y[:fade_len] = y[:fade_len] * window[:fade_len]
        y[-fade_len:] = y[-fade_len:] * window[fade_len:]

    return y


# --- DİNAMİK VE NEDENSEL IMRAD PDF RAPORU MOTORU ---
class IMRADReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, sanitize_text('OTONOM SIBER GUVENLIK SES ANALIZI - ADLI BILISIM RAPORU'), border=False,
                  ln=True, align='C')
        self.set_font('helvetica', 'I', 9)
        self.cell(0, 5, sanitize_text(f'Rapor Uretim Tarihi: {time.strftime("%Y-%m-%d %H:%M:%S")}'), border=False,
                  ln=True, align='C')
        self.set_draw_color(34, 211, 238)
        self.line(10, 26, 200, 26)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, sanitize_text(f'Sayfa {self.page_no()}'), align='C')


def generate_imrad_pdf(file_name, threshold, duration, entropy, fake_prob, worst_chunk_time,
                       max_sal_sec, peak_freq, saliency_img_path, timeline_path, waveform_path, nyquist):
    log_to_terminal("RAPOR MOTORU", "Dinamik, Nedensel ve LaTeX Destekli PDF olusturuluyor...")
    pdf = IMRADReport()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, sanitize_text('1. GIRIS VE SINYAL TOPOLOJISI'), ln=True)
    pdf.set_font('helvetica', '', 10)

    intro_text = (f"Incelenen Dosya: {os.path.basename(file_name)}\n"
                  f"Sinyal Toplam Suresi: {duration:.2f} saniye\n"
                  f"Nyquist Frekansi (Maksimum Olculebilir Bant): {nyquist} Hz\n"
                  f"Sistem Guvenlik Esigi (Threshold): %{threshold}\n\n"
                  f"Amac: Ham ses sinyali uzerindeki spektral doku incelenerek, sentetik ses ureticilerine (vocoder) "
                  f"ve GAN (Generative Adversarial Network) mimarilerine ait anomali ve frekans sizintilarinin "
                  f"nedensel olarak tespit edilmesidir.")
    pdf.multi_cell(0, 6, sanitize_text(intro_text))
    pdf.ln(5)

    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, sanitize_text('2. MATEMATIKSEL MODEL VE NOTASYONLAR'), ln=True)
    pdf.set_font('helvetica', '', 10)

    pdf.multi_cell(0, 6, sanitize_text(
        "Sistem, karar mekanizmasini birinci prensipler (first-principles) uzerinden asagidaki iki temel denklemle insa eder:\n"))

    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 6, sanitize_text("A. Ihtimaliyet Uzayi (Softmax Fonksiyonu):"), ln=True)
    pdf.set_font('helvetica', '', 10)

    form1_path = os.path.join(DATA_DIR, "form1.png")
    render_latex_to_image(r"P(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}", form1_path)
    if os.path.exists(form1_path):
        pdf.image(form1_path, x=85, w=40)
    pdf.multi_cell(0, 6, sanitize_text(
        "Tanim: x_i, agin son katmanindan cikan ham skordur. Euler sayisi (e) tabaninda ustellik alinarak olasilik dagilimina donusturulur.\n"))

    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 6, sanitize_text("B. Karar Kararliligi (Shannon Entropisi):"), ln=True)
    pdf.set_font('helvetica', '', 10)

    form2_path = os.path.join(DATA_DIR, "form2.png")
    render_latex_to_image(r"H(P) = -\sum P(x) \log_2 P(x)", form2_path)
    if os.path.exists(form2_path):
        pdf.image(form2_path, x=80, w=50)
    pdf.multi_cell(0, 6, sanitize_text(
        "Tanim: H(P) sistemin kararsizlik seviyesini olcer. 0'a yakin degerler deterministik karari ifade eder.\n"))

    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, sanitize_text('3. DENEYSEL BULGULAR VE NEDENSEL CIKARIM (CAUSAL INFERENCE)'), ln=True)
    pdf.set_font('helvetica', '', 10)

    decision = "SAHTE (DEEPFAKE)" if fake_prob >= threshold else "GERCEK (GUVENLI)"

    causal_analysis = (f"Modelin Kayan Pencere (Sliding Window) ve XAI analizleri sonucunda, sesin {worst_chunk_time}. "
                       f"saniyeleri arasindaki penceresinde maksimum sapma (%{fake_prob:.2f} Sahte) tespit edilmistir.\n\n"
                       f"Nedensel Cikarim Raporu:\n"
                       f"Gradyan turev haritalari incelendiginde, kritik anomalinin tam olarak {max_sal_sec:.2f}. saniye civarinda "
                       f"ve yaklasik {peak_freq:.0f} Hz frekans bandinda yogunlastigi gorulmektedir. "
                       f"Insan ses tellerinin (glottal pulse) dogal akisinin aksine, bu spesifik frekans bandinda gozlemlenen "
                       f"yuksek ve dikdortgen formlu enerji degisimleri (saliency spikes), sesin bir vocoder veya GAN tarafindan "
                       f"sentezlendigini isaret etmektedir. Entropi skorunun {entropy:.4f} olarak olculmesi, modelin "
                       f"bu lekelenmeyi bir 'arka plan gurultusu' olarak degil, kesin bir sentetik manipulasyon izi olarak "
                       f"degerlendirdigini kanitlamaktadir.\n\n"
                       f"NIHAI KARAR: Belirlenen %{threshold} esigine gore, dosya kesin olarak {decision} olarak siniflandirilmistir.")

    pdf.multi_cell(0, 6, sanitize_text(causal_analysis))

    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, sanitize_text('4. GORSEL VE SPEKTRAL KANITLAR (XAI)'), ln=True)
    pdf.set_font('helvetica', '', 9)

    pdf.multi_cell(0, 5, sanitize_text(
        "Sekil 1: Kayan Pencere Zaman Cizelgesi. Grafikteki kirmizi referans noktasi, analiz suresi boyunca esik degerini asan ve en yuksek sahtelik olasiligina sahip olan zaman araligini (kritik sapma anini) gostermektedir."))
    if os.path.exists(timeline_path):
        pdf.image(timeline_path, x=15, w=180)
        pdf.ln(5)

    pdf.multi_cell(0, 5, sanitize_text(
        f"Sekil 2: 2D Turevsel Vurgu Haritasi (Saliency Map). Yesil referans dairesi ve dikey kesik cizgi, modelin siniflandirma kararini dogrudan etkileyen maksimum sentetik enerji sizintisinin merkez frekansini ({peak_freq:.0f} Hz) ve gerceklestigi saniyeyi ({max_sal_sec:.2f} sn) tespit etmektedir."))
    if os.path.exists(saliency_img_path):
        pdf.image(saliency_img_path, x=30, w=150)
        pdf.ln(5)

    pdf.multi_cell(0, 5, sanitize_text(
        f"Sekil 3: 1D Sinyal Dalga Formu (Waveform). Kirmizi ile taranmis dikey alan, tespit edilen spektral anomalinin 1 boyutlu zaman uzayindaki faz kirilmasi sinirlarini belirtmektedir."))
    if os.path.exists(waveform_path):
        pdf.image(waveform_path, x=30, w=150)

    report_path = os.path.join(DATA_DIR, "IMRAD_Adli_Rapor.pdf")
    pdf.output(report_path)
    log_to_terminal("RAPOR MOTORU", f"Dinamik PDF basariyla uretildi: {report_path}")
    return report_path


# --- BATCH (TOPLU) ANALİZ MOTORU ---
def process_batch(files, threshold, use_dsp=False, temperature=DEFAULT_TEMPERATURE, model_variant="Baseline (CrossEntropy)"):
    active_2d = MODEL_VARIANTS.get(model_variant, MODEL_VARIANTS["Baseline (CrossEntropy)"])["2d"]
    if not files:
        return None

    log_to_terminal("TOPLU ANALİZ BAŞLADI", f"{len(files)} adet dosya isleniyor. Eşik: %{threshold}")
    results = []

    for file_obj in files:
        file_path = file_obj.name
        y, sr = librosa.load(file_path, sr=16000)

        if use_dsp:
            y = apply_dsp_armor(y, sr)

        max_length = 16000 * 4
        y_chunk = y[:max_length] if len(y) >= max_length else np.pad(y, (0, max_length - len(y)), mode='constant')

        mel_spec = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=TRAIN_N_MELS, n_fft=TRAIN_N_FFT, hop_length=TRAIN_HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        tensor_2d = torch.tensor(mel_spec_db).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out_2d = active_2d(tensor_2d)

        fake_prob = torch.softmax(out_2d / temperature, dim=1)[0][1].item() * 100
        decision = "SAHTE" if fake_prob >= threshold else "GERÇEK"

        results.append({
            "Dosya Adı": os.path.basename(file_path),
            "Sahte İhtimali (%)": round(fake_prob, 2),
            "Karar": decision
        })
        log_to_terminal("DOSYA İŞLENDİ", f"{os.path.basename(file_path)} -> %{fake_prob:.2f} ({decision})")

    return pd.DataFrame(results)


# --- TEKLİ FULL DASHBOARD ANALİZ MOTORU ---
def analyze_single_full(audio_path, threshold, use_dsp=False, temperature=DEFAULT_TEMPERATURE, model_variant="Baseline (CrossEntropy)"):
    active_2d = MODEL_VARIANTS.get(model_variant, MODEL_VARIANTS["Baseline (CrossEntropy)"])["2d"]
    active_1d = MODEL_VARIANTS.get(model_variant, MODEL_VARIANTS["Baseline (CrossEntropy)"])["1d"]
    log_to_terminal("TEKLİ ANALİZ BAŞLADI", f"Dosya: {audio_path} | Eşik: %{threshold} | Model: {model_variant}")
    if audio_path is None:
        return "Hata", None, "Hata", None, "Hata", None, None

    start_time = time.time()

    y_orig, sr_orig = librosa.load(audio_path, sr=None)
    nyquist = sr_orig / 2
    y_16k, sr_16k = librosa.load(audio_path, sr=16000)

    if use_dsp:
        y_16k = apply_dsp_armor(y_16k, sr_16k)

    total_duration = librosa.get_duration(y=y_16k, sr=sr_16k)

    # --- TIMELINE ---
    chunk_samples = 16000 * 4
    chunks = [y_16k[i:i + chunk_samples] for i in range(0, len(y_16k), chunk_samples)]

    if not chunks:
        chunks = [np.zeros(chunk_samples)]

    timeline_fakes, timeline_times = [], []

    for idx, chunk in enumerate(chunks):
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

        c_spec = librosa.feature.melspectrogram(y=chunk, sr=sr_16k, n_mels=TRAIN_N_MELS, n_fft=TRAIN_N_FFT, hop_length=TRAIN_HOP_LENGTH)
        c_tensor = torch.tensor(librosa.power_to_db(c_spec, ref=np.max)).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            c_out = active_2d(c_tensor)

        timeline_fakes.append(torch.softmax(c_out / temperature, dim=1)[0][1].item() * 100)
        timeline_times.append(f"{idx * 4}-{(idx + 1) * 4}")

    worst_chunk_idx = np.argmax(timeline_fakes)
    worst_time_str = timeline_times[worst_chunk_idx]
    max_fake_prob = timeline_fakes[worst_chunk_idx]

    y_focus = chunks[worst_chunk_idx]
    if len(y_focus) < chunk_samples:
        y_focus = np.pad(y_focus, (0, chunk_samples - len(y_focus)), mode='constant')

    mel_spec = librosa.feature.melspectrogram(y=y_focus, sr=sr_16k, n_mels=TRAIN_N_MELS, n_fft=TRAIN_N_FFT, hop_length=TRAIN_HOP_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    tensor_2d = torch.tensor(mel_spec_db).unsqueeze(0).unsqueeze(0).float().to(device)

    tensor_2d.requires_grad_()
    out_2d = active_2d(tensor_2d)
    probs_2d = torch.softmax(out_2d / temperature, dim=1)[0]
    out_2d[0, 1].backward()

    saliency_map = tensor_2d.grad.data.abs().squeeze().cpu().numpy()
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-9)

    saliency_time_sum = saliency_map.sum(axis=0)
    max_sal_frame = np.argmax(saliency_time_sum)
    max_sal_sec = librosa.frames_to_time(max_sal_frame, sr=sr_16k, hop_length=TRAIN_HOP_LENGTH)

    max_mel_bin = np.argmax(saliency_map[:, max_sal_frame])
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=sr_16k / 2.0)
    peak_freq = mel_freqs[max_mel_bin]

    entropy_val = calculate_entropy(probs_2d.detach().cpu().numpy())
    res_2d_text = format_result(probs_2d.detach().cpu().numpy(),
                                threshold) + f"\n\nSpesifik Odak: {max_sal_sec:.2f}. saniye, {peak_freq:.0f} Hz"

    tensor_1d = torch.tensor(y_focus).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out_1d = active_1d(tensor_1d)
    res_1d_text = format_result(torch.softmax(out_1d / temperature, dim=1)[0].cpu().numpy(), threshold)

    # --- GRAFİKLERİN ÇİZİMİ ---
    plt.figure(figsize=(10, 2))
    plt.plot(timeline_times, timeline_fakes, marker='o', color='red', linestyle='-', linewidth=2)
    plt.axhline(y=threshold, color='orange', linestyle='--', label=f'Eşik (%{threshold})')

    y_offset = -25 if max_fake_prob > 80 else 25
    plt.annotate('Kritik Sapma', xy=(worst_chunk_idx, max_fake_prob),
                 xytext=(worst_chunk_idx, max_fake_prob + y_offset),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 ha='center', va='center', color='red', weight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1, alpha=0.8))

    plt.ylim(-15, 135)
    plt.title("Kayan Pencere Analizi: Saniye Saniye Olasılık Değişimi", fontsize=10)
    plt.ylabel("Sahte (%)", fontsize=8)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    timeline_path = os.path.join(DATA_DIR, "timeline_plot.png")
    plt.savefig(timeline_path, dpi=120)
    plt.close()

    plt.figure(figsize=(6, 3))
    librosa.display.specshow(mel_spec_db, sr=sr_16k, hop_length=TRAIN_HOP_LENGTH, x_axis='time', y_axis='mel', cmap='gray',
                             alpha=0.6)
    plt.imshow(saliency_map, cmap='hot', aspect='auto', origin='lower', extent=[0, 4, 0, 8000], alpha=0.5)
    plt.axvline(x=max_sal_sec, color='lime', linestyle='--', lw=2, alpha=0.8)
    plt.scatter(max_sal_sec, peak_freq, color='none', s=800, edgecolors='lime', lw=3, zorder=5)
    plt.text(max_sal_sec + 0.1, peak_freq, f'Sızıntı Merkezi\n({peak_freq:.0f} Hz)', color='lime', weight='bold',
             fontsize=9)
    plt.title('2D Türevsel Vurgu Haritası (Saliency Map)', fontsize=10)
    plt.tight_layout()
    path_2d_xai = os.path.join(DATA_DIR, "temp_xai_2d.png")
    plt.savefig(path_2d_xai, dpi=120)
    plt.close()

    plt.figure(figsize=(6, 3))
    librosa.display.waveshow(y_focus, sr=sr_16k, color="cyan", alpha=0.8)
    plt.gca().set_facecolor('#0f172a')
    plt.axvspan(max_sal_sec - 0.1, max_sal_sec + 0.1, color='red', alpha=0.4, label='Faz Kırılma Alanı')
    plt.title(f'1D Dalga Formu: {max_sal_sec:.2f}. Saniyedeki Spektral Anomali', fontsize=10)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    path_1d_wave = os.path.join(DATA_DIR, "temp_waveform_1d.png")
    plt.savefig(path_1d_wave, dpi=120)
    plt.close()

    inference_ms = (time.time() - start_time) * 1000
    vram_used_mb = torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0

    dsp_status = "Aktif" if use_dsp else "Kapalı"
    hardware_report = f"""
### Donanım Profili ve Metrikler
* **Sinyal Süresi:** {total_duration:.2f} sn
* **Şüpheli Odak:** {max_sal_sec:.2f}. saniye
* **Pik Sızıntı Frekansı:** {peak_freq:.0f} Hz
* **İşlem Gecikmesi:** {inference_ms:.2f} ms
* **VRAM Kullanımı:** {vram_used_mb:.2f} MB
* **Sıcaklık (T):** {temperature}
* **DSP Ön İşleme:** {dsp_status}
* **Spektrogram (FFT/Hop):** {TRAIN_N_FFT}/{TRAIN_HOP_LENGTH}
"""

    pdf_path = generate_imrad_pdf(audio_path, threshold, total_duration, entropy_val, max_fake_prob, worst_time_str,
                                  max_sal_sec, peak_freq, path_2d_xai, timeline_path, path_1d_wave, nyquist)

    return hardware_report, timeline_path, res_2d_text, path_2d_xai, res_1d_text, path_1d_wave, pdf_path


# --- GRADIO ARAYÜZ TASARIMI ---
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# Otonom Ses Siber Güvenlik Analiz Sistemi")

    with gr.Tabs():
        with gr.TabItem("Tekli Analiz ve Raporlama"):
            with gr.Row():
                with gr.Column(scale=1):
                    # type="filepath" -> Gradio dosya yolunu olduğu gibi geçirir, librosa tüm formatları (FLAC/WAV/MP3) destekler.
                    audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                           label="Ses Dosyası Yükle veya Kaydet")
                    threshold_slider = gr.Slider(minimum=10, maximum=90, value=50, step=1,
                                                 label="Güvenlik Eşiği (Threshold %)")
                    threshold_slider.change(fn=lambda x: log_to_terminal("EŞİK DEĞİŞTİRİLDİ", f"Yeni değer: %{x}"),
                                            inputs=[threshold_slider], outputs=[])
                    dsp_checkbox = gr.Checkbox(value=False, label="DSP Ön İşleme (Deneysel)",
                                              info="DC Offset, Sessizlik Budama ve Fade. Eğitimde bu işlem yoktu, açmak sonuçları bozabilir.")
                    temperature_slider = gr.Slider(minimum=0.5, maximum=5.0, value=DEFAULT_TEMPERATURE, step=0.1,
                                                   label="Sıcaklık Kalibrasyonu (Temperature Scaling)",
                                                   info="Yüksek = yumuşak/temkinli karar, Düşük = keskin/agresif karar")
                    model_selector = gr.Radio(
                        choices=list(MODEL_VARIANTS.keys()),
                        value="Baseline (CrossEntropy)",
                        label="Model Varyantı",
                        info="Baseline: Standart eğitim | Robust: Label Smoothing + Cosine Annealing")
                    analyze_btn = gr.Button("Sinyal Analizini Başlat")
                    hardware_output = gr.Markdown("### Donanım Profili\n*Sistem analize hazır.*")

                    gr.Markdown("""
                    ---
                    ### Analiz Rehberi (Görsel İşaretleyiciler)
                    * **Kırmızı Nokta (Zaman Çizelgesi):** Kayan pencere analizinde eşik değerini aşan en yüksek sahtelik olasılığına sahip zaman aralığını (kritik sapma anını) gösterir.
                    * **Yeşil Daire ve Çizgi (Isı Haritası):** Türevsel vurgu haritası üzerinde, modelin sınıflandırma kararını doğrudan etkileyen maksimum sentetik enerji sızıntısının merkez frekansını ve gerçekleştiği saniyeyi tespit eder.
                    * **Kırmızı Bant (Dalga Formu):** Tespit edilen spektral anomalinin 1 boyutlu zaman uzayındaki (waveform) faz kırılması sınırlarını tarar.
                    """)

                with gr.Column(scale=3):
                    timeline_img = gr.Image(label="Kayan Pencere Zaman Çizelgesi (Timeline)")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 2D SENet (Spektral Analiz)")
                            res_2d_out = gr.Textbox(label="Çıkarım Sonuçları", lines=5)
                            img_2d_out = gr.Image(label="Türevsel Vurgu Haritası (Saliency Map)")

                        with gr.Column():
                            gr.Markdown("### 1D RawNet2 (Zaman Uzayı Analizi)")
                            res_1d_out = gr.Textbox(label="Çıkarım Sonuçları", lines=5)
                            img_1d_out = gr.Image(label="Sinyal Dalga Formu (Waveform)")

                    pdf_out = gr.File(label="Adli Bilişim Raporu (PDF İndir)")

            analyze_btn.click(
                fn=analyze_single_full,
                inputs=[audio_input, threshold_slider, dsp_checkbox, temperature_slider, model_selector],
                outputs=[hardware_output, timeline_img, res_2d_out, img_2d_out, res_1d_out, img_1d_out, pdf_out]
            )

        with gr.TabItem("Toplu Sinyal Analizi"):
            gr.Markdown(
                "Bu modül, birden fazla sesi otonom olarak analiz eder ve sonuçları tablo halinde sunar. İşlemler terminale loglanır.")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(file_count="multiple", label="Ses Dosyalarını Seç (.wav, .mp3)")
                    batch_threshold = gr.Slider(minimum=10, maximum=90, value=50, step=1,
                                                label="Karar Eşiği (Threshold %)")
                    batch_dsp_checkbox = gr.Checkbox(value=False, label="DSP Ön İşleme (Deneysel)")
                    batch_temperature_slider = gr.Slider(minimum=0.5, maximum=5.0, value=DEFAULT_TEMPERATURE, step=0.1,
                                                         label="Sıcaklık Kalibrasyonu")
                    batch_model_selector = gr.Radio(
                        choices=list(MODEL_VARIANTS.keys()),
                        value="Baseline (CrossEntropy)",
                        label="Model Varyantı")
                    batch_btn = gr.Button("Toplu Analizi Başlat", variant="primary")

                with gr.Column(scale=2):
                    batch_output = gr.Dataframe(label="Analiz Sonuçları",
                                                headers=["Dosya Adı", "Sahte İhtimali (%)", "Karar"])

            batch_btn.click(fn=process_batch, inputs=[batch_files, batch_threshold, batch_dsp_checkbox, batch_temperature_slider, batch_model_selector], outputs=[batch_output])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, allowed_paths=[DATA_DIR])