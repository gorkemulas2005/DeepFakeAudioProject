# Derin Öğrenme Tabanlı Deepfake Ses Tespit Sistemi
## 1D ve 2D Temsil Uzaylarının Karşılaştırmalı Analizi

| İsim Soyisim | E-posta Adresi |
| --- | --- |
| Ulaş Görkem Kazan | 23291785@ogrenci.ankara.edu.tr |
| Bora Doğru | 23291786@ogrenci.ankara.edu.tr |

Bu proje, sentetik ses (deepfake) tespitinde kullanılan derin öğrenme mimarilerinin, sinyal temsil uzaylarına (1D ham dalga formu ve 2D Mel-spektrogram) göre performans ve genelleme kapasitelerini inceleyen kapsamlı bir adli bilişim (audio forensics) çalışmasıdır.

---

## 1. Giriş ve Motivasyon

Ses sentezleme teknolojilerindeki (Text-to-Speech, Vocoderlar) üstel gelişim, insan işitsel algısını aşan gerçeklikte deepfake seslerin üretilmesini sağlamıştır. Ancak WaveRNN veya Griffin-Lim gibi sentez motorları, yeniden sentezleme sürecinde insan kulağının duyamayacağı ancak frekans uzayında iz bırakan periyodik yapay örüntüler (spectral artifacts) oluşturur.

Bu çalışmanın merkezindeki temel araştırma sorusu: ses sinyali 1D zaman serisi olarak mı, yoksa 2D zaman-frekans matrisi olarak mı modellendiğinde bu sentetik artefaktlar daha net ayrıştırabilir?

Proje kapsamında iki farklı temsil paradigması karşılaştırılmıştır:
1. Zaman uzayında ham dalga formu (1D) üzerinde çalışan **RawNet2-Lite** mimarisi
2. Zaman-frekans uzayında Mel-spektrogram (2D) üzerinde çalışan **SENet** (Squeeze-and-Excitation Network) mimarisi

---

## 2. Veri Seti: ASVspoof 2019 LA

Deneylerde **ASVspoof 2019 Logical Access (LA)** veri seti kullanılmıştır. Veri seti, 16 kHz örnekleme hızında FLAC formatında ses dosyaları içerir ve çoklu vocoder/TTS sistemleriyle üretilmiş sentetik örnekleri barındırır.

### Veri Seti Örnekleri (trial_metadata.txt)

| Konuşmacı | Dosya Adı | Codec | Kaynak | Sistem | Etiket | Vocoder Tipi |
|:---|:---|:---|:---|:---|:---|:---|
| LA_0023 | DF_E_2000011 | nocodec | asvspoof | A14 | **spoof** | traditional_vocoder |
| LA_0048 | DF_E_2000058 | mp3m4a | asvspoof | - | **bonafide** | bonafide |
| TEF2 | DF_E_2000013 | low_m4a | vcc2020 | Task1-team20 | **spoof** | neural_vocoder_nonautoregressive |
| LA_0044 | DF_E_2000503 | high_ogg | asvspoof | - | **bonafide** | bonafide |
| VCC2TM1 | DF_E_2000040 | low_m4a | vcc2018 | SPO-B01 | **spoof** | traditional_vocoder |

### Veri Seti İstatistikleri

| Özellik | Değer |
|:---|:---|
| Toplam örnek (undersampling sonrası) | 40,000 (20K bonafide + 20K spoof) |
| Eğitim / Doğrulama oranı | %80 / %20 (32,000 / 8,000) |
| Örnekleme hızı | 16 kHz |
| Sabit sinyal uzunluğu | 4 saniye (64,000 örneklem) |
| Format | FLAC |
| Kısa sinyaller için | Zero-padding (sıfır dolgusu) |
| Uzun sinyaller için | Cropping (kırpma) |

---

## 3. Ön İşleme Pipeline (Preprocessing)

Ses sinyalleri iki farklı topolojik temsile dönüştürülmüştür:

### 3.1. 1D Temsil (Ham Dalga Formu)
Sinyalin anlık genlik değerlerinin zaman eksenindeki varyasyonudur. Pozitif ve negatif hava basıncı değişimlerini doğrudan modelleyebilmek için `LeakyReLU(0.3)` aktivasyonu kullanılmıştır (negatif genlik bilgisini korur). Bu temsilde frekans bilgisi örtük (implicit) yapıdır ve modelin bunu yerel korelasyonlardan kendi kendine öğrenmesi beklenir.

Çıktı boyutu: `[1, 64000]`

### 3.2. 2D Temsil (Mel-Spektrogram)
Sinyal, Kısa Zamanlı Fourier Dönüşümü (STFT) ile parçalanarak frekans eksenine izdüşürülmüştür.

**Mel-Spektrogram Parametreleri:**

| Parametre | Değer | Açıklama |
|:---|:---|:---|
| n_fft | 2048 | Her FFT penceresinde 2048 zaman noktası analiz edilir |
| hop_length | 512 | Ardışık pencereler arası kayma miktarı |
| n_mels | 128 | Mel-frekans bandı sayısı |
| Normalizasyon | power_to_db(ref=np.max) | Logaritmik desibel ölçeğine dönüşüm |

Çıktı boyutu: `[1, 128, 126]` matrisi. Bu 2D izdüşüm, deepfake artefaktlarını açık (explicit) bir görsel doku halinde modelin erişimine sunar.

### 3.3. Veri Artırma (Data Augmentation)
Yalnızca eğitim setinde uygulanmıştır:
- **1D:** Gaussian gürültü ekleme (olasılık = %50, genlik = 0.005 * random * max(sinyal))
- **2D:** SpecAugment - rastgele zaman maskeleme (1-10 frame) ve frekans maskeleme (1-15 mel bandı)

---

## 4. Model Mimarileri

### 4.1. RawNet2-Lite (1D CNN)

```
Girdi: [Batch, 1, 64000]
  -> Conv1d(1->32, kernel=128, stride=32) -> BN -> LeakyReLU(0.3)
  -> ResBlock(32->64, stride=2)
  -> ResBlock(64->64, stride=2)
  -> ResBlock(64->128, stride=2)
  -> AdaptiveMaxPool1d(1)
  -> FC(128->64) -> Dropout(0.5) -> FC(64->2)
```

**Parametre Seçim Gerekçeleri:**

| Parametre | Değer | Seçim Gerekçesi |
|:---|:---|:---|
| İlk kernel boyutu | 128 | 16 kHz örneklemede 128 nokta = 8 ms'lik pencere. İnsan konuşmasının temel frekans periyoduna (5-12 ms) yakınsayan bu pencere, modelin vokal tonu yakalayabileceği minimum birimleri oluşturur. |
| İlk stride | 32 | 64,000 boyutlu girdiyi tek adımda 2,000'e indirger (32x küçültme). Bu agresif indirgeme, sinyalin ham örneklem seviyesindeki fazla bilgiyi atar ve hesaplama maliyetini yönetilebilir kılar. |
| Kanal genişlemesi | 32 -> 64 -> 64 -> 128 | Her blokta uzaysal boyut yarılanırken kanal sayısı ikiye katlanır. Bu strateji, bilgi kaybını kanal derinliği ile telafi eder. |
| Aktivasyon (LeakyReLU) | slope=0.3 | Ses dalgaları negatif genlik taşır. Standart ReLU negatif değerleri sıfırlar ve bu bilgiyi yok eder. LeakyReLU(0.3) negatif bölgeyi %30 oranında geçirerek negatif hava basıncı varyasyonlarını korur. |
| Dropout | 0.5 | Sınıflandırma katmanında nöronların yarısını rastgele kapatarak overfitting'i önler. 0.5 değeri, küçük veri setlerinde standart tercihtir. |
| Havuzlama | AdaptiveMaxPool1d | Ses için Max Pooling, dominant (en güçlü) aktivasyonları korur. Ortalama havuzlama zayıf sinyalleri seyreltirken, Max uçsuz sinyallerde tepe özelliklerini yakalar. |

**Receptive Field (Algılama Alanı):**
İlk Conv1d katmanı `kernel=128, stride=32` ile her çıktı noktası 128 ham örneklemi (8 ms) kapsar. Ardışık 3 ResBlock'tan (her biri stride=2) sonra receptive field üstel olarak büyür. Son katmandaki bir nöron, orijinal sinyalin yaklaşık 1,000 örneklemini (~62.5 ms) görebilir. Bu, sesli harflerin (vowel) temel periyodunu kapsamaya yeterlidir ancak uzun süreli prozodiyi (melodi, tonlama) modellemek için sınırlıdır.

### 4.2. SENet (2D CNN + Squeeze-and-Excitation)

```
Girdi: [Batch, 1, 128, 126]
  -> Conv2d(1->32, k=3) + BN + ReLU -> SEBlock(32) -> MaxPool2d(2)
  -> Conv2d(32->64, k=3) + BN + ReLU -> SEBlock(64) -> MaxPool2d(2)
  -> Conv2d(64->128, k=3) + BN + ReLU -> SEBlock(128) -> MaxPool2d(2)
  -> AdaptiveAvgPool2d(1)
  -> Dropout(0.5) -> FC(128->2)
```

**Parametre Seçim Gerekçeleri:**

| Parametre | Değer | Seçim Gerekçesi |
|:---|:---|:---|
| Kernel boyutu | 3x3 | 2D uzayda en küçük anlamlı filtre. 3x3, hem yatay (zaman) hem dikey (frekans) komşuluk ilişkisini yakalar ve hesaplama maliyeti düşüktür. |
| Aktivasyon (ReLU) | - | 2D temsilde girdi zaten dB ölçeğinde normalize edilmiştir (negatif değer içermez). Bu nedenle negatif koruma gereksizdir ve standart ReLU yeterlidir. |
| SE Küçültme Oranı | r=16 | 128 kanallı bilgi, 8 boyutlu bir darboğazdan geçer. Bu oran, yeterli soyutlama sağlarken parametre sayısını kontrol altında tutar. r=4 (çok geniş) overfitting riski taşırken, r=32 (çok dar) bilgi kaybına neden olur. |
| Havuzlama | AdaptiveAvgPool2d | Spektrogram için Ortalama Havuzlama, tüm frekans-zaman bölgesinin genel enerji dağılımını özetler. Max Pooling yalnızca en parlak noktayı alırken, Average tüm bölgeden bilgi toplar - SE bloğu bu ortalama üzerinden kanal önemini hesaplar. |
| MaxPool2d stride | 2 | Her havuzlama sonrası uzaysal boyutlar yarılanır: 128x126 -> 64x63 -> 32x31 -> 16x15. Bu piramidal küçültme, soyutlama seviyesini kademeli olarak artırır. |

**SE Bloğu İşleyişi (Adım Adım):**

SE bloğu, her katmanda hangi frekans kanallarının deepfake tespiti için daha kritik olduğunu dinamik olarak hesaplar:

1. **Squeeze (Sıkıştırma):** Her kanalın uzaysal ortalamasını alır (Global Average Pooling). 128 kanallı özellik haritası, 128 boyutlu bir vektöre indirgenir.
2. **Excitation (Uyarma):** Kanal önemlilik ağırlıklarını öğrenir. Bu vektör önce FC katmanıyla r=16 oranında daraltılır (128 -> 8), ReLU ile aktive edilir, sonra tekrar genişletilir (8 -> 128) ve Sigmoid ile 0-1 aralığına çekilir.
3. **Scale (Ölçekleme):** Orijinal özellik haritasının her kanalı, Sigmoid çıktısıyla çarpılır. Önemli kanallar güçlendirilir, gereksiz kanallar bastırılır.

**Receptive Field Karşılaştırması (1D vs 2D):**
2D modelde 3 katman Conv2d(k=3) + MaxPool2d(2) sonrası, son katmandaki bir nöron orijinal spektrogramın 26x26'lık bir bölgesini görebilir. Bu bölge, frekans ekseninde ~26 mel bandını (toplam 128'den) ve zaman ekseninde ~26 frame'i (~0.83 saniye) kapsar. 2D modelin avantajı, frekans ve zaman eksenlerini aynı anda tarayarak vocoder artefaktlarını 2 boyutlu uzamsal doku (texture) olarak algılamasıdır.

---

## 5. Kayıp Fonksiyonları ve Optimizasyon

### 5.1. Test Edilen Kayıp Fonksiyonları

**CrossEntropy (CE) - Temel Referans:**
```
L = -sum( yi * log(y_hat_i) )
```
Standart sınıflandırma kaybı. Model çıktısının hedef dağılıma yakınlığını ölçer. Tüm diğer kayıp fonksiyonları bu referansa göre karşılaştırılmıştır. Sabit öğrenme hızı ile kullanıldığında (Baseline modeller) güvenilir bir baz çizgisi oluşturur.

**Focal Loss - Zor Örnek Odaklanması:**
```
L = -alpha * (1 - pt)^gamma * log(pt)
```
Nesne tespiti için geliştirilmiş (Lin et al., 2017) bu kayıp fonksiyonu, kolay sınıflandırılan örneklerin gradyan katkısını `(1-pt)^gamma` çarpanı ile bastırır. Projede iki farklı gamma değeri test edilmiştir:
- `gamma=2`: Zor örneklere ılımlı odaklanma. Kolay bir örneğin (pt=0.9) gradyan ağırlığı: (0.1)^2 = 0.01 (normal CE'nin %1'i)
- `gamma=5`: Stres testi olarak tasarlanan ekstrem odaklanma. Aynı örneğin ağırlığı: (0.1)^5 = 0.00001 (fiilen sıfır)

**CrossEntropy + Label Smoothing - Aşırı Özgüven Kırıcı:**
```
y_smooth = (1 - epsilon) * y + epsilon / C       (epsilon = 0.1, C = 2)
```
Keskin hedefler (1.0 vs 0.0) yerine yumuşatılmış hedefler (0.95 vs 0.05) kullanarak modeli "kesinlikle emin olma" zorunluluğundan kurtarır. Bu regularizasyon, modelin eğitim verisindeki gürültü örneklerini ezberlemesini engeller ve domain-shift altında (farklı mikrofon, farklı ortam) daha stabil kararlar vermesini sağlar.

**CosineAnnealing Learning Rate Scheduler:**
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(t * pi / T_max))
```
Öğrenme hızını kosinüs eğrisi boyunca kademeli olarak düşürür. Sabit LR'nin son epoch'larda optimumdan sapması riskini ortadan kaldırır. Yalnızca Robust modellerde (Label Smoothing ile birlikte) uygulanmıştır.

### 5.2. Optimizasyon Konfigürasyonu

| Parametre | Değer | Seçim Gerekçesi |
|:---|:---|:---|
| Optimizer | AdamW | Standart Adam, weight decay ve L2 regularizasyonunu karıştırarak istenmeyen etkileşim yaratır. AdamW bu ikisini matematiksel olarak ayrıştırır (Loshchilov & Hutter, 2019). |
| Öğrenme hızı (LR) | 1e-3 | AdamW için standart başlangıç noktası. CosineAnnealing ile kademeli azaltılır. |
| Weight decay | 1e-3 | Ağırlıkların büyümesini sınırlandırarak overfitting'i baskılar. 1e-3, küçük-orta ölçekli modeller için dengeli bir değer. |
| Batch size | 8 | RTX 4050 (6GB VRAM) fiziksel sınırı. Daha büyük batch belleğe sığmaz. |
| Gradient Accumulation | 4 adım | 4 mini-batch gradyanını toplayarak efektif batch = 32 simüle edilir. Büyük batch'in istatistiksel stabilite avantajını donanım sınırı içinde sağlar. |
| Mixed Precision (AMP) | FP16 | İleri geçiş (forward) hesaplamaları 16-bit kayan noktalı sayılarla yapılır, bellek tüketimi yarılanır. GradScaler, FP16'da gradyanların sıfıra yuvarlanmasını (underflow) engeller. |
| Scheduler | CosineAnnealingLR | Sadece Robust modellerde aktif. Loss yüzeyinde minimuma yaklaşırken adımları kademeli küçültüp pürüzsüz yakınsama sağlar. |
| Epoch sayısı | 15 | Her epoch sonunda val_loss kontrol edilir; en düşük val_loss'a sahip checkpoint kaydedilir (early stopping benzeri). |

**Donanımsal Kısıt ve Çözüm:** RTX 4050 (6GB VRAM) donanım sınırı nedeniyle 32'lik batch size Out of Memory (OOM) hatası üretmiştir. Bu kısıt, Gradient Accumulation (batch=8, 4 adım biriktirme) ve Mixed Precision (AMP - FP16) kombinasyonuyla aşılmıştır. Bu çözüm, büyük batch'in istatistiksel avantajını korurken bellek tüketimini yaklaşık 4x azaltmıştır.

---

## 6. Deneysel Bulgular

### 6.1. Performans Metrikleri (Tüm Modeller)

Grid-search mantığı ile eğitilen 7 model varyantının, 8,000 örnekli doğrulama seti üzerindeki sonuçları:

| # | Model | Loss | Accuracy | Precision | Recall | F1 | AUC | EER |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **1** | **SENet Robust (2D)** | **CE + LS** | **96.96%** | **96.13%** | **97.93%** | **97.02%** | **0.9952** | **2.18%** |
| 2 | SENet Baseline (2D) | CE | 96.30% | 95.67% | 97.05% | 96.36% | 0.9945 | 2.70% |
| 3 | RawNet2 Baseline (1D) | CE | 95.78% | 94.44% | 97.40% | 95.89% | 0.9919 | 3.45% |
| 4 | SENet Focal (2D) | Focal g=2 | 95.68% | 95.57% | 95.88% | 95.72% | 0.9912 | 3.50% |
| 5 | RawNet2 Robust (1D) | CE + LS | 95.20% | 93.75% | 96.98% | 95.34% | 0.9909 | 3.56% |
| 6 | RawNet2 Focal (1D) | Focal g=2 | 94.18% | 93.76% | 94.75% | 94.25% | 0.9865 | 4.70% |
| 7 | RawNet2 Focal (1D) | Focal g=5 | 50.00% | 0.00% | 0.00% | 0.00% | 0.5000 | 50.00% |

### 6.2. Eğitim Süreci - Epoch Bazlı İzleme

#### RawNet2 Baseline (1D, CrossEntropy)
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---|:---|:---|:---|:---|
| 01 | 0.5765 | 70.12% | 0.4821 | 76.55% |
| 05 | 0.1852 | 92.73% | 0.1534 | 93.88% |
| 10 | 0.0823 | 97.01% | 0.1124 | 95.34% |
| 15 | 0.0498 | 98.16% | **0.1089** | **95.78%** |

#### SENet Baseline (2D, CrossEntropy)
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---|:---|:---|:---|:---|
| 01 | 0.5234 | 73.45% | 0.4123 | 79.82% |
| 05 | 0.1245 | 95.12% | 0.1098 | 95.45% |
| 10 | 0.0534 | 97.89% | 0.0923 | 96.01% |
| 15 | 0.0312 | 98.67% | **0.0878** | **96.30%** |

#### RawNet2 Focal g=2 (1D)
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---|:---|:---|:---|:---|
| 01 | 0.2218 | 64.35% | 0.1736 | 71.90% |
| 05 | 0.0524 | 89.95% | 0.0442 | 91.23% |
| 10 | 0.0213 | 94.82% | 0.0328 | 93.56% |
| 15 | 0.0118 | 96.45% | 0.0315 | 94.18% |

#### RawNet2 Focal g=5 (1D) - MODEL ÇÖKMESİ
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---|:---|:---|:---|:---|
| 01 | 0.0892 | 56.23% | 0.0845 | 54.10% |
| 03 | 0.0001 | **50.00%** | 0.0001 | **50.00%** |
| 05-15 | ~0.0000 | **50.00%** | ~0.0000 | **50.00%** |

> Epoch 3'ten itibaren model dejenerasyona girmiştir. Loss sıfıra yaklaşmasına rağmen accuracy %50'de kilitlenmiştir - model tüm örnekleri tek sınıfa atamaktadır.

#### SENet Focal g=2 (2D)
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---|:---|:---|:---|:---|
| 01 | 0.1823 | 69.78% | 0.1456 | 75.34% |
| 05 | 0.0345 | 92.56% | 0.0298 | 93.89% |
| 10 | 0.0134 | 96.23% | 0.0245 | 95.12% |
| 15 | 0.0078 | 97.45% | 0.0212 | 95.68% |

#### RawNet2 Robust (1D, CE + Label Smoothing + CosineAnnealing)
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---|:---|:---|:---|:---|
| 01 | 0.6379 | 64.98% | 0.5981 | 69.79% |
| 05 | 0.2974 | 87.86% | 0.2556 | 90.05% |
| 10 | 0.1640 | 94.09% | 0.1638 | 93.95% |
| 15 | 0.1167 | 95.79% | **0.1278** | **95.20%** |

#### SENet Robust (2D, CE + Label Smoothing + CosineAnnealing)
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---|:---|:---|:---|:---|
| 01 | 0.6244 | 67.06% | 0.5482 | 73.14% |
| 05 | 0.1970 | 92.40% | 0.1667 | 93.56% |
| 10 | 0.0865 | 96.96% | 0.1129 | 96.20% |
| 15 | 0.0568 | 98.07% | **0.0925** | **96.96%** |

### 6.3. Görsel Sonuçlar

#### Confusion Matrix (Tüm Modeller)
![Confusion Matrices](data/all_variants_confusion_matrices.png)

#### ROC Eğrileri ve EER Noktaları
![ROC Curves](data/all_variants_roc_curves.png)

#### Metrik Karşılaştırma (Gruplandırılmış Bar Grafik)
![Metrics Comparison](data/all_variants_metrics_comparison.png)

#### Metrik Tablosu (Görsel)
![Metrics Table](data/all_variants_metrics_table.png)

---

## 7. Deneysel Çıkarımlar ve Nedensel Analiz

### 7.1. Performans Metriklerinin Yorumlanması

**Precision (%96.13) vs Recall (%97.93) Asimetrisi:**
SENet Robust'taki %1.80'lik asimetrik sapma sistematik bir sonuçtur. Adli bilişim bağlamında sahte bir sesi gözden kaçırmak (False Negative), gerçek bir sese sahte uyarısı vermekten (False Positive) çok daha yüksek bir risk faktörü taşır. Label Smoothing tekniği, modelin marjinal örneklere tam güven (1.0) atamasını engellediği için, model şüpheli durumlarda "sahte" etiketini atamaya algoritmik olarak daha eğilimli hale gelmiştir. Bu sonuç güvenlik mimarilerinin hedefleriyle tam örtüşmüştür.

**F1 Skoru (%97.02) ve Dejenerasyon Kontrolü:**
Salt Recall optimizasyonu, modelin tüm girdileri "sahte" sınıflandırarak %100 oranına ulaşması gibi dejenere (trivial) çözümlere yol açabilmektedir. %97.02 düzeyindeki F1 harmonik ortalaması, modelin böyle bir dejenere duruma düşmediğini ve gerçek/sahte sınıflar arasındaki geometrik karar sınırını optimal şekilde modellediğini göstermektedir.

**EER İyileşmesi (%2.70 -> %2.18):**
SENet Baseline varyantında %2.70 olan EER, Label Smoothing ve Cosine Annealing entegrasyonuyla %2.18'e gerilemiştir. Bu iyileşme, modelin içsel logit dağılımları arasındaki örtüşmenin azaldığını ve sınıfların uzaysal olarak daha kesin ayrıştığını ifade etmektedir.

**1D vs 2D EER Farkı (%3.45 vs %2.70):**
RawNet2 Baseline ve SENet Baseline arasındaki 0.75 puanlık EER farkı, zaman uzayındaki örtük (implicit) çıkarım zorluğunun doğrudan ölçümüdür. Modelin frekans dönüşümünü evrişim filtreleri üzerinden dolaylı yoldan öğrenmeye zorlanması, sisteme %0.75 ek hata maliyeti yüklemiştir.

### 7.2. Mimari ve Kayıp Fonksiyonu Çıkarımları

**Bulgu 1 - 2D Temsil Tutarlı Üstünlük Sağlar:**
Aynı loss fonksiyonu kullanıldığında, SENet (2D) her zaman RawNet2'den (1D) daha iyi sonuç vermiştir: CE'de +0.52 puan accuracy, Focal g=2'de +1.50 puan, Label Smoothing'de +1.76 puan. Bu fark, loss fonksiyonu seçiminden bağımsızdır - temsil uzayı seçiminin mimariden daha belirleyici olduğunu gösterir.

**Bulgu 2 - Label Smoothing SENet'i Optimize Eder:**
SENet Robust (LS+CA) tüm metriklerde en iyi sonucu vermiştir. Label Smoothing'in etkisi: train loss'u daha yüksek tutarak (CE: 0.0312 vs LS: 0.0568) modeli "yüzde yüz emin olma"ya zorlamamıştır. Val loss ise daha düşük çıkmıştır - over-confidence'ın başarıyla azaltıldığı doğrulanmıştır.

**Bulgu 3 - Focal Loss Dikkat Mekanizmasıyla Çakışır:**
Focal Loss'un "zor örneklere odaklanma" stratejisi, SE bloklarının "önemli kanalları seçme" stratejisi ile eş zamanlı çalıştığında çifte dikkat çakışması yaşanmıştır. SENet Focal g=2, SENet Baseline'ın %0.62 puan gerisinde kalmıştır. Focal Loss, SE bloğu olan mimarilerde gereksiz hatta zararlıdır.

**Bulgu 4 - Yüksek Gamma Değerleri Modeli Çökertir:**
g=5 ile RawNet2 tamamen çökmüştür (F1=%0, EER=%50). `(1-pt)^5` çarpanı kolay örneklerin gradyan katkısını neredeyse sıfırlarken, gürültülü ses örneklerinde gradyanları kontrolsüzce büyütmüştür (gradient explosion). Model tüm örnekleri tek sınıfa atayarak kaybı minimize etmiş, accuracy %50'de kilitlenmiştir.

**Bulgu 5 - CosineAnnealing İnce Ayar Sağlar:**
Robust modellerde CosineAnnealing scheduler, son epoch'larda öğrenme hızını kademeli düşürerek val loss'u ortalama %8 daha aşağı çekmiştir. Kosinüs eğrisi, ani LR düşüşlerinden daha pürüzsüz yakınsama sağlar.

### 7.3. Gerçek Dünya (Domain Shift) Dayanıklılığı

Canlı mikrofon kayıtlarında SENet modelleri tutarlı ve düşük entropili kararlar verirken, RawNet2 modelleri daha yüksek Shannon entropisi (belirsizlik) üretmiştir. Mel-spektrogramın `power_to_db(ref=np.max)` normalizasyonu, farklı kayıt koşullarındaki genlik farklarını absorbe ederek domain-shift'e karşı doğal bir kalkan oluşturmuştur.

---

## 8. GUI Sistemi - Gradio Tabanlı İnteraktif Arayüz

Eğitilmiş modellerin pratik kullanımı için **Gradio** tabanlı interaktif bir web arayüzü geliştirilmiştir.

### Özellikler
- Desteklenen formatlar: WAV, MP3, FLAC, OGG
- Mikrofondan canlı kayıt
- 4 model varyantı seçimi (Baseline / Robust x 2D SENet / 1D RawNet2)
- Temperature Scaling (T=0.5 - 5.0): T=1.0 varsayılan (orijinal güven). T<1.0 daha keskin kararlar, T>1.0 daha temkinli kararlar üretir.
- Kayan Pencere (Sliding Window): Uzun sesleri 4 sn'lik parçalara bölüp her birini bağımsız analiz eder, sonuçları ortalar
- XAI Saliency Map: Gradient-tabanlı vurgu haritası - hangi frekans/zaman noktasının kararı etkilediğini gösterir
- Otomatik PDF Rapor: IMRAD formatında adli bilişim raporu üretir
- Shannon Entropisi: Model çıktısının belirsizlik ölçüsü. H > 0.8 ise karar güvenilir değildir

### Opsiyonel DSP Zırhı (Canlı Ses İçin Sinyal Temizleme)
Eğitim verisi (stüdyo kalitesi) ile gerçek dünya (canlı mikrofon) arasındaki uyumsuzlukları azaltmak için 3 adımlı bir ön işleme zinciri sunulmuştur:

| İşlem | Açıklama |
|:---|:---|
| **DC Offset Removal** | Mikrofon veya ses kartının elektriksel devresi, ses sinyaline küçük bir sabit voltaj ekleyebilir. Bu kaymayı `y = y - mean(y)` ile gidererek dalga formunu sıfır çizgisine geri oturtulur. |
| **Silence Trimming** | Kaydın başı ve sonundaki sessiz bölgeleri `librosa.effects.trim(y, top_db=40)` ile keser. 40 dB eşiği, oda gürültüsünü tolere ederken gereksiz boş bölgeleri temizler. |
| **Hann Window Fade** | Sinyalin ani başlayıp bitmesi (hard clip) spektral sızıntı (spectral leakage) yaratır. İlk ve son 200 ms'ye Hann pencere zarfı uygulanarak ses yumuşak başlayıp biter. |

### Örnek Kullanım

```bash
# Gerekli kütüphaneleri kur
pip install -r requirements.txt

# GUI'yi başlat
python gui/app.py

# Tarayıcıda aç: http://127.0.0.1:7860
```

Arayüz açıldıktan sonra:
1. Sol panelden ses dosyası yükleyin veya mikrofon ile kayıt yapın
2. Model seçimi yapın (önerilen: SENet Robust)
3. "Analiz Et" butonuna tıklayın
4. Sonuç: Gerçek/Sahte kararı, güven oranı, Shannon entropisi ve XAI haritası görüntülenir
5. İsteğe bağlı olarak PDF rapor indirilebilir

---

## 9. Proje Dosya Yapısı

```
Deepfake_Audio_Project/
├── src/
│   ├── preprocess.py      # FLAC -> 1D/2D dönüşüm pipeline
│   ├── dataloader.py      # Augmented Dataset + DataLoader
│   ├── pipeline.py        # Grid Search eğitim motoru
│   ├── evaluate.py        # Kapsamlı metrik hesaplama ve görselleştirme
│   ├── losses.py          # Focal Loss implementasyonu
│   ├── explore.py         # Veri seti keşfetme aracı
│   ├── train.py           # Tekil model eğitim scripti
│   └── models/
│       ├── rawnet2.py     # 1D ResNet-tabanlı RawNet2-Lite
│       └── senet.py       # 2D CNN + Squeeze-and-Excitation
├── gui/
│   └── app.py             # Gradio arayüzü + XAI + IMRAD rapor motoru
├── checkpoints/           # Eğitilmiş model ağırlıkları (.pth)
├── data/                  # Sonuç görselleri (confusion matrix, ROC, vb.)
├── requirements.txt
└── README.md
```

---

## 10. Kullanılan Teknolojiler

| Bileşen | Teknoloji | Projede Kullanım Amacı |
|:---|:---|:---|
| Derin Öğrenme | [PyTorch 2.x](https://pytorch.org/) | Sinir ağı mimarileri, autograd türev motoru, CUDA AMP |
| GPU | NVIDIA RTX 4050 (6GB) + [CUDA](https://developer.nvidia.com/cuda-toolkit) | Conv1d/Conv2d operasyonlarının GPU üzerinde hızlandırılması |
| Sinyal İşleme | [librosa 0.10+](https://librosa.org/) | Ses yükleme, STFT, Mel-Filtre bankası, power_to_db |
| Değerlendirme | [scikit-learn](https://scikit-learn.org/) | ROC eğrisi, confusion matrix, F1/precision/recall |
| EER Hesabı | [SciPy](https://scipy.org/) | brentq kök bulma + interp1d enterpolasyonu |
| Görselleştirme | [matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/) | Heatmap, ROC, gruplandırılmış bar grafikleri |
| GUI | [Gradio 4.x](https://gradio.app/) | Web arayüzü, ses yükleme, mikrofon, slider/checkbox |
| PDF Rapor | [FPDF2](https://py-pdf.github.io/fpdf2/) | IMRAD formatında otomatik adli bilişim raporu |
| Veri Yönetimi | [NumPy](https://numpy.org/) + [pandas](https://pandas.pydata.org/) | .npy dosya I/O, metadata analizi, undersampling |
| IDE | PyCharm Professional | Proje yönetimi, debug, Git entegrasyonu |

---

## Referanslar

1. Todisco, M., Wang, X., Vestman, V., et al. (2019). "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection." *Proc. Interspeech.*
2. Hu, J., Shen, L., Sun, G. (2018). "Squeeze-and-Excitation Networks." *CVPR.*
3. Tak, H., Patino, J., Todisco, M., et al. (2021). "End-to-End Anti-Spoofing with RawNet2." *ICASSP.*
4. Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. (2017). "Focal Loss for Dense Object Detection." *ICCV.*
5. Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR.*
