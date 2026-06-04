# Derin Öğrenme Tabanlı Deepfake Ses Tespit Sistemi: 1D ve 2D Temsil Uzaylarının Karşılaştırmalı Analizi

Bu proje, sentetik ses (deepfake) tespitinde kullanılan derin öğrenme mimarilerinin, sinyal temsil uzaylarına (1D ham dalga formu ve 2D Mel-spektrogram) göre performans ve genelleme kapasitelerini inceleyen kapsamlı ve deneysel bir adli bilişim (audio forensics) çalışmasıdır.

## 1. Giriş ve Temel Hipotez (Introduction)

Ses sentezleme teknolojilerindeki (Text-to-Speech, Vocoderlar) üstel gelişim, insan işitsel algısını aşan gerçeklikte deepfake seslerin üretilmesini sağlamıştır. Ancak WaveRNN veya Griffin-Lim gibi sentez motorları, yeniden sentezleme (re-synthesis) sürecinde insan kulağının duyamayacağı ancak frekans uzayında iz bırakan **periyodik yapay örüntüler (spectral artifacts)** oluştururlar. 

Bu çalışmanın merkezindeki temel araştırma sorusu şudur: *Ses sinyali 1D zaman serisi olarak mı, yoksa 2D zaman-frekans matrisi olarak mı modellendiğinde bu sentetik artefaktlar daha net ayrıştırılabilir?* Bu amaçla, ham dalga formunu doğrudan işleyen **RawNet2** mimarisi ile, Mel-spektrogramı geometrik bir doku olarak işleyen **SENet** (Squeeze-and-Excitation Network) mimarisi çapraz testlere tabi tutulmuştur.

## 2. Metodoloji ve Fiziksel Temsil (Methods)

### 2.1. Veri Seti ve Ön İşleme Mimarisi
Eğitim ve doğrulama süreçlerinde **ASVspoof 2019 Logical Access (LA)** veri seti kullanılmıştır. Veri seti 16 kHz örnekleme hızında, 4 saniyelik sabit uzunluklara (64,000 örneklem) getirilmiş ve veri dengesizliğini önlemek için undersampling uygulanarak toplam 40,000 örneğe (20K Bonafide, 20K Spoof) indirgenmiştir. 

Sinyallerin analizi için iki farklı topolojik temsil kullanılmıştır:
*   **1D Zaman Uzayı (Ham Dalga Formu):** Sinyalin anlık genlik değerlerinin zaman eksenindeki varyasyonudur. Pozitif ve negatif hava basıncı değişimlerini doğrudan modelleyebilmek için 1D mimarisinde negatif değerleri sıfırlamayan `LeakyReLU` aktivasyonu tercih edilmiştir. Bu temsilde frekans bilgisi "gizli" (implicit) yapıdadır ve modelin bunu yerel korelasyonlardan kendi kendine öğrenmesi beklenir.
*   **2D Zaman-Frekans Uzayı (Mel-Spektrogram):** Sinyal, Kısa Zamanlı Fourier Dönüşümü (STFT) ile parçalanarak frekans eksenine izdüşürülmüştür. $n_{fft}=2048$ ve $hop\_length=512$ kullanılarak, zamansal ve frekansal çözünürlük arasında (Heisenberg belirsizlik ilkesi bağlamında) optimum denge kurulmuştur. Sinyal, 128 mel-frekans bandına bölünerek `power_to_db` normalizasyonu ile matris formuna ($128 \times 126$) getirilmiştir. Bu 2D izdüşüm, deepfake artefaktlarını açık (explicit) bir görsel çizgi deseni halinde modelin erişimine sunar.

Aşırı uyumu engellemek adına eğitim setine stokastik veri artırma uygulanmıştır. 1D temsilde **Gaussian Gürültü** eklenerek çevresel bozulmalar simüle edilirken; 2D temsilde **SpecAugment** (rastgele zaman ve frekans bantlarını sıfırlama) kullanılarak modelin eksik bilgilere karşı direnci artırılmıştır.

### 2.2. Ağ Topolojileri ve Karar Mekanizmaları
*   **RawNet2 (1D CNN):** $stride=32$ parametreli ilk evrişim (convolution) katmanı, 64,000 boyutlu devasa girdiyi ani bir boyutsal indirgeme ile soyutlar. Ardışık $stride=2$ oranlı Residual bloklar (ResBlock), yerel zaman korelasyonlarını hiyerarşik olarak birleştirerek özellikleri süzer.
*   **SENet (2D CNN + SE):** Squeeze-and-Excitation blokları ile güçlendirilmiş 2D mimari. SE blokları, her katmanda hangi frekans kanallarının deepfake tespiti için daha kritik olduğunu dinamik olarak hesaplar. Küçültme oranının (reduction ratio) $r=16$ seçilmesiyle, model 128 kanallık bilgiyi 8 boyutlu bir darboğazdan (bottleneck) geçmeye zorlanır. Bu durum, modelin gereksiz gürültüleri elemesini ve sadece karar sınırını belirleyen temel frekanslara odaklanmasını sağlar.

### 2.3. Optimizasyon Süreci, Kısıtlar ve Hiperparametreler
Sınırlı donanım kaynakları (6GB VRAM kapasiteli RTX 4050) eğitim optimizasyonunu şekillendirmiştir. 32 boyutlu batch-size bellek sınırını aştığından, donanımsal çözüm olarak **Gradient Accumulation** (batch=8, 4 adım biriktirme) ve bellek tüketimini yarıya indiren **Karma Hassasiyet (AMP - FP16)** bir arada kullanılmıştır. 

Öğrenme stratejisi olarak standart Adam yerine, L2 regülarizasyonunu ağırlık sönümleme (weight decay, $10^{-3}$) mekanizmasıyla doğru ayrıştıran **AdamW** seçilmiştir. Bununla birlikte, öğrenme hızının sabit kalması yerine kosinüs eğrisi boyunca kademeli düşmesini sağlayan **Cosine Annealing** (T_max=EPOCHS) algoritması ile kayıp yüzeyindeki (loss surface) minimuma stabil bir iniş sağlanmıştır.

## 3. Deneysel Bulgular (Results)

Grid-search mantığıyla eğitilen 7 model varyantının, tamamen görülmemiş 8,000 örnekli (doğrulama) set üzerindeki başarımları aşağıda sunulmuştur. Biyometrik güvenlik standartlarına uygun olarak en önemli değerlendirme kriteri **Equal Error Rate (EER)** olarak belirlenmiştir.

| Deney # | Mimari Tipi | Kayıp Fonksiyonu (Loss) | Accuracy | Precision | Recall | F1 Score | EER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **SENet Robust (2D)** | **CE + Label Smoothing** | **96.96%** | **96.13%** | **97.93%** | **97.02%** | **2.18%** |
| 2 | SENet Baseline (2D) | CrossEntropy (CE) | 96.30% | 95.67% | 97.05% | 96.36% | 2.70% |
| 3 | SENet Focal (2D) | Focal Loss ($\gamma=2$) | 95.68% | 95.57% | 95.88% | 95.72% | 3.50% |
| 4 | RawNet2 Baseline (1D) | CrossEntropy (CE) | 95.78% | 94.44% | 97.40% | 95.89% | 3.45% |
| 5 | RawNet2 Robust (1D) | CE + Label Smoothing | 95.20% | 93.75% | 96.98% | 95.34% | 3.56% |
| 6 | RawNet2 Focal (1D) | Focal Loss ($\gamma=2$) | 94.18% | 93.76% | 94.75% | 94.25% | 4.70% |
| 7 | RawNet2 Focal (1D) | Focal Loss ($\gamma=5$) | 50.00% | 0.00% | 0.00% | 0.00% | 50.00% |

## 4. Akademik Analiz ve Çıkarımlar (Discussion)

Elde edilen sayısal veriler ve eğitim sırasındaki model davranışları, salt başarım rakamlarının ötesinde spesifik nedensel (causal) çıkarımlar sunmaktadır:

**A. Sayısal Metriklerin Nedensel (Causal) Analizi:**
*   **Precision (Hassasiyet - %96.13) vs. Recall (Duyarlılık - %97.93) Asimetrisi:** SENet Robust modelinde gözlemlenen %1.80'lik asimetrik sapma sistematik bir sonuçtur. Adli bilişim (forensics) bağlamında sahte bir sesi gözden kaçırmak (False Negative), gerçek bir sese sahte uyarısı vermekten (False Positive) çok daha yüksek bir risk faktörüdür. Label Smoothing tekniği, modelin marjinal örneklere tam güven (1.0) atamasını engellediğinden, model şüpheli durumlarda "sahte" (spoof) etiketini atamaya algoritmik olarak daha eğilimli hale gelmiştir. Bu durum Recall oranını maksimize ederken Precision'dan cüzi bir ödün verilmesine neden olmuş; bu sonuç güvenlik mimarilerinin hedefleriyle tam örtüşmüştür.
*   **F1 Skoru (%97.02) ve Degenerasyon Kontrolü:** Salt Recall optimizasyonu, modelin tüm girdileri "sahte" olarak sınıflandırarak %100 oranına ulaşması gibi dejenere (trivial) çözümlere yol açabilmektedir. %97.02 düzeyindeki F1 harmonik ortalaması, modelin bu tip bir dejenere duruma düşmediğini ve gerçek/sahte sınıflar arasındaki geometrik karar sınırını (decision boundary) optimal şekilde modellediğini göstermektedir.
*   **EER (Equal Error Rate - %2.18):** ASVspoof standartlarında birincil değerlendirme ölçütü olan EER, SENet Baseline varyantında %2.70 iken, Label Smoothing ve Cosine Annealing entegrasyonuyla %2.18'e gerilemiştir. Bu iyileşme, modelin içsel (internal) logit dağılımları arasındaki örtüşmenin (overlap) azaldığını ve sınıfların uzaysal olarak daha kesin ayrıştığını ifade etmektedir.
*   **Temsil Uzaylarına Bağlı EER Varyansı (%3.45 vs %2.70):** RawNet2 Baseline ve SENet Baseline arasındaki 0.75 puanlık EER farkı, zaman uzayındaki örtük (implicit) çıkarım zorluğunun doğrudan ölçümüdür. Modelin frekans dönüşümünü evrişim filtreleri üzerinden dolaylı yoldan öğrenmeye zorlanması, sisteme %0.75 oranında ek hata maliyeti yüklemiştir.

**B. Temsil Uzayının Geometrik Avantajı:** 
Tüm karşılaştırma matrislerinde, 2D temsili kullanan SENet mimarisi 1D RawNet2'ye üstünlük sağlamıştır. Bunun temel nedeni, vocoder sentezleme süreçlerinin oluşturduğu ardışık faz hatalarının, Mel-spektrogram matrisinde kolayca yakalanabilen 2 boyutlu uzamsal dokular (spatially-correlated textures) yaratmasıdır. 1D zaman serisinde bu yapısal anomalilerin modellenmesi topolojik olarak çok daha maliyetlidir.

**C. "Çifte Dikkat" Çakışması ve Focal Loss Degenerasyonu:**
Nesne tespitinde başarılı olan Focal Loss formülasyonu ($\mathcal{L} = -(1 - p_t)^\gamma \log(p_t)$), ses alanında kritik sorunlar yaratmıştır. $\gamma=5$ gibi ekstrem bir odaklanma parametresi uygulandığında 1D RawNet2 tamamen çökmüştür (%50 Accuracy). Bunun matematiksel sebebi, $(1-p_t)^5$ çarpanının kolay/temiz ses örneklerindeki gradyan akışını neredeyse sıfırlarken, doğal akustik gürültü barındıran "zor" örneklerin gradyan ağırlığını kontrolsüzce büyütmesidir (Gradient Explosion). 
2D SENet mimarisinde model çökmemiş olsa da başarım düşmüştür. Squeeze-and-Excitation bloklarının içsel kanal dikkat (attention) mekanizması ile Focal Loss'un veri odaklı dışsal dikkatinin aynı anda çalışması birbiriyle çakışarak modelin optimizasyonunu bozmuştur.

**C. Label Smoothing ve Domain Kayması (Domain Shift) Direnci:**
Focal Loss'un başarısızlığına karşın, etiket yumuşatma (Label Smoothing, $\epsilon=0.1$) tekniği modeli stabilize etmiştir. Standart CrossEntropy'nin modeli %100 özgüvenli (over-confident) olmaya ittiği durumlarda (1.0 vs 0.0), etiketlerin 0.95 ve 0.05 hedeflerine çekilmesi modelin eğitim verisini ezberlemesini önlemiştir. Bu regülarizasyon, mikrofon farklılıkları ve ortam gürültüsü (Domain Shift) içeren canlı testlerde modelin sahte/gerçek kararını çok daha stabil vermesini sağlamıştır.

**D. Sinyal Ön İşleme (DSP) ve Karar Güvenilirliği:**
Eğitim ortamı (stüdyo kalitesi) ile gerçek dünya (canlı mikrofon) arasındaki veri uyumsuzluklarını önlemek amacıyla test pipeline'ına opsiyonel bir DSP Zırhı eklenmiştir:
*   *DC Offset Removal:* Kaymanın (voltaj ötelemesi) genlik merkezini bozmasını engeller.
*   *Hann Window Fade:* Sinyal başı ve sonundaki ani kesilmelerin (hard clip) spektral sızıntı (spectral leakage) yaratmasını önler.
Ayrıca sınırda (belirsiz) kararları tespit edebilmek için nihai softmax çıktılarına **Shannon Entropisi** uygulanmıştır. Entropinin yüksek çıkması durumunda ($H > 0.8$), kararın deterministik olmadığı matematiksel olarak kullanıcıya bildirilmektedir. Yüksek Recall ve yüksek F1 skoru, modelin sahte sesleri yakalamadaki kararlılığını kanıtlamaktadır; zira güvenlik sistemlerinde bir sahte sesi kaçırmak (False Negative), gerçek bir sese sahte demekten (False Positive) çok daha maliyetlidir.

## 5. Proje Organizasyonu ve Teknolojik Altyapı

*   `preprocess.py`: Ham veriden (FLAC) Numpy matrislerine (1D ve 2D) dönüşüm ve normalizasyon.
*   `dataloader.py`: PyTorch `Dataset`, veri maskeleme, batch ve dinamik gürültü/SpecAugment işlemleri.
*   `pipeline.py` / `evaluate.py`: Tüm konfigürasyonların (Grid Search) eğitimi ve IMRAD grafiksel metrik üretim motoru.
*   `app.py`: Modellerin birleştirildiği tam bağımsız (sliding-window destekli) Gradio tabanlı interaktif adli analiz arayüzü ve PDF rapor üreticisi.

**Kullanılan Referans Kütüphaneler:**
*   [PyTorch](https://pytorch.org/): CUDA tabanlı hızlandırılmış Tensör işlemleri ve `autograd` türev motoru.
*   [librosa](https://librosa.org/): Sinyal okuma, STFT, Mel-Filtre bankası ve spektrogram analizi.
*   [scikit-learn](https://scikit-learn.org/) & [SciPy](https://scipy.org/): Performans eğrilerinin (ROC), karmaşıklık matrislerinin ve Enterpolasyon tabanlı EER noktalarının hesaplanması.
*   [Gradio](https://gradio.app/): Sistem entegrasyonu, canlı demo ve GUI inşası.
*   [pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/): Veri işleme, metadata kontrolü ve tensör manipülasyonları.
