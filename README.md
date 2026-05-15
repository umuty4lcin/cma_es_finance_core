# 🚀 Hibrit Algoritmik Ticaret Motoru (Hybrid Algo-Trading Engine)

**Sürüm:** 2.0 *(Production-Grade)*  
**Geliştiriciler:** Umut Yalçın & Rafet Emir Dilsiz  

---

# 📖 1. Projeye Giriş

Bu proje, finansal piyasalardaki (Borsa İstanbul, Kripto vb.) karmaşık fiyat hareketlerini analiz edip, insan duygularından arındırılmış tam otomatik alım-satım kararları veren ileri düzey bir yapay zeka sistemidir.

Sistem, geleneksel “tek hisse ezberleyen” botların aksine **“Global Beyin + Yerel Risk” (Global Model + Local Risk)** felsefesiyle çalışır:

---

## 🌍 Global Yapay Zeka

Bütün piyasanın (farklı sektörlerin) verilerini tek bir mega havuzda birleştirerek piyasanın genel davranışlarını öğrenir.

Model yalnızca tek bir hisseyi ezberlemez; piyasanın:

- Trend davranışlarını
- Volatilite karakterini
- Momentum geçişlerini
- Psikolojik döngülerini

öğrenmeye çalışır.

---

## 🛡️ Yerel Risk Yöneticisi

Her hisse senedinin kendi mikro-yapısına göre:

- Volatilite
- Testere (Whipsaw) eğilimi
- Momentum sertliği
- Trend davranışı

analiz edilerek, genetik algoritma yardımıyla dinamik:

- **Stop-Loss (SL)**
- **Take-Profit (TP)**

oranları belirlenir.

---

# 📚 2. Terimler Sözlüğü

## 📈 Finans ve Mimari Terimleri

### 🔹 Olay-Döngülü (Event-Driven) Backtest

Sanal bakiye illüzyonlarını ve vektörel hataları engelleyen, piyasa zaman akışını mum mum (*bar-by-bar*) simüle eden gerçekçi test motoru.

---

### 🔹 Out-of-Universe (OOU) Test

Yapay zekanın eğitim sırasında hiç görmediği yepyeni bir hisse veya sektör üzerinde test edilmesi.

Örnek:

> Model eğitimde FROTO görmedi ama testte başarılı sonuç verdi.

Bu durum modelin gerçekten genelleme yapabildiğini gösterir.

---

### 🔹 Calmar Ratio

Risk/ödül optimizasyon metriği:

```math
\text{Calmar Ratio} = \frac{\text{Net Getiri}}{\text{Maximum Drawdown}}
```

Genetik algoritmanın optimize ettiği ana performans skorudur.

---

### 🔹 Stop-Loss (SL) & Take-Profit (TP)

- **SL:** Zararı sınırlayan otomatik çıkış noktası
- **TP:** Kârı realize eden otomatik çıkış noktası

---

## 🤖 Yapay Zeka Terimleri

### 🔹 LSTM (Long Short-Term Memory)

Zaman serisi analizi yapan özel sinir ağı mimarisi.

Örnek:

> Son 60 mum → Gelecek fiyat davranışı tahmini

---

### 🔹 Kalman Filtresi

Fiyat hareketlerindeki anlamsız sıçramaları ve gürültüyü temizleyen istatistiksel filtreleme yöntemi.

---

### 🔹 CMA-ES

“En iyi parametre hayatta kalır” mantığıyla çalışan evrimsel optimizasyon algoritması.

Amaç:

> Maksimum getiri + Minimum risk

---

# 🏗️ 3. Sistem Mimarisi (Katmanlar)

Proje, tam otomatik ve modüler çalışan 5 ana katmandan oluşur.

---

## 🔹 1. Otomatik Veri Hattı ve Veritabanı  
`data_fetcher.py` & `db_manager.py`

- CSV tabanlı yapı terk edilmiştir
- SQLite tabanlı yüksek performanslı veri mimarisi kurulmuştur
- Veriler SARDIS BIST API üzerinden artımlı (*incremental*) çekilir
- Eksik zaman dilimleri (*gap*) otomatik doldurulur
- Tüm piyasa verileri `market_data.db` içinde saklanır

---

## 🔹 2. Sinyal ve Öznitelik Mühendisliği  
`signal_filters.py` & `feature_engineering.py`

Ham piyasa verileri doğrudan modele verilmez.

Önce:

- Kalman filtresi uygulanır
- Gürültü azaltılır
- Matematiksel öznitelikler üretilir

Üretilen bazı feature’lar:

- Volatilite
- Momentum farkları
- Trend sapmaları
- Fiyat değişim hızları

---

## 🔹 3. Süper Beyin: Global LSTM Modeli  
`train_model.py`

Yapay zeka yalnızca tek bir hisse üzerinde eğitilmez.

Farklı sektörlerden seçilmiş “öğretmen hisseler” kullanılarak:

- Havacılık
- Bankacılık
- Sanayi
- Teknoloji

gibi alanlardan büyük bir **Mega Dataset** oluşturulur.

### 📌 Overfitting Önleme Teknikleri

- Shuffle
- Early Stopping
- Multi-sector eğitim yaklaşımı

---

## 🔹 4. Genetik Optimizasyon  
`cma_optimizer.py`

Her hisse için ayrı ayrı çalışır.

Optimizasyon hedefi:

- Maksimum kâr
- Minimum drawdown
- Gerçekçi risk yönetimi

### 📌 Güvenlik Kuralları

Örnek:

- `SL > %0.75` gibi mantıksız parametreler reddedilir
- Hileli optimizasyonlar filtrelenir
- Sistem batma riskine karşı korunur

---

## 🔹 5. Gerçekçi Backtest Motoru  
`backtest_engine.py` & `main.py`

### 📌 Sabit Kasa Modeli (Fixed Position Sizing)

Her işlem sabit sermaye ile açılır.

Örnek:

> Her işlem = 10.000 TL

Bu yaklaşım:

- Logaritmik büyüme illüzyonlarını
- Gerçek dışı bileşik getirileri
- Sahte performans sonuçlarını

engeller.

### 📊 Üretilen Çıktılar

- Net Kâr
- Gerçek İşlem Sayısı
- Win Rate
- Max Drawdown
- Risk/Ödül Analizi

---

## 📂 4. Proje Yapısı


```plaintext
cma_es_finance_core/
├── database/                   # Veritabanı Yönetimi
│   ├── db_manager.py           # SQLite tablo ve sorgu yöneticisi
│   └── market_data.db          # Milyonlarca mumu tutan yerel veritabanı
│
├── core/                       # Motorun İç Parçaları
│   ├── data_fetcher.py         # SARDIS API'den veri indiren ahtapot
│   ├── data_pipeline.py        # Eksik verileri dolduran tesisat
│   ├── signal_filters.py       # Kalman filtresi ve sinyal temizleyiciler
│   ├── feature_engineering.py  # LSTM için matematiksel öznitelik üreticisi
│   ├── ai_prep.py              # Veri ölçekleme (MinMax) ve paketleme
│   ├── ai_models.py            # LSTM mimarisinin tasarımı
│   └── backtest_engine.py      # Olay-döngülü (Event-Driven) simülasyon motoru
│
├── optimizers/                 # Karar Mekanizmaları
│   └── cma_optimizer.py        # Genetik Algoritma (Calmar Optimizasyonu)
│
├── train_model.py              # Global Süper Beyin'i eğiten modül
├── main.py                     # Optimizasyon ve backtest orkestrasyonu
└── backtest_sonuclari_v2.xlsx  # Nihai performans raporu
```

---

# 💡 5. Sistemin Felsefesi ve Çıktı Analizi

Bu sürüm, eski vektörel backtest sistemlerinin ürettiği:

- Gerçek dışı milyarlık getiriler
- Sonsuz bileşik büyüme
- Zaman akışı hataları

gibi problemleri tamamen reddeder.

Bunun yerine:

> Akademik olarak daha gerçekçi ve sürdürülebilir sonuçlar üretir.

---

## 📊 FROTO OOU Testi

Model eğitim sırasında FROTO verisini hiç görmediği halde:

- `%0.75 SL`
- `%16.24 TP`

gibi parametreler bulup:

> Aylar içerisinde sabit kasayla yaklaşık `%35` net getiri sağlayabilir.

Bu, modelin yalnızca ezber yapmadığını; gerçekten piyasa davranışını öğrendiğini gösterir.

---

## ⚖️ Asimetrik Risk Yönetimi

Sistem:

- Düşük kazanma oranıyla bile
- Uzun vadede kâr edebilir

Örnek:

> %42 Win Rate ile bile pozitif getiri mümkündür.

Sebep:

- Küçük zararlar
- Büyük kazançlar
- Sıkı risk kontrolü

---

# 🧠 Sistem Felsefesi

> “Biz yapay zekaya piyasanın kaosunu öğrettik, evrimsel algoritmaya ise o kaosta nasıl hayatta kalacağını...”

---


# 📜 Lisans
MIT License