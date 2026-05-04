# 🚀 Hibrit Algoritmik Ticaret Motoru (Hybrid Algo-Trading Engine)

**Sürüm:** 1.0  
**Geliştiriciler:** Umut Yalçın & Rafet Emir Dilsiz  

---

## 📖 1. Projeye Giriş

Bu proje, finansal piyasalardaki (Borsa İstanbul, Kripto vb.) karmaşık fiyat hareketlerini analiz edip, insan duygularından (korku ve açgözlülük) arındırılmış **tam otomatik alım-satım kararları** veren bir yapay zeka sistemidir.

Sistem yalnızca şu soruya cevap vermez:

> “Fiyat yükselecek mi?”

Aynı zamanda şunları da optimize eder:

- Yanılırsam **zararımı nasıl minimize ederim?**
- Haklıysam **kârımı nerede realize ederim?**

Bunu **genetik algoritmalar** ile kendi kendine öğrenerek yapar.

---

## 📚 2. Terimler Sözlüğü

### 📈 Finans Terimleri

- **Backtest (Geriye Dönük Test):**  
  Sistem geçmiş veriler üzerinde test edilir.  
  _"Geçmişte çalışsaydı ne olurdu?"_

- **Stop-Loss (SL):**  
  Zararı sınırlamak için otomatik çıkış noktası  
  _(Örn: %0.11)_

- **Take-Profit (TP):**  
  Kâr hedefi noktası  
  _(Örn: %12.63)_

- **Max Drawdown:**  
  Kasanın en yüksek noktadan en düşük noktaya düşüş yüzdesi

- **Whipsaw (Testere Piyasası):**  
  Yönsüz, zigzag hareketli piyasa koşulları

---

### 🤖 Yapay Zeka ve Yazılım Terimleri

- **LSTM (Long Short-Term Memory):**  
  Zaman serisi analizi yapan özel sinir ağı  
  (Son 60 mum → Gelecek tahmin)

- **Kalman Filtresi:**  
  Gürültü temizleyici (trend çıkarımı)

- **CMA-ES:**  
  Evrimsel optimizasyon algoritması  
  (“En iyi parametre hayatta kalır” mantığı)

- **Overfitting:**  
  Modelin geçmişi ezberleyip gelecekte başarısız olması

- **Early Stopping:**  
  Ezberlemeyi önlemek için erken durdurma tekniği

---

## 🏗️ 3. Sistem Mimarisi

Sistem 5 katmandan oluşur:

---

### 🔹 1. Veri Toplama ve Temizleme (`data_pipeline.py`)

- Eksik zaman dilimleri doldurulur
- Look-ahead bias engellenir
- 15 dakikalık veri kullanılır

---

### 🔹 2. Gürültü İptali ve Feature Engineering  
(`signal_filters.py`, `feature_engineering.py`)

- Kalman filtresi uygulanır
- Yeni özellikler üretilir:
  - Volatilite
  - Trend sapmaları
  - Fiyat farkları

---

### 🔹 3. Yapay Zeka Modeli  
(`ai_prep.py`, `ai_models.py`)

- LSTM modeli eğitilir
- Girdi: Son 60 mum (~15 saat)
- Çıktı: Gelecek 15 mum için olasılık

Örnek: %55 ihtimalle fiyat yükselecek


---

### 🔹 4. Genetik Optimizasyon  
(`cma_optimizer.py`)

Amaç:

> Maksimum kâr + Minimum risk

Optimize edilen parametreler:

- Confidence Threshold
- Stop-Loss
- Take-Profit

Kullanılan metrik:

**Calmar Ratio = Getiri / Drawdown**

---

### 🔹 5. Backtest Motoru  
(`backtest_engine.py`, `main.py`)

- Gerçekçi simülasyon
- Komisyon: binde 2
- Çıktılar:
  - Net kâr
  - İşlem sayısı
  - Drawdown

---

## 📂 4. Proje Yapısı

```plaintext
cma_es_finance_core/
├── data/
│   ├── 15m/
│   │   ├── THYAO.csv
│   │   └── ASELS.csv
│   └── best_lstm_model_15min.keras
│
├── core/
│   ├── data_pipeline.py
│   ├── signal_filters.py
│   ├── feature_engineering.py
│   ├── ai_prep.py
│   ├── ai_models.py
│   └── backtest_engine.py
│
├── optimizers/
│   └── cma_optimizer.py
│
├── main.py
└── finansal_backtest_raporu_v1.xlsx
```

## 🎯 5. Çıktıların Anlamı

### 📊 Örnek Sonuçlar (THYAO)

- **Win Rate:** %39.68  
  → Düşük gibi görünür ama kârlıdır

- **Max Drawdown:** %-17.34  
  → Risk kontrol altında

- **Net Kâr:** 20.670 TL  
  → 10.000 TL → 30.670 TL

---

### 💡 Kritik İçgörü

Bu sistem:

- Az kazanır ama **büyük kazanır**
- Çok kaybeder ama **küçük kaybeder**

---

## 🧠 Sistem Felsefesi

Bu proje şunu kabul eder:

> Finansal piyasalar tamamen tahmin edilemez.

Ama şunu kanıtlar:

> Doğru istatistik + yapay zeka + risk yönetimi = sürdürülebilir kâr

---

## ⚠️ Uyarı

Bu sistem yatırım tavsiyesi değildir.  
Gerçek piyasalarda kullanılmadan önce kapsamlı test yapılmalıdır.

---

## 📌 Gelecek Geliştirmeler

- [ ] Reinforcement Learning entegrasyonu  
- [ ] Multi-asset trading (aynı anda birden fazla varlık)  
- [ ] Gerçek zamanlı işlem (live trading)  
- [ ] Web dashboard (izleme paneli)  

---

## ⭐ Katkı

Projeye katkıda bulunmak için PR açabilirsiniz.

---

## 📜 Lisans

MIT License
