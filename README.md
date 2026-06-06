# Hibrit Algoritmik Ticaret Motoru

**Surum:** 2.0 (Production-Grade)  
**Gelistiriciler:** Umut Yalcin & Rafet Emir Dilsiz  

---

## 1. Projeye Giris

Bu proje, finansal piyasalardaki (Borsa Istanbul, Kripto vb.) karmasik fiyat hareketlerini analiz edip, insan duygularından arindirilmis tam otomatik alim-satim kararlari veren ileri duzey bir yapay zeka sistemidir.

Sistem, geleneksel "tek hisse ezberleyen" botlarin aksine **"Global Beyin + Yerel Risk" (Global Model + Local Risk)** felsefesiyle calisir.

---

## Global Yapay Zeka

Butun piyasanin (farkli sektorlerin) verilerini tek bir mega havuzda birlestirerek piyasanin genel davranislarini ogrenir.

Model yalnizca tek bir hisseyi ezberlememez; piyasanin:

- Trend davranislarini
- Volatilite karakterini
- Momentum gecislerini
- Psikolojik dongulerini

ogrenmeye calisir.

---

## Yerel Risk Yoneticisi

Her hisse senedinin kendi mikro-yapisina gore dinamik Stop-Loss (SL) ve Take-Profit (TP) oranlari belirlenir:

- Volatilite
- Testere (Whipsaw) egilimi
- Momentum sertligi
- Trend davranisi

---

## 2. Terimler Sozlugu

### Olay-Dongusu (Event-Driven) Backtest

Sanal bakiye iluzyonlarini ve vektorel hatalari engelleyen, piyasa zaman akisini mum mum (bar-by-bar) simule eden gercekci test motoru.

---

### Out-of-Universe (OOU) Test

Yapay zekanin egitim sirasinda hic gormedigi yepyeni bir hisse veya sektor uzerinde test edilmesi.

Ornek:
> Model egitimde FROTO'yu gormedi, testte basarili sonuc verdi.

Bu durum modelin gercekten genelleme yapabildigini gosterir.

---

### Calmar Ratio

Risk/odul optimizasyon metrigi:

```
Calmar Ratio = Net Getiri / Maximum Drawdown
```

Genetik algoritmanin optimize ettigi ana performans skorudur.

---

### Stop-Loss (SL) ve Take-Profit (TP)

- **SL:** Zarari sinirlandiran otomatik cikis noktasi
- **TP:** Kari realize eden otomatik cikis noktasi

---

### LSTM (Long Short-Term Memory)

Zaman serisi analizi yapan ozel sinir agi mimarisi.

Ornek:
> Son 60 mum -> Gelecek fiyat davranisi tahmini

---

### Kalman Filtresi

Fiyat hareketlerindeki anlamsiz sicramalari ve gurultuyu temizleyen istatistiksel filtreleme yontemi.

---

### CMA-ES

"En iyi parametre hayatta kalir" mantigiyla calisan evrimsel optimizasyon algoritmasi.

Amac: Maksimum getiri + Minimum risk

---

## 3. Sistem Mimarisi

Proje, tam otomatik ve moduler calisan 5 ana katmandan olusur.

---

### Katman 1: Otomatik Veri Hatti ve Veritabani
`data_fetcher.py` ve `db_manager.py`

- SQLite tabanli yuksek performansli veri mimarisi
- Veriler SARDIS BIST API uzerinden artimli (incremental) cekilir
- Eksik zaman dilimleri (gap) otomatik doldurulur
- Tum piyasa verileri `market_data.db` icinde saklanir

---

### Katman 2: Sinyal ve Oznitelik Muhendisligi
`signal_filters.py` ve `feature_engineering.py`

Ham piyasa verileri dogrudan modele verilmez:

- Kalman filtresi uygulanir
- Gurultu azaltilir
- Matematiksel oznitelikler uretilir

Uretilen oznitelikler:

- Volatilite
- Momentum farklari
- Trend sapmalar
- Fiyat degisim hizlari

---

### Katman 3: Super Beyin - Global LSTM Modeli
`train_model.py`

Farkli sektorlerden secilmis hisseler kullanilarak buyuk bir Mega Dataset olusturulur:

- Havacilik (THYAO)
- Bankacilik (GARAN, HALKB, ISCTR, VAKBN)
- Savunma (ASELS)
- Enerji / Petrokimya (TUPRS, SASA, SISE)

Overfitting onleme teknikleri: Shuffle, Early Stopping, cok-sektor egitim.

---

### Katman 4: Genetik Optimizasyon
`cma_optimizer.py`

Her hisse icin ayri ayri calisir.

Optimizasyon hedefi:
- Maksimum kar
- Minimum drawdown
- Gercekci risk yonetimi

Guvenlik kurallari:
- SL minimum %0.75 altindaki parametreler reddedilir
- Az islem yapan (< 10) parametreler elenir

---

### Katman 5: Event-Driven Backtest Motoru
`backtest_engine.py` ve `main.py`

Sabit Kasa Modeli (Fixed Position Sizing): Her islem sabit sermaye ile acilir (10.000 TL).

Bu yaklasim logaritmik buyume iluzyonlarini ve sahte performans sonuclarini engeller.

Uretilen ciktilar:
- Net Kar
- Gercek Islem Sayisi
- Win Rate
- Max Drawdown
- Risk/Odul Analizi

---

## 4. Proje Yapisi

```
cma_es_finance_core/
|-- database/
|   |-- db_manager.py           # SQLite tablo ve sorgu yoneticisi
|   `-- market_data.db          # Milyonlarca mumu tutan yerel veritabani
|
|-- core/
|   |-- data_fetcher.py         # SARDIS API'den veri indiren motor
|   |-- data_pipeline.py        # Eksik verileri dolduran tesisat
|   |-- signal_filters.py       # Kalman filtresi
|   |-- feature_engineering.py  # LSTM icin oznitelik uretici
|   |-- ai_prep.py              # Veri olcekleme (MinMax) ve paketleme
|   |-- ai_models.py            # LSTM mimarisi
|   `-- backtest_engine.py      # Event-driven simulasyon motoru
|
|-- optimizers/
|   `-- cma_optimizer.py        # Genetik Algoritma (Calmar Optimizasyonu)
|
|-- data/
|   `-- global_lstm_model_15min.keras  # Egitilmis global model
|
|-- train_model.py              # Global Super Beyin'i egiten modul
|-- main.py                     # Optimizasyon ve backtest orkestrasyonu
|-- dashboard.py                # Streamlit gorsel panel
`-- backtest_sonuclari_v2.xlsx  # Nihai performans raporu
```

---

## 5. Sistemin Felsefesi

Bu surum, eski vektorel backtest sistemlerinin urettigi gercek disi milyarlik getiriler, sonsuz bilesik buyume ve zaman akisi hatalarini tamamen reddeder.

Bunun yerine akademik olarak savunulabilir, gercekci sonuclar uretir.

---

## FROTO OOU Testi

Model egitim sirasinda FROTO verisini hic gormedigi halde optimize parametreler bulup aylar icinde sabit kasayla yaklasik %35 net getiri saglayabilir.

Bu, modelin yalnizca ezber yapmadigini; gercekten piyasa davranisini ogrendigini gosterir.

---

## Asimetrik Risk Yonetimi

Sistem dusuk kazanma oraniyla bile uzun vadede kar edebilir.

Ornek: %42 Win Rate ile pozitif getiri mumkundur.

Sebep: Kucuk zararlar, buyuk kazanclar, siki risk kontrolu.

---

## Lisans

MIT License
