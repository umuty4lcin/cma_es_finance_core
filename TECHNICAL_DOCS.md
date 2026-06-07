# Hibrit Algoritmik Ticaret Motoru — Teknik Dokümantasyon

**Proje:** Gelismis Istatistiksel Sinyal Isleme ve Yapay Zeka Teknikleri Hibriti ile Finansal Piyasalarda Ongu ve Risk Yonetimi Sistemi  
**Geliştiriciler:** Umut Yalcin & Rafet Emir Dilsiz  
**Kurum:** Erciyes Universitesi Bilgisayar Muhendisligi  
**Surum:** 2.0

---

## Icerik

1. Projenin Amaci ve Felsefesi
2. Kullanilan Teknolojiler
3. Proje Haritasi (Dosya Yapisi)
4. Uctan Uca Veri Akisi
5. Modul ve Dosya Detaylari
6. Yapay Zeka ve Istatistik Teknikleri
7. Is Mantigi ve Kopru Mekanizmasi
8. Backtest Motoru ve Risk Yonetimi
9. Performans Metrikleri
10. Sistem Sinir ve Kisitlamalari

---

## 1. Projenin Amaci ve Felsefesi

### 1.1 Temel Problem

Geleneksel algoritmik ticaret sistemleri tek bir hissenin gecmis verisini ezberler (Overfitting). Piyasa yapisal bir degisime ugradigi veya "Testere Piyasasi (Whipsaw)" kosullari olustugunzda bu sistemler hizla bozulur. Ozel olarak banka veya enerji sektorunu ogrenen bir model, diger sektorlerde veya kriz donemlerinde calismaz.

### 1.2 Cozum: Global Beyin + Yerel Risk Mimarisi

Proje iki temel sutuna dayanir:

**Kuresel Bilgelik (Yapay Zeka Katmani):**  
Havayollari, bankacilik, savunma, petrokimya ve cam gibi birbirinden farkli sektordeki 9 hissenin verisi tek bir mega havuzda birlestirilerek LSTM sinir agi egitilir. Model tek bir hissenin fiyat hareketini degil, piyasanin genel mikro-yapisini, momentum gecislerini, volatilite karakterini ve trend davranislarini ogrenir.

**Yerel Bireysellestirme (Evrimsel Optimizasyon Katmani):**  
Kuresel modelden gelen yukseklik tahminleri, her hissenin kendi DNA'sina (volatilite, likidite, trend sertligi) gore ayri ayri optimize edilir. CMA-ES algoritmasi, her hisse icin en yuksek Calmar Orani'ni veren Esik, Stop-Loss ve Take-Profit degerlerini bulur.

### 1.3 Temel Felsefe

> "Hakli olmak ile para kazanmak ayni sey degildir."

Yapay zeka dogruluk orani ne kadar yuksek olursa olsun, siki bir risk yonetimi olmadan surdurulebilir kazanc elde edilemez. Sistem bu gercegi kodun her katmanina isler.

---

## 2. Kullanilan Teknolojiler

### 2.1 Programlama Dili ve Ortam

| Bilesek | Versiyon | Rol |
|---------|----------|-----|
| Python | 3.12 | Ana gelistirme dili |
| venv | — | Izole paket ortami |

### 2.2 Yapay Zeka ve Makine Ogrenmesi

| Kutuphane | Versiyon | Kullanim |
|-----------|----------|---------|
| TensorFlow / Keras | 2.21+ | LSTM sinir agi mimarisi ve egitim |
| scikit-learn | 1.9+ | MinMaxScaler, class_weight hesaplama |
| NumPy | 2.4+ | Matris operasyonlari, dizi islemleri |

### 2.3 Evrimsel Optimizasyon

| Kutuphane | Versiyon | Kullanim |
|-----------|----------|---------|
| cma | 4.4+ | CMA-ES algoritmasinin referans implementasyonu |

### 2.4 Veri ve Veritabani

| Kutuphane | Versiyon | Kullanim |
|-----------|----------|---------|
| SQLite3 | stdlib | Yerel veritabani motoru |
| pandas | 3.0+ | DataFrame islemleri, zaman serisi yonetimi |
| requests | 2.34+ | SARDIS REST API ile HTTP iletisimi |

### 2.5 Gorsellestime ve Arayuz

| Kutuphane | Versiyon | Kullanim |
|-----------|----------|---------|
| Streamlit | 1.58+ | Interaktif web tabanli panel |
| Plotly | 6.8+ | Interaktif grafik ve Equity Curve gorseli |

### 2.6 Dis Veri Kaynagi

**SARDIS BIST API** (`http://89.167.3.33:8005`)  
Borsa Istanbul hisselerine ait 15 dakikalik OHLCV (Acilis, Yuksek, Dusuk, Kapanis, Hacim) mum verilerini saglayan ozel REST API. Veriler artimli (incremental) cekilir; sistem her sorguda sadece son bilinen mumdan sonrasini ister.

---

## 3. Proje Haritasi (Dosya Yapisi)

```
cma_es_finance_core/
|
|-- database/
|   |-- db_manager.py           [KATMAN 1] SQLite yonetimi
|   `-- market_data.db          Tum hisse mumlarini saklayan veritabani
|
|-- core/
|   |-- data_fetcher.py         [KATMAN 1] API ile veri cekme
|   |-- data_pipeline.py        [KATMAN 1] GAP tespiti ve doldurma
|   |-- signal_filters.py       [KATMAN 2] Kalman filtresi
|   |-- feature_engineering.py  [KATMAN 2] Oznitelik ve hedef uretimi
|   |-- ai_prep.py              [KATMAN 2] LSTM veri hazirlama
|   |-- ai_models.py            [KATMAN 3] LSTM mimarisi tanimi
|   `-- backtest_engine.py      [KATMAN 5] Event-driven backtest motoru
|
|-- optimizers/
|   `-- cma_optimizer.py        [KATMAN 4] CMA-ES genetik optimizasyon
|
|-- data/
|   `-- global_lstm_model_15min.keras   Egitilmis global model
|
|-- train_model.py              [CALISTIRICI] Model egitim orkestratoru
|-- main.py                     [CALISTIRICI] Optimizasyon + backtest orkestratoru
|-- dashboard.py                [ARAYUZ] Streamlit gorsel panel
|-- requirements.txt            Proje bagimliliklari
`-- TECHNICAL_DOCS.md           Bu dokuman
```

### 3.1 Katman - Dosya Eslesmesi

| Katman | Sorumluluk | Dosyalar |
|--------|-----------|---------|
| Katman 1 | Veri toplama ve depolama | `db_manager.py`, `data_fetcher.py`, `data_pipeline.py` |
| Katman 2 | Sinyal isleme ve ozellik muhendisligi | `signal_filters.py`, `feature_engineering.py`, `ai_prep.py` |
| Katman 3 | Global yapay zeka modeli | `ai_models.py`, `train_model.py` |
| Katman 4 | Yerel risk optimizasyonu | `cma_optimizer.py` |
| Katman 5 | Gercekci backtest ve sonuc | `backtest_engine.py`, `main.py` |
| Arayuz | Gorsel panel | `dashboard.py` |

---

## 4. Uctan Uca Veri Akisi

```
[SARDIS BIST API]
        |
        | HTTP GET /candles?symbol=X&period=15m
        v
[data_fetcher.py] --> artimli cekme, timestamp takibi
        |
        | INSERT OR IGNORE
        v
[market_data.db]  --> SQLite: candles tablosu (symbol, period, timestamp, OHLCV)
        |
        | SELECT * ORDER BY timestamp ASC
        v
[data_pipeline.py] --> GAP tespiti (tatil/yarim gun), ffill ile doldurma
        |
        | temizlenmis DataFrame
        v
[signal_filters.py] --> Kalman Filtresi: gurultu iptali, trend cikarimi
        |
        | DataFrame + kalman_close sutunu
        v
[feature_engineering.py] --> 5 oznitelik + target etiketi
        |
        | (N, 6) DataFrame
        v
[ai_prep.py] --> MinMax olcekleme, (Samples, 60, 5) sliding window
        |
        |  X_train, y_train, X_test, y_test
        v
[train_model.py] --> Mega Dataset birlestirme, LSTM egitimi
        |
        | global_lstm_model_15min.keras
        v
[main.py / dashboard.py] --> model.predict(X_test) -> predictions (N, 1)
        |
        |  predictions + test_df
        v
[cma_optimizer.py] --> 3 boyutlu CMA-ES, Calmar orani maksimizasyonu
        |
        | opt_thresh, opt_sl, opt_tp
        v
[backtest_engine.py] --> event-driven simulasyon, equity curve
        |
        v
[Sonuc] --> Net Kar, Win Rate, Max Drawdown, Islem Sayisi
```

---

## 5. Modul ve Dosya Detaylari

---

### 5.1 `database/db_manager.py` — SQLite Yonetim Katmani

**Amac:** Veritabani baglantisi, tablo olusturma ve CRUD islemleri icin merkezi modul.

#### Fonksiyonlar

**`get_connection()`**
```
Girdi : Yok
Cikti : sqlite3.Connection nesnesi
```
Her cagirida `database/market_data.db` dosyasina yeni bir baglanti acar. Baglanti yolu `os.path.abspath(__file__)` ile dinamik hesaplanir; proje hangi dizinden calistirilirse calistirilsin dogru dosyayi bulur.

---

**`init_db()`**
```
Girdi : Yok
Cikti : Yok (yan etki: tablo olusturulur)
```
`candles` tablosunu olusturur. Birlesik PRIMARY KEY `(symbol, period, timestamp)` sayesinde ayni muma ait mukerrer kayit fiziksel olarak engellenir. Tablo zaten varsa hata vermez (`CREATE TABLE IF NOT EXISTS`).

Tablo semasi:
```sql
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT,
    period    TEXT,
    timestamp INTEGER,   -- Unix millisaniye
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    REAL,
    PRIMARY KEY (symbol, period, timestamp)
)
```

---

**`get_latest_timestamp(symbol, period)`**
```
Girdi : symbol (str), period (str)
Cikti : int | None  -- en yeni mumun Unix ms timestamp'i
```
`MAX(timestamp)` sorgusu ile o sembole ait en guncel mumun zamanini dondurur. `data_fetcher.py` bu degeri artimli indirme icin baslangic noktasi olarak kullanir.

---

**`save_candles(symbol, period, candles_list)`**
```
Girdi : symbol (str), period (str), candles_list (list[dict])
Cikti : int  -- yeni eklenen satir sayisi
```
`INSERT OR IGNORE` ile coklu satir ekler. PRIMARY KEY cakismasi varsa sessizce atlar, `cursor.rowcount` ile sadece gercekten yeni eklenen satir sayisini dondurur.

---

**`load_data_as_df(symbol, period)`**
```
Girdi : symbol (str), period (str)
Cikti : pandas.DataFrame  -- timestamp index'li, OHLCV sutunlu
```
`ORDER BY timestamp ASC` ile siralanmis veriyi ceker, `pd.to_datetime(unit='ms')` ile timestamp'i gercek tarih-saate donusturur ve `date` sutununu index olarak atar.

---

### 5.2 `core/data_fetcher.py` — SARDIS API Veri Cekici

**Amac:** SARDIS REST API'den 15 dakikalik BIST mum verisi cekerek veritabanina kaydetmek.

#### Fonksiyonlar

**`fetch_candles_from_api(symbol, period, start_ts, limit)`**
```
Girdi : symbol (str), period (str), start_ts (int|None), limit (int)
Cikti : list[dict]  -- API'den gelen ham mum listesi
```
`/candles` endpoint'ine GET istegi gonderir. `start_ts` varsa sadece o tarihten sonrasini ister (artimli cekme). Hata durumunda bos liste doner, exception'i yutar.

---

**`sync_symbol_data(symbol, period)`**
```
Girdi : symbol (str), period (str)
Cikti : Yok (yan etki: DB guncellenir)
```
Artimli senkronizasyon dongusu:
1. `get_latest_timestamp()` ile DB'deki en son mumu bul
2. `start_ts = latest_ts + 1` ile API'ye sadece yeni verileri sor
3. Gelen verileri `save_candles()` ile kaydet
4. API 100.000 mum dondurduyse daha fazlasi olabilir, donguye devam et
5. 100.000'den az geldiyse tur tamamlandi, dur

Her API cagrisinda 0.5 saniye beklenerek sunucu yuk dengesi korunur.

---

### 5.3 `core/data_pipeline.py` — GAP Tespiti ve Doldurma Hatti

**Amac:** Ham veritabani verisindeki zaman bosluglarini (GAP) tespit edip borsa takvimine uygun sekilde doldurmak.

#### Fonksiyon: `load_and_fill_gaps(symbol, timeframe)`

```
Girdi : symbol (str), timeframe (str) -- ornegin '15m'
Cikti : pandas.DataFrame  -- temizlenmis, tam zamanli veri
```

**Islem adimlari:**

1. `load_data_as_df()` ile ham veri cekilir
2. UTC -> Europe/Istanbul saat dilimi donusumu yapilir (`tz_localize + tz_convert`)
3. Veri gun bazinda gruplandirılır (`groupby(df.index.date)`)
4. Her islem gunu icin `pd.date_range()` ile olması gereken tum 15 dakikalik mumlar olusturulur (gunun ilk ve son mumunu baz alarak sadece o gunun icini kapsayan aralik)
5. DataFrame yeni eksiksiz index'e gore yeniden boyutlandirilir (`reindex`)
6. Eksik close fiyatlari onceki gecerli fiyatla doldurulur (`ffill`) — flat candle yaratır
7. open/high/low sutunlari eksik ise close ile esitlenir (doji mum)
8. volume eksikleri 0 ile doldurulur

**Neden ffill?**  
Borsanin kapali oldugu anlarda (hafta sonu, resmi tatil, yarim gun) fiyat "donmus" kabul edilir. Bu donemler icin en gerçekci yaklasim bir onceki fiyati tekrarlamaktir.

---

### 5.4 `core/signal_filters.py` — Kalman Filtresi

**Amac:** Finansal fiyat serisindeki stokastik gurultuyu istatistiksel olarak temizleyerek altta yatan gercek trendi ortaya cikarmak.

#### Fonksiyon: `apply_kalman_filter(prices, process_variance, measurement_variance)`

```
Girdi : prices (pd.Series), Q (float), R (float)
Cikti : pd.Series  -- filtrelenmis fiyat serisi (kalman_close)
```

**Kalman Filtresi Teorisi:**

Kalman Filtresi, bir sistemin "gizli gercek durumunu" gurultulu olcumlerden tahmin eden Bayesci bir algoritmadır. Finans icin:

- **Gizli Durum:** Gerçek fiyat trendinin matematiksel tahmini (`xhat`)
- **Olcum:** Borsa verisinden gelen gurultulu kapanış fiyatı
- **Process Variance (Q):** Trendin ne kadar hizli degisebilecegi (dusuk Q → daha yukumlu trend)
- **Measurement Variance (R):** Borsadaki ani spekulatif sıçramaların boyutu (yuksek R → olcume az guven)

**Her adimda iki asama:**

*Tahmin Asaması (Predict):*
```
x_hat_minus[k] = x_hat[k-1]           -- onceki tahmin ilerletilir
P_minus[k]     = P[k-1] + Q           -- belirsizlik birikir
```

*Guncelleme Asamasi (Update):*
```
K[k]    = P_minus[k] / (P_minus[k] + R)        -- Kalman Kazanci
x_hat[k] = x_hat_minus[k] + K[k] * (olcum - x_hat_minus[k])
P[k]    = (1 - K[k]) * P_minus[k]              -- belirsizlik guncellenir
```

**Kalman Kazanci (K)** 0 ile 1 arasindadir:
- K ≈ 1: Olcume tam guvenir (ani degisim gercek kabul edilir)
- K ≈ 0: Tahmine guvenir (ani degisim gurultu kabul edilir)

**Diger filtrelerden farki:**
- SMA/EMA: Gecmis fiyatların agirlikli ortalamasi; "lag" (gecikme) problemi yaratir
- Kalman: Gecmis + belirsizlik modeli; minimum gecikmeyle trend cikarir, roket navigasyonundan uyarlanan teknik

**Parametreler (mevcut degerler):**
```python
process_variance    = 1e-5   # Trendin degisim hizi (cok yukumlu)
measurement_variance = 0.01² # Fiyattaki beklenen gurultu (%1 standart sapma)
```

---

### 5.5 `core/feature_engineering.py` — Oznitelik ve Hedef Uretimi

**Amac:** Ham ve Kalman-filtrelenmis fiyat verisinden LSTM'in anlayacagi matematiksel oznitelikleri ve ikili siniflandirma hedefini uretmek.

#### Fonksiyon: `create_features_and_target(df, lookahead=15, threshold=0.001)`

```
Girdi : df (DataFrame), lookahead (int), threshold (float)
Cikti : DataFrame -- oznitelikler + target sutunlari ekli
```

**Uretilen 5 Oznitelik (X):**

| Oznitelik | Formul | Anlam |
|-----------|--------|-------|
| `close` | Ham fiyat | Mutlak fiyat seviyesi |
| `kalman_close` | Kalman ciktisi | Gurultusuz trend cizgisi |
| `feature_kalman_diff` | `(close - kalman_close) / kalman_close` | Fiyatin trendden sapma yuzdesi; overbuying/overselling sinyali |
| `feature_return_1m` | `close.pct_change(1)` | Bir onceki muma gore anlik momentum |
| `feature_volatility_15m` | `feature_return_1m.rolling(15).std()` | Son 15 mumdaki fiyat degiskenliginin standart sapmasi |

**Hedef Degisken (Y — Siniflandirma):**

```python
future_close  = close.shift(-lookahead)          # 15 mum sonraki fiyat
future_return = (future_close - close) / close   # Gelecek getiri
target = 1 if future_return >= threshold else 0  # %0.1'den fazla yukselirse AL
```

- `lookahead=15` mum: 15 x 15 dakika = 225 dakika ≈ 4 saat ilerisi hedeflenir
- `threshold=0.001`: %0.1'lik minimum yukseklik hareketi; cok kucuk oynakliklari kirpmak icin

---

### 5.6 `core/ai_prep.py` — LSTM Veri Hazirlayici

**Amac:** Ham oznitelik DataFrame'ini LSTM'in beklediği 3 boyutlu tensor formatina donusturmek.

#### Fonksiyon: `prepare_lstm_data(df, feature_cols, target_col, window_size, train_ratio)`

```
Girdi : df (DataFrame), feature_cols (list), window_size (int)=60, train_ratio (float)=0.8
Cikti : X_train, y_train, X_test, y_test (numpy arrays), scaler, class_weights, test_df
```

**Kritik Tasarim Kararlari:**

*1. Kronolojik Bolunme (%80 Train / %20 Test):*
```python
split_idx = int(len(df) * 0.8)
train_df  = df.iloc[:split_idx]   # Zaman cizgisinin ilk %80'i
test_df   = df.iloc[split_idx:]   # Geri kalan %20 hic gorulmemis
```
Finansal serilerde rastgele bolunme (random split) kullanilmaz — model gelecek verilerini egitim sirasinda "gormus" olur (data leakage). Kronolojik bolunme bunu onler.

*2. MinMax Olcekleme (Yalnizca Train Uzerinde Egitilir):*
```python
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[feature_cols])  # Egitilir
test_scaled  = scaler.transform(test_df[feature_cols])       # Sadece donusturulur
```
Scaler'in test verisiyle temas etmesi "data leakage" yaratir. Eger test verisinin min/max degerleri olceklemeye dahil edilseydi model, test donemindeki fiyat araligini ogrenmus olurdu.

*3. Kayan Pencere (Sliding Window):*
```
X[i] = scaled_data[i-60 : i]   -- Son 60 mumun 5 ozniteligi: (60, 5)
y[i] = target[i]                -- Bu blogun tahmini: 0 veya 1
```
Cikti tensoru boyutu: `(N - 60, 60, 5)` → `(Ornek Sayisi, Zaman Adimi, Oznitelik Sayisi)`

*4. Sinif Agirliklandirmasi (Class Weighting):*
Finansal veride "AL" sinyali (target=1) nadir olusur; veri %45-55 araliginda dengesizdir. Class weight hesaplamasiyla modele daha nadir olan "AL" sinyallerini daha fazla onem vermesi soylenir:
```python
weight_for_1 = (1 / class_1_count) * (len(y_train) / 2.0)
```

---

### 5.7 `core/ai_models.py` — LSTM Mimarisi

**Amac:** TensorFlow/Keras ile iki katmanli LSTM sinir agi modelini tanimlamak.

#### Fonksiyon: `build_lstm_model(input_shape)`

```
Girdi : input_shape (tuple) -- (window_size, n_features) = (60, 5)
Cikti : keras.Sequential model
```

**Model Mimarisi:**

```
Input:   (60, 5)           -- 60 mum, 5 oznitelik

LSTM(64, return_sequences=True)
  -- 64 hafiza birimli LSTM katmani
  -- return_sequences=True: bir sonraki LSTM icin her adimda cikti uretir
  -- Giris, Cikis ve Unutma Kapilari sayesinde uzun/kisa vadeli bagimliliklari ogrenir

Dropout(0.2)
  -- Egitimde rastgele %20 noronu kapatir
  -- Overfitting onleme: model belirli noronlara bagimli kalmaz

LSTM(32, return_sequences=False)
  -- 32 birimli ikinci LSTM katmani
  -- return_sequences=False: 60 adimlik diziyi tek bir vektore ozler

Dropout(0.2)
  -- Yine %20 dropout

Dense(16, activation='relu')
  -- 16 noronlu tam baglantili karar katmani
  -- ReLU: negatif aktivasyonlari sifirlar, gradyan patlamalarini onler

Dense(1, activation='sigmoid')
  -- Tek cikti: 0-1 arasi yukselis olasiligi
  -- sigmoid: binary classification icin standart aktivasyon

Kayip Fonksiyonu: binary_crossentropy
Optimizer       : Adam (adaptive learning rate)
Metrik          : accuracy
```

**LSTM'in Avantaji:**  
Standart ANN her girdiyi bagimsiz isler. LSTM'deki Giris Kapisi (hangi bilgi saklansin), Unutma Kapisi (hangi eski bilgi silinsin) ve Cikis Kapisi (hangi bilgi iletilsin) mekanizmalari sayesinde 60 mumlu zaman serisindeki uzun vadeli kaliplari ogrenir.

---

### 5.8 `train_model.py` — Global Model Egitim Orkestratoru

**Amac:** Farkli sektördeki 9 hissenin verisini birlestirerek "piyasanin genel davranisini" ogrenen global LSTM modelini egitmek.

#### Fonksiyon: `train_global_model(timeframe)`

**Egitim Sembolleri:**
```
ASELS (savunma), GARAN (banka), HALKB (banka), ISCTR (banka),
THYAO (havacilik), TUPRS (enerji/petrokimya), VAKBN (banka),
SASA (petrokimya), SISE (cam/sanayi)
```

**Islem Akisi:**

1. Her sembol icin bagimsiz olarak:
   - Veri yukle → Kalman filtresi uygula → Oznitelik cikar → LSTM icin hazirla
   - Her sembol kendi scaler'iyla olceklenir (fiyat seviyeleri arasindaki fark normallesmis olur)
   
2. Tum sembollerin X_train ve y_train dizileri `np.vstack` ile ust uste yigilir:
   ```
   Mega Train Set: (288.490, 60, 5) -- 9 sektorden birlesik veri
   Mega Test Set : (71.721, 60, 5)
   ```

3. Global sinif agirlikları hesaplanir (tum sektorlerin dengesizligi icin)

4. Egitim parametreleri:
   - `epochs=30`: Maksimum 30 tur (EarlyStopping devreye girerse daha az)
   - `batch_size=512`: Her adimda 512 ornek paralel islenir (GPU olmadan da hizli)
   - `shuffle=True`: Farkli hisselerin verileri karistirilir — modelin sektor sirasini ogrenip ezberlemesi engellenir
   - `class_weight`: Nadir AL sinyallerine daha fazla agirlik
   - `EarlyStopping(patience=3)`: val_loss 3 epoch ardilik iyilesmezse dur, en iyi agirliklara geri don
   - `ModelCheckpoint`: En dusuk val_loss'u veren modeli kaydet

---

### 5.9 `optimizers/cma_optimizer.py` — CMA-ES Genetik Optimizasyon

**Amac:** Gradyan hesaplanamayan, gurultulu ve suereksiz bir arama uzayinda (alim-satim parametreleri) en yuksek Calmar Oranini bulmak.

#### Fonksiyon: `fast_backtest_evaluator(params, predictions, test_df, window_size, commission)` — Fitness Fonksiyonu

```
Girdi : params [threshold, stop_loss, take_profit], predictions (array), test_df (DataFrame)
Cikti : float -- minimize edilecek deger (dusuk = iyi)
```

CMA-ES bir minimizasyon algoritmasi oldugu icin:
- Kotu sonuclar → pozitif buyuk sayı (ornegin 999999.0)
- Iyi sonuclar → negatif sayı (Calmar oraninin negatifi)

**Sinirlar (Evrimsel Mutasyon Korumasi):**
```python
threshold   : [0.45, 0.99]   -- %45 ile %99 arasinda guven esigi
stop_loss   : [0.0075, ∞)    -- minimum %0.75 stop-loss
take_profit : [0.015, ∞)     -- minimum %1.5 kar hedefi
```

**Ceza Sartlari:**
```python
if trade_count < 10: return 999999.0   -- Cok az islem = gecersiz strateji
if equity <= 0:      return 999999.0   -- Iflas = gecersiz strateji
```

**Event-Driven Simulasyon (Mini Backtest):**
Her parametre seti icin tam bir olay dongusu calistirilir:
1. Olasılik esigi gecilirse pozisyon ac
2. Her mumda Take-Profit / Stop-Loss / 15 mum zaman limiti kontrol et
3. Cikis sartlarından biri gerceklestiyse pozisyonu kapat, sermaye guncelle

**Fitness Hesaplama:**
```python
calmar_ratio = net_profit / (max_drawdown + 1.0)
return -calmar_ratio   # Minimize icin negatif
```

`+1.0` ifadesi max_drawdown'in sifir olmasi durumunda sifira bolme hatasini onler.

---

#### Fonksiyon: `run_cma_optimization(predictions, test_df, window_size)` — CMA-ES Ana Dongusu

```
Girdi : predictions (numpy array), test_df (DataFrame), window_size (int)
Cikti : (opt_thresh, opt_sl, opt_tp) -- en iyi parametre uclusu
```

**CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Teorisi:**

CMA-ES biyolojik evrimden ilham alan bir kara-kutu optimizasyon algoritmasidir:

- **Populasyon:** Her nesilde 20 aday parametre seti (20 "birey")
- **Seçilim:** En iyi Calmar skorunu veren bireylerin ortalaması ve ortak varyansı hesaplanır
- **Kovaryans Matrisi:** Arama elipsi bu kârli bolgeye dogru buyutur ve yonlenir. Boyutlar arasi korelasyon ogrenilir (ornegin "yuksek SL ile yuksek TP genellikle birlikte iyi calisir")
- **Sigma (σ):** Arama adim buyuklugu; iyi bolge bulununca kucutur, genis arama gerektiginde buyutur

**Baslangiç Parametreleri:**
```python
initial_params = [0.505, 0.015, 0.15]  # Esik:%50.5, SL:%1.5, TP:%15
sigma0         = 0.02                  # Baslangic arama yariçapi
popsize        = 20                    # Nesil basi birey sayisi
maxiter        = 30                    # Maksimum nesil sayisi
```

**30 Nesil Sonunda:** CMA-ES o hissenin kârli bölgesine "yakinlasmis" ve en yuksek Calmar Oranlı parametreyi bulmuştur.

---

### 5.10 `core/backtest_engine.py` — Event-Driven Backtest Motoru

**Amac:** CMA-ES tarafindan bulunan optimal parametrelerle gercekci bir ticaret simulasyonu yaparak performansi olcmek.

#### Fonksiyon: `run_backtest(predictions, test_df, threshold, stop_loss, take_profit, window_size)`

```
Girdi : predictions (array), test_df (DataFrame), threshold, stop_loss, take_profit, window_size
Cikti : DataFrame -- signal, strategy_return, equity sutunlari ekli
```

**Event-Driven (Olay-Dongusu) Yaklasimi:**

Geleneksel "vektörel" backtest tum sinyal barlarini esasmanda paralel olarak degerlendirip ayni anda birden fazla pozisyon acar — gercekte imkânsiz bir sermaye kaynagi varsayar.

Bu motorun calisma mantigi:
```
Her mum icin:
  EGER pozisyonda degil:
    EGER olasılık > esik:
      Pozisyon ac, giris fiyatini kaydet, bar sayacini sifirla
  EGER pozisyondaysa:
    bar sayacini artir
    guncel getirir hesapla
    EGER getiri >= take_profit VEYA getiri <= -stop_loss VEYA bar >= 15:
      Pozisyonu kapat
      Net getiri = guncel_getiri - komisyon (binde 2)
      actual_position = min(10000, max(0, equity))  -- Borclanma engeli
      equity += actual_position * net_getiri
  equity_curve[i] = equity
```

**Tasarim Kararlari:**

| Kural | Gercek Etki |
|-------|------------|
| Ayni anda tek pozisyon | Cakisan islem illüzyonu yok |
| Sabit 10.000 TL pozisyon | Logaritmik bilesik buyume yok |
| `min(10000, max(0, equity))` | Borclanarak islem engeli |
| Komisyon: binde 2 (her islemde) | Gercek maliyet dahil |
| 15 mum zaman limiti | AI'nin vizyon siniri (4 saat) |

---

### 5.11 `main.py` — Ana Orkestrator

**Amac:** Tum katmanlari siraliyla tetikleyen, 10 hisse icin optimizasyon + backtest dongusu calıştıran ana program.

**Akis:**
```
Model yukle
Her sembol icin:
  1. Veri yukle + temizle (load_and_fill_gaps)
  2. Kalman filtresi uygula
  3. Oznitelik + hedef uret
  4. LSTM formatina donustur
  5. Tahmin yap (predictions = model.predict)  -- tek seferlik
  6. CMA-ES optimizasyonu (predictions kullanilir)
  7. Backtest (ayni predictions kullanilir)     -- ikinci kez predict YOK
  8. Metrikleri hesapla
  9. Sonuclari listele
Sonuclari terminale tabloyla yazdir
```

---

### 5.12 `dashboard.py` — Streamlit Gorsel Panel

**Amac:** Kullanicinin sembol secip risk parametrelerini manuel olarak test edebildigi interaktif web paneli.

**Bilesenler:**
- Yan panel (Sidebar): Sembol secimi, Threshold/SL/TP kaydiracilar
- Ana alan: 4 KPI karti (Net Kar, Islem Sayisi, Win Rate, Max Drawdown), Equity Curve grafigi

**Akis:** Simülasyon Baslat dugmesine basilinca:
1. Model bellege yukle
2. Secili sembolun verisi islenir
3. Manuel parametrelerle backtest calistirilir
4. Sonuclar anlik gosterilir

---

## 6. Yapay Zeka ve Istatistik Teknikleri

### 6.1 Kalman Filtresi (Istatistiksel Sinyal Isleme)

**Alan:** Kontrol teorisi, roket navigasyonu  
**Kullanim:** Gurultulu fiyat serisinden saf trend cikarimi

Klasik indikatörlerin (SMA, EMA) "lag" probleminin aksine Kalman Filtresi minimum gecikmeyle calisir cunku gecmise degil sisteme ait bir modele dayanir.

### 6.2 LSTM (Long Short-Term Memory)

**Alan:** Derin ogrenme, zaman serisi analizi  
**Kullanim:** 60 mumlik gecmise bakarak gelecek 4 saatin yukselis olasiligini hesaplama

LSTM'in standart RNN'e ustunlugu: "Gradyan Kayboluyor (Vanishing Gradient)" problemini Giris/Unutma/Cikis kapilarıyla cozer. Uzun vadeli bagimliliklari (200+ adim geride) hatirlayabilir.

### 6.3 MinMax Olcekleme

**Alan:** Makine ogrenmesi veri on isleme  
**Kullanim:** Farkli fiyat seviyelerindeki hisseleri (10 TL vs 500 TL) ayni [0,1] araligina getirme

Olmadan LSTM baskın olan buyuk degerli hissenin desenine kilitlenirdi.

### 6.4 Kayan Pencere (Sliding Window)

**Alan:** Zaman serisi tahmin  
**Kullanim:** Her tahmin noktası icin 60 mumlik gecmis baglamın sakinmasi

LSTM'e "son 60 muma bak, 15 mum sonrasini tahmin et" semantigini verir.

### 6.5 Sinif Agirlandirmasi (Class Weighting)

**Alan:** Dengesiz siniflandirma  
**Kullanim:** "AL" sinyalinin "BEKLE"ye oranla daha nadir olmasi durumunda modelin coglunluk sinifina yonelmesini engellemek

### 6.6 EarlyStopping + ModelCheckpoint

**Alan:** Derin ogrenme egitim teknigi  
**Kullanim:** val_loss 3 ardisik epoch iyilesmezse egitimi durdur, o ana kadarki en iyi modeli kaydet

Overfitting'i cerrahi bir mudahaleyle keser.

### 6.7 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**Alan:** Evrimsel hesaplama, kara-kutu optimizasyon  
**Kullanim:** Turev alinamaz, gurultulu, cok-modalli parametre uzayinda global optimum arama

Gradient Descent calismaz cunku alim-satim parametrelerinin harita yuzeyı surer surer degildir (non-differentiable). CMA-ES turev gerektirmez.

### 6.8 Calmar Orani (Fitness Kriteri)

**Alan:** Finansal performans metrigi  
**Formulü:** `Net Getiri / Maximum Drawdown`

CMA-ES'in optimize ettigi kriter. Sadece kari degil, o kari kac birim risk alarak kazandigini olcer. Cok kazanip cok kaybeden bir sistem yerine az kazanip cok az kaybeden bir sistem tercih edilir.

---

## 7. Is Mantigi ve Kopru Mekanizmasi

### 7.1 LSTM ile CMA-ES'in El Sikismasi

Sistemin en ozgun yeri iki algoritmánin birbirine baglandigi nokta:

**LSTM'in Ciktisi:**
```
pred_prob = [0.45, 0.52, 0.81, 0.33, 0.65, ...]
```
Her mum icin 0 ile 1 arasinda bir "yukselme olasiligi" vektoru. LSTM esigi bilmez — sadece not verir.

**CMA-ES'in Sorusu:**
```python
signal = np.where(pred_prob > threshold, 1, 0)
```
"Bu notlardan kac puan alana 'AL' demeliyim ki en yuksek Calmar Oranini elde edeyim?"

CMA-ES her nesilde farkli bir esik dener:
- Esik=0.70: Sadece cok emin olunan mumlar seçilir → az islem, sek islemler
- Esik=0.50: Daha fazla islem → daha fazla komisyon, daha fazla maruz kalinan risk

Algoritma 30 nesil sonunda her hissenin "altin gecme notunu" (optimal esigini) bulur.

### 7.2 Asimetrik Risk Yonetimi

CMA-ES dogrudan TP/SL oranlarini da optimize eder. Bu asimetrik bir risk profili yaratir:

```
Kotu senaryo : -SL kaybedilir (ornegin %2)
Iyi senaryo  : +TP kazanilir (ornegin %15)

Beklenti degeri = (Win_Rate * TP) - ((1 - Win_Rate) * SL)
```

%40 kazanma orani bile surdurulebilir olabilir: `0.40 * 15 - 0.60 * 2 = 6 - 1.2 = +4.8%`

### 7.3 "Out-of-Universe" (OOU) Testi ve Genelleme

FROTO hissesi egitim setinde bulunmaz. Sistem FROTO'yu hic gormedigi halde:
1. Global model FROTO'nun mumlarini diger hisselerle ayni matematiksel strukture sahip zaman serisi olarak isler
2. CMA-ES FROTO'nun ozel volatilite karakterine gore SL/TP bulur
3. Basarili sonuc: Modelin ezberlemediginin, piyasa davranisini genellettiginin kamiti

---

## 8. Backtest Motoru ve Risk Yonetimi

### 8.1 Sabit Kasa Modeli (Fixed Position Sizing)

Her islem `10.000 TL` sabit buyuklukte acilir. Kasa buyuse de kucilse de bu deger degismez.

**Neden?**  
Logaritmik (bilesik buyume) modelde her islem mevcut kasanin tamami ile acilir — bu "simülasyon illüzyonu" yaratir. Gercekte her islem icin ayni sermayeyi riske atarak daha gercekci bir performans resmi ortaya cikar.

### 8.2 Borclanma Engellemesi

```python
actual_position = min(position_size, max(0, equity))
```
Kasa sifirin altina duserse `max(0, equity)=0` olur ve yeni islem acilamaz. Sonsuz borclanma illüzyonu ortadan kalkar.

### 8.3 Zaman Limiti Cikisi

LSTM 15 mum (225 dakika) ileriye tahmin eder. Bu sure dolunca ne TP ne de SL vurulmamis olsa bile pozisyon kapanir. Modelin "vizyon sınırı" bir trade yonetim kurali olarak uygulanir.

---

## 9. Performans Metrikleri

| Metrik | Formul | Anlam |
|--------|--------|-------|
| Net Kar (TL) | `equity_son - 10.000` | Toplam kazanc/kayip |
| Win Rate (%) | `kazanclı_cikislar / toplam_cikislar * 100` | Islemlerin yüzdesi karda kapandi |
| Max Drawdown (%) | `max((equity - cummax) / cummax) * 100` | En kotu tepe-dip dususu; risk olcusu |
| Calmar Orani | `Net Getiri / Max Drawdown` | Birim risk basina kazanc; optimizasyon kriteri |
| Islem Sayisi | `len(exit_bars)` | Acilip kapanan toplam pozisyon |

**Tutarlilik:** Win rate ve islem sayisi her yerde cikis barlari baz alinarak hesaplanir (`strategy_return != 0`).

---

## 10. Deneysel Sonuclar ve Ablasyon Calismalari

Bu bolum, sistemin her bileseninin katkisini nicel olarak olcen kontrollu
deneyleri icerir. Tum deneyler `analysis/` dizinindeki scriptlerle uretilmis
ve tekrarlanabilirlik icin CMA-ES seed'i 42'ye sabitlenmistir.

### 10.1 Ongoru Kalitesi (LSTM Tahmin Metrikleri)

Global model 10 hissenin test setinde (gorulmemis %20) degerlendirildi.
Esik 0.50 (sigmoid dogal siniri) ile siniflandirma metrikleri:

| Sembol | Accuracy | Precision | Recall | F1 |
|--------|---------:|----------:|-------:|-----:|
| ASELS | %51.20 | %48.99 | %47.01 | %47.98 |
| GARAN | %55.75 | %52.24 | %7.49 | %13.10 |
| HALKB | %54.62 | %49.89 | %14.07 | %21.95 |
| ISCTR | %50.66 | %44.43 | %49.37 | %46.77 |
| THYAO | %59.95 | %52.19 | %13.42 | %21.35 |
| TUPRS | %51.36 | %53.93 | %2.99 | %5.67 |
| VAKBN | %53.30 | %51.29 | %24.77 | %33.41 |
| SASA | %58.25 | %43.36 | %20.49 | %27.83 |
| SISE | %53.03 | %47.69 | %42.18 | %44.76 |
| FROTO | %53.99 | %47.40 | %28.63 | %35.70 |
| **Ortalama** | **%54.21** | — | — | **%29.85** |

**Yorum:** Model ortalama %54 dogrulukla calisir — finansal piyasalarin
yari-etkinligi (semi-efficient market) goz onune alindiginda rastgeleden
(%50) anlamli sekilde iyidir, ancak mucize bir tahminci degildir. Bu
**bilincli ve dürüst** bir sonuctur: sistemin tezi "kusursuz tahmin" degil,
"zayif ama tutarli bir sinyali siki risk yonetimiyle kara cevirmektir".

### 10.2 CMA-ES Ablasyonu — Risk Optimizasyonunun Katkisi

Ayni LSTM tahminleriyle iki senaryo karsilastirildi:
- **Sabit parametre (optimize edilmemis):** Esik %50, SL %2, TP %4 (insan-secimi makul degerler)
- **CMA-ES optimize:** Her hisse icin evrimle bulunan parametreler

| Olcut | Sabit Parametre | CMA-ES Optimize |
|-------|----------------:|----------------:|
| Toplam Net Kar | **-9.769 TL** (zarar) | **+17.327 TL** |
| Ortalama Calmar | 16.29 | 214.26 |
| Karli hisse sayisi | 4 / 10 | 9 / 10 |

**Sonuc:** Optimize edilmemis "makul" parametrelerle sistem **para kaybeder**
(10 hissenin 6'si zararda). CMA-ES bunu kara cevirir. Bu, projenin merkez
iddiasinin — hibrit risk optimizasyon katmaninin degeri — **cürütülemez
ampirik kanitidir.** Bu sonuc, modelden bagimsiz olarak (hem Kalman'li hem
Kalman'siz modelde) tekrarlanmis ve dogrulanmistir.

### 10.3 Kalman Ablasyonu — Cok-Seed Calisma

"Istatistiksel isaret isleme (Kalman) katmani gercekten katki sagliyor mu?"
sorusunu kesin yanitlamak icin, Kalman'li (5 ozellik) ve Kalman'siz (3 ozellik)
modeller **3 farkli egitim seed'i** (42, 7, 123) ile egitilip ortalandi.

| Konfigurasyon | F1 (ort±std) | Accuracy | Toplam Kar (ort±std) | Calmar (ort±std) |
|---------------|-------------:|---------:|---------------------:|-----------------:|
| Kalman'li (5 oz.) | %24.4 ± 5.1 | %54.4 | 20.994 ± **2.815** TL | 310 ± **96** |
| Kalman'siz (3 oz.) | %25.5 ± 5.5 | %53.8 | 26.564 ± **7.941** TL | 534 ± **185** |

**Iki kademeli sonuc:**

1. **Tahmin dogrulugu — istatistiksel olarak ayirt edilemez.** F1 farki
   (1.1 puan) standart sapmanin (~5 puan) cok icindedir. Kalman, ham tahmin
   gucunu degistirmez — cunku LSTM zaten zamansal gürültü temizlemeyi içsel
   olarak ogrenir; disaridan Kalman ozelligi bilgiyi tekrarlar.

2. **Kalman bir STABİLİZATÖR (varyans azaltici) gorevi gorur.** Kalman'siz
   modelin kar standart sapmasi (7.941) Kalman'liya gore ~3 kat yuksektir.
   Yani Kalman'siz model yuksek ortalamayi **dengesizlik pahasina** alir
   (seed'e gore 17k-32k arasi savrulur). Kalman'li model cok daha **tutarli
   ve tekrarlanabilir** sonuc verir.

**Tezsel cikarim:** Kalman filtresi dogruluk icin degil, **kararlilik ve
robustluk** icin kullanilir. Bu, klasik istatistiksel isaret islemenin derin
ogrenme ile birlestiriminde hangi rolu üstlendigini netlestiren özgün bir
katkidir: bir regularizör/stabilizatör.

### 10.4 Genelleme Testi (Out-of-Universe)

FROTO hissesi egitim setinde **hic bulunmaz** (9 hisse ile egitim, FROTO
sadece test). Buna ragmen sistem FROTO'da CMA-ES optimizasyonu sonrasi
**+1.141 ila +6.491 TL** (seed'e gore) kar uretmistir. Bu, modelin tek tek
hisseleri ezberlemedigini, piyasanin **genel mikro-yapisini** ogrendigini
kanitlar.

### 10.5 Deney Scriptleri (Tekrarlanabilirlik)

| Script | Uretttigi Sonuc |
|--------|-----------------|
| `analysis/evaluate_model.py` | Ongoru metrikleri + CMA-ES ablasyonu (`--no-kalman` destekli) |
| `analysis/quick_metrics.py` | Hizli F1/Accuracy (CMA'siz) |
| `analysis/train_ablation.py` | Kalman'siz (3 ozellik) model egitimi |
| `analysis/multiseed_ablation.py` | Cok-seed Kalman ablasyon calismasi |

---

## 11. Sistem Sinirlari ve Kisitlamalar

| Konu | Sinir | Aciklama |
|------|-------|----------|
| Veri frekansi | 15 dakikalık | API'nin destekledigi en yuksek frekans |
| Egitim donemı | 2021-2026 | Veritabanindaki mevcut veri araligi |
| GPU destegi | Yok (Windows native) | TF >= 2.11 Windows GPU desteklemiyor; WSL2 veya DirectML gerekir |
| Coklu pozisyon | Var | `core/portfolio_backtest.py`: paylasimli sermaye, es zamanli N pozisyon |
| Short (Aciga Satis) | Var | `run_backtest(allow_short=True)`: prob < (1-esik) ise asaga bahis |
| Cuzdan modu | Sabit / Dinamik | `mode='fixed'` (baseline) veya `mode='compound'` (bilesik buyume) |
| API erişimi | SARDIS ozel API | Kamu erisimi yok; alternatif veri kaynagi gerekebilir |
| Egitim olasılık dogrulugu | ~%54 | Piyasa kaosunda beklenen deger; TP/SL asimetrisi ile telafi edilir |

---

## Hizli Referans: Fonksiyon Indeksi

| Fonksiyon | Dosya | Girdi | Cikti |
|-----------|-------|-------|-------|
| `init_db()` | db_manager.py | — | DB hazir |
| `save_candles()` | db_manager.py | symbol, period, list | int (kaydedilen) |
| `load_data_as_df()` | db_manager.py | symbol, period | DataFrame |
| `sync_symbol_data()` | data_fetcher.py | symbol, period | — (DB guncellenir) |
| `load_and_fill_gaps()` | data_pipeline.py | symbol, timeframe | DataFrame (temiz) |
| `apply_kalman_filter()` | signal_filters.py | pd.Series | pd.Series (filtreli) |
| `create_features_and_target()` | feature_engineering.py | DataFrame | DataFrame (oznitelikli) |
| `prepare_lstm_data()` | ai_prep.py | DataFrame | X_train, X_test, scaler, ... |
| `build_lstm_model()` | ai_models.py | input_shape | keras.Model |
| `train_global_model()` | train_model.py | timeframe | — (model kaydedilir) |
| `fast_backtest_evaluator()` | cma_optimizer.py | params, preds, df | float (fitness) |
| `run_cma_optimization()` | cma_optimizer.py | preds, df | (thresh, sl, tp) |
| `run_backtest()` | backtest_engine.py | preds, df, params | DataFrame (equity_curve) |
