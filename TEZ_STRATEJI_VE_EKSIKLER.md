# Tez Strateji ve Eksiklik Yol Haritasi

**Durum:** Tez Onerisi (Mart 2026) ve Ara Rapor (Mayis 2026) JURIYE GONDERILDI
ve artik DEGISTIRILEMEZ. Bu belge, gonderilmis belgelerle projenin bugunku
hali arasindaki farklarin, **final tez** ve **sozlu savunma** asamasinda nasil
yonetilecegine dair stratejiyi icerir.

---

## 0. Temel Strateji Ilkesi

> **Ara rapor bir ilerleme anidir (snapshot); final tez nihai otoritedir.**

Bir bitirme projesinde ara rapor ile final tez arasinda fark olmasi **normaldir
ve beklenir** — ara raporun amaci zaten "devam eden calisma"yi gostermektir.
Dolayisiyla cogu "tutarsizlik" aslinda **sorun degil, ILERLEME kanitidir.**
Anahtar: bu farklari final tezde dogru cercevelemek ve savunmada
"projeyi gelistirdik" diye sahiplenmek (savunmaya gecmek degil).

Her bosluk icin 4 stratejiden biri uygulanir:

| Strateji | Anlami | Ne zaman |
|----------|--------|----------|
| **KAPAT** | Dusuk eforla gercekten uret/ekle | Elimizde zaten varsa |
| **YENIDEN CERCEVELE** | "Gelecek Calismalar"a tasi + gerekcelendir | Buyuk, yapilmayacak kapsam |
| **DUZELT + ANLAT** | Final tezde guncel hali yaz, evrimi olumlu anlat | Belge-kod celiskisi |
| **SAVUN** | Juriye hazir sozlu cevap | Soru gelmesi muhtemel |

---

## 1. KAPAT — Dusuk Eforla Gercekten Uretilecekler

Bunlar zaten projede mevcut; sadece final teze tasimak gerekiyor. **En yuksek
getiri/efor orani burada.**

| Madde | Durum | Aksiyon |
|-------|-------|---------|
| Tablo 3.1 (backtest) bayat | Yeni model + adil baseline hazir | Yeni rakamlarla tabloyu yeniden uret |
| Sekiller Listesi bos | Dashboard + analysis grafikleri var | Candlestick, equity, drawdown, confusion matrix, ablasyon grafiklerini ekle |
| Ongoru metrikleri yok | P/R/F1/Confusion hesaplandi | 3. Bolume "Ongoru Kalitesi" tablosu ekle |
| Ablasyon yok | CMA-ES + Kalman cok-seed yapildi | Yeni alt bolum: "Ablasyon Calismalari" |
| OOU / coklu sembol yok | FROTO testi + 9 sektor mega dataset var | Metodoloji bolumune ekle |

---

## 2. YENIDEN CERCEVELE — "Gelecek Calismalar" (yapilmayacak buyuk kapsam)

Tez Onerisi'nde vaat edilen ama final surede yapilmayacak buyuk bilesenler.
Bunlari SAKLAMAK yerine ACIKCA gelecek calismalara tasimak ve gerekcelendirmek
gerekir. Gerekce her zaman ayni: **"Genislik yerine derinlik"** — cekirdek
hibrit yontemi (Kalman+LSTM+CMA-ES) saglam kurmaya odaklandik.

| Onerideki vaat | Final tezdeki cerceve |
|----------------|----------------------|
| Temel analiz (bilanco, gelir tablosu), cok boyutluluk | "Cekirdek teknik+istatistiksel hibrit dogrulandiktan sonra, temel analiz katmani gelecek calismaya birakildi." |
| Gercek zamanli WebSocket akisi | "Sistem backtest ve metodoloji dogrulamasina odaklandi; canli akis entegrasyonu mimaride hazir (modüler), gelecek is." |
| Kripto piyasalari | "Kapsam, daha kontrollu ve duzenlenmis bir pazar olan BIST'e odaklandi." |
| Web scraping (Selenium) | "Veri kaynagi olarak kararli bir API (SARDIS) tercih edildi." |
| Duygu analizi (haber) | Ara raporda zaten gelecek calisma olarak gecti — tutarli devam ettir. |

**Onemli:** Final tezin "Giris" bolumunde, ara rapordaki gibi kapsami
NETLESTIR ve daralmayi bir KARAR olarak sun (eksiklik olarak degil).

### 2.1 KARAR: Temel Analiz -> Yol A (Gelecek Calisma) [KILITLENDI]

Temel analiz, projeye EKLENMEYECEK; tezde gelecek calisma olarak cercevelenecek.
Gerekce sadece "zaman yetmedi" degil, **teknik olarak dogru olan** sudur:

1. **Zaman olcegi uyusmazligi:** Temel veriler (bilanco, F/K, EPS) ceyreklik
   guncellenir (yilda 4 kez). Sistem 15-dk barlarla 4 saat sonrasini tahmin eder.
   Bir ceyrek boyunca binlerce intraday bar icin temel oran SABIT kalir -> kisa
   vadeli harekete neredeyse sifir sinyal verir.
2. **Look-ahead bias riski:** Bugunku oranlari 2021-2026 backtest'ine uygulamak,
   modele sirketin gelecegini sizdirir (metodolojik hata). Dogru yontem
   "point-in-time" ceyreklik veridir; BIST icin erisimi zordur.
3. **Dogru entegrasyon ayri mimari ister:** Temel + teknik birlestirme, ikisini
   tek modele besleyerek degil, IKI SEVIYELI karar ile yapilir (temel = hisse
   secimi/filtre [yavas], teknik = zamanlama [hizli]). Bu, ayri bir cok-olcekli
   mimari konusudur.

**Teze hazir paragraf (Gelecek Calismalar):**
> "Bu calismada cekirdek hibrit yontem (Kalman + LSTM + CMA-ES) intraday teknik
> veriler uzerinde dogrulanmistir. Tez onerisinde ongorulen temel analiz
> entegrasyonu, temel verilerin ceyreklik zaman olceginin sistemin 4 saatlik
> intraday tahmin ufkuyla dogrudan birlesememesi nedeniyle ayri bir calisma
> olarak birakilmistir. Onerilen dogru yaklasim, temel ve teknik analizi tek bir
> modele beslemek yerine iki ayri karar seviyesinde kullanmaktir: temel saglamlik
> skoru ile hisse evreninin filtrelenmesi (yavas, secim) ve teknik model ile
> giris-cikis zamanlamasi (hizli). Bu cok-olcekli mimari, point-in-time temel
> veri kaynagiyla birlikte gelecek calismaya birakilmistir."

---

## 3. DUZELT + ILERLEME OLARAK ANLAT — Belge/Kod Celiskileri

Bunlar gonderilmis belgelerde yanlis/eski; final tezde DOGRU hali yazilacak.
Jüri sorarsa cevap: **"Ara rapordan bu yana sistemi su sekilde gelistirdik..."**
(Bu bir ilerleme anlatisidir, savunma degil.)

### 3.1 "Vektorel" -> "Olay-Gudumlu (Event-Driven)" backtest
- Ara rapor Ozet: "vektorel backtest motoruna entegre edilmistir."
- Gercek: olay-gudumlu (stateful) motor. Eski vektorel surum hatal iydi (ayni
  anda sinirsiz pozisyon varsayimi -> gercekci degildi).
- **Final tez:** "olay-gudumlu" yaz. Evrim anlatisi: "Vektorel yaklasimin
  gercekci olmayan es-zamanli pozisyon varsayimi nedeniyle, tek-pozisyonlu,
  komisyon ve zaman limiti iceren olay-gudumlu motora gecildi." -> Bu OLUMLU.

### 3.2 Tablo 2.1 (CMA-ES baslangic degerleri) guncelle
- Eski tablo: Esik %70, SL %1, TP %2 baslangic; min %0.1.
- Gercek kod: Esik %50.5, SL %1.5, TP %15 baslangic; SL min %0.75, TP min %1.5.
- **Final tez:** kodla birebir uyumlu tablo.

### 3.3 RMSE -> Precision/Recall/F1
- Oneri RMSE diyor; ama model IKILI SINIFLANDIRMA yapiyor -> RMSE (regresyon
  metrigi) yanlis.
- **Final tez:** RMSE dilini cikar; siniflandirma metrikleri (Accuracy, Precision,
  Recall, F1, Confusion Matrix) kullan. Ara rapor zaten dogru yonde (win rate/DD).

### 3.4 Kalman'in katkisi: "kanitlandi" -> nuanced (stabilizator)
- Ara rapor 3.2: Kalman'in faydasi "kanitlanmistir" (fazla iddiali).
- Cok-seed ablasyon: Kalman dogrulugu artirmiyor (F1 %24.4 vs %25.5, berabere)
  ama kar varyansini ~3x dusuruyor (stabilizator/regularizor).
- **Final tez:** Durust + nuanced anlat. Bu DAHA GUCLU bir akademik bulgu:
  "klasik isaret islemenin derin ogrenme ile birlesiminde rolu netlestirildi."

### 3.5 Tablo 3.1 (THYAO sonuclari) yeniden uret
- Eski: once -9.999 TL (-%100 DD, 5276 islem), sonra +261.96 TL.
- Sorun: eski model + eski parametre + bozuk baseline (%55 esik -> asiri islem
  -> iflas). Yeni model + adil baseline (%50) ile rakamlar farkli.
- **Final tez:** guncel modelle yeniden uret. -%100 yerine adil baseline kullan.

---

## 4. SAVUN — Juri Soru-Cevap Hazirligi (en olasi 5 soru)

**S1: "Onerideki cok boyutlu temel analiz nerede?"**
C: "Bilincli bir karardir. Temel veriler ceyreklik olcekte guncellenir; sistemimiz
ise 15 dakikalik barlarla 4 saatlik tahmin yapar. Bir ceyrek boyunca sabit kalan
bir temel oran, bu intraday harekete anlamli sinyal vermez; ayrica gecmis
backtest'e bugunku oranlari uygulamak look-ahead bias yaratir. Dogru yaklasim
ikisini tek modele beslemek degil, iki seviyede kullanmaktir: temel analiz hisse
secimi (yavas), teknik model zamanlama (hizli). Biz cekirdek hibrit yontemi 9
sektor + OOU testiyle dogruladik; cok-olcekli temel entegrasyon gelecek calisma."

**S2: "Modelin dogrulugu sadece %54, bu dusuk degil mi?"**
C: "Finansal piyasalar yari-etkin; %54 rastgeleden anlamli iyi. Ama tezin
tezi 'kusursuz tahmin' degil: zayif ama tutarli bir sinyali, CMA-ES risk
optimizasyonuyla kara cevirmek. Ablasyon bunu kanitliyor: optimize edilmemis
parametre -9.769 TL kaybederken, CMA-ES +17.327 TL kazandiriyor."

**S3: "Kalman gercekten gerekli mi?"**
C: "Cok-seed ablasyon yaptik. Kalman dogrulugu artirmiyor ama sonuc varyansini
~3x dusuruyor — yani bir stabilizator/regularizor. Bu, klasik isaret islemenin
derin ogrenme ile birlesimindeki rolunu net gosteren ozgun bir bulgu."

**S4: "Ara raporda vektorel yaziyordu, simdi event-driven?"**
C: "Evet, ara rapordan sonra gelistirdik. Vektorel surum es-zamanli sinirsiz
pozisyon varsayiyordu, gercekci degildi. Komisyon, tek pozisyon ve zaman limiti
iceren olay-gudumlu motora gectik."

**S5: "Asiri uyum (overfitting) yok mu?"**
C: "FROTO hissesini egitimde HIC kullanmadik (Out-of-Universe testi). Model
FROTO'da da kar uretti -> ezberlemedigini, piyasa mikro-yapisini ogrendigini
gosterir. Ayrica kronolojik train/test bolme + EarlyStopping + Dropout."

---

## 5. Onceliklendirilmis Yapilacaklar (Final Tez)

**Yuksek oncelik (kapatma + duzeltme):**
1. Tablo 3.1'i guncel model + adil baseline ile yeniden uret
2. Tablo 2.1'i gercek CMA-ES degerleriyle guncelle
3. "Vektorel" -> "olay-gudumlu" + RMSE -> F1 dil duzeltmeleri
4. Sekiller Listesi'ni doldur (mevcut grafikler)

**Orta oncelik (guclendirme):**
5. Yeni bolum: Ablasyon Calismalari (CMA-ES + Kalman cok-seed)
6. Ongoru metrikleri tablosu (P/R/F1/Confusion)
7. Metodoloji: coklu sembol mega dataset + OOU testi
8. Kalman "stabilizator" bulgusunu 3.2'ye isle

**Dusuk oncelik (cerceveleme):**
9. Giris/Sonuc'ta kapsam daralmasini gerekcelendir (genislik->derinlik)
10. Gelecek Calismalar: temel analiz, real-time, kripto, duygu analizi

---

## 6. Ozet Felsefe

Gonderilmis belgeleri DUZELTEMEYIZ — ama gerek de yok. Ara rapor bir kontrol
noktasidir. Final tez ve savunmada:
- Gercekten yapilanlari (ablasyon, OOU, portfoy) ONE CIKAR (bunlar belgede yoktu,
  artida).
- Belge-kod celiskilerini "ilerleme" olarak ANLAT (vektorel->event-driven gibi).
- Yapilmayan buyuk kapsami "bilincli karar + gelecek calisma" olarak CERCEVELE.
- Juri sorularina hazir, durust, sayisal kanitli cevaplar ver.

Bu yaklasimla, belgelerdeki "eksikler" bir zayiflik degil, projenin olgunlasma
hikayesi haline gelir.
