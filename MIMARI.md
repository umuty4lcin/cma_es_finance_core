# Mimari ve Ilerleme Dokumani

Bu belge, projenin React + FastAPI gecisinden sonraki mimarisini ve simdiye
kadar tamamlanan adimlari aciklar. (Eski Streamlit tek-sayfa panel kaldirildi.)

---

## 1. Genel Bakis — Uc Katmanli Mimari

Sistem artik birbirinden ayri uc katmandan olusur:

```
┌─────────────────────┐     HTTP/JSON      ┌──────────────────────┐     Python      ┌────────────────────────┐
│   FRONTEND (React)  │  ───────────────>  │   BACKEND (FastAPI)  │  ────────────>  │   MOTOR (core/, opt/)  │
│   frontend/         │   /api proxy       │   api/               │   import        │   degismedi            │
│   :5173             │  <───────────────  │   :8000              │  <────────────  │   LSTM, CMA-ES, ...    │
└─────────────────────┘                    └──────────────────────┘                 └────────────────────────┘
   Grafikler, UI                              REST API, sarmalama                      Hesaplama, AI
```

**Temel ilke:** Hesaplama motoru (`core/`, `optimizers/`) HIC degismedi. FastAPI
bu motoru REST API olarak sarmalar; React de bu API'yi tuketip gorsellestirir.
Boylece is mantigi (Python) ile sunum (React) tamamen ayrildi.

---

## 2. Katman 1 — Frontend (`frontend/`)

**Teknoloji:** Vite + React 19 + TypeScript

| Dosya | Gorev |
|-------|-------|
| `src/App.tsx` | Ana sayfa; durum yonetimi, panellerin yerlesimi |
| `src/api.ts` | Backend cagrilari (`fetchSymbols`, `analyze`) |
| `src/types.ts` | API yanit tipleri (TypeScript arayuzleri) |
| `src/components/Controls.tsx` | Sol panel: hisse secimi, mod, short, CMA-ES anahtarlari |
| `src/components/KpiCards.tsx` | 6 metrik karti (Net Kar, Calmar, Win Rate...) |
| `src/components/PriceChart.tsx` | Candlestick + Kalman + sinyaller + hacim |
| `src/components/PlotlyCharts.tsx` | Equity/Drawdown, Radar, Olasilik histogrami |
| `src/App.css` | Koyu tema, panel/grid duzeni |
| `vite.config.ts` | `/api` -> `localhost:8000` proxy |

**Grafik kutuphaneleri:**
- **lightweight-charts** (v5): candlestick + hacim (TradingView tarzi)
- **plotly.js-dist-min**: radar (scatterpolar), equity/drawdown, histogram
  (react-plotly.js'in CJS/ESM interop sorunlari nedeniyle, Plotly'yi dogrudan
  saran kendi `Plot` bilesenimizi kullaniyoruz — `Plotly.react`)

---

## 3. Katman 2 — Backend (`api/`)

**Teknoloji:** FastAPI + uvicorn

| Dosya | Gorev |
|-------|-------|
| `api/main.py` | FastAPI uygulamasi, endpoint'ler, CORS, Pydantic istek modeli |
| `api/engine.py` | Motor servisi: pipeline'i calistirir, JSON-uyumlu yanit uretir |

**Endpoint'ler:**
- `GET /api/health` — saglik kontrolu
- `GET /api/symbols` — hisse listesi
- `POST /api/analyze` — tek sembol tam analiz (tahmin + opsiyonel CMA-ES + backtest + grafik verileri)

**`api/engine.py` ne yapar:**
1. `_prepare(symbol)`: veri yukle -> Kalman -> oznitelik -> LSTM tahmin (sembol basina onbellek)
2. `auto_optimize` ise CMA-ES ile en iyi Esik/SL/TP'yi bulur
3. `run_backtest` ile event-driven simulasyon
4. Metrikler + OHLC + equity/drawdown + radar + tahmin dagilimi + sinyalleri JSON olarak doner

Model bir kez yuklenir (lazy + global cache); ilk istek ~15-20 sn, sonrakiler hizli.

---

## 4. Katman 3 — Motor (`core/`, `optimizers/`)

Bu katman gecisten ETKILENMEDI. Onceki tum is mantigi burada:
- `core/data_pipeline.py`, `data_fetcher.py`, `signal_filters.py` (Kalman),
  `feature_engineering.py`, `ai_prep.py`, `ai_models.py`, `backtest_engine.py`
- `optimizers/cma_optimizer.py`
- `core/portfolio_backtest.py` (paylasimli-sermaye portfoy motoru)

FastAPI bu modulleri dogrudan import ederek kullanir.

---

## 5. Veri Akisi (Bir Analiz Istegi)

```
Kullanici "Simulasyonu Baslat"a basar (Controls.tsx)
   -> App.tsx: analyze(request) cagrilir (api.ts)
      -> POST /api/analyze  (Vite proxy -> :8000)
         -> api/main.py: AnalyzeRequest dogrulanir
            -> api/engine.py: analyze_symbol()
               -> core/ motorlari: tahmin + CMA-ES + backtest
            <- JSON (metrikler, OHLC, equity, radar, ...)
      <- AnalyzeResponse
   -> App.tsx: result state guncellenir
      -> KpiCards, PriceChart, EquityDrawdown, Radar, Histogram render edilir
```

---

## 6. Calistirma

```powershell
# Terminal 1 - Backend
.\run_backend.bat        # veya: venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000

# Terminal 2 - Frontend
.\run_frontend.bat       # veya: cd frontend; npm run dev
```
Tarayici: http://localhost:5173

---

## 7. Tamamlananlar ve Sirada Olanlar

| Adim | Durum | Aciklama |
|------|-------|----------|
| **R0** | ✅ | Iskelet: FastAPI + React baglantisi, uctan uca dogrulandi |
| **R1** | ✅ | Tek-hisse sayfasi: candlestick, equity/drawdown, radar, histogram, KPI'lar |
| **R2** | ⬜ | Portfoy sayfasi: kosu formu, split metrikleri, trade blotter, split radari |
| **Short modeli** | ⬜ | 3-sinifli (yukari/yatay/asagi) model egitimi — short gercek sinyale dayansin |
| **R3** | ⬜ (ops.) | Canli log + kosu kuyrugu (vakit kalirsa) |

---

## 8. Onemli Tasarim Notlari

- **node_modules** git'e dahil DEGIL (frontend/.gitignore) — depo sismez.
- **Motor degismedi:** React'e gecis hicbir hesaplama kodunu bozmadi; ayni
  sonuclar (ornegin FROTO +1141 TL) API uzerinden birebir uretiliyor.
- **Tekrarlanabilirlik:** CMA-ES seed=42 sabit; ayni girdi ayni sonucu verir.
- **Short su an deneysel:** mevcut model long-only egitildigi icin short zarar
  edebilir; dogru short icin 3-sinifli model egitimi planli (bkz. sirada).
