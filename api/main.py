"""
CMA-ES Finance API — FastAPI backend.

Mevcut core/ ve optimizers/ motorlarini React onyuzune REST API olarak sunar.
Hicbir motor kodu burada tekrarlanmaz; sadece sarmalanir.

Calistirma (proje kokunden):
    venv\\Scripts\\python.exe -m uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="CMA-ES Finance API", version="0.1.0")

# React (Vite dev sunucusu) ile haberlesme icin CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sistemin calistigi BIST sembolleri (FROTO = egitimde gorulmemis OOU testi)
SYMBOLS = ["ASELS", "GARAN", "HALKB", "ISCTR", "THYAO",
           "TUPRS", "VAKBN", "SASA", "SISE", "FROTO"]


@app.get("/api/health")
def health():
    """Basit saglik kontrolu."""
    return {"status": "ok", "service": "cma-es-finance-api"}


@app.get("/api/symbols")
def get_symbols():
    """Kullanilabilir hisse sembollerini dondurur."""
    return {"symbols": SYMBOLS}


class AnalyzeRequest(BaseModel):
    symbol: str
    auto_optimize: bool = True
    threshold: float = 0.50
    stop_loss: float = 0.02
    take_profit: float = 0.04
    mode: str = "fixed"          # 'fixed' | 'compound'
    allow_short: bool = False


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Tek sembol icin tam analiz: tahmin + (opsiyonel CMA-ES) + backtest + grafik verileri."""
    if req.symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Bilinmeyen sembol: {req.symbol}")
    # Agir importu (TensorFlow) sadece gerektiginde yap
    from api.engine import analyze_symbol
    try:
        return analyze_symbol(
            symbol=req.symbol, auto_optimize=req.auto_optimize,
            threshold=req.threshold, stop_loss=req.stop_loss, take_profit=req.take_profit,
            mode=req.mode, allow_short=req.allow_short,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatasi: {e}")
