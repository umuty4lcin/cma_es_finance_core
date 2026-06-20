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
    model_type: str = "2class"        # '2class' (long-only) | '3class' (long+short)
    auto_optimize: bool = True
    # 2-sinifli parametreler
    threshold: float = 0.50
    stop_loss: float = 0.02
    take_profit: float = 0.04
    mode: str = "fixed"               # 'fixed' | 'compound'
    # 3-sinifli ek parametreler
    long_threshold: float = 0.50
    short_threshold: float = 0.50


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Tek sembol icin tam analiz: tahmin + (opsiyonel CMA-ES) + backtest + grafik verileri.

    model_type='2class' -> long-only ikili siniflandirma (mevcut R1)
    model_type='3class' -> long+short softmax siniflandirma
    """
    if req.symbol not in SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Bilinmeyen sembol: {req.symbol}")
    if req.model_type not in ("2class", "3class"):
        raise HTTPException(status_code=400, detail=f"Gecersiz model_type: {req.model_type}")

    try:
        if req.model_type == "3class":
            from api.engine import analyze_symbol_3class
            return analyze_symbol_3class(
                symbol=req.symbol, auto_optimize=req.auto_optimize,
                long_threshold=req.long_threshold, short_threshold=req.short_threshold,
                stop_loss=req.stop_loss, take_profit=req.take_profit, mode=req.mode,
            )
        else:
            from api.engine import analyze_symbol
            return analyze_symbol(
                symbol=req.symbol, auto_optimize=req.auto_optimize,
                threshold=req.threshold, stop_loss=req.stop_loss, take_profit=req.take_profit,
                mode=req.mode,
            )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503,
                            detail=f"Model bulunamadi: {e}. 3-sinifli model henuz egitilmemis olabilir.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatasi: {e}")


class PortfolioRequest(BaseModel):
    symbols: list[str] = SYMBOLS
    max_positions: int = 4
    auto_optimize: bool = True
    sizing: str = "fixed"   # 'fixed' | 'dynamic'


@app.post("/api/portfolio")
def portfolio(req: PortfolioRequest):
    """Paylasimli-sermaye portfoy backtest'i (2-sinifli, long-only). Tez kapsami disi (gelecek calisma)."""
    bad = [s for s in req.symbols if s not in SYMBOLS]
    if bad:
        raise HTTPException(status_code=400, detail=f"Bilinmeyen sembol(ler): {bad}")
    if not req.symbols:
        raise HTTPException(status_code=400, detail="En az bir sembol secilmeli.")
    from api.engine import run_portfolio
    try:
        return run_portfolio(req.symbols, max_positions=req.max_positions,
                             auto_optimize=req.auto_optimize, sizing=req.sizing)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfoy hatasi: {e}")
