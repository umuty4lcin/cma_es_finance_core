"""
CMA-ES Finance API — FastAPI backend.

Mevcut core/ ve optimizers/ motorlarini React onyuzune REST API olarak sunar.
Hicbir motor kodu burada tekrarlanmaz; sadece sarmalanir.

Calistirma (proje kokunden):
    venv\\Scripts\\python.exe -m uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
