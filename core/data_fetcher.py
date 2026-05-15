import requests
import time
from database.db_manager import get_latest_timestamp, save_candles, init_db

# SARDIS API Base URL
BASE_URL = "http://89.167.3.33:8005"

def fetch_candles_from_api(symbol, period, start_ts=None, limit=100000):
    """SARDIS API'den belirtilen parametrelerle mum verisi çeker."""
    url = f"{BASE_URL}/candles"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }
    
    # Eğer elimizde eski veri varsa, sadece o tarihten sonrasını iste
    if start_ts:
        params["start_ts"] = start_ts
        
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # 400, 404, 422 gibi hataları yakalar
        return response.json()
    except Exception as e:
        print(f"API Hatası [{symbol} - {period}]: {e}")
        return []

def sync_symbol_data(symbol, period="15m"):
    """
    Belirtilen sembolün verilerini API'den çekip SQLite'a eşitler (Senkronize eder).
    """
    print(f"\n[{symbol} - {period}] Veri senkronizasyonu başlatılıyor...")
    
    # 1. Elimizdeki en son mumun tarihini bul
    latest_ts = get_latest_timestamp(symbol, period)
    
    # Eğer veri varsa, son mumdan +1 milisaniye sonrasını çekmeye başla
    start_ts = latest_ts + 1 if latest_ts else None
    
    total_fetched = 0
    total_saved = 0
    
    while True:
        # API'ye İstek At
        candles = fetch_candles_from_api(symbol, period, start_ts=start_ts)
        
        if not candles:
            break
            
        # Veritabanına kaydet
        saved_count = save_candles(symbol, period, candles)
        total_fetched += len(candles)
        total_saved += saved_count
        
        print(f"  -> {len(candles)} mum çekildi. Veritabanına yeni eklenen: {saved_count}")
        
        # Eğer gelen mum sayısı limitin (100.000) altındaysa, tüm geçmişi çekmişiz demektir.
        if len(candles) < 100000:
            break
            
        # Eğer 100.000 mum geldiyse, daha eskiye (veya yeniye) gitmek için son mumun zamanını al
        start_ts = candles[-1]['timestamp'] + 1
        time.sleep(0.5) # API'yi yormamak için yarım saniye bekle
        
    print(f"✅ [{symbol}] Senkronizasyon Tamamlandı. Toplam İndirilen: {total_fetched} | Yeni Eklenen: {total_saved}")

# Sadece bu dosya çalıştırılırsa test etmek için
if __name__ == "__main__":
    init_db() # Veritabanını oluştur
    
    # Test için birkaç sembolü API'den çekelim
    test_symbols = ['FROTO']
    for sym in test_symbols:
        sync_symbol_data(sym, period="15m")