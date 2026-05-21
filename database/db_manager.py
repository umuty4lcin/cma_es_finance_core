import sqlite3
import pandas as pd
import os

# db_manager.py dosyasının bulunduğu klasörün tam yolunu otomatik bulur
DB_DIR = os.path.dirname(os.path.abspath(__file__))
# Veritabanı dosyasının bu klasörün içinde olduğunu belirtir
DB_PATH = os.path.join(DB_DIR, 'market_data.db')

def get_connection():
    # Artık sistem nereden çalışırsa çalışsın, her zaman database/market_data.db'yi bulacak
    conn = sqlite3.connect(DB_PATH)
    return conn
def init_db():
    """Veritabanını ve tabloları oluşturur."""
    conn = get_connection()
    cursor = conn.cursor()
    # Mum verileri için tablo
    # PRIMARY KEY (symbol, period, timestamp) mükerrer kaydı (duplicate) engeller.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT,
            period TEXT,
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, period, timestamp)
        )
    ''')
    conn.commit()
    conn.close()
    print("[DB] Veritabanı altyapısı hazır.")

def get_latest_timestamp(symbol, period):
    """Elimizdeki en son (en güncel) mumun zaman damgasını getirir."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(timestamp) FROM candles WHERE symbol=? AND period=?', (symbol, period))
    result = cursor.fetchone()[0]
    conn.close()
    return result

def save_candles(symbol, period, candles_list):
    """API'den gelen mumları veritabanına kaydeder."""
    if not candles_list:
        return 0
        
    conn = get_connection()
    cursor = conn.cursor()
    
    # INSERT OR IGNORE: Eğer aynı timestamp varsa üzerine yazma, es geç.
    query = '''
        INSERT OR IGNORE INTO candles (symbol, period, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''
    
    data = [
        (symbol, period, c['timestamp'], c['open'], c['high'], c['low'], c['close'], c['volume'])
        for c in candles_list
    ]
    
    cursor.executemany(query, data)
    conn.commit()
    saved_count = cursor.rowcount
    conn.close()
    
    return saved_count

def load_data_as_df(symbol, period):
    """Eğitim ve test aşamasında veritabanından pandas DataFrame olarak veriyi okur."""
    conn = get_connection()
    query = 'SELECT * FROM candles WHERE symbol=? AND period=? ORDER BY timestamp ASC'
    df = pd.read_sql(query, conn, params=(symbol, period))
    conn.close()
    
    # Timestamp'i gerçek tarihe çevir
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    return df