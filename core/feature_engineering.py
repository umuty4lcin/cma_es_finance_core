import pandas as pd
import numpy as np

def create_features_and_target(df, lookahead=4, threshold=0.02):
    """
    Yapay zeka için öznitelikleri (X) ve hedef etiketleri (Y) oluşturur.
    
    Parametreler:
    - df: Temizlenmiş ve Kalman filtresi uygulanmış DataFrame.
    - lookahead: Geleceğe kaç mum (dakika) bakılacağı (Örn: 15).
    - threshold: '1' etiketi vermek için gereken minimum yüzdesel getiri (Örn: 0.001 yani %0.1).
    """
    
    # Kopyasını alıyoruz ki orijinal veri bozulmasın
    data = df.copy()
    
    # ---------------------------------------------------------
    # BÖLÜM 1: ÖZNİTELİKLER (FEATURES - X) - Sistemin Girdileri
    # ---------------------------------------------------------
    
    # 1. Kalman Sapması: Gerçek fiyat, Kalman (trend) çizgisinden ne kadar saptı? (Yüzdesel)
    data['feature_kalman_diff'] = (data['close'] - data['kalman_close']) / data['kalman_close']
    
    # 2. Anlık Getiri: Fiyat bir önceki dakikaya göre ne kadar değişti?
    data['feature_return_1m'] = data['close'].pct_change(1)
    
    # 3. Volatilite (Hareketlilik): Son 15 dakikadaki fiyat değişimlerinin standart sapması
    data['feature_volatility_15m'] = data['feature_return_1m'].rolling(window=15).std()
    
    # ---------------------------------------------------------
    # BÖLÜM 2: HEDEF (TARGET - Y) - Sistemin Çıktısı (Sınıflandırma)
    # ---------------------------------------------------------
    
    # Gelecekteki fiyatı bulmak için veriyi yukarı kaydırıyoruz (shift ile eksi değer kullanarak)
    # Yani şu anki satıra, 'lookahead' kadar sonraki kapanış fiyatını yazıyoruz.
    data['future_close'] = data['close'].shift(-lookahead)
    
    # Gelecekteki fiyat ile şu anki fiyat arasındaki yüzdesel farkı hesaplıyoruz
    data['future_return'] = (data['future_close'] - data['close']) / data['close']
    
    # EĞER gelecekteki getiri bizim belirlediğimiz eşikten (threshold) büyükse 1 yap, değilse 0 yap.
    # np.where(koşul, doğruysa_ne_yazılsın, yanlışsa_ne_yazılsın)
    data['target'] = np.where(data['future_return'] >= threshold, 1, 0)
    
    # Hesaplamalar sırasında oluşan NaN (boş) değerleri temizle 
    # (Örneğin son 15 dakikanın gelecekteki fiyatı belli olmadığı için boş kalır)
    data.dropna(inplace=True)
    
    return data

# import numpy as np
# import pandas as pd

# def create_features_and_target(df, lookahead=15, threshold=0.001):
#     df = df.copy()
    
#     # --- HEDEF (TARGET) DEĞİŞKENİ ---
#     df['future_close'] = df['close'].shift(-lookahead)
#     df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
#     # İŞTE EKSİK OLAN KRİTİK SATIR:
#     # Eğer gelecek getiri eşiği geçiyorsa 1 (AL), geçmiyorsa 0 (BEKLE)
#     df['target'] = np.where(df['future_return'] >= threshold, 1, 0)
    
#     # --- MEVCUT ÖZNİTELİKLER ---
#     df['feature_return_1m'] = df['close'].pct_change()
#     df['feature_volatility_15m'] = df['feature_return_1m'].rolling(window=15).std()
    
#     if 'kalman_close' in df.columns:
#         df['feature_kalman_diff'] = (df['close'] - df['kalman_close']) / df['kalman_close']

#     # ========================================================
#     # --- YENİ EKLENEN KLASİK İNDİKATÖRLER (DENEYSEL) ---
#     # ========================================================
    
#     # 1. RSI (Relative Strength Index - 14 Mum)
#     delta = df['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     df['feature_rsi_14'] = 100 - (100 / (1 + rs))

#     # 2. MACD (12, 26) - Hisseler arası uyum için fiyata bölünerek normalize edildi
#     ema12 = df['close'].ewm(span=12, adjust=False).mean()
#     ema26 = df['close'].ewm(span=26, adjust=False).mean()
#     df['feature_macd_norm'] = (ema12 - ema26) / df['close'] 

#     # 3. Bollinger Bantları Konumu (%B - 20 Mum)
#     sma20 = df['close'].rolling(window=20).mean()
#     std20 = df['close'].rolling(window=20).std()
#     upper_band = sma20 + (std20 * 2)
#     lower_band = sma20 - (std20 * 2)
#     df['feature_bb_position'] = (df['close'] - lower_band) / (upper_band - lower_band)

#     # 4. Stochastic Oscillator (%K - 14 Mum)
#     low14 = df['low'].rolling(window=14).min()
#     high14 = df['high'].rolling(window=14).max()
#     df['feature_stoch_k'] = (df['close'] - low14) / (high14 - low14)

#     # 5. SMA 50 Filtresi (Trend Uzaklığı)
#     sma50 = df['close'].rolling(window=50).mean()
#     df['feature_sma50_diff'] = (df['close'] - sma50) / sma50

#     # ========================================================

#     # Yeni indikatörlerin hesaplanması için gereken ilk ~50 satırlık NaN verileri temizle
#     df.dropna(inplace=True)
    
#     return df