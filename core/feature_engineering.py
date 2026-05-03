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