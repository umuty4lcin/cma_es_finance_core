import numpy as np
import pandas as pd

def apply_kalman_filter(prices, process_variance=1e-5, measurement_variance=0.01**2):
    """
    1 Boyutlu temel Kalman Filtresi uygulayarak gürültülü fiyat serisini yumuşatır.
    
    Parametreler:
    - prices: pandas Series (Örn: df['close'])
    - process_variance (Q): Sistemin kendi içindeki değişim hızı (Küçüldükçe daha pürüzsüz olur)
    - measurement_variance (R): Ölçümdeki (fiyattaki) beklenen gürültü miktarı
    """
    n_iter = len(prices)
    sz = (n_iter,)
    
    # Kalman değişkenleri için bellek tahsisi
    xhat = np.zeros(sz)      # Sonraki tahmin (Filtrelenmiş fiyat)
    P = np.zeros(sz)         # Tahmin hatası varyansı
    xhatminus = np.zeros(sz) # Önceki tahmin
    Pminus = np.zeros(sz)    # Önceki hata
    K = np.zeros(sz)         # Kalman Kazancı (Gain)
    
    # Başlangıç değerleri
    xhat[0] = prices.iloc[0]
    P[0] = 1.0
    
    # Filtreyi tüm zaman serisi üzerinde döngüyle uygula
    for k in range(1, n_iter):
        # Zaman Güncellemesi (Tahmin)
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + process_variance
        
        # Ölçüm Güncellemesi (Düzeltme)
        K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
        xhat[k] = xhatminus[k] + K[k] * (prices.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    # Filtrelenmiş veriyi Pandas Series olarak döndür
    return pd.Series(xhat, index=prices.index, name='kalman_close')