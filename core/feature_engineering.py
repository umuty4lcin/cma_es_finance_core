import pandas as pd
import numpy as np


def create_features_and_target(df, lookahead=15, threshold=0.001):
    """
    Yapay zeka icin oznitelikleri (X) ve hedef etiketleri (Y) olusturur.

    lookahead : kac mum ilerisi hedefleniyor (varsayilan: 15 mum ~ 4 saat)
    threshold : '1' etiketi icin gereken minimum yuzdsel getiri (varsayilan: %0.1)
    """
    data = df.copy()

    # Kalman sapmasi: gercek fiyat trend cizgisinden ne kadar sapti?
    data['feature_kalman_diff'] = (data['close'] - data['kalman_close']) / data['kalman_close']

    # Anlik getiri: bir onceki muma gore degisim
    data['feature_return_1m'] = data['close'].pct_change(1)

    # Volatilite: son 15 mumdaki fiyat degisimlerinin standart sapmasi
    data['feature_volatility_15m'] = data['feature_return_1m'].rolling(window=15).std()

    # Hedef: lookahead mum sonraki kapanisi simdi ile karsilastir
    data['future_close'] = data['close'].shift(-lookahead)
    data['future_return'] = (data['future_close'] - data['close']) / data['close']
    data['target'] = np.where(data['future_return'] >= threshold, 1, 0)

    data.dropna(inplace=True)

    return data


def create_features_and_target_3class(df, lookahead=15, up_threshold=0.005, down_threshold=-0.005):
    """
    3-sinifli yon siniflandirmasi icin oznitelik + hedef uretir (short destegi).

    Etiketler:
        2 = YUKARI  (future_return >= up_threshold)
        1 = YATAY   (down_threshold < future_return < up_threshold)
        0 = ASAGI   (future_return <= down_threshold)

    Varsayilan esikler:
        up_threshold   = +%0.5
        down_threshold = -%0.5

    NOT: 2-sinifli pipeline (create_features_and_target) ile yan yana yasar;
    sadece 'target' sutunu farkli tanimlanir (cok-sinifli etiket).
    """
    data = df.copy()

    data['feature_kalman_diff'] = (data['close'] - data['kalman_close']) / data['kalman_close']
    data['feature_return_1m'] = data['close'].pct_change(1)
    data['feature_volatility_15m'] = data['feature_return_1m'].rolling(window=15).std()

    data['future_close'] = data['close'].shift(-lookahead)
    data['future_return'] = (data['future_close'] - data['close']) / data['close']

    # 3-sinifli etiket
    conditions = [
        data['future_return'] >= up_threshold,
        data['future_return'] <= down_threshold,
    ]
    choices = [2, 0]  # 2=YUKARI, 0=ASAGI
    data['target'] = np.select(conditions, choices, default=1)  # 1=YATAY

    data.dropna(inplace=True)
    return data
