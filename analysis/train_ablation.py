"""
FAZ 1 - Kalman Ablasyonu icin Egitim Scripti

Kalman tabanli ozellikleri (kalman_close, feature_kalman_diff) CIKARARAK
sadece 3 ham ozellikle bir LSTM varyanti egitir:
    [close, feature_return_1m, feature_volatility_15m]

Amac: "Istatistiksel isaret isleme (Kalman) katmani modele gercekten katki
sagliyor mu?" sorusunu nicel olarak yanitlamak. Tam model (5 ozellik) ile
bu varyant (3 ozellik) ayni egitim/test bolunmesiyle karsilastirilir.

Cikti modeli: data/ablation_no_kalman_15min.keras
"""

import os
import sys
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.ai_models import build_lstm_model


TRAIN_SYMBOLS = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE']
# Kalman ozellikleri CIKARILDI:
FEATURES_NO_KALMAN = ['close', 'feature_return_1m', 'feature_volatility_15m']
MODEL_SAVE_PATH = 'data/ablation_no_kalman_15min.keras'
WINDOW = 60


def train_no_kalman_model(timeframe='15m'):
    print("=" * 60)
    print("ABLASYON EGITIMI: KALMAN OZELLIKLERI OLMADAN (3 ozellik)")
    print("=" * 60)

    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []

    for symbol in TRAIN_SYMBOLS:
        try:
            print(f"  -> {symbol}")
            clean_df = load_and_fill_gaps(symbol, timeframe=timeframe)
            # Kalman yine hesaplanir cunku create_features_and_target target uretirken
            # kalman_close'a ihtiyac duymaz; ama fonksiyon feature_kalman_diff uretiyor.
            # Bu varyantta o sutunu KULLANMIYORUZ, sadece feature listesine almiyoruz.
            clean_df['kalman_close'] = apply_kalman_filter(clean_df['close'])
            ai_df = create_features_and_target(clean_df, lookahead=15, threshold=0.001)

            X_train, y_train, X_test, y_test, _, _, _ = prepare_lstm_data(
                df=ai_df, feature_cols=FEATURES_NO_KALMAN, window_size=WINDOW
            )
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_test_list.append(X_test)
            y_test_list.append(y_test)
        except Exception as e:
            print(f"  [!] {symbol} atlandi: {e}")

    if not X_train_list:
        print("[KRITIK HATA] Hicbir sembol islenemedi.")
        return

    X_train_g = np.vstack(X_train_list)
    y_train_g = np.concatenate(y_train_list)
    X_test_g = np.vstack(X_test_list)
    y_test_g = np.concatenate(y_test_list)

    print(f"\nMega Train: {X_train_g.shape} | Mega Test: {X_test_g.shape}")

    classes = np.unique(y_train_g)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_g)
    class_weights = dict(zip(classes, weights))

    input_shape = (X_train_g.shape[1], X_train_g.shape[2])
    model = build_lstm_model(input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)

    model.fit(
        X_train_g, y_train_g,
        epochs=30, batch_size=512,
        validation_data=(X_test_g, y_test_g),
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint],
        verbose=1, shuffle=True
    )

    print(f"\n[BASARILI] Ablasyon modeli kaydedildi: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_no_kalman_model()
