"""
3-sinifli global LSTM modelini egitir (short destegi icin).

Etiketler: 0=ASAGI, 1=YATAY, 2=YUKARI (esikler: +/- %0.5, 15 mum sonra)

Cikis modeli: data/global_lstm_3class_15min.keras

Mevcut 2-sinifli train_model.py'dan bagimsiz; ikisi yan yana yasar.
"""

import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target_3class
from core.ai_prep import prepare_lstm_data
from core.ai_models import build_lstm_3class_model


TRAIN_SYMBOLS = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE']
FEATURES = ['close', 'kalman_close', 'feature_kalman_diff',
            'feature_return_1m', 'feature_volatility_15m']
WINDOW = 60
MODEL_SAVE_PATH = 'data/global_lstm_3class_15min.keras'
UP_THR = 0.005     # +%0.5 (v1: dengeli long+short, daha iyi val accuracy)
DOWN_THR = -0.005  # -%0.5


def train(timeframe='15m'):
    print("=" * 60)
    print("3-SINIFLI GLOBAL LSTM EGITIMI BASLIYOR (short destegi)")
    print(f"Esikler: yukari >= +%{UP_THR*100:.1f}, asagi <= %{DOWN_THR*100:.1f}, lookahead=15")
    print("=" * 60)

    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []

    for symbol in TRAIN_SYMBOLS:
        try:
            print(f"\n  -> Isleniyor: {symbol}")
            clean = load_and_fill_gaps(symbol, timeframe=timeframe)
            clean['kalman_close'] = apply_kalman_filter(clean['close'])
            ai = create_features_and_target_3class(clean, lookahead=15,
                                                    up_threshold=UP_THR,
                                                    down_threshold=DOWN_THR)
            # Sinif dagilimini gozlemle
            dist = ai['target'].value_counts().sort_index()
            print(f"     Sinif dagilimi: 0(asagi)={dist.get(0,0)}, 1(yatay)={dist.get(1,0)}, 2(yukari)={dist.get(2,0)}")

            X_tr, y_tr, X_te, y_te, _, _, _ = prepare_lstm_data(
                df=ai, feature_cols=FEATURES, window_size=WINDOW)
            X_train_list.append(X_tr); y_train_list.append(y_tr)
            X_test_list.append(X_te); y_test_list.append(y_te)
        except Exception as e:
            print(f"     [!] {symbol} atlandi: {e}")

    if not X_train_list:
        print("[KRITIK HATA] Hicbir sembol islenemedi.")
        return

    print("\nMega veri seti olusturuluyor...")
    X_train_g = np.vstack(X_train_list)
    y_train_g = np.concatenate(y_train_list)
    X_test_g = np.vstack(X_test_list)
    y_test_g = np.concatenate(y_test_list)

    print(f"  Mega Train: {X_train_g.shape}")
    print(f"  Mega Test:  {X_test_g.shape}")

    # Global sinif dagilimi
    unique, counts = np.unique(y_train_g, return_counts=True)
    print(f"  Global sinif dagilimi: {dict(zip(unique.tolist(), counts.tolist()))}")

    classes = np.unique(y_train_g)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_g)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    print(f"  Global sinif agirliklari: {class_weights}")

    print("\nLSTM (3-sinifli) insa ediliyor ve egitim basliyor...")
    input_shape = (X_train_g.shape[1], X_train_g.shape[2])
    model = build_lstm_3class_model(input_shape, n_classes=3)

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)

    model.fit(
        X_train_g, y_train_g,
        epochs=30, batch_size=512,
        validation_data=(X_test_g, y_test_g),
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint],
        verbose=2, shuffle=True,
    )

    print(f"\n[BASARILI] 3-sinifli model kaydedildi: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
