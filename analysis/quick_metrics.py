"""
Hizli Ongoru Metrikleri (CMA-ES YOK)

Verilen bir modeli ve ozellik listesini kullanarak sadece BOLUM A
(Precision/Recall/F1/Accuracy/Confusion) hesaplar. CMA optimizasyonu
calistirmadigi icin saniyeler icinde biter.

Kullanim:
  python quick_metrics.py                       -> tam model (5 ozellik)
  python quick_metrics.py --no-kalman           -> Kalman ablasyon modeli (3 ozellik)
  python quick_metrics.py --model PATH --feats f1,f2
"""

import os
import sys
import io
import argparse
import contextlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data

SYMBOLS = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE', 'FROTO']
WINDOW = 60

FULL_FEATURES = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
NO_KALMAN_FEATURES = ['close', 'feature_return_1m', 'feature_volatility_15m']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='data/global_lstm_model_15min.keras')
    ap.add_argument('--no-kalman', action='store_true', help='3 ozellikli ablasyon modeli')
    ap.add_argument('--feats', default=None, help='Virgulle ayrilmis ozellik listesi (override)')
    ap.add_argument('--tag', default='', help='Cikti dosyasi etiketi')
    args = ap.parse_args()

    if args.no_kalman and args.model == 'data/global_lstm_model_15min.keras':
        args.model = 'data/ablation_no_kalman_15min.keras'

    if args.feats:
        features = args.feats.split(',')
    elif args.no_kalman:
        features = NO_KALMAN_FEATURES
    else:
        features = FULL_FEATURES

    print(f"Model: {args.model}\nOzellikler ({len(features)}): {features}\n")
    model = load_model(args.model)

    rows = []
    for symbol in SYMBOLS:
        try:
            clean_df = load_and_fill_gaps(symbol, timeframe='15m')
            clean_df['kalman_close'] = apply_kalman_filter(clean_df['close'])
            ai_df = create_features_and_target(clean_df, lookahead=15, threshold=0.001)
            with contextlib.redirect_stdout(io.StringIO()):
                X_tr, y_tr, X_te, y_te, sc, cw, tdf = prepare_lstm_data(
                    df=ai_df, feature_cols=features, window_size=WINDOW)
            preds = model.predict(X_te, verbose=0)
            y_pred = (preds.flatten() > 0.5).astype(int)
            cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            rows.append({
                'Sembol': symbol,
                'Accuracy': accuracy_score(y_te, y_pred) * 100,
                'Precision': precision_score(y_te, y_pred, zero_division=0) * 100,
                'Recall': recall_score(y_te, y_pred, zero_division=0) * 100,
                'F1': f1_score(y_te, y_pred, zero_division=0) * 100,
                'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            })
            print(f"  {symbol:6s} F1={rows[-1]['F1']:5.2f}%  Acc={rows[-1]['Accuracy']:5.2f}%  "
                  f"Prec={rows[-1]['Precision']:5.2f}%  Rec={rows[-1]['Recall']:5.2f}%")
        except Exception as e:
            print(f"  [HATA] {symbol}: {e}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nORTALAMA F1: {df['F1'].mean():.2f}%  |  ORTALAMA Accuracy: {df['Accuracy'].mean():.2f}%  |  "
          f"ORTALAMA Recall: {df['Recall'].mean():.2f}%")


if __name__ == "__main__":
    main()
