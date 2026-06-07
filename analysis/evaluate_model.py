"""
FAZ 1 - Akademik Degerlendirme Scripti

Iki analiz uretir:
  BOLUM A : Ongoru (Prediction) metrikleri -> Precision / Recall / F1 / Accuracy / Confusion
  BOLUM B : CMA-ES ablasyonu -> sabit parametre vs evrimsel optimize (Calmar, Net Kar, DD)

Sonuclar hem ekrana yazilir hem analysis/faz1_results.md dosyasina kaydedilir.
Bu script mevcut egitilmis modeli kullanir, yeniden egitim YAPMAZ.
"""

import os
import sys
import io
import argparse
import contextlib
import numpy as np
import pandas as pd

# Proje kokunu path'e ekle (script analysis/ altinda)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.backtest_engine import run_backtest
from optimizers.cma_optimizer import run_cma_optimization


SYMBOLS = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE', 'FROTO']
WINDOW = 60
FULL_FEATURES = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
NO_KALMAN_FEATURES = ['close', 'feature_return_1m', 'feature_volatility_15m']

# Ablasyon icin "insan tarafindan secilmis" sabit parametreler (optimize edilmemis baseline).
# Esik 0.50 = sigmoid'in dogal karar siniri; SL %2 / TP %4 = ders kitabi 1:2 risk-getiri.
# (Not: esik 0.60 ile model neredeyse hic islem yapmiyor cunku olasiliklar %50 civarinda
#  kumeleniyor; bu yuzden adil bir baseline icin dogal sinir olan 0.50 secildi.)
BASELINE_PARAMS = {'threshold': 0.50, 'stop_loss': 0.02, 'take_profit': 0.04}


def trading_metrics(df, initial_capital=10000.0):
    """run_backtest ciktisindan sessizce metrik hesaplar."""
    equity = df['equity'].values
    sr = df['strategy_return'].values
    net = equity[-1] - initial_capital
    rolling_max = np.maximum.accumulate(equity)
    dd = np.where(rolling_max > 0, (equity - rolling_max) / rolling_max, 0)
    maxdd = abs(dd.min()) * 100
    trades = int((sr != 0).sum())
    wins = int((sr > 0).sum())
    wr = wins / trades * 100 if trades > 0 else 0.0
    calmar = net / (maxdd + 1.0)
    return {'net': net, 'maxdd': maxdd, 'trades': trades, 'win_rate': wr, 'calmar': calmar}


def quiet_backtest(predictions, test_df, threshold, stop_loss, take_profit):
    """run_backtest'i ciktisini bastirarak calistirir."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = run_backtest(predictions, test_df,
                          threshold=threshold, stop_loss=stop_loss,
                          take_profit=take_profit, window_size=WINDOW)
    return df


def quiet_cma(predictions, test_df):
    with contextlib.redirect_stdout(io.StringIO()):
        t, sl, tp = run_cma_optimization(predictions, test_df, window_size=WINDOW)
    return t, sl, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='data/global_lstm_model_15min.keras')
    ap.add_argument('--no-kalman', action='store_true',
                    help='3 ozellikli Kalman ablasyon modeli (otomatik model yolu + ozellik)')
    ap.add_argument('--tag', default='full', help='Cikti dosyasi etiketi (faz1_results_<tag>.md)')
    args = ap.parse_args()

    if args.no_kalman:
        if args.model == 'data/global_lstm_model_15min.keras':
            args.model = 'data/ablation_no_kalman_15min.keras'
        features = NO_KALMAN_FEATURES
        if args.tag == 'full':
            args.tag = 'nokalman'
    else:
        features = FULL_FEATURES

    if not os.path.exists(args.model):
        print(f"[HATA] Model bulunamadi: {args.model}")
        return

    print(f"Model: {args.model}\nOzellikler ({len(features)}): {features}\n")
    model = load_model(args.model)

    pred_rows = []   # BOLUM A
    ablation_rows = []  # BOLUM B

    for symbol in SYMBOLS:
        print(f"\n>>> {symbol} degerlendiriliyor...")
        try:
            clean_df = load_and_fill_gaps(symbol, timeframe='15m')
            clean_df['kalman_close'] = apply_kalman_filter(clean_df['close'])
            ai_df = create_features_and_target(clean_df, lookahead=15, threshold=0.001)

            with contextlib.redirect_stdout(io.StringIO()):
                X_train, y_train, X_test, y_test, scaler, cw, test_df = prepare_lstm_data(
                    df=ai_df, feature_cols=features, window_size=WINDOW
                )

            predictions = model.predict(X_test, verbose=0)
            y_pred = (predictions.flatten() > 0.5).astype(int)

            # --- BOLUM A: Ongoru metrikleri ---
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp_ = cm.ravel()
            pred_rows.append({
                'Sembol': symbol,
                'Accuracy': accuracy_score(y_test, y_pred) * 100,
                'Precision': precision_score(y_test, y_pred, zero_division=0) * 100,
                'Recall': recall_score(y_test, y_pred, zero_division=0) * 100,
                'F1': f1_score(y_test, y_pred, zero_division=0) * 100,
                'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp_,
                'Pozitif_Oran': y_test.mean() * 100,
            })

            # --- BOLUM B: CMA-ES ablasyonu ---
            base_df = quiet_backtest(predictions, test_df,
                                     BASELINE_PARAMS['threshold'],
                                     BASELINE_PARAMS['stop_loss'],
                                     BASELINE_PARAMS['take_profit'])
            base_m = trading_metrics(base_df)

            ot, osl, otp = quiet_cma(predictions, test_df)
            opt_df = quiet_backtest(predictions, test_df, ot, osl, otp)
            opt_m = trading_metrics(opt_df)

            ablation_rows.append({
                'Sembol': symbol,
                'Sabit_Calmar': base_m['calmar'], 'Sabit_Kar': base_m['net'],
                'Sabit_DD': base_m['maxdd'], 'Sabit_Islem': base_m['trades'],
                'Opt_Esik': ot * 100, 'Opt_SL': osl * 100, 'Opt_TP': otp * 100,
                'Opt_Calmar': opt_m['calmar'], 'Opt_Kar': opt_m['net'],
                'Opt_DD': opt_m['maxdd'], 'Opt_Islem': opt_m['trades'],
            })

            print(f"    F1: {pred_rows[-1]['F1']:.1f}% | "
                  f"Sabit Calmar: {base_m['calmar']:.2f} -> Opt Calmar: {opt_m['calmar']:.2f}")

        except Exception as e:
            print(f"    [HATA] {symbol}: {e}")

    # --- Raporlama ---
    pred_df = pd.DataFrame(pred_rows)
    abl_df = pd.DataFrame(ablation_rows)

    print("\n\n" + "=" * 70)
    print("BOLUM A - ONGORU METRIKLERI (esik 0.5)")
    print("=" * 70)
    print(pred_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nORTALAMA F1: {pred_df['F1'].mean():.2f}% | "
          f"ORTALAMA Accuracy: {pred_df['Accuracy'].mean():.2f}%")

    print("\n\n" + "=" * 70)
    print("BOLUM B - CMA-ES ABLASYONU (Sabit Parametre vs Evrimsel Optimize)")
    print("=" * 70)
    print(abl_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nORTALAMA Calmar  Sabit: {abl_df['Sabit_Calmar'].mean():.2f}  ->  "
          f"Optimize: {abl_df['Opt_Calmar'].mean():.2f}")
    print(f"TOPLAM Net Kar   Sabit: {abl_df['Sabit_Kar'].sum():.0f} TL  ->  "
          f"Optimize: {abl_df['Opt_Kar'].sum():.0f} TL")

    # --- Markdown ciktisi ---
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'faz1_results_{args.tag}.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# FAZ 1 Deneysel Sonuclar ({args.tag})\n\n")
        f.write(f"Model: `{args.model}` | Ozellik sayisi: {len(features)}\n\n")
        f.write("## Bolum A - Ongoru Metrikleri (esik 0.5)\n\n")
        f.write(pred_df.to_markdown(index=False, floatfmt=".2f"))
        f.write(f"\n\n**Ortalama F1:** {pred_df['F1'].mean():.2f}%  |  "
                f"**Ortalama Accuracy:** {pred_df['Accuracy'].mean():.2f}%\n\n")
        f.write("## Bolum B - CMA-ES Ablasyonu\n\n")
        f.write(abl_df.to_markdown(index=False, floatfmt=".2f"))
        f.write(f"\n\n**Ortalama Calmar** Sabit: {abl_df['Sabit_Calmar'].mean():.2f} -> "
                f"Optimize: {abl_df['Opt_Calmar'].mean():.2f}\n")
        f.write(f"**Toplam Net Kar** Sabit: {abl_df['Sabit_Kar'].sum():.0f} TL -> "
                f"Optimize: {abl_df['Opt_Kar'].sum():.0f} TL\n")

    print(f"\n[KAYDEDILDI] {out_path}")


if __name__ == "__main__":
    main()
