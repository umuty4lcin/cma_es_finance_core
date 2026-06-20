"""
3-Sinifli Model Teshis Scripti

Amac: "Neden bazi sembollerde CMA-ES hic islem uretmiyor?" sorusunu yanitlamak.

Her sembol icin:
  1. Olasilik dagilim istatistikleri (p_up, p_down) — min/max/median/p90/p95
  2. Long_threshold icin "kac mum gecer" tablosu (0.35 / 0.40 / 0.45 / 0.50)
  3. Short_threshold icin ayni tablo
  4. CMA-ES sonucu (varsa kar/zarar, yoksa sebep)

Cikti: analysis/diag_3class.md
"""

import os
import sys
import io
import contextlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target_3class
from core.ai_prep import prepare_lstm_data
from core.backtest_engine_3class import run_backtest_3class
from optimizers.cma_optimizer_3class import run_cma_optimization_3class

SYMBOLS = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE', 'FROTO']
MODEL_PATH = "data/global_lstm_3class_15min.keras"
FEATURES = ['close', 'kalman_close', 'feature_kalman_diff',
            'feature_return_1m', 'feature_volatility_15m']
WINDOW = 60


def main():
    print("Model yukleniyor...")
    model = load_model(MODEL_PATH)

    rows = []
    for sym in SYMBOLS:
        print(f"\n>>> {sym} teshis...")
        try:
            clean = load_and_fill_gaps(sym, timeframe='15m')
            clean['kalman_close'] = apply_kalman_filter(clean['close'])
            # train_model_3class.py ile ayni esikleri kullan (v1: +/-%0.5)
            ai = create_features_and_target_3class(clean, lookahead=15,
                                                    up_threshold=0.005, down_threshold=-0.005)
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, X_te, _, _, _, tdf = prepare_lstm_data(ai, FEATURES, window_size=WINDOW)
            preds = model.predict(X_te, verbose=0)  # (N, 3)
            p_down = preds[:, 0]
            p_flat = preds[:, 1]
            p_up = preds[:, 2]

            # 1. Dagilim istatistikleri
            row = {
                'Sembol': sym,
                'N_bar': len(preds),
                # p_up
                'up_min': round(float(p_up.min()), 3),
                'up_med': round(float(np.median(p_up)), 3),
                'up_p90': round(float(np.percentile(p_up, 90)), 3),
                'up_p95': round(float(np.percentile(p_up, 95)), 3),
                'up_max': round(float(p_up.max()), 3),
                # p_down
                'dn_min': round(float(p_down.min()), 3),
                'dn_med': round(float(np.median(p_down)), 3),
                'dn_p90': round(float(np.percentile(p_down, 90)), 3),
                'dn_p95': round(float(np.percentile(p_down, 95)), 3),
                'dn_max': round(float(p_down.max()), 3),
                # Esik gecme sayilari (long)
                'long_>0.35': int((p_up > 0.35).sum()),
                'long_>0.40': int((p_up > 0.40).sum()),
                'long_>0.45': int((p_up > 0.45).sum()),
                'long_>0.50': int((p_up > 0.50).sum()),
                # Esik gecme sayilari (short)
                'shrt_>0.35': int((p_down > 0.35).sum()),
                'shrt_>0.40': int((p_down > 0.40).sum()),
                'shrt_>0.45': int((p_down > 0.45).sum()),
                'shrt_>0.50': int((p_down > 0.50).sum()),
            }

            # 2. CMA-ES sonucu
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    lt, st, sl, tp = run_cma_optimization_3class(preds, tdf, window_size=WINDOW)
                with contextlib.redirect_stdout(io.StringIO()):
                    rdf = run_backtest_3class(preds, tdf,
                                              long_threshold=lt, short_threshold=st,
                                              stop_loss=sl, take_profit=tp,
                                              window_size=WINDOW)
                eq = rdf['equity'].values
                sr = rdf['strategy_return'].values
                row['CMA_long_thr'] = round(lt, 3)
                row['CMA_short_thr'] = round(st, 3)
                row['CMA_SL'] = round(sl, 3)
                row['CMA_TP'] = round(tp, 3)
                row['Trades'] = int((sr != 0).sum())
                row['Long_giris'] = int((rdf['signal'].values == 1).sum())
                row['Short_giris'] = int((rdf['signal'].values == -1).sum())
                row['Net_Kar'] = round(float(eq[-1] - 10000), 0)
            except Exception as e:
                row['CMA_long_thr'] = None
                row['Trades'] = 0
                row['Net_Kar'] = None
                row['Hata'] = str(e)[:50]

            rows.append(row)
            print(f"  p_up max={row['up_max']:.3f}, p_down max={row['dn_max']:.3f}, "
                  f"long>0.40={row['long_>0.40']}, shrt>0.40={row['shrt_>0.40']}, "
                  f"trades={row.get('Trades', '?')}, kar={row.get('Net_Kar', '?')}")

        except Exception as e:
            print(f"  [HATA] {sym}: {e}")

    df = pd.DataFrame(rows)

    # Iki ayri tablo: olasilik dagilimi + CMA sonucu
    print("\n" + "=" * 80)
    print("BOLUM A — OLASILIK DAGILIM ISTATISTIKLERI")
    print("=" * 80)
    cols_a = ['Sembol', 'N_bar', 'up_med', 'up_p90', 'up_p95', 'up_max',
              'dn_med', 'dn_p90', 'dn_p95', 'dn_max']
    print(df[cols_a].to_string(index=False))

    print("\n" + "=" * 80)
    print("BOLUM B — ESIK GECME SAYILARI (kac mum hangi esigi gecti?)")
    print("=" * 80)
    cols_b = ['Sembol', 'long_>0.35', 'long_>0.40', 'long_>0.45', 'long_>0.50',
              'shrt_>0.35', 'shrt_>0.40', 'shrt_>0.45', 'shrt_>0.50']
    print(df[cols_b].to_string(index=False))

    print("\n" + "=" * 80)
    print("BOLUM C — CMA-ES SONUCLARI")
    print("=" * 80)
    cols_c = ['Sembol', 'CMA_long_thr', 'CMA_short_thr', 'CMA_SL', 'CMA_TP',
              'Trades', 'Long_giris', 'Short_giris', 'Net_Kar']
    print(df[cols_c].to_string(index=False))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_3class.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write("# 3-Sinifli Model Teshis Sonuclari\n\n")
        f.write("## A — Olasilik Dagilim Istatistikleri\n\n")
        f.write(df[cols_a].to_markdown(index=False))
        f.write("\n\n## B — Esik Gecme Sayilari\n\n")
        f.write(df[cols_b].to_markdown(index=False))
        f.write("\n\n## C — CMA-ES Sonuclari\n\n")
        f.write(df[cols_c].to_markdown(index=False))
        f.write("\n")
    print(f"\n[KAYDEDILDI] {out}")


if __name__ == "__main__":
    main()
