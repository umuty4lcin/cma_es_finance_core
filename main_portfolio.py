"""
Portfoy (Coklu Es-Zamanli Pozisyon) Demo Orkestratoru

Tek-sembol main.py'nin aksine, bu script tum hisseleri ORTAK bir sermaye
havuzunda es zamanli simule eder. Her hisse icin:
  1. LSTM tahmini uretilir
  2. CMA-ES ile o hisseye ozel optimal parametreler bulunur
  3. Tum semboller paylasimli sermaye ile portfoy backtest'ine verilir

Karsilastirma: ayni parametrelerle tek-tek (izole) backtest toplam kari ile
portfoy (paylasimli, max N pozisyon) sonucu yan yana raporlanir.
"""

import os
import io
import contextlib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.backtest_engine import run_backtest
from core.portfolio_backtest import run_portfolio_backtest
from optimizers.cma_optimizer import run_cma_optimization

SYMBOLS = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE', 'FROTO']
MODEL_PATH = 'data/global_lstm_model_15min.keras'
FEATURES = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
WINDOW = 60
MAX_POSITIONS = 4


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[HATA] Model bulunamadi: {MODEL_PATH}")
        return

    print("Model yukleniyor...")
    model = load_model(MODEL_PATH)

    symbol_data = {}
    isolated_total = 0.0

    for sym in SYMBOLS:
        print(f"\n>>> {sym}: tahmin + CMA-ES optimizasyon...")
        try:
            clean = load_and_fill_gaps(sym, timeframe='15m')
            clean['kalman_close'] = apply_kalman_filter(clean['close'])
            ai = create_features_and_target(clean, lookahead=15, threshold=0.001)
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, X_test, _, _, _, test_df = prepare_lstm_data(ai, FEATURES, window_size=WINDOW)
            preds = model.predict(X_test, verbose=0)

            t, sl, tp = quiet(run_cma_optimization, preds, test_df, WINDOW)
            symbol_data[sym] = {'predictions': preds, 'test_df': test_df,
                                'threshold': t, 'stop_loss': sl, 'take_profit': tp}

            # Izole (tek-tek) backtest karsilastirma icin
            iso_df = quiet(run_backtest, preds, test_df, t, sl, tp, WINDOW)
            iso_profit = iso_df['equity'].iloc[-1] - 10000.0
            isolated_total += iso_profit
            print(f"    Esik %{t*100:.1f} | SL %{sl*100:.2f} | TP %{tp*100:.2f} | "
                  f"Izole kar: {iso_profit:.0f} TL")
        except Exception as e:
            print(f"    [HATA] {sym}: {e}")

    if not symbol_data:
        print("[HATA] Hicbir sembol islenemedi.")
        return

    print("\n" + "=" * 60)
    print(f"PORTFOY BACKTEST (paylasimli sermaye, max {MAX_POSITIONS} es zamanli pozisyon)")
    print("=" * 60)
    eq_df, tr_df, summary = run_portfolio_backtest(
        symbol_data, max_positions=MAX_POSITIONS, window_size=WINDOW, allow_short=False)

    print(f"\nBaslangic Sermayesi   : 10.000 TL (TEK havuz, tum hisseler paylasir)")
    print(f"Son Ozkaynak          : {summary['final_equity']:.0f} TL")
    print(f"Net Kar               : {summary['net_profit']:.0f} TL")
    print(f"Max Drawdown          : %{summary['max_drawdown']:.2f}")
    print(f"Toplam Islem          : {summary['total_trades']}")
    print(f"Kazanma Orani         : %{summary['win_rate']:.2f}")
    print(f"Calmar Orani          : {summary['calmar']:.1f}")
    print(f"Es zamanli max pozisyon: {summary['max_concurrent']}")

    print("\n" + "-" * 60)
    print("KARSILASTIRMA")
    print("-" * 60)
    print(f"Izole (her hisse ayri 10.000 TL, {len(symbol_data)} havuz): "
          f"toplam {isolated_total:.0f} TL kar")
    print(f"Portfoy (TEK 10.000 TL havuz, paylasimli): "
          f"{summary['net_profit']:.0f} TL kar ({summary['net_profit']/10000*100:.1f}% getiri)")

    if not tr_df.empty:
        by_sym = tr_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
        print("\nHisse bazinda portfoy PnL katkisi:")
        print(by_sym.to_string())


if __name__ == "__main__":
    main()
