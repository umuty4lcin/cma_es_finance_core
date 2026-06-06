import os
import numpy as np
import pandas as pd
import openpyxl
from tensorflow.keras.models import load_model

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.backtest_engine import run_backtest
from optimizers.cma_optimizer import run_cma_optimization


def save_to_excel(results_list, filename='backtest_sonuclari_v2.xlsx'):
    df = pd.DataFrame(results_list)
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Performans Raporu', index=False)
        worksheet = writer.sheets['Performans Raporu']
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + i)].width = column_len
    print(f"\n[BASARILI] Tum sonuclar '{filename}' dosyasina kaydedildi.")


def main():
    print("=" * 50)
    print("BIREYSEL OPTIMIZASYONLU BACKTEST MOTORU BASLATILIYOR")
    print("=" * 50)

    symbol_list = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE', 'FROTO']

    model_path = 'data/global_lstm_model_15min.keras'

    if not os.path.exists(model_path):
        print(f"\n[KRITIK HATA] Egitilmis model ({model_path}) bulunamadi!")
        return

    print("Super Beyin (Global Model) yukleniyor...\n")
    best_model = load_model(model_path)
    results_summary = []

    features_to_use = [
        'close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m'
    ]

    for symbol in symbol_list:
        print(f"\n>>> {symbol} ICIN OZEL ANALIZ VE EVRIM BASLIYOR <<<")

        try:
            clean_df = load_and_fill_gaps(symbol, timeframe='15m')
            clean_df['kalman_close'] = apply_kalman_filter(clean_df['close'])
            ai_df = create_features_and_target(clean_df, lookahead=15, threshold=0.001)

            X_train, y_train, X_test, y_test, scaler, class_weights, test_df = prepare_lstm_data(
                df=ai_df, feature_cols=features_to_use, window_size=60
            )

            # Tahmin bir kez yapilir; hem CMA-ES hem backtest bu diziyi kullanir
            print(f"[{symbol}] Model tahmini yapiliyor...")
            predictions = best_model.predict(X_test, verbose=0)

            print(f"[{symbol}] Kendi dinamiklerine gore genetik optimizasyon yapiliyor (Bu biraz surebilir)...")
            opt_thresh, opt_sl, opt_tp = run_cma_optimization(predictions, test_df, window_size=60)
            print(f"[{symbol} OZEL ORANLARI] Esik: %{opt_thresh*100:.2f} | SL: %{opt_sl*100:.2f} | TP: %{opt_tp*100:.2f}")

            results_df = run_backtest(
                predictions, test_df,
                threshold=opt_thresh, stop_loss=opt_sl, take_profit=opt_tp, window_size=60
            )

            net_profit = results_df['equity'].iloc[-1] - 10000.0
            rolling_max = results_df['equity'].cummax()
            max_drawdown = abs(((results_df['equity'] - rolling_max) / rolling_max).min()) * 100

            exit_bars = results_df[results_df['strategy_return'] != 0]
            total_trades = len(exit_bars)
            win_rate = (len(exit_bars[exit_bars['strategy_return'] > 0]) / total_trades * 100) if total_trades > 0 else 0.0

            results_summary.append({
                'Sembol': symbol,
                'Opt. Esik (%)': round(opt_thresh * 100, 2),
                'Opt. SL (%)': round(opt_sl * 100, 2),
                'Opt. TP (%)': round(opt_tp * 100, 2),
                'Net Kar (TL)': round(net_profit, 2),
                'Max Risk (%)': round(max_drawdown, 2),
                'Islem Sayisi': total_trades,
                'Kazanma Orani (%)': round(win_rate, 2)
            })
            print(f"[{symbol} TAMAM] Kar: {net_profit:.2f} TL | Risk: %{max_drawdown:.2f}")

        except Exception as e:
            print(f"[HATA] {symbol} islenirken sorun olustu: {e}")

    print("\n" + "=" * 50)
    print("HIBRIT PERFORMANS RAPORU (Global Beyin + Yerel Risk)")
    print("=" * 50)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    save_to_excel(results_summary)


if __name__ == "__main__":
    main()
