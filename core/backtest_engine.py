import numpy as np


def run_backtest(predictions, test_df, threshold=0.60, stop_loss=0.01, take_profit=0.02, window_size=60):
    """
    Event-driven (stateful) backtest motoru.
    Ayni anda tek pozisyon, sabit 10.000 TL kasa, komisyon dahil.

    predictions : model.predict() ciktisi numpy array, (N, 1) veya (N,)
    test_df     : prepare_lstm_data'dan donen ham test DataFrame'i
    """
    print(f"\n[BACKTEST] Esik: %{threshold*100:.2f} | SL: %{stop_loss*100:.2f} | TP: %{take_profit*100:.2f}")

    aligned_df = test_df.iloc[window_size:].copy()
    aligned_df['pred_prob'] = predictions.flatten()

    closes = aligned_df['close'].values
    probs = aligned_df['pred_prob'].values

    initial_capital = 10000.0
    position_size = 10000.0
    commission = 0.002

    equity = initial_capital
    in_position = False
    entry_price = 0.0
    bars_held = 0

    signals = np.zeros(len(closes))
    strat_returns = np.zeros(len(closes))
    equity_curve = np.zeros(len(closes))

    for i in range(len(closes)):
        if not in_position:
            if probs[i] > threshold:
                in_position = True
                entry_price = closes[i]
                signals[i] = 1
                bars_held = 0
        else:
            bars_held += 1
            current_return = (closes[i] - entry_price) / entry_price

            if current_return >= take_profit or current_return <= -stop_loss or bars_held >= 15:
                net_trade_return = current_return - commission
                strat_returns[i] = net_trade_return

                actual_position = min(position_size, max(0, equity))
                equity += actual_position * net_trade_return

                in_position = False
                bars_held = 0

        equity_curve[i] = equity

    aligned_df['signal'] = signals
    aligned_df['strategy_return'] = strat_returns
    aligned_df['equity'] = equity_curve

    # Metrikler: cikis barlarina gore tutarli sayim
    exit_mask = strat_returns != 0
    total_trades = int(exit_mask.sum())
    winning_trades = int((strat_returns > 0).sum())
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    net_profit = equity_curve[-1] - initial_capital

    rolling_max = aligned_df['equity'].cummax()
    drawdown = np.where(rolling_max > 0, (aligned_df['equity'] - rolling_max) / rolling_max, 0)
    max_drawdown = abs(drawdown.min()) * 100

    print("\n" + "=" * 30)
    print("BACKTEST SONUCLARI (EVENT-DRIVEN)")
    print("=" * 30)
    print(f"Toplam Islem Sayisi   : {total_trades}")
    print(f"Kazanma Orani (Win %) : %{win_rate:.2f}")
    print(f"Max Drawdown (Risk)   : %{max_drawdown:.2f}")
    print("-" * 30)
    print(f"Net Kar               : {net_profit:.2f} TL")
    print(f"Son Kasa Bakiyesi     : {equity_curve[-1]:.2f} TL")
    print("=" * 30)

    return aligned_df
