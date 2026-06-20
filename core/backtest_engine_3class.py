"""
3-sinifli model icin event-driven backtest motoru (short destegi).

Mevcut 2-sinifli core/backtest_engine.py'dan bagimsiz; her ikisi yan yana yasar.

Karar mantigi:
    P = [P_asagi, P_yatay, P_yukari]  (softmax cikisi)

    Pozisyonda degilken:
      LONG  ac: eger P_yukari > long_threshold
      SHORT ac: eger P_asagi  > short_threshold

    Pozisyondayken: TP / SL / zaman limiti ile cikis (yon-aware).
"""

import numpy as np


def run_backtest_3class(predictions, test_df,
                       long_threshold=0.50, short_threshold=0.50,
                       stop_loss=0.02, take_profit=0.04,
                       window_size=60, time_limit=15,
                       mode='fixed', initial_capital=10000.0):
    """
    predictions : (N, 3) numpy array — softmax olasiliklari [P_asagi, P_yatay, P_yukari]
    test_df     : prepare_lstm_data'dan donen ham test DataFrame
    long_threshold, short_threshold : long / short giris esikleri
    mode        : 'fixed' (her islem 10k) | 'compound' (kasanin tamami reinvest)
    """
    print(f"\n[3CLASS BACKTEST] long_thr={long_threshold:.3f} | short_thr={short_threshold:.3f} | "
          f"SL=%{stop_loss*100:.2f} | TP=%{take_profit*100:.2f} | mode={mode}")

    aligned_df = test_df.iloc[window_size:].copy()
    preds = np.asarray(predictions)
    if preds.shape[1] != 3:
        raise ValueError(f"3-sinifli model bekleniyor; shape={preds.shape}")

    aligned_df['p_down'] = preds[:, 0]
    aligned_df['p_flat'] = preds[:, 1]
    aligned_df['p_up']   = preds[:, 2]

    closes = aligned_df['close'].values
    p_up   = preds[:, 2]
    p_down = preds[:, 0]

    position_size = 10000.0
    commission = 0.002

    equity = initial_capital
    in_position = False
    position_dir = 0   # +1 long, -1 short
    entry_price = 0.0
    bars_held = 0

    signals = np.zeros(len(closes))            # +1 long giris, -1 short giris
    strat_returns = np.zeros(len(closes))
    equity_curve = np.zeros(len(closes))

    for i in range(len(closes)):
        if not in_position:
            # Long fursati once kontrol et, sonra short (ayni anda ikisi de zor)
            if p_up[i] > long_threshold:
                in_position = True
                position_dir = 1
                entry_price = closes[i]
                signals[i] = 1
                bars_held = 0
            elif p_down[i] > short_threshold:
                in_position = True
                position_dir = -1
                entry_price = closes[i]
                signals[i] = -1
                bars_held = 0
        else:
            bars_held += 1
            price_change = (closes[i] - entry_price) / entry_price
            current_return = price_change * position_dir  # short icin ters cevrilir

            if current_return >= take_profit or current_return <= -stop_loss or bars_held >= time_limit:
                net_trade_return = current_return - commission
                strat_returns[i] = net_trade_return

                if mode == 'compound':
                    equity *= (1.0 + net_trade_return)
                else:
                    actual_position = min(position_size, max(0, equity))
                    equity += actual_position * net_trade_return

                in_position = False
                position_dir = 0
                bars_held = 0

        equity_curve[i] = equity

    aligned_df['signal'] = signals
    aligned_df['strategy_return'] = strat_returns
    aligned_df['equity'] = equity_curve

    # Metrikler
    sr = strat_returns
    total_trades = int((sr != 0).sum())
    wins = int((sr > 0).sum())
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    net_profit = equity_curve[-1] - initial_capital

    rolling_max = np.maximum.accumulate(equity_curve)
    drawdown = np.where(rolling_max > 0, (equity_curve - rolling_max) / rolling_max, 0)
    max_drawdown = abs(drawdown.min()) * 100

    long_entries = int((signals == 1).sum())
    short_entries = int((signals == -1).sum())

    print(f"  Trades: {total_trades} (long={long_entries}, short={short_entries}) | "
          f"Win: %{win_rate:.2f} | DD: %{max_drawdown:.2f} | Net: {net_profit:.2f} TL")

    return aligned_df
