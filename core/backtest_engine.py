import numpy as np


def run_backtest(predictions, test_df, threshold=0.60, stop_loss=0.01, take_profit=0.02,
                 window_size=60, mode='fixed', initial_capital=10000.0, allow_short=False):
    """
    Event-driven (stateful) backtest motoru.
    Ayni anda tek pozisyon, komisyon dahil.

    predictions : model.predict() ciktisi numpy array, (N, 1) veya (N,)
    test_df     : prepare_lstm_data'dan donen ham test DataFrame'i
    mode        : 'fixed'    -> her islem sabit 10.000 TL ile (dogrulanmis baseline)
                  'compound' -> dinamik cuzdan: her islemde mevcut kasanin tamami reinvest edilir
                                (geometrik/bilesik buyume). equity *= (1 + net_getiri)
    allow_short : False -> sadece uzun (LONG) pozisyon (mevcut/baseline davranis)
                  True  -> simetrik short: prob < (1 - esik) ise asaga bahis (SHORT) acilir.
                           Short getirisi fiyat duserse pozitiftir.

    signal sutunu: +1 (long giris), -1 (short giris), 0 (sinyal yok)
    """
    print(f"\n[BACKTEST] Mod: {mode} | Short: {allow_short} | Esik: %{threshold*100:.2f} | "
          f"SL: %{stop_loss*100:.2f} | TP: %{take_profit*100:.2f}")

    aligned_df = test_df.iloc[window_size:].copy()
    aligned_df['pred_prob'] = predictions.flatten()

    closes = aligned_df['close'].values
    probs = aligned_df['pred_prob'].values

    position_size = 10000.0  # 'fixed' modunda her islemin sabit buyuklugu
    commission = 0.002

    equity = initial_capital
    in_position = False
    entry_price = 0.0
    bars_held = 0
    position_dir = 0  # +1 long, -1 short, 0 flat

    signals = np.zeros(len(closes))
    strat_returns = np.zeros(len(closes))
    equity_curve = np.zeros(len(closes))

    for i in range(len(closes)):
        if not in_position:
            if probs[i] > threshold:
                in_position = True
                position_dir = 1
                entry_price = closes[i]
                signals[i] = 1
                bars_held = 0
            elif allow_short and probs[i] < (1.0 - threshold):
                in_position = True
                position_dir = -1
                entry_price = closes[i]
                signals[i] = -1
                bars_held = 0
        else:
            bars_held += 1
            # Yone gore pozisyon getirisi: long -> +fiyat degisimi, short -> -fiyat degisimi
            price_change = (closes[i] - entry_price) / entry_price
            current_return = price_change * position_dir

            if current_return >= take_profit or current_return <= -stop_loss or bars_held >= 15:
                net_trade_return = current_return - commission
                strat_returns[i] = net_trade_return

                if mode == 'compound':
                    # Dinamik cuzdan: tum kasa reinvest -> geometrik buyume
                    equity *= (1.0 + net_trade_return)
                else:
                    # Sabit kasa (baseline): her islem 10.000 TL, borclanma engelli
                    actual_position = min(position_size, max(0, equity))
                    equity += actual_position * net_trade_return

                in_position = False
                position_dir = 0
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

    long_entries = int((signals == 1).sum())
    short_entries = int((signals == -1).sum())

    print("\n" + "=" * 30)
    print("BACKTEST SONUCLARI (EVENT-DRIVEN)")
    print("=" * 30)
    print(f"Toplam Islem Sayisi   : {total_trades}")
    if allow_short:
        print(f"  Long / Short giris  : {long_entries} / {short_entries}")
    print(f"Kazanma Orani (Win %) : %{win_rate:.2f}")
    print(f"Max Drawdown (Risk)   : %{max_drawdown:.2f}")
    print("-" * 30)
    print(f"Net Kar               : {net_profit:.2f} TL")
    print(f"Son Kasa Bakiyesi     : {equity_curve[-1]:.2f} TL")
    print("=" * 30)

    return aligned_df
