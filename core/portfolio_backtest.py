"""
Portfoy (Coklu Es-Zamanli Pozisyon) Backtest Motoru

Tek-sembol backtest_engine.py'nin aksine, bu motor BIRDEN FAZLA hisseyi
ORTAK bir sermaye havuzu ve ORTAK zaman cizelgesi uzerinde es zamanli
simule eder. Ayni anda en fazla `max_positions` pozisyon acik olabilir;
her pozisyon paylasimli kasadan pay alir.

Temel mantik:
  - Tum sembollerin test verileri zaman damgasina gore birlestirilir (master timeline).
  - Her zaman adiminda once acik pozisyonlarin cikis sartlari kontrol edilir,
    sonra bos slot varsa yeni giris sinyalleri degerlendirilir.
  - Kasa (cash) + acik pozisyonlarin anlik degeri = toplam ozkaynak (equity).
  - Borclanma yok: yeni pozisyon en fazla mevcut nakit kadar buyuk olabilir.

Bu motor backtest_engine.py'yi DEGISTIRMEZ; ayri bir yetenek olarak eklenir.
"""

import numpy as np
import pandas as pd


def run_portfolio_backtest(symbol_data, max_positions=3, initial_capital=10000.0,
                           commission=0.002, window_size=60, time_limit=15,
                           sizing='fixed', allow_short=False):
    """
    symbol_data : dict
        {
          'GARAN': {'predictions': np.array, 'test_df': pd.DataFrame,
                    'threshold': 0.52, 'stop_loss': 0.02, 'take_profit': 0.04},
          ...
        }
        predictions, prepare_lstm_data'dan gelen X_test ile ayni uzunlukta olmali.
        test_df, prepare_lstm_data'nin dondurdugu ham test DataFrame'i olmali.

    max_positions : ayni anda acik olabilecek en fazla pozisyon sayisi
    sizing        : 'fixed'   -> her pozisyon initial_capital/max_positions kadar
                    'dynamic' -> her pozisyon (mevcut ozkaynak)/max_positions kadar
    allow_short   : True ise prob < (1 - esik) oldugunda short acilir

    Donus: (equity_curve_df, trades_df, summary_dict)
    """
    # 1) Her sembol icin zaman damgasi -> (close, prob, params) tablosu kur
    per_symbol = {}
    all_timestamps = set()
    for sym, d in symbol_data.items():
        aligned = d['test_df'].iloc[window_size:].copy()
        preds = np.asarray(d['predictions']).flatten()
        if len(preds) != len(aligned):
            raise ValueError(f"{sym}: tahmin uzunlugu ({len(preds)}) ile "
                             f"hizalanmis veri ({len(aligned)}) uyusmuyor.")
        aligned['prob'] = preds
        # Hizli erisim icin dict: ts -> (close, prob)
        lookup = {ts: (c, p) for ts, c, p in
                  zip(aligned.index, aligned['close'].values, aligned['prob'].values)}
        per_symbol[sym] = {
            'lookup': lookup,
            'threshold': d['threshold'],
            'stop_loss': d['stop_loss'],
            'take_profit': d['take_profit'],
        }
        all_timestamps.update(aligned.index)

    master = sorted(all_timestamps)
    if not master:
        raise ValueError("Master zaman cizelgesi bos.")

    # 2) Durum
    cash = initial_capital
    positions = {}   # sym -> dict(entry_price, dir, bars_held, size, last_close)
    trades = []
    equity_curve = []

    fixed_slot = initial_capital / max_positions

    for ts in master:
        # --- (a) Acik pozisyonlari yonet (cikis kontrolu) ---
        for sym in list(positions.keys()):
            if ts not in per_symbol[sym]['lookup']:
                continue  # bu sembolun bu zaman damgasinda mumu yok -> tut
            close, _ = per_symbol[sym]['lookup'][ts]
            pos = positions[sym]
            pos['last_close'] = close
            pos['bars_held'] += 1
            price_change = (close - pos['entry_price']) / pos['entry_price']
            cur_ret = price_change * pos['dir']

            sl = per_symbol[sym]['stop_loss']
            tp = per_symbol[sym]['take_profit']
            if cur_ret >= tp or cur_ret <= -sl or pos['bars_held'] >= time_limit:
                net_ret = cur_ret - commission
                cash += pos['size'] * (1.0 + net_ret)  # pozisyonu kapat, nakde don
                trades.append({'symbol': sym, 'exit_ts': ts, 'dir': pos['dir'],
                               'net_return': net_ret, 'pnl': pos['size'] * net_ret,
                               'bars_held': pos['bars_held']})
                del positions[sym]

        # --- (b) Yeni giris sinyalleri (bos slot varsa) ---
        for sym in symbol_data.keys():
            if len(positions) >= max_positions:
                break
            if sym in positions:
                continue
            if ts not in per_symbol[sym]['lookup']:
                continue
            close, prob = per_symbol[sym]['lookup'][ts]
            thr = per_symbol[sym]['threshold']

            direction = 0
            if prob > thr:
                direction = 1
            elif allow_short and prob < (1.0 - thr):
                direction = -1
            if direction == 0:
                continue

            slot = fixed_slot if sizing == 'fixed' else (cash + _open_value(positions)) / max_positions
            size = min(slot, cash)
            if size <= 0:
                continue
            cash -= size
            positions[sym] = {'entry_price': close, 'dir': direction,
                              'bars_held': 0, 'size': size, 'last_close': close}

        # --- (c) Anlik ozkaynak (nakit + acik pozisyon degeri) ---
        equity = cash + _open_value(positions)
        equity_curve.append({'timestamp': ts, 'equity': equity,
                             'cash': cash, 'open_positions': len(positions)})

    # 3) Raporlama
    eq_df = pd.DataFrame(equity_curve).set_index('timestamp')
    tr_df = pd.DataFrame(trades)

    final_equity = eq_df['equity'].iloc[-1]
    net_profit = final_equity - initial_capital
    rolling_max = eq_df['equity'].cummax()
    dd = np.where(rolling_max > 0, (eq_df['equity'] - rolling_max) / rolling_max, 0)
    max_dd = abs(dd.min()) * 100
    total_trades = len(tr_df)
    win_rate = (tr_df['net_return'] > 0).mean() * 100 if total_trades > 0 else 0.0
    calmar = net_profit / (max_dd + 1.0)

    summary = {
        'final_equity': final_equity, 'net_profit': net_profit, 'max_drawdown': max_dd,
        'total_trades': total_trades, 'win_rate': win_rate, 'calmar': calmar,
        'max_concurrent': int(eq_df['open_positions'].max()),
    }
    return eq_df, tr_df, summary


def _open_value(positions):
    """Acik pozisyonlarin anlik (mark-to-market) toplam degeri."""
    total = 0.0
    for pos in positions.values():
        price_change = (pos['last_close'] - pos['entry_price']) / pos['entry_price']
        cur_ret = price_change * pos['dir']
        total += pos['size'] * (1.0 + cur_ret)
    return total
