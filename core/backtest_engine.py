# # import numpy as np

# # def run_vectorized_backtest(model, X_test, test_df, threshold=0.60, stop_loss=0.01, take_profit=0.02, window_size=60):
# #     print(f"\n[SİMÜLASYON] Eşik: %{threshold*100:.2f} | SL: %{stop_loss*100:.2f} | TP: %{take_profit*100:.2f}")
    
# #     predictions = model.predict(X_test, verbose=0)
# #     aligned_df = test_df.iloc[window_size:].copy()
# #     aligned_df['pred_prob'] = predictions.flatten()
    
# #     # 1. Sinyal Üretimi
# #     aligned_df['signal'] = np.where(aligned_df['pred_prob'] > threshold, 1, 0)
# #     commission = 0.002
    
# #     # 2. Vektörel Risk Yönetimi (Clip)
# #     capped_returns = np.clip(aligned_df['future_return'], -stop_loss, take_profit)
    
# #     aligned_df['strategy_return'] = np.where(
# #         aligned_df['signal'] == 1,
# #         capped_returns - commission,
# #         0
# #     )
    
# #     # --- YENİ: SABİT KASA (FIXED POSITION) YÖNETİMİ ---
# #     initial_capital = 10000.0
# #     position_size = 10000.0 # Sistem kâr etse bile her işleme sadece 10.000 TL'lik büyüklükle girer
    
# #     # Her işlemin TL bazında net kâr/zararını (PnL) hesapla
# #     aligned_df['trade_pnl'] = aligned_df['strategy_return'] * position_size
    
# #     # Kasayı kümülatif çarpım (cumprod) ile değil, kümülatif toplam (cumsum) ile büyüt
# #     aligned_df['equity'] = initial_capital + aligned_df['trade_pnl'].cumsum()
    
# #     # Metrikler
# #     total_trades = aligned_df['signal'].sum()
# #     winning_trades = len(aligned_df[(aligned_df['signal'] == 1) & (aligned_df['strategy_return'] > 0)])
# #     win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
# #     net_profit = aligned_df['equity'].iloc[-1] - initial_capital
    
# #     rolling_max = aligned_df['equity'].cummax()
    
# #     # Sıfıra bölme hatasını önlemek için güvenlik önlemi
# #     drawdown = np.where(rolling_max > 0, (aligned_df['equity'] - rolling_max) / rolling_max, 0)
# #     max_drawdown = drawdown.min() * 100
    
# #     print("\n" + "="*30)
# #     print("🚀 FİNAL BACKTEST SONUÇLARI")
# #     print("="*30)
# #     print(f"Toplam İşlem Sayısı   : {total_trades}")
# #     print(f"Kazanma Oranı (Win %) : %{win_rate:.2f}")
# #     print(f"Max Drawdown (Risk)   : %{max_drawdown:.2f}")
# #     print("-" * 30)
# #     print(f"Net Kâr               : {net_profit:.2f} TL")
# #     print(f"Son Kasa Bakiyesi     : {aligned_df['equity'].iloc[-1]:.2f} TL")
# #     print("="*30)
    
# #     return aligned_df

import numpy as np

def run_vectorized_backtest(model, X_test, test_df, threshold=0.60, stop_loss=0.01, take_profit=0.02, window_size=60):
    print(f"\n[SİMÜLASYON] Eşik: %{threshold*100:.2f} | SL: %{stop_loss*100:.2f} | TP: %{take_profit*100:.2f}")
    
    predictions = model.predict(X_test, verbose=0)
    aligned_df = test_df.iloc[window_size:].copy()
    aligned_df['pred_prob'] = predictions.flatten()
    
    # Numpy array'lere çeviriyoruz (Hız için)
    closes = aligned_df['close'].values
    probs = aligned_df['pred_prob'].values
    
    # Gerçekçi takip değişkenleri (Stateful Variables)
    initial_capital = 10000.0
    position_size = 10000.0
    commission = 0.002
    
    equity = initial_capital
    in_position = False
    entry_price = 0.0
    bars_held = 0
    
    # Sonuçları tutacağımız listeler
    signals = np.zeros(len(closes))
    strat_returns = np.zeros(len(closes))
    equity_curve = np.zeros(len(closes))
    
    # --- GERÇEK DÜNYA OLAY DÖNGÜSÜ (EVENT-DRIVEN) ---
    for i in range(len(closes)):
        if not in_position:
            # İşlemde değiliz, alım kovalıyoruz
            if probs[i] > threshold:
                in_position = True
                entry_price = closes[i]
                signals[i] = 1 # Sadece girdiğimiz anı 1 işaretle
                bars_held = 0
        else:
            # İçerideyiz, satım/çıkış kovalıyoruz
            bars_held += 1
            current_return = (closes[i] - entry_price) / entry_price
            
            # 3 Çıkış Şartı: Take-Profit vurdu, Stop-Loss vurdu, veya AI'nin limiti (15 mum) doldu
            if current_return >= take_profit or current_return <= -stop_loss or bars_held >= 15:
                # İşlemi Kapat
                net_trade_return = current_return - commission
                strat_returns[i] = net_trade_return 
                
                # YENİ: Kasada ne kadar varsa (maksimum 10000) onunla işleme gir. Bakiye eksiyse işlem yapma.
                actual_position = min(position_size, max(0, equity))
                equity += (actual_position * net_trade_return)
                
                in_position = False
                bars_held = 0
                
        # Her mumda güncel kasayı kaydet
        equity_curve[i] = equity

    # Hesaplanan gerçekçi verileri DataFrame'e yaz
    aligned_df['signal'] = signals
    aligned_df['strategy_return'] = strat_returns
    aligned_df['equity'] = equity_curve
    
    # Gerçekçi Metrikler
    total_trades = int(aligned_df['signal'].sum())
    winning_trades = len(aligned_df[aligned_df['strategy_return'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    net_profit = aligned_df['equity'].iloc[-1] - initial_capital
    
    rolling_max = aligned_df['equity'].cummax()
    drawdown = np.where(rolling_max > 0, (aligned_df['equity'] - rolling_max) / rolling_max, 0)
    max_drawdown = drawdown.min() * 100
    
    print("\n" + "="*30)
    print("🚀 FİNAL BACKTEST SONUÇLARI (GERÇEKÇİ)")
    print("="*30)
    print(f"Toplam İşlem Sayısı   : {total_trades}")
    print(f"Kazanma Oranı (Win %) : %{win_rate:.2f}")
    print(f"Max Drawdown (Risk)   : %{max_drawdown:.2f}")
    print("-" * 30)
    print(f"Net Kâr               : {net_profit:.2f} TL")
    print(f"Son Kasa Bakiyesi     : {aligned_df['equity'].iloc[-1]:.2f} TL")
    print("="*30)
    
    return aligned_df