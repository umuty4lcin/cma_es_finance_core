# import numpy as np
# import pandas as pd

# def run_vectorized_backtest(model, X_test, test_df, threshold=0.60, window_size=60):
#     """
#     Eğitilmiş modelin test verisi üzerindeki tahminlerini kullanarak sanal bir portföy yönetir.
    
#     Parametreler:
#     - threshold: İşleme girmek için gereken minimum AI güven seviyesi (Örn: %60 eminse AL)
#     """
#     print(f"\nModel test verisi üzerinde tahminler yapıyor (Güven Eşiği: %{threshold*100})...")
    
#     # 1. Modelin tahminlerini al (0 ile 1 arası olasılıklar döner)
#     predictions = model.predict(X_test)
    
#     # 2. Test verisini tahminlerle hizala (İlk 60 mumu veri bloklamak için kullandığımızdan kesiyoruz)
#     aligned_df = test_df.iloc[window_size:].copy()
#     aligned_df['pred_prob'] = predictions.flatten()
    
#     # 3. Alım Sinyalleri Üret 
#     # (Eğer olasılık threshold'dan büyükse 1 (AL), değilse 0 (BEKLE))
#     aligned_df['signal'] = np.where(aligned_df['pred_prob'] > threshold, 1, 0)
    
#     # 4. Kâr/Zarar ve Komisyon Hesaplama
#     # Borsa İstanbul için ortalama komisyon oranını binde 2 (0.002) varsayalım (Alış+Satış)
#     commission = 0.002
    
#     # Sinyal varsa (1 ise), o anki 'future_return' (gelecekteki getiri) alınır ve komisyon düşülür.
#     aligned_df['strategy_return'] = np.where(
#         aligned_df['signal'] == 1,
#         aligned_df['future_return'] - commission,
#         0
#     )
    
#     # 5. Kümülatif Büyüme (Sermaye Eğrisi)
#     initial_capital = 10000.0 # 10.000 TL başlangıç sermayesi
#     aligned_df['equity'] = initial_capital * (1 + aligned_df['strategy_return']).cumprod()
    
#     # --- PROFESYONEL PERFORMANS METRİKLERİ ---
#     total_trades = aligned_df['signal'].sum()
#     winning_trades = len(aligned_df[(aligned_df['signal'] == 1) & (aligned_df['strategy_return'] > 0)])
    
#     win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
#     net_profit = aligned_df['equity'].iloc[-1] - initial_capital
    
#     # Maksimum Düşüş (Max Drawdown - Kasanın zirveden ne kadar eridiği)
#     rolling_max = aligned_df['equity'].cummax()
#     drawdown = (aligned_df['equity'] - rolling_max) / rolling_max
#     max_drawdown = drawdown.min() * 100
    
#     print("\n" + "="*30)
#     print("🚀 BACKTEST SONUÇLARI")
#     print("="*30)
#     print(f"Başlangıç Sermayesi   : {initial_capital:.2f} TL")
#     print(f"Toplam İşlem Sayısı   : {total_trades}")
#     print(f"Kazanma Oranı (Win %) : %{win_rate:.2f}")
#     print(f"Max Drawdown (Risk)   : %{max_drawdown:.2f}")
#     print("-" * 30)
#     print(f"Net Kâr               : {net_profit:.2f} TL")
#     print(f"Son Kasa Bakiyesi     : {aligned_df['equity'].iloc[-1]:.2f} TL")
#     print("="*30)
    
#     return aligned_df

import numpy as np
import pandas as pd

def run_vectorized_backtest(model, X_test, test_df, threshold=0.60, stop_loss=0.01, take_profit=0.02, window_size=60):
    print(f"\n[SİMÜLASYON] Eşik: %{threshold*100:.2f} | SL: %{stop_loss*100:.2f} | TP: %{take_profit*100:.2f}")
    
    predictions = model.predict(X_test, verbose=0)
    aligned_df = test_df.iloc[window_size:].copy()
    aligned_df['pred_prob'] = predictions.flatten()
    
    # 1. Sinyal Üretimi
    aligned_df['signal'] = np.where(aligned_df['pred_prob'] > threshold, 1, 0)
    commission = 0.002
    
    # 2. Vektörel Risk Yönetimi (Clip)
    # Eğer getiri SL'den kötüyse zararı kes (-stop_loss), TP'den iyiyse kârı al (take_profit)
    capped_returns = np.clip(aligned_df['future_return'], -stop_loss, take_profit)
    
    aligned_df['strategy_return'] = np.where(
        aligned_df['signal'] == 1,
        capped_returns - commission,
        0
    )
    
    # Kümülatif Kasa Hesaplama
    initial_capital = 10000.0
    aligned_df['equity'] = initial_capital * (1 + aligned_df['strategy_return']).cumprod()
    
    # Metrikler
    total_trades = aligned_df['signal'].sum()
    winning_trades = len(aligned_df[(aligned_df['signal'] == 1) & (aligned_df['strategy_return'] > 0)])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    net_profit = aligned_df['equity'].iloc[-1] - initial_capital
    
    rolling_max = aligned_df['equity'].cummax()
    drawdown = (aligned_df['equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    print("\n" + "="*30)
    print("🚀 FİNAL BACKTEST SONUÇLARI")
    print("="*30)
    print(f"Toplam İşlem Sayısı   : {total_trades}")
    print(f"Kazanma Oranı (Win %) : %{win_rate:.2f}")
    print(f"Max Drawdown (Risk)   : %{max_drawdown:.2f}")
    print("-" * 30)
    print(f"Net Kâr               : {net_profit:.2f} TL")
    print(f"Son Kasa Bakiyesi     : {aligned_df['equity'].iloc[-1]:.2f} TL")
    print("="*30)
    
    return aligned_df