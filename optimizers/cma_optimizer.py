# import cma
# import numpy as np
# import pandas as pd

# # def fast_backtest_evaluator(params, predictions, test_df, window_size=60, commission=0.002):
# #     threshold, stop_loss, take_profit = params[0], params[1], params[2]
    
# #     # Genetik Sınırlar (Mutasyon mantık dışına çıkarsa cezalandır)
# #     if threshold < 0.50 or threshold > 0.99 or stop_loss < 0.001 or take_profit < 0.001:
# #         return 999999.0  
        
# #     aligned_df = test_df.iloc[window_size:].copy()
# #     aligned_df['pred_prob'] = predictions.flatten()
# #     aligned_df['signal'] = np.where(aligned_df['pred_prob'] > threshold, 1, 0)
    
# #     total_trades = aligned_df['signal'].sum()
# #     if total_trades < 5: # Çok tembel botları engelle
# #         return 999999.0
        
# #     capped_returns = np.clip(aligned_df['future_return'], -stop_loss, take_profit)
    
# #     aligned_df['strategy_return'] = np.where(
# #         aligned_df['signal'] == 1,
# #         capped_returns - commission,
# #         0
# #     )
    
# #     equity = 10000.0 * (1 + aligned_df['strategy_return']).cumprod()
# #     net_profit = equity.iloc[-1] - 10000.0
    
# #     return -net_profit
# def fast_backtest_evaluator(params, predictions, test_df, window_size=60, commission=0.002):
#     threshold, stop_loss, take_profit = params[0], params[1], params[2]
    
#     # Genetik Sınırlar
#     if threshold < 0.50 or threshold > 0.99 or stop_loss < 0.001 or take_profit < 0.001:
#         return 999999.0  
        
#     aligned_df = test_df.iloc[window_size:].copy()
#     aligned_df['pred_prob'] = predictions.flatten()
#     aligned_df['signal'] = np.where(aligned_df['pred_prob'] > threshold, 1, 0)
    
#     total_trades = aligned_df['signal'].sum()
#     if total_trades < 1: # 5 yerine 1 yaptık çünkü algoritma işlem bulmakta zorlanıyordu algoritma biraz nefes alsın
#         return 999999.0
        
#     capped_returns = np.clip(aligned_df['future_return'], -stop_loss, take_profit)
    
#     aligned_df['strategy_return'] = np.where(
#         aligned_df['signal'] == 1,
#         capped_returns - commission,
#         0
#     )
    
#     equity = 10000.0 * (1 + aligned_df['strategy_return']).cumprod()
#     net_profit = equity.iloc[-1] - 10000.0
    
#     # --- YENİ EKLENEN RİSK YÖNETİMİ (DRAWDOWN) ---
#     rolling_max = equity.cummax()
#     drawdown = (equity - rolling_max) / rolling_max
#     max_drawdown = abs(drawdown.min()) * 100
    
#     # 1. KURAL: Zarar eden modelleri doğrudan ele
#     if net_profit <= 0:
#         return -net_profit
        
#     # 2. KURAL (KIRMIZI ÇİZGİ): Drawdown %15'i geçerse, ne kadar kâr ederse etsin modeli çöpe at!
#     if max_drawdown > 15.0:
#         return 999999.0
        
#     # 3. KURAL (OPTİMİZASYON HEDEFİ): Kârı, alınan riske böl (Return over Max Drawdown - RoMD)
#     # Algoritma artık kârı maksimize etmeye değil, bu oranı maksimize etmeye çalışacak.
#     risk_adjusted_score = net_profit / (max_drawdown + 0.1) 
    
#     return -risk_adjusted_score

# def run_cma_optimization(predictions, test_df, window_size=60):
#     print("\n" + "="*50)
#     print("🧬 3 BOYUTLU CMA-ES EVRİMİ BAŞLIYOR (Threshold, SL, TP)")
#     print("="*50)
    
#     # Başlangıç Genleri: [Threshold: %70, Stop-Loss: %0.5, Take-Profit: %1.5]
#     #initial_params = [0.70, 0.05, 0.015] 
#     #sigma0 = 0.05 
#     #değiştirdik çünkü algoritma işlem bulmakta zorlanıyordu. Eşiği biraz düşürdük, SL ve TP'yi biraz artırdık.

#     # Başlangıç Genleri: [Threshold: %52, Stop-Loss: %2, Take-Profit: %5]
#     # Eşiği 0.70'den 0.52'ye çektik ki algoritma işlem bulabilsin.
#     initial_params = [0.52, 0.02, 0.05] 
    
#     # Standart sapmayı (arama adım büyüklüğünü) biraz küçülttük
#     sigma0 = 0.02
    
#     es = cma.CMAEvolutionStrategy(initial_params, sigma0, {'popsize': 20, 'maxiter': 30, 'verbose': -9})
    
#     generation = 1
#     while not es.stop():
#         solutions = es.ask()
#         fitness_scores = [fast_backtest_evaluator(x, predictions, test_df, window_size) for x in solutions]
#         es.tell(solutions, fitness_scores)
        
#         best_idx = np.argmin(fitness_scores)
#         best_p = solutions[best_idx]
#         best_profit = -fitness_scores[best_idx]
        
#         # Ceza yemiş (999999) sonuçları ekrana basma
#         if best_profit != -999999.0:
#             print(f"Nesil {generation:02d} | Kâr: {best_profit:>7.2f} TL | "
#                   f"Genler -> Eşik: %{best_p[0]*100:.1f}, SL: %{best_p[1]*100:.2f}, TP: %{best_p[2]*100:.2f}")
#         generation += 1
        
#     best_params = es.result.xbest
    
#     print("="*50)
#     print("🏆 OPTİMİZASYON TAMAMLANDI!")
#     print("="*50)
    
#     return best_params[0], best_params[1], best_params[2]

import cma
import numpy as np
import pandas as pd

def fast_backtest_evaluator(params, predictions, test_df, window_size=60, commission=0.002):
    threshold, stop_loss, take_profit = params[0], params[1], params[2]
    
    # Sınırların dışına çıkarsa öldür
    if threshold < 0.50 or threshold > 0.99 or stop_loss < 0.001 or take_profit < 0.001:
        return 999999.0  
        
    aligned_df = test_df.iloc[window_size:].copy()
    aligned_df['pred_prob'] = predictions.flatten()
    aligned_df['signal'] = np.where(aligned_df['pred_prob'] > threshold, 1, 0)
    
    # En az 10 işlem yapsın ki şans eseri kâr edenleri eleyelim
    if aligned_df['signal'].sum() < 10: 
        return 999999.0
        
    capped_returns = np.clip(aligned_df['future_return'], -stop_loss, take_profit)
    
    aligned_df['strategy_return'] = np.where(
        aligned_df['signal'] == 1,
        capped_returns - commission,
        0
    )
    
    equity = 10000.0 * (1 + aligned_df['strategy_return']).cumprod()
    net_profit = equity.iloc[-1] - 10000.0
    
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100
    
    # --- YENİ PROFESYONEL FİTNESS (CALMAR ORANI) ---
    if net_profit <= 0:
        # Zarar ediyorsa zararı direkt döndür (minimize etmesi için abs alıyoruz)
        return abs(net_profit) 
        
    # Eğer kâr varsa, Kâr / Risk oranını hesapla (Risk ne kadar küçükse oran o kadar artar)
    # Sıfıra bölme hatası olmasın diye +1.0 ekliyoruz
    calmar_ratio = net_profit / (max_drawdown + 1.0)
    
    # CMA-ES her zaman "en küçük" değeri aradığı için en yüksek Calmar oranını eksiyle döndürüyoruz
    return -calmar_ratio

def run_cma_optimization(predictions, test_df, window_size=60):
    print("\n" + "="*50)
    print("🧬 3 BOYUTLU CMA-ES EVRİMİ BAŞLIYOR (Calmar Oranı Aktif)")
    print("="*50)
    
    # Başlangıç noktasını, az önce 4380 TL bulduğu o "altın madenine" yakın kuruyoruz ki 
    # aramaya o kârlı bölgeden başlayıp riskleri törpülesin.
    initial_params = [0.505, 0.015, 0.15] 
    sigma0 = 0.02 
    
    es = cma.CMAEvolutionStrategy(initial_params, sigma0, {'popsize': 20, 'maxiter': 30, 'verbose': -9})
    
    generation = 1
    while not es.stop():
        solutions = es.ask()
        fitness_scores = [fast_backtest_evaluator(x, predictions, test_df, window_size) for x in solutions]
        es.tell(solutions, fitness_scores)
        
        best_idx = np.argmin(fitness_scores)
        best_p = solutions[best_idx]
        best_score = -fitness_scores[best_idx]
        
        if best_score != -999999.0:
            print(f"Nesil {generation:02d} | Calmar Skoru: {best_score:>7.2f} | "
                  f"Genler -> Eşik: %{best_p[0]*100:.1f}, SL: %{best_p[1]*100:.2f}, TP: %{best_p[2]*100:.2f}")
        generation += 1
        
    best_params = es.result.xbest
    
    print("="*50)
    print("🏆 OPTİMİZASYON TAMAMLANDI!")
    print("="*50)
    
    return best_params[0], best_params[1], best_params[2]