import cma
import numpy as np

def fast_backtest_evaluator(params, predictions, test_df, window_size=60, commission=0.002):
    threshold, stop_loss, take_profit = params[0], params[1], params[2]
    
    # 1. Genetik Sınırlar (Mutasyon mantık dışına çıkarsa öldür)
    if threshold < 0.45 or threshold > 0.99 or stop_loss < 0.0075 or take_profit < 0.015:
        return 999999.0  
        
    aligned_df = test_df.iloc[window_size:].copy()
    closes = aligned_df['close'].values
    probs = predictions.flatten()
    
    # Gerçekçi takip değişkenleri (Stateful Variables)
    initial_capital = 10000.0
    position_size = 10000.0
    
    equity = initial_capital
    in_position = False
    entry_price = 0.0
    bars_held = 0
    trade_count = 0
    
    equity_curve = np.zeros(len(closes))
    
    # --- GERÇEK DÜNYA OLAY DÖNGÜSÜ (EVENT-DRIVEN) ---
    for i in range(len(closes)):
        if not in_position:
            # İşlemde değiliz, Eşik (Threshold) geçilirse alım yap
            if probs[i] > threshold:
                in_position = True
                entry_price = closes[i]
                bars_held = 0
                trade_count += 1
        else:
            # İçerideyiz, satım kovalıyoruz
            bars_held += 1
            current_return = (closes[i] - entry_price) / entry_price
            
            # 3 Çıkış Şartı: TP, SL veya 15 Mum Limiti
            if current_return >= take_profit or current_return <= -stop_loss or bars_held >= 15:
                net_trade_return = current_return - commission
                
                # Kasada ne kadar para varsa onunla işleme gir (Eksi bakiyeyi engelle)
                actual_position = min(position_size, max(0, equity))
                equity += (actual_position * net_trade_return)
                
                in_position = False
                bars_held = 0
                
        # Kasayı her mumda kaydet
        equity_curve[i] = equity

    # Ceza 1: Eğer çok az işlem yapmışsa (tembelse) veya kasa sıfırlanmışsa
    if trade_count < 10 or equity <= 0: 
        return 999999.0

    # Risk (Drawdown) Hesaplama
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = np.where(rolling_max > 0, (equity_curve - rolling_max) / rolling_max, 0)
    max_drawdown = abs(drawdowns.min()) * 100
    net_profit = equity - initial_capital
    
    # --- YENİ PROFESYONEL FİTNESS (CALMAR ORANI) ---
    if net_profit <= 0:
        return abs(net_profit) # Zararı ceza puanı olarak döndür
        
    calmar_ratio = net_profit / (max_drawdown + 1.0)
    return -calmar_ratio # CMA-ES minimize ettiği için kârı (Calmar'ı) eksiyle döndürüyoruz

def run_cma_optimization(predictions, test_df, window_size=60):
    print("\n" + "="*50)
    print("🧬 3 BOYUTLU CMA-ES EVRİMİ BAŞLIYOR (Gerçekçi Calmar Oranı)")
    print("="*50)
    
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
            print(f"Nesil {generation:02d} | Gerçek Calmar Skoru: {best_score:>7.2f} | "
                  f"Genler -> Eşik: %{best_p[0]*100:.1f}, SL: %{best_p[1]*100:.2f}, TP: %{best_p[2]*100:.2f}")
        generation += 1
        
    best_params = es.result.xbest
    
    print("="*50)
    print("🏆 GERÇEKÇİ OPTİMİZASYON TAMAMLANDI!")
    print("="*50)
    
    return best_params[0], best_params[1], best_params[2]