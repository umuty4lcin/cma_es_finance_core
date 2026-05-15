import os
import numpy as np
import cma
from tensorflow.keras.models import load_model

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data

def optimize_global_parameters():
    print("="*60)
    print("🧬 GLOBAL CMA-ES OPTİMİZASYONU BAŞLIYOR (SEPET STRATEJİSİ)")
    print("="*60)

    model_path = 'data/global_lstm_model_15min.keras'
    if not os.path.exists(model_path):
        print("[HATA] Global model bulunamadı. Lütfen önce train_model.py'yi çalıştırın.")
        return

    model = load_model(model_path)

    # Optimizasyon Sepeti: Piyasayı yansıtan 3-4 zıt karakterli hisse seçiyoruz
    basket = ['THYAO', 'ASELS', 'GARAN']
    symbol_data = {}

    print("1. Sepetteki hisselerin tahminleri (Olasılıklar) önceden hesaplanıyor...")
    for sym in basket:
        try:
            df = load_and_fill_gaps(sym, timeframe='15min')
            df['kalman_close'] = apply_kalman_filter(df['close'])
            ai_df = create_features_and_target(df, lookahead=15, threshold=0.001)
            
            features = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
            _, _, X_test, _, _, _, test_df = prepare_lstm_data(ai_df, features, window_size=60)
            
            # Eğitimi hızlandırmak için tahminleri 1 kere yapıp hafızada (RAM) tutuyoruz
            preds = model.predict(X_test, verbose=0).flatten()
            symbol_data[sym] = {'preds': preds, 'test_df': test_df}
            print(f"  -> {sym} tahmini hazırlandı.")
        except Exception as e:
            print(f"  [!] {sym} hazırlanamadı: {e}")

    # Sepet bazlı Calmar Oranı hesaplama fonksiyonu
    def global_fitness(params):
        thresh, sl, tp = params[0], params[1], params[2]

        # Evrimsel Sınırlar: Eşiği %45'lere kadar indiriyoruz ki yeni model işlem bulabilsin
        if thresh < 0.45 or thresh > 0.99 or sl < 0.0075 or tp < 0.001:
            return 999999.0

        calmar_scores = []

        for sym, data in symbol_data.items():
            preds = data['preds']
            test_df = data['test_df'].iloc[60:].copy() 
            
            test_df['signal'] = np.where(preds > thresh, 1, 0)
            
            # Eğer bir hissede hiç işleme girmezse algoritmayı cezalandır (Tembellik cezası)
            if test_df['signal'].sum() < 5: 
                calmar_scores.append(-100.0)
                continue

            capped_returns = np.clip(test_df['future_return'], -sl, tp)
            test_df['strat_ret'] = np.where(test_df['signal'] == 1, capped_returns - 0.002, 0)
            
            equity = 10000.0 * (1 + test_df['strat_ret']).cumprod()
            net_profit = equity.iloc[-1] - 10000.0
            
            rolling_max = equity.cummax()
            drawdown = (equity - rolling_max) / rolling_max
            max_dd = abs(drawdown.min()) * 100

            if net_profit <= 0:
                calmar_scores.append(net_profit) # Zararı puan olarak yansıt
            else:
                calmar = net_profit / (max_dd + 1.0)
                calmar_scores.append(calmar)

        # Sepetteki tüm hisselerin ORTALAMA Calmar skorunu döndür (Minimize için negatif)
        avg_calmar = np.mean(calmar_scores)
        return -avg_calmar 

    # Başlangıç Genleri: Yeni modelin nabzına göre ayarlandı
    initial_params = [0.49, 0.015, 0.10]
    sigma0 = 0.02
    es = cma.CMAEvolutionStrategy(initial_params, sigma0, {'popsize': 20, 'maxiter': 30, 'verbose': -9})

    generation = 1
    print("\n2. Evrimsel Süreç Başlıyor (Ortalama Calmar Oranı Hedefleniyor)...")
    while not es.stop():
        solutions = es.ask()
        scores = [global_fitness(x) for x in solutions]
        es.tell(solutions, scores)
        
        best_idx = np.argmin(scores)
        best_p = solutions[best_idx]
        best_score = -scores[best_idx]
        
        if best_score != -999999.0:
            print(f"Nesil {generation:02d} | Ort. Sepet Skoru: {best_score:>7.2f} | "
                  f"Genler -> Eşik: %{best_p[0]*100:.1f}, SL: %{best_p[1]*100:.2f}, TP: %{best_p[2]*100:.2f}")
        generation += 1

    best_params = es.result.xbest
    print("\n" + "="*50)
    print("🏆 GLOBAL OPTİMİZASYON TAMAMLANDI!")
    print(f"Yeni Altın Oranlar -> Eşik: %{best_params[0]*100:.2f} | SL: %{best_params[1]*100:.2f} | TP: %{best_params[2]*100:.2f}")
    print("Lütfen bu oranları main.py içindeki OPT_THRESH, OPT_SL, OPT_TP değişkenlerine yazın!")
    print("="*50)

if __name__ == "__main__":
    optimize_global_parameters()