# import os
# from core.data_pipeline import load_and_fill_gaps
# from core.signal_filters import apply_kalman_filter
# from core.feature_engineering import create_features_and_target
# from core.ai_prep import prepare_lstm_data
# from core.ai_models import build_lstm_model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# def main():
#     print("--- CMA-ES Finance Engine Başlatılıyor ---\n")
    
#     raw_data_path = 'data/15m/THYAO.csv' 
#     timeframe = '15min' 
    
#     try:
#         # Önceki Adımlar
#         clean_df = load_and_fill_gaps(raw_data_path, timeframe=timeframe)
#         clean_df['kalman_close'] = apply_kalman_filter(clean_df['close'])
#         ai_df = create_features_and_target(clean_df, lookahead=15, threshold=0.001)
        
#         features_to_use = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
        
#         X_train, y_train, X_test, y_test, scaler, class_weights, test_df = prepare_lstm_data(
#             df=ai_df, 
#             feature_cols=features_to_use, 
#             window_size=60
#         )
        
#         # --- YENİ ADIM: YAPAY ZEKA EĞİTİMİ ---
#         print("\nAdım 5: LSTM Modeli İnşa Ediliyor ve Eğitim Başlıyor...")
        
#         # Girdi şeklini (60, 5) olarak modele veriyoruz
#         input_shape = (X_train.shape[1], X_train.shape[2])
#         model = build_lstm_model(input_shape)
        
#         # Callbacks (Eğitim sırasındaki akıllı asistanlar)
#         # 1. EarlyStopping: Eğer model 3 tur (patience) boyunca test verisinde gelişme göstermezse eğitimi durdur (zaman tasarrufu).
#         early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
#         # 2. ModelCheckpoint: En iyi ağırlıkları fiziksel olarak bilgisayarına kaydeder.
#         checkpoint = ModelCheckpoint('data/best_lstm_model_15min.keras', monitor='val_loss', save_best_only=True)
        
#         # Model Eğitimi (Fit)
#         # Not: Bilgisayarının gücüne göre bu işlem biraz uzun sürebilir.
#         print("\nEğitim Başlıyor! (Epochs: 20, Batch Size: 256)")
#         history = model.fit(
#             X_train, y_train,
#             epochs=20,                      # Verinin üzerinden 20 kez geçilecek
#             batch_size=256,                 # Her seferinde 256 satırlık paketler halinde okuyacak
#             validation_data=(X_test, y_test), # Modelin hiç görmediği test verisiyle kendini sınaması
#             class_weight=class_weights,     # Dengesiz veriyi eşitlemek için verdiğimiz ağırlıklar
#             callbacks=[early_stop, checkpoint],
#             verbose=1                       # Terminalde ilerleme çubuğunu (progress bar) gösterir
#         )
        
#         print("\n[BAŞARILI] Eğitim tamamlandı ve en iyi model 'data/best_lstm_model_15min.keras' olarak kaydedildi.")
        
#     except Exception as e:
#         print(f"HATA OLUŞTU: {e}")

# if __name__ == "__main__":
#     main()

# import os
# from core.data_pipeline import load_and_fill_gaps
# from core.signal_filters import apply_kalman_filter
# from core.feature_engineering import create_features_and_target
# from core.ai_prep import prepare_lstm_data
# from core.backtest_engine import run_vectorized_backtest
# from optimizers.cma_optimizer import run_cma_optimization  # YENİ MODÜL
# from tensorflow.keras.models import load_model

# def main():
#     print("--- CMA-ES Finance Engine Başlatılıyor ---\n")
#     raw_data_path = 'data/15m/THYAO.csv' 
#     model_path = 'data/best_lstm_model_15min.keras'
#     timeframe = '15min' 
    
#     try:
#         clean_df = load_and_fill_gaps(raw_data_path, timeframe=timeframe)
#         clean_df['kalman_close'] = apply_kalman_filter(clean_df['close'])
#         ai_df = create_features_and_target(clean_df, lookahead=15, threshold=0.001)
        
#         features_to_use = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
        
#         X_train, y_train, X_test, y_test, scaler, class_weights, test_df = prepare_lstm_data(
#             df=ai_df, 
#             feature_cols=features_to_use, 
#             window_size=60
#         )
        
#         print(f"\nAdım 6: Kaydedilmiş en iyi model yükleniyor ({model_path})...")
#         if os.path.exists(model_path):
#             best_model = load_model(model_path)
            
#             # 1. HIZLI TAHMİN (Sadece 1 kere çalışır)
#             print("Yapay zeka test verisi üzerinde tahminlerini (olasılıkları) hesaplıyor...")
#             predictions = best_model.predict(X_test)
            
#             # 2. CMA-ES OPTİMİZASYONU
#             # Bize en yüksek kârı getirecek olan "Threshold" değerini bulacak
#             opt_thresh, opt_sl, opt_tp = run_cma_optimization(predictions, test_df, window_size=60)
            
#             # 3. FİNAL BACKTEST
#             # Bulunan bu mükemmel eşik değeriyle nihai sonuçları görelim
#             print("\nEn İyi Parametrelerle Final Simülasyonu Çalıştırılıyor...")
#             results_df = run_vectorized_backtest(
#                 best_model, X_test, test_df, 
#                 threshold=opt_thresh, 
#                 stop_loss=opt_sl, 
#                 take_profit=opt_tp, 
#                 window_size=60
#             )
            
#         else:
#             print(f"HATA: {model_path} bulunamadı.")
            
#     except Exception as e:
#         print(f"HATA OLUŞTU: {e}")

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import pandas as pd
import openpyxl
from tensorflow.keras.models import load_model

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.backtest_engine import run_vectorized_backtest

def main():
    print("="*50)
    print("🚀 ÇOKLU SEMBOL BACKTEST MOTORU BAŞLATILIYOR")
    print("="*50)
    
    # 1. TEST EDİLECEK SEMBOLLERİ BURAYA YAZ (Dosya adlarıyla aynı olmalı)
    # Örneğin data/15m/ klasöründe GARAN.csv ve ASELS.csv varsa:
    symbol_list = ['ASELS', 'GARAN', 'HALKB', 'ISCTR','THYAO', 'TUPRS', 'VAKBN']
    
    # THYAO'dan elde ettiğimiz o altın oranlar
    OPT_THRESH = 0.5002
    OPT_SL = 0.0011
    OPT_TP = 0.1263
    
    # Eğitilmiş ortak beynimiz
    model_path = 'data/best_lstm_model_15min.keras'
    
    if not os.path.exists(model_path):
        print(f"HATA: Eğitilmiş model ({model_path}) bulunamadı!")
        return

    print("🧠 Eğitilmiş ana model yükleniyor...\n")
    best_model = load_model(model_path)
    
    # Sonuçları toplu görmek için bir tablo oluşturacağız
    results_summary = []

    # 2. SEMBOLLER ÜZERİNDE DÖNGÜ (Döngü her bir sembol için sırayla çalışır)
    for symbol in symbol_list:
        raw_data_path = f'data/15m/{symbol}.csv'
        
        # Eğer klasörde o isimde bir dosya yoksa atla
        if not os.path.exists(raw_data_path):
            print(f"[UYARI] {symbol}.csv bulunamadı, atlanıyor...")
            continue
            
        print(f"\n>>> {symbol} İÇİN ANALİZ BAŞLIYOR <<<")
        
        try:
            # Veri Hazırlığı
            clean_df = load_and_fill_gaps(raw_data_path, timeframe='15min')
            clean_df['kalman_close'] = apply_kalman_filter(clean_df['close'])
            ai_df = create_features_and_target(clean_df, lookahead=15, threshold=0.001)
            
            features_to_use = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
            
            # Veriyi modele uygun hale getir (Scaler vs.)
            X_train, y_train, X_test, y_test, scaler, class_weights, test_df = prepare_lstm_data(
                df=ai_df, 
                feature_cols=features_to_use, 
                window_size=60
            )
            
            # Sadece X_test (hiç görülmemiş veri) üzerinde simülasyon yapacağız
            results_df = run_vectorized_backtest(
                best_model, X_test, test_df, 
                threshold=OPT_THRESH, 
                stop_loss=OPT_SL, 
                take_profit=OPT_TP, 
                window_size=60
            )
            
            # Sonuçları listeye kaydet
            net_profit = results_df['equity'].iloc[-1] - 10000.0
            rolling_max = results_df['equity'].cummax()
            drawdown = (results_df['equity'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) * 100
            total_trades = results_df['signal'].sum()
            win_rate = (results_df[results_df['signal'] == 1]['strategy_return'] > 0).mean() * 100
            
            results_summary.append({
                'Sembol': symbol,
                'Net Kar (TL)': round(net_profit, 2),
                'Max Risk (%)': round(max_drawdown, 2),
                'Islem Sayisi': total_trades,
                'Kazanma Orani': round(win_rate, 2)
            })
            
            print(f"[{symbol} TAMAMLANDI] Kâr: {net_profit:.2f} TL | Risk: %{max_drawdown:.2f}")

        except Exception as e:
            print(f"[HATA] {symbol} işlenirken sorun oluştu: {e}")

    # 3. FİNAL RAPORUNU EKRANA BAS
    print("\n" + "="*50)
    print("📊 GENEL PERFORMANS RAPORU (Out-of-Market Test)")
    print("="*50)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    save_to_excel(results_summary)


def save_to_excel(results_list, filename='backtest_sonuclari.xlsx'):
    """
    Backtest sonuçlarını profesyonel bir Excel dosyasına kaydeder.
    """
    df = pd.DataFrame(results_list)
    
    # Excel dosyasını oluştur
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Performans Raporu', index=False)
        
        # Sütun genişliklerini otomatik ayarla
        worksheet = writer.sheets['Performans Raporu']
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65+i)].width = column_len

    print(f"\n[BAŞARILI] Tüm sonuçlar '{filename}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()