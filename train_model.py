import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.ai_models import build_lstm_model

def train_global_model(timeframe='15m'):
    print("="*60)
    print("🌍 GLOBAL YAPAY ZEKA (SUPER BEYİN) EĞİTİMİ BAŞLIYOR")
    print("="*60)
    
    # Eğitime sokacağımız "Öğretmen" semboller
    # Piyasayı temsil eden en hacimli ve farklı sektörlerden hisseler
    train_symbols = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE']
    model_save_path = 'data/global_lstm_model_15min.keras'
    
    # Tüm hisselerin verilerini toplayacağımız devasa havuzlar
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    print("Adım 1: Her bir sembol izole olarak işleniyor ve paketleniyor...")
    
    for symbol in train_symbols:
        try:
            print(f"  -> İşleniyor: {symbol}...")
            clean_df = load_and_fill_gaps(symbol, timeframe=timeframe)
            clean_df['kalman_close'] = apply_kalman_filter(clean_df['close'])
            ai_df = create_features_and_target(clean_df, lookahead=15, threshold=0.001)
            
            features_to_use = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
            
            # Her sembol kendi içinde bölünür ve ölçeklenir (Scaler)
            X_train, y_train, X_test, y_test, _, _, _ = prepare_lstm_data(
                df=ai_df, 
                feature_cols=features_to_use, 
                window_size=60
            )
            
            # Elde edilen paketleri ana havuza at
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_test_list.append(X_test)
            y_test_list.append(y_test)
            
        except Exception as e:
            print(f"  [!] {symbol} atlandı. Hata: {e}")
            continue

    if not X_train_list:
        print("\n[KRİTİK HATA] Hiçbir sembol işlenemedi. Veritabanını kontrol et.")
        return

    # Adım 2: Toplanan paketleri numpy array ile Mega Veri Seti haline getir
    print("\nAdım 2: Paketler Mega Veri Setine (Mega Dataset) dönüştürülüyor...")
    X_train_global = np.vstack(X_train_list)
    y_train_global = np.concatenate(y_train_list)
    X_test_global = np.vstack(X_test_list)
    y_test_global = np.concatenate(y_test_list)
    
    print(f"  Mega Train Set Boyutu: {X_train_global.shape}")
    print(f"  Mega Test Set Boyutu:  {X_test_global.shape}")

    # Adım 3: Global Sınıf Dengesizliği Hesaplama
    # Tüm hisselerin toplamındaki Alım/Bekleme oranını dengeliyoruz
    classes = np.unique(y_train_global)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_global)
    global_class_weights = dict(zip(classes, weights))
    print(f"  Global Sınıf Ağırlıkları: {global_class_weights}")

    # Adım 4: Model İnşası ve Mega Eğitim
    print("\nAdım 3: LSTM Modeli İnşa Ediliyor ve Eğitim Başlıyor...")
    input_shape = (X_train_global.shape[1], X_train_global.shape[2])
    model = build_lstm_model(input_shape)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
    
    print("\nEğitim Başlıyor! (Kemerlerinizi bağlayın, bu biraz uzun sürebilir...)")
    history = model.fit(
        X_train_global, y_train_global,
        epochs=30, # Mega set olduğu için epoch sayısını biraz artırabiliriz
        batch_size=512, # Veri çok büyük olduğu için batch_size'ı 256'dan 512'ye çıkardık
        validation_data=(X_test_global, y_test_global),
        class_weight=global_class_weights,     
        callbacks=[early_stop, checkpoint],
        verbose=1,
        shuffle=True # FARKLI HİSRELERİN VERİLERİNİ KARIŞTIR (Çok Önemli!)
    )
    
    print(f"\n[BAŞARILI] Global Süper Beyin eğitildi ve kaydedildi: {model_save_path}")

if __name__ == "__main__":
    train_global_model()