import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df, feature_cols, target_col='target', window_size=20, train_ratio=0.8):
    """
    Veriyi LSTM ağının anlayacağı 3 Boyutlu (Örneklem, Zaman Adımı, Öznitelik) hale getirir
    ve Eğitim/Test olarak böler.
    
    - window_size: Modelin karar vermek için geçmişe dönük kaç dakikaya bakacağı (Örn: Son 60 dakika)
    - train_ratio: Verinin yüzde kaçının eğitim için kullanılacağı (Kronolojik bölünür)
    """
    
    # 1. Eğitim ve Test verisini KRONOLOJİK olarak böl.
    # (Finansal verilerde asla rastgele bölme (shuffle) yapılmaz, yoksa model geleceği önceden görür)
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # 2. Sadece modele girecek öznitelikleri (Features) seç ve Ölçeklendir (0 ile 1 arasına al)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scaler'ı SADECE eğitim verisiyle eğitiyoruz (Train set sızıntısını önlemek için)
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols]) # Test verisini sadece dönüştürüyoruz
    
    # Hedef (Y) etiketlerini numpy dizisine alalım
    train_y = train_df[target_col].values
    test_y = test_df[target_col].values
    
    # 3. Kayan Pencere (Sliding Window) Algoritması
    # Veriyi LSTM'in beklediği [Samples, TimeSteps, Features] formatına dönüştürür.
    def create_sequences(X_data, y_data, window):
        X_seq, y_seq = [], []
        for i in range(window, len(X_data)):
            # Son 'window' kadar satırı al ve bir blok yap
            X_seq.append(X_data[i-window:i])
            # Bu bloğun hemen sonrasındaki hedef etiketi al
            y_seq.append(y_data[i])
        return np.array(X_seq), np.array(y_seq)
        
    X_train, y_train = create_sequences(train_scaled, train_y, window_size)
    X_test, y_test = create_sequences(test_scaled, test_y, window_size)
    
    # Sınıf Dengesizliği (Class Imbalance) için ağırlıkları hesapla.
    # Hem 2-sinifli (0/1) hem cok-sinifli (0/1/2 ...) durumlari destekler.
    unique_classes = np.unique(y_train)
    n_samples = len(y_train)
    class_weights = {}
    for c in unique_classes:
        count = np.sum(y_train == c)
        # sklearn 'balanced' formulu ile ayni: n_samples / (n_classes * count)
        class_weights[int(c)] = (1 / count) * (n_samples / len(unique_classes))

    print("\n--- Veri Bölünme Raporu ---")
    print(f"Eğitim Seti (Train): {len(X_train)} blok")
    print(f"Test Seti (Test): {len(X_test)} blok")
    print(f"Sınıf Ağırlıkları: {class_weights}")
    print(f"LSTM Girdi Şekli (X_train): {X_train.shape} -> (Satır Sayısı, Zaman Adımı, Öznitelik Sayısı)")

    return X_train, y_train, X_test, y_test, scaler, class_weights, test_df