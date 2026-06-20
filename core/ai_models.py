import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TensorFlow'un gereksiz uyarılarını gizler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_lstm_model(input_shape):
    """
    Algoritmik ticaret için optimize edilmiş LSTM Sinir Ağı Mimarisi.
    """
    model = Sequential()
    
    # Girdi Katmanı (Zaman Adımı, Öznitelik Sayısı) -> Örn: (60, 5)
    model.add(Input(shape=input_shape))
    
    # 1. LSTM Katmanı
    # return_sequences=True yapıyoruz çünkü bir sonraki LSTM katmanına zincirleme veri aktaracağız.
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2)) # Aşırı öğrenmeyi (overfitting) engellemek için nöronların %20'sini rastgele kapatır
    
    # 2. LSTM Katmanı
    # return_sequences=False yapıyoruz çünkü artık diziyi tek bir sonuca özetleyeceğiz.
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Tam Bağlantılı (Dense) Katman - Karar aşaması
    model.add(Dense(units=16, activation='relu'))
    
    # Çıkış Katmanı
    # Sigmoid fonksiyonu 0 ile 1 arasında bir "olasılık" üretir. (Örn: %85 ihtimalle yön yukarı)
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Modeli Derleme
    # binary_crossentropy: Sadece 1 ve 0 olan sınıflandırmalar için en iyi kayıp (hata) hesaplama yöntemidir.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_lstm_3class_model(input_shape, n_classes=3):
    """
    3-sinifli yon siniflandirmasi icin LSTM mimarisi (short destegi).

    Cikis: softmax ile [P_asagi, P_yatay, P_yukari] olasiliklari (toplam = 1).

    2-sinifli build_lstm_model ile ayni govde; sadece cikis katmani ve kayip
    fonksiyonu cok-sinifli icin uyarlandi.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=16, activation='relu'))

    # Cikis: cok-sinifli softmax
    model.add(Dense(units=n_classes, activation='softmax'))

    # sparse_categorical_crossentropy: y'nin one-hot olmasi gerekmez, tam sayi etiketler yeterli
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model