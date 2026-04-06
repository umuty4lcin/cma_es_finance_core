import os
from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter

def main():
    print("--- CMA-ES Finance Engine Başlatılıyor ---\n")
    
    raw_data_path = 'data/THYAO.csv' 
    processed_data_path = 'data/THYAO_cleaned.csv'
    timeframe = '1min' 
    
    try:
        # 1. Aşama: Veri Boru Hattı (Data Pipeline)
        print("Adım 1: Veri temizleniyor...")
        clean_df = load_and_fill_gaps(raw_data_path, timeframe=timeframe)
        
        # 2. Aşama: İstatistiksel Sinyal İşleme (Signal Processing)
        print("Adım 2: Sinyal işleme (Kalman Filtresi) uygulanıyor...")
        clean_df['kalman_close'] = apply_kalman_filter(clean_df['close'])
        
        # 3. Aşama: Kaydetmeden önce indeksi Unix Timestamp'e dönüştür
        export_df = clean_df.copy()
        export_df.index = export_df.index.astype('int64') // 10**6
        export_df.index.name = 'timestamp'
        
        # Temizlenmiş ve filtrelenmiş veriyi kaydet
        export_df.to_csv(processed_data_path)
        print(f"\n[BAŞARILI] İşlenmiş veri kaydedildi: {processed_data_path}")
        
        # Sonucu kontrol edelim (Hem normal kapanış hem kalman kapanışı yan yana)
        print(export_df[['close', 'kalman_close']].head(10))
        
    except FileNotFoundError:
        print(f"HATA: {raw_data_path} bulunamadı.")

if __name__ == "__main__":
    main()