import os
from core.data_pipeline import load_and_fill_gaps

def main():
    print("--- CMA-ES Finance Engine Başlatılıyor ---\n")
    
    raw_data_path = 'data/THYAO.csv' 
    processed_data_path = 'data/THYAO_cleaned.csv'
    timeframe = '1min' 
    
    try:
        # 1. Aşama: Veriyi Oku ve Temizle
        clean_df = load_and_fill_gaps(raw_data_path, timeframe=timeframe)
        
        # 2. Aşama: Kaydetmeden önce indeksi Unix Timestamp'e (Milisaniye) dönüştür
        export_df = clean_df.copy()
        # Datetime objesini int64 yapınca nanosaniye döner, 10**6'ya bölüp milisaniyeye çeviriyoruz
        export_df.index = export_df.index.astype('int64') // 10**6
        export_df.index.name = 'timestamp'
        
        # 3. Aşama: Temizlenmiş veriyi kaydet
        export_df.to_csv(processed_data_path)
        print(f"\n[BAŞARILI] Temizlenmiş veri kaydedildi: {processed_data_path}")
        
        # Sonucu kontrol edelim
        print(export_df.head())
        print("\nVeri boyutu:", export_df.shape)
        
    except FileNotFoundError:
        print(f"HATA: {raw_data_path} bulunamadı. Lütfen 'data' klasörüne verinizi ekleyin.")

if __name__ == "__main__":
    main()