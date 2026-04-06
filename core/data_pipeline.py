import pandas as pd

def load_and_fill_gaps(file_path, timeframe='1min'):
    print(f"Veri yükleniyor: {file_path}")
    
    # 1. Veriyi oku ve Timestamp'i İstanbul saatine çevir
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Istanbul')
    df.set_index('timestamp', inplace=True)
    
    # 2. Veriyi gün bazında grupla
    daily_groups = df.groupby(df.index.date)
    
    all_expected_timestamps = []
    
    # 3. Her bir işlem günü için olması gereken zaman aralığını oluştur
    for date, daily_data in daily_groups:
        start_time = daily_data.index.min() # O günün ilk verisi
        end_time = daily_data.index.max()   # O günün son verisi
        
        # Sadece o günün başlangıç ve bitişi arasında 1 dakikalık indeks üret
        daily_index = pd.date_range(start=start_time, end=end_time, freq=timeframe)
        all_expected_timestamps.extend(daily_index)
        
    # 4. Tüm günlerin indekslerini tek bir yapıda birleştir
    full_bist_index = pd.DatetimeIndex(all_expected_timestamps)
    
    # Eksik mum sayısını hesapla
    missing_candles_count = len(full_bist_index) - len(df)
    print(f"Tatiller ve yarım günler hariç tutularak tespit edilen eksik mum (GAP) sayısı: {missing_candles_count}")
    
    # 5. DataFrame'i yeni ve kusursuz indekse göre yeniden boyutlandır
    df_reindexed = df.reindex(full_bist_index)
    
    # 6. Fiyat (Close) boşluklarını bir önceki fiyat ile doldur (Forward Fill)
    df_reindexed['close'] = df_reindexed['close'].ffill()
    
    # Open, High ve Low boşluklarını Close ile aynı yap
    for col in ['open', 'high', 'low']:
        df_reindexed[col] = df_reindexed[col].fillna(df_reindexed['close'])
        
    # Hacim boşluklarını 0 ile doldur
    if 'volume' in df_reindexed.columns:
        df_reindexed['volume'] = df_reindexed['volume'].fillna(0)
        
    print("Boşluklar başarıyla dolduruldu.\n")
    return df_reindexed