import pandas as pd
from database.db_manager import load_data_as_df

def load_and_fill_gaps(symbol, timeframe='15m'):
    """
    Veriyi SQLite veritabanından çeker ve borsa işlem saatlerine göre 
    oluşan zaman boşluklarını (GAP) tespit edip doldurur.
    """
    print(f"Veri yükleniyor: Veritabanı -> {symbol} ({timeframe})")
    
    # 1. Veriyi veritabanından pandas DataFrame olarak çek
    df = load_data_as_df(symbol, timeframe)
    
    if df.empty:
        raise ValueError(f"[HATA] {symbol} için veritabanında hiç veri bulunamadı!")
    
    # db_manager veriyi çekerken 'date' adında bir indeks oluşturuyor. 
    # API'den gelen bu ham zaman UTC'dir. Bunu İstanbul saatine çeviriyoruz.
    df.index = df.index.tz_localize('UTC').tz_convert('Europe/Istanbul')
    
    # 2. Veriyi gün bazında grupla
    daily_groups = df.groupby(df.index.date)
    
    all_expected_timestamps = []
    
    # Pandas '15m' yerine '15min' formatını sever, bunu otomatik düzeltelim
    pd_freq = timeframe.replace('m', 'min') if 'm' in timeframe else timeframe
    
    # 3. Her bir işlem günü için olması gereken zaman aralığını oluştur
    for date, daily_data in daily_groups:
        start_time = daily_data.index.min() # O günün ilk verisi
        end_time = daily_data.index.max()   # O günün son verisi
        
        # Sadece o günün başlangıç ve bitişi arasında belirlenen periyotta indeks üret
        daily_index = pd.date_range(start=start_time, end=end_time, freq=pd_freq)
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
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].fillna(df_reindexed['close'])
            
    # Hacim boşluklarını 0 ile doldur
    if 'volume' in df_reindexed.columns:
        df_reindexed['volume'] = df_reindexed['volume'].fillna(0)
        
    print("Boşluklar başarıyla dolduruldu.\n")
    return df_reindexed