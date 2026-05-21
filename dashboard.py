import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from tensorflow.keras.models import load_model

# --- Kendi Çekirdek Modüllerimiz ---
from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.backtest_engine import run_vectorized_backtest

# Sayfa ayarları (Geniş ekran)
st.set_page_config(page_title="Hibrit Algo-Trading", layout="wide", page_icon="📈")

st.title("Hibrit Algoritmik Ticaret Paneli")
st.markdown("Global Yapay Zeka Modeli (LSTM) ve Yerel Risk Yönetimi Simülatörü")
st.markdown("---")

# --- YAN PANEL (SIDEBAR) ---
st.sidebar.header("Simülasyon Ayarları")
symbol = st.sidebar.selectbox("Hisse Senedi Seçin", 
                              ["FROTO", "TUPRS", "ASELS", "THYAO", "GARAN", "ISCTR", "HALKB", "VAKBN", "SASA", "SISE"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Parametreleri**\n\n*(Jüriye göstermek için buradaki değerlerle oynayabilirsiniz)*")

# Optimizasyonla bulduğumuz değerleri manuel test etmek için kaydırıcılar
threshold = st.sidebar.slider("Güven Eşiği (Threshold) %", 45.0, 90.0, 50.0, 0.1) / 100.0
stop_loss = st.sidebar.slider("Zarar Kes (Stop-Loss) %", 0.1, 15.0, 1.5, 0.1) / 100.0
take_profit = st.sidebar.slider("Kâr Al (Take-Profit) %", 1.0, 30.0, 15.0, 0.1) / 100.0

# --- ANA EKRAN VE SİMÜLASYON ---
if st.sidebar.button("Simülasyonu Başlat"):
    # Yükleme animasyonu
    with st.spinner(f'{symbol} verileri veritabanından çekiliyor ve yapay zeka analiz yapıyor...'):
        try:
            # 1. Modeli Yükle
            model_path = 'data/global_lstm_model_15min.keras'
            if not os.path.exists(model_path):
                st.error("[HATA] Global model bulunamadı! Lütfen önce train_model.py çalıştırın.")
                st.stop()
            
            model = load_model(model_path)

            # 2. Veriyi Hazırla
            df = load_and_fill_gaps(symbol, timeframe='15m')
            df['kalman_close'] = apply_kalman_filter(df['close'])
            ai_df = create_features_and_target(df, lookahead=15, threshold=0.001)

            # SADECE 5 TEMEL ÖZELLİK (Klasik indikatörleri sildiğimiz temiz versiyon)
            features = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
            _, _, X_test, _, _, _, test_df = prepare_lstm_data(ai_df, features, window_size=60)

            # 3. Gerçekçi Backtest'i Çalıştır
            results_df = run_vectorized_backtest(
                model, X_test, test_df,
                threshold=threshold, stop_loss=stop_loss, take_profit=take_profit, window_size=60
            )

            # 4. Metrikleri Hesapla
            initial_capital = 10000.0
            final_equity = results_df['equity'].iloc[-1]
            net_profit = final_equity - initial_capital
            
            # Sadece çıkış yapılan barları say (Kazanma oranı için)
            exit_bars = results_df[results_df['strategy_return'] != 0]
            total_trades = len(exit_bars)
            if total_trades > 0:
                win_rate = (len(exit_bars[exit_bars['strategy_return'] > 0]) / total_trades) * 100
            else:
                win_rate = 0.0

            rolling_max = results_df['equity'].cummax()
            drawdowns = np.where(rolling_max > 0, (results_df['equity'] - rolling_max) / rolling_max, 0)
            max_drawdown = abs(drawdowns.min()) * 100

            # 5. KPI (Anahtar Performans Göstergeleri) Kartlarını Çizdir
            st.success("Simülasyon Başarıyla Tamamlandı!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Kâr durumuna göre renk (Yeşil veya Kırmızı)
            profit_color = "normal" if net_profit >= 0 else "inverse"
            col1.metric(label="Net Kâr (TL)", value=f"{net_profit:.2f} ₺", delta=f"% {(net_profit/initial_capital)*100:.2f}", delta_color=profit_color)
            col2.metric(label="İşlem Sayısı", value=f"{total_trades} Adet")
            col3.metric(label="Kazanma Oranı", value=f"% {win_rate:.2f}")
            col4.metric(label="Max Risk (Drawdown)", value=f"% {max_drawdown:.2f}", delta="Risk", delta_color="inverse")

            # 6. Kasa Büyüme Grafiğini Çizdir (Plotly)
            st.markdown("### Kasa Büyüme Eğrisi (Equity Curve)")
            fig = px.line(results_df, x=results_df.index, y='equity', 
                          title=f"{symbol} Portföy Performansı (Başlangıç: 10.000 TL)")
            
            # Grafiği güzelleştirme
            fig.update_layout(
                xaxis_title="Zaman (BIST İşlem Saatleri)",
                yaxis_title="Kasa Bakiyesi (TL)",
                template="plotly_dark",
                hovermode="x unified"
            )
            # Başlangıç çizgisini (10.000 TL) yatay olarak belirt
            fig.add_hline(y=10000, line_dash="dash", line_color="gray", annotation_text="Başlangıç Sermayesi")
            
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Sistem bir hata ile karşılaştı: {e}")