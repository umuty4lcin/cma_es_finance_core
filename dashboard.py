import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
from tensorflow.keras.models import load_model

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.backtest_engine import run_backtest

st.set_page_config(page_title="Hibrit Algo-Trading", layout="wide")

st.title("Hibrit Algoritmik Ticaret Paneli")
st.markdown("Global Yapay Zeka Modeli (LSTM) ve Yerel Risk Yonetimi Simulatoru")
st.markdown("---")

st.sidebar.header("Simulasyon Ayarlari")
symbol = st.sidebar.selectbox(
    "Hisse Senedi Secin",
    ["FROTO", "TUPRS", "ASELS", "THYAO", "GARAN", "ISCTR", "HALKB", "VAKBN", "SASA", "SISE"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Parametreleri**")

threshold = st.sidebar.slider("Guven Esigi (Threshold) %", 45.0, 90.0, 50.0, 0.1) / 100.0
stop_loss = st.sidebar.slider("Zarar Kes (Stop-Loss) %", 0.1, 15.0, 1.5, 0.1) / 100.0
take_profit = st.sidebar.slider("Kar Al (Take-Profit) %", 1.0, 30.0, 15.0, 0.1) / 100.0

if st.sidebar.button("Simulasyonu Baslat"):
    with st.spinner(f'{symbol} verileri veritabanindan cekiliyor ve yapay zeka analiz yapiyor...'):
        try:
            model_path = 'data/global_lstm_model_15min.keras'
            if not os.path.exists(model_path):
                st.error("[HATA] Global model bulunamadi! Lutfen once train_model.py calistirin.")
                st.stop()

            model = load_model(model_path)

            df = load_and_fill_gaps(symbol, timeframe='15m')
            df['kalman_close'] = apply_kalman_filter(df['close'])
            ai_df = create_features_and_target(df, lookahead=15, threshold=0.001)

            features = [
                'close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m'
            ]
            _, _, X_test, _, _, _, test_df = prepare_lstm_data(ai_df, features, window_size=60)

            predictions = model.predict(X_test, verbose=0)

            results_df = run_backtest(
                predictions, test_df,
                threshold=threshold, stop_loss=stop_loss, take_profit=take_profit, window_size=60
            )

            initial_capital = 10000.0
            final_equity = results_df['equity'].iloc[-1]
            net_profit = final_equity - initial_capital

            exit_bars = results_df[results_df['strategy_return'] != 0]
            total_trades = len(exit_bars)
            win_rate = (len(exit_bars[exit_bars['strategy_return'] > 0]) / total_trades * 100) if total_trades > 0 else 0.0

            rolling_max = results_df['equity'].cummax()
            drawdowns = np.where(rolling_max > 0, (results_df['equity'] - rolling_max) / rolling_max, 0)
            max_drawdown = abs(drawdowns.min()) * 100

            st.success("Simulasyon Basariyla Tamamlandi!")

            col1, col2, col3, col4 = st.columns(4)

            profit_color = "normal" if net_profit >= 0 else "inverse"
            col1.metric(
                label="Net Kar (TL)",
                value=f"{net_profit:.2f} TL",
                delta=f"% {(net_profit / initial_capital) * 100:.2f}",
                delta_color=profit_color
            )
            col2.metric(label="Islem Sayisi", value=f"{total_trades} Adet")
            col3.metric(label="Kazanma Orani", value=f"% {win_rate:.2f}")
            col4.metric(label="Max Risk (Drawdown)", value=f"% {max_drawdown:.2f}", delta="Risk", delta_color="inverse")

            st.markdown("### Kasa Buyume Egrisi (Equity Curve)")
            fig = px.line(
                results_df, x=results_df.index, y='equity',
                title=f"{symbol} Portfoy Performansi (Baslangic: 10.000 TL)"
            )
            fig.update_layout(
                xaxis_title="Zaman (BIST Islem Saatleri)",
                yaxis_title="Kasa Bakiyesi (TL)",
                template="plotly_dark",
                hovermode="x unified"
            )
            fig.add_hline(y=10000, line_dash="dash", line_color="gray", annotation_text="Baslangic Sermayesi")

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Sistem bir hata ile karsilasti: {e}")
