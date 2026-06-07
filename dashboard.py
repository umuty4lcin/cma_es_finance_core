import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from tensorflow.keras.models import load_model

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.backtest_engine import run_backtest
from core.portfolio_backtest import run_portfolio_backtest
from optimizers.cma_optimizer import run_cma_optimization
from core.viz import compute_metrics, gauge, price_volume_chart, equity_drawdown_chart

MODEL_PATH = 'data/global_lstm_model_15min.keras'
FEATURES = ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m']
WINDOW = 60
INITIAL_CAPITAL = 10000.0
ALL_SYMBOLS = ["FROTO", "TUPRS", "ASELS", "THYAO", "GARAN", "ISCTR", "HALKB", "VAKBN", "SASA", "SISE"]

st.set_page_config(page_title="Hibrit Algo-Trading", layout="wide")


# ----------------------------- Onbellekli yardimcilar -----------------------------
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_symbol_data(symbol):
    """Sembol verisini hazirlar ve tahminleri uretir (onbellekli -> tekrar tekrar hesaplanmaz)."""
    df = load_and_fill_gaps(symbol, timeframe='15m')
    df['kalman_close'] = apply_kalman_filter(df['close'])
    ai_df = create_features_and_target(df, lookahead=15, threshold=0.001)
    _, _, X_test, _, _, _, test_df = prepare_lstm_data(ai_df, FEATURES, window_size=WINDOW)
    model = get_model()
    predictions = model.predict(X_test, verbose=0).flatten()
    return predictions, test_df


# ----------------------------- Baslik -----------------------------
st.title("Hibrit Algoritmik Ticaret Paneli")
st.caption("Global LSTM Tahmin Modeli + CMA-ES Evrimsel Risk Optimizasyonu | BIST 15dk")

model = get_model()
if model is None:
    st.error("[HATA] Global model bulunamadi! Lutfen once train_model.py calistirin.")
    st.stop()

# ----------------------------- Yan panel -----------------------------
st.sidebar.header("Simulasyon Ayarlari")

mode_label = st.sidebar.radio(
    "Cuzdan Modu",
    ["Sabit Kasa (10.000 TL)", "Dinamik Cuzdan (Bilesik)"],
    help="Sabit: her islem 10.000 TL ile. Dinamik: mevcut kasanin tamami reinvest edilir (geometrik buyume)."
)
mode = 'compound' if mode_label.startswith("Dinamik") else 'fixed'

allow_short = st.sidebar.checkbox("Short Islem (Aciga Satis)", value=False,
                                  help="Acik: prob < (1 - esik) oldugunda asaga bahis (short) acilir.")

st.sidebar.markdown("---")
auto_opt = st.sidebar.checkbox("Otomatik Optimize Et (CMA-ES)", value=True,
                               help="Acik: CMA-ES en iyi parametreleri evrimle bulur. Kapali: manuel kaydiraclar.")

if not auto_opt:
    st.sidebar.markdown("**Manuel Risk Parametreleri**")
    threshold = st.sidebar.slider("Guven Esigi (Threshold) %", 45.0, 90.0, 50.0, 0.1) / 100.0
    stop_loss = st.sidebar.slider("Zarar Kes (Stop-Loss) %", 0.1, 15.0, 1.5, 0.1) / 100.0
    take_profit = st.sidebar.slider("Kar Al (Take-Profit) %", 1.0, 30.0, 15.0, 0.1) / 100.0
else:
    threshold = stop_loss = take_profit = None

# ----------------------------- Sekmeler -----------------------------
tab_single, tab_portfolio = st.tabs(["Tek Hisse Analizi", "Portfoy Karsilastirma"])

# ============================ SEKME 1: TEK HISSE ============================
with tab_single:
    symbol = st.selectbox("Hisse Senedi Secin", ALL_SYMBOLS)

    if st.button("Simulasyonu Baslat", type="primary"):
        with st.spinner(f'{symbol} analiz ediliyor...'):
            try:
                predictions, test_df = get_symbol_data(symbol)

                if auto_opt:
                    with st.spinner("CMA-ES evrimsel optimizasyon calisiyor (30 nesil)..."):
                        t, sl, tp = run_cma_optimization(predictions.reshape(-1, 1), test_df, window_size=WINDOW)
                else:
                    t, sl, tp = threshold, stop_loss, take_profit

                results_df = run_backtest(predictions.reshape(-1, 1), test_df,
                                          threshold=t, stop_loss=sl, take_profit=tp,
                                          window_size=WINDOW, mode=mode, allow_short=allow_short)
                m = compute_metrics(results_df)

                st.success("Simulasyon tamamlandi!")
                st.info(f"Kullanilan Parametreler -> Esik: %{t*100:.2f} | SL: %{sl*100:.2f} | "
                        f"TP: %{tp*100:.2f} | Mod: {mode}")

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Net Kar (TL)", f"{m['net_profit']:.0f}",
                          delta=f"%{(m['net_profit']/INITIAL_CAPITAL)*100:.1f}",
                          delta_color="normal" if m['net_profit'] >= 0 else "inverse")
                c2.metric("Islem Sayisi", f"{m['total_trades']}")
                c3.metric("Kazanma Orani", f"%{m['win_rate']:.1f}")
                c4.metric("Max Risk", f"%{m['max_dd']:.1f}", delta="Drawdown", delta_color="inverse")
                c5.metric("Calmar Orani", f"{m['calmar']:.1f}")

                # Gauge gostergeleri
                g1, g2, g3 = st.columns(3)
                g1.plotly_chart(gauge(m['win_rate'], "Kazanma Orani", 0, 100, "%", True),
                                use_container_width=True)
                g2.plotly_chart(gauge(min(m['calmar'], 500), "Calmar Orani", 0, 500, "", True),
                                use_container_width=True)
                g3.plotly_chart(gauge(m['max_dd'], "Max Drawdown", 0, 100, "%", False),
                                use_container_width=True)

                # Candlestick + Kalman + sinyaller + hacim
                st.plotly_chart(price_volume_chart(results_df, symbol), use_container_width=True)

                # Kasa + drawdown paneli
                st.plotly_chart(equity_drawdown_chart(results_df), use_container_width=True)

                # LSTM olasilik dagilimi
                st.markdown("#### LSTM Yukselis Olasiligi Dagilimi")
                hist = px.histogram(x=predictions, nbins=40, template="plotly_dark")
                hist.add_vline(x=t, line_dash="dash", line_color="red",
                               annotation_text=f"Esik %{t*100:.0f}")
                hist.update_layout(xaxis_title="Yukselis Olasiligi", yaxis_title="Mum Sayisi", height=280)
                st.plotly_chart(hist, use_container_width=True)

            except Exception as e:
                st.error(f"Hata: {e}")

# ============================ SEKME 2: PORTFOY ============================
with tab_portfolio:
    st.markdown("**Paylasimli sermaye** simulasyonu: tum hisseler TEK 10.000 TL havuzu "
                "paylasir, ayni anda en fazla N pozisyon acik olabilir.")

    cset1, cset2 = st.columns([1, 1])
    max_positions = cset1.slider("Es zamanli max pozisyon", 1, 8, 4)
    show_isolated = cset2.checkbox("Izole (her hisse ayri 10k) karsilastirmasini goster", value=True)
    sizing = 'dynamic' if mode == 'compound' else 'fixed'

    if st.button("Portfoy Simulasyonu Calistir", type="primary"):
        symbol_data = {}
        param_rows = []
        isolated_total = 0.0
        prog = st.progress(0.0, text="Hisseler analiz ediliyor...")

        for idx, sym in enumerate(ALL_SYMBOLS):
            try:
                preds, tdf = get_symbol_data(sym)
                if auto_opt:
                    t, sl, tp = run_cma_optimization(preds.reshape(-1, 1), tdf, window_size=WINDOW)
                else:
                    t, sl, tp = threshold, stop_loss, take_profit
                symbol_data[sym] = {'predictions': preds.reshape(-1, 1), 'test_df': tdf,
                                    'threshold': t, 'stop_loss': sl, 'take_profit': tp}
                param_rows.append({'Sembol': sym, 'Esik %': round(t*100, 1),
                                   'SL %': round(sl*100, 2), 'TP %': round(tp*100, 2)})
                if show_isolated:
                    iso = run_backtest(preds.reshape(-1, 1), tdf, threshold=t, stop_loss=sl,
                                       take_profit=tp, window_size=WINDOW, mode=mode, allow_short=allow_short)
                    isolated_total += iso['equity'].iloc[-1] - INITIAL_CAPITAL
            except Exception as e:
                st.warning(f"{sym} atlandi: {e}")
            prog.progress((idx + 1) / len(ALL_SYMBOLS), text=f"{sym} tamamlandi")

        if not symbol_data:
            st.error("Hicbir sembol islenemedi.")
        else:
            eq_df, tr_df, summ = run_portfolio_backtest(
                symbol_data, max_positions=max_positions, window_size=WINDOW,
                sizing=sizing, allow_short=allow_short)

            st.success("Portfoy simulasyonu tamamlandi!")
            ret_pct = summ['net_profit'] / INITIAL_CAPITAL * 100

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Net Kar (TL)", f"{summ['net_profit']:.0f}", delta=f"%{ret_pct:.1f}",
                      delta_color="normal" if summ['net_profit'] >= 0 else "inverse")
            k2.metric("Max Risk", f"%{summ['max_drawdown']:.1f}")
            k3.metric("Calmar", f"{summ['calmar']:.1f}")
            k4.metric("Kazanma Orani", f"%{summ['win_rate']:.1f}")
            k5.metric("Max Es Zamanli", f"{summ['max_concurrent']} / {max_positions}")

            # Portfoy kasa + drawdown egrisi
            st.plotly_chart(equity_drawdown_chart(eq_df), use_container_width=True)

            cc1, cc2 = st.columns([3, 2])
            with cc1:
                if not tr_df.empty:
                    pnl = tr_df.groupby('symbol')['pnl'].sum().reset_index().sort_values('pnl', ascending=False)
                    bar = px.bar(pnl, x='symbol', y='pnl', color='pnl',
                                 color_continuous_scale='RdYlGn', template="plotly_dark",
                                 title="Hisse Bazinda Portfoy PnL Katkisi (TL)")
                    st.plotly_chart(bar, use_container_width=True)
            with cc2:
                st.markdown("**Hisse Bazinda Optimal Parametreler**")
                st.dataframe(pd.DataFrame(param_rows), use_container_width=True, height=400)

            if show_isolated:
                iso_ret = isolated_total / (INITIAL_CAPITAL * len(symbol_data)) * 100
                st.markdown("#### Karsilastirma: Sermaye Verimliligi")
                comp = pd.DataFrame([
                    {'Senaryo': 'Izole (her hisse ayri 10k)',
                     'Sermaye (TL)': INITIAL_CAPITAL * len(symbol_data),
                     'Net Kar (TL)': round(isolated_total, 0), 'Sermaye Getirisi %': round(iso_ret, 1)},
                    {'Senaryo': f'Portfoy (tek 10k, max {max_positions} pozisyon)',
                     'Sermaye (TL)': INITIAL_CAPITAL,
                     'Net Kar (TL)': round(summ['net_profit'], 0), 'Sermaye Getirisi %': round(ret_pct, 1)},
                ])
                st.dataframe(comp, use_container_width=True)
                st.caption("Not: Mutlak kar yaniltici (izole 10x sermaye kullanir). "
                           "Adil olcut SERMAYE GETIRISI %'dir.")
