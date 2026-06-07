"""
Gorsellestirme yardimcilari (Plotly).

Dashboard'dan ayri tutulur ki Streamlit calistirmadan test edilebilsinler.
Tum fonksiyonlar plotly Figure dondurur; Streamlit baglantisi yoktur.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INITIAL_CAPITAL = 10000.0


def compute_metrics(results_df, initial_capital=INITIAL_CAPITAL):
    """run_backtest ciktisindan ozet metrikler."""
    final_equity = results_df['equity'].iloc[-1]
    net_profit = final_equity - initial_capital
    exit_bars = results_df[results_df['strategy_return'] != 0]
    total_trades = len(exit_bars)
    win_rate = (len(exit_bars[exit_bars['strategy_return'] > 0]) / total_trades * 100) if total_trades > 0 else 0.0
    rolling_max = results_df['equity'].cummax()
    dd = np.where(rolling_max > 0, (results_df['equity'] - rolling_max) / rolling_max, 0)
    max_dd = abs(dd.min()) * 100
    calmar = net_profit / (max_dd + 1.0)
    return {
        'net_profit': net_profit, 'final_equity': final_equity, 'total_trades': total_trades,
        'win_rate': win_rate, 'max_dd': max_dd, 'calmar': calmar
    }


def gauge(value, title, vmin, vmax, suffix="", good_high=True):
    """Plotly ibreli gosterge (gauge)."""
    steps = ([{'range': [vmin, (vmin+vmax)/2], 'color': '#3b2f2f'},
              {'range': [(vmin+vmax)/2, vmax], 'color': '#1f3b2f'}] if good_high else
             [{'range': [vmin, (vmin+vmax)/2], 'color': '#1f3b2f'},
              {'range': [(vmin+vmax)/2, vmax], 'color': '#3b2f2f'}])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={'suffix': suffix},
        title={'text': title, 'font': {'size': 14}},
        gauge={'axis': {'range': [vmin, vmax]},
               'bar': {'color': '#00cc96' if good_high else '#ef553b'},
               'steps': steps}))
    fig.update_layout(template="plotly_dark", height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def price_volume_chart(df, symbol):
    """Candlestick + Kalman + alim/short/cikis isaretleri + hacim alt paneli (range slider'li)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04,
                        subplot_titles=(f"{symbol} Fiyat (Candlestick) + Kalman + Sinyaller", "Hacim"))

    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='OHLC',
                                 increasing_line_color='#26a69a', decreasing_line_color='#ef5350'),
                  row=1, col=1)
    if 'kalman_close' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['kalman_close'], name='Kalman',
                                 line=dict(color='#ffa600', width=1.5)), row=1, col=1)

    buys = df[df['signal'] == 1]
    shorts = df[df['signal'] == -1]
    exits = df[df['strategy_return'] != 0]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['low'] * 0.998, mode='markers', name='Long Giris',
                             marker=dict(color='#00e5ff', size=11, symbol='triangle-up')), row=1, col=1)
    if len(shorts) > 0:
        fig.add_trace(go.Scatter(x=shorts.index, y=shorts['high'] * 1.002, mode='markers', name='Short Giris',
                                 marker=dict(color='magenta', size=11, symbol='triangle-down')), row=1, col=1)
    fig.add_trace(go.Scatter(x=exits.index, y=exits['close'], mode='markers', name='Cikis',
                             marker=dict(color='orange', size=8, symbol='x')), row=1, col=1)

    vol_colors = np.where(df['close'] >= df['open'], '#26a69a', '#ef5350')
    fig.add_trace(go.Bar(x=df.index, y=df.get('volume', pd.Series(0, index=df.index)),
                         marker_color=vol_colors, name='Hacim'), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=560, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      xaxis2_rangeslider_visible=True, xaxis2_rangeslider_thickness=0.06)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    return fig


def equity_drawdown_chart(df, initial_capital=INITIAL_CAPITAL):
    """Kasa egrisi + altinda drawdown (dolgu) paneli."""
    eq = df['equity']
    roll = eq.cummax()
    dd = np.where(roll > 0, (eq - roll) / roll * 100, 0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05, subplot_titles=("Kasa Buyume Egrisi", "Drawdown (%)"))
    fig.add_trace(go.Scatter(x=df.index, y=eq, name='Kasa', line=dict(color='#00cc96'),
                             fill='tozeroy', fillcolor='rgba(0,204,150,0.1)'), row=1, col=1)
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=dd, name='Drawdown', line=dict(color='#ef553b'),
                             fill='tozeroy', fillcolor='rgba(239,85,59,0.3)'), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=440, hovermode="x unified", showlegend=False)
    return fig
