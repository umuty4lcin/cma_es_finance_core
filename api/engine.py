"""
API motor servisi — mevcut core/ ve optimizers/ kodunu sarmalar.

Tek bir sembol icin tam analiz pipeline'ini calistirir ve React onyuzunun
ihtiyac duydugu tum verileri (metrikler, OHLC, equity/drawdown, radar,
tahmin dagilimi, sinyaller) JSON-uyumlu sozluk olarak dondurur.

Agir nesneler (model, hazirlanmis veri) bellekte onbeleklenir.
"""

import io
import contextlib
from functools import lru_cache

import numpy as np
import pandas as pd

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target, create_features_and_target_3class
from core.ai_prep import prepare_lstm_data
from core.backtest_engine import run_backtest
from core.backtest_engine_3class import run_backtest_3class
from optimizers.cma_optimizer import run_cma_optimization
from optimizers.cma_optimizer_3class import run_cma_optimization_3class

MODEL_PATH = "data/global_lstm_model_15min.keras"
MODEL_PATH_3CLASS = "data/global_lstm_3class_15min.keras"
FEATURES = ["close", "kalman_close", "feature_kalman_diff",
            "feature_return_1m", "feature_volatility_15m"]
WINDOW = 60
INITIAL_CAPITAL = 10000.0

_model = None
_model_3class = None


def get_model():
    """Egitilmis global LSTM modelini bir kez yukler (lazy + cache)."""
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model


def get_model_3class():
    """3-sinifli egitilmis modeli bir kez yukler (lazy + cache)."""
    global _model_3class
    if _model_3class is None:
        from tensorflow.keras.models import load_model
        _model_3class = load_model(MODEL_PATH_3CLASS)
    return _model_3class


@lru_cache(maxsize=32)
def _prepare(symbol: str):
    """Sembol verisini hazirlar ve tahminleri uretir (sembol basina onbellek)."""
    clean = load_and_fill_gaps(symbol, timeframe="15m")
    clean["kalman_close"] = apply_kalman_filter(clean["close"])
    ai = create_features_and_target(clean, lookahead=15, threshold=0.001)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, X_test, _, _, _, test_df = prepare_lstm_data(ai, FEATURES, window_size=WINDOW)
    preds = get_model().predict(X_test, verbose=0).flatten()
    return test_df, preds


def _ts(index) -> list:
    """tz-aware DatetimeIndex -> UNIX saniye listesi (lightweight-charts formati)."""
    return [int(t.timestamp()) for t in index]


def _pct_rank(series: np.ndarray, value: float) -> float:
    """Bir degerin seri icindeki yuzdelik sirasini (0-100) dondurur."""
    s = series[~np.isnan(series)]
    if len(s) == 0:
        return 50.0
    return float((s <= value).mean() * 100.0)


def _rsi(close: np.ndarray, period: int = 14) -> float:
    """Son RSI degeri (0-100)."""
    diff = np.diff(close)
    if len(diff) < period:
        return 50.0
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _radar(df: pd.DataFrame) -> dict:
    """
    Hissenin guncel cok-boyutlu 'durumunu' 0-100 olcekli radar profili olarak
    hesaplar. Her boyut, son degerin tum test serisindeki yuzdelik sirasidir
    (RSI haric, o ham 0-100'dur).
    """
    close = df["close"].values
    ret = df["close"].pct_change().fillna(0).values
    vol = df.get("volume", pd.Series(0, index=df.index)).values
    kalman = df["kalman_close"].values

    # Momentum: son 20-bar getiri
    mom_series = pd.Series(close).pct_change(20).values
    momentum = _pct_rank(mom_series, mom_series[-1])

    # Volatilite: 20-bar getiri standart sapmasi
    vol_series = pd.Series(ret).rolling(20).std().values
    volatility = _pct_rank(vol_series, vol_series[-1])

    # Hacim: 20-bar ortalama hacim
    volm_series = pd.Series(vol).rolling(20).mean().values
    volume_score = _pct_rank(volm_series, volm_series[-1])

    # Trend gucu: 20-bar kalman egimi (mutlak)
    slope_series = np.abs(pd.Series(kalman).pct_change(20).values)
    trend = _pct_rank(slope_series, slope_series[-1])

    # RSI (ham 0-100)
    rsi = _rsi(close)

    return {
        "labels": ["Momentum", "Volatilite", "Hacim", "Trend Gucu", "RSI"],
        "values": [round(momentum, 1), round(volatility, 1), round(volume_score, 1),
                   round(trend, 1), round(rsi, 1)],
    }


def analyze_symbol(symbol: str, auto_optimize: bool = True,
                   threshold: float = 0.5, stop_loss: float = 0.02,
                   take_profit: float = 0.04, mode: str = "fixed") -> dict:
    """Tek sembol icin tam analiz (2-sinifli, sadece long); JSON-uyumlu sozluk dondurur."""
    test_df, preds = _prepare(symbol)

    if auto_optimize:
        with contextlib.redirect_stdout(io.StringIO()):
            t, sl, tp = run_cma_optimization(preds.reshape(-1, 1), test_df, window_size=WINDOW)
    else:
        t, sl, tp = float(threshold), float(stop_loss), float(take_profit)

    with contextlib.redirect_stdout(io.StringIO()):
        rdf = run_backtest(preds.reshape(-1, 1), test_df, threshold=t, stop_loss=sl,
                           take_profit=tp, window_size=WINDOW, mode=mode)

    # --- Metrikler ---
    equity = rdf["equity"].values
    sr = rdf["strategy_return"].values
    net_profit = float(equity[-1] - INITIAL_CAPITAL)
    rolling_max = np.maximum.accumulate(equity)
    dd = np.where(rolling_max > 0, (equity - rolling_max) / rolling_max * 100.0, 0.0)
    max_dd = float(abs(dd.min()))
    total_trades = int((sr != 0).sum())
    wins = int((sr > 0).sum())
    win_rate = float(wins / total_trades * 100.0) if total_trades > 0 else 0.0
    calmar = float(net_profit / (max_dd + 1.0))

    # --- Zaman serileri (lightweight-charts: time = unix saniye) ---
    times = _ts(rdf.index)
    candles = [{"time": tt, "open": round(float(o), 4), "high": round(float(h), 4),
                "low": round(float(l), 4), "close": round(float(c), 4)}
               for tt, o, h, l, c in zip(times, rdf["open"], rdf["high"], rdf["low"], rdf["close"])]
    volume = [{"time": tt, "value": float(v)} for tt, v in zip(times, rdf["volume"])]
    kalman = [{"time": tt, "value": round(float(k), 4)} for tt, k in zip(times, rdf["kalman_close"])]
    equity_series = [{"time": tt, "value": round(float(e), 2)} for tt, e in zip(times, equity)]
    drawdown_series = [{"time": tt, "value": round(float(d), 3)} for tt, d in zip(times, dd)]

    sig = rdf["signal"].values
    long_markers = [{"time": tt, "price": round(float(p), 4)}
                    for tt, p, s in zip(times, rdf["low"], sig) if s == 1]
    short_markers = [{"time": tt, "price": round(float(p), 4)}
                     for tt, p, s in zip(times, rdf["high"], sig) if s == -1]
    exit_markers = [{"time": tt, "price": round(float(p), 4)}
                    for tt, p, r in zip(times, rdf["close"], sr) if r != 0]

    return {
        "symbol": symbol,
        "params": {"threshold": round(t, 4), "stop_loss": round(sl, 4), "take_profit": round(tp, 4),
                   "mode": mode, "auto_optimize": auto_optimize},
        "metrics": {"net_profit": round(net_profit, 2), "final_equity": round(float(equity[-1]), 2),
                    "win_rate": round(win_rate, 2), "max_drawdown": round(max_dd, 2),
                    "calmar": round(calmar, 2), "total_trades": total_trades,
                    "long_count": int((sig == 1).sum()), "short_count": int((sig == -1).sum())},
        "candles": candles,
        "volume": volume,
        "kalman": kalman,
        "equity": equity_series,
        "drawdown": drawdown_series,
        "markers": {"long": long_markers, "short": short_markers, "exit": exit_markers},
        "predictions": [round(float(p), 4) for p in preds],
        "radar": _radar(rdf),
        "initial_capital": INITIAL_CAPITAL,
    }


# ----------------------------------------------------------------------------
#  3-SINIFLI YOL (short destegi)
# ----------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _prepare_3class(symbol: str):
    """3-sinifli model icin sembol verisini hazirlar + softmax tahminleri uretir."""
    clean = load_and_fill_gaps(symbol, timeframe="15m")
    clean["kalman_close"] = apply_kalman_filter(clean["close"])
    # train_model_3class.py ile ayni esikleri kullan (v1: +/-%0.5)
    ai = create_features_and_target_3class(clean, lookahead=15,
                                            up_threshold=0.005, down_threshold=-0.005)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, X_test, _, _, _, test_df = prepare_lstm_data(ai, FEATURES, window_size=WINDOW)
    preds = get_model_3class().predict(X_test, verbose=0)  # (N, 3) softmax
    return test_df, preds


def analyze_symbol_3class(symbol: str, auto_optimize: bool = True,
                           long_threshold: float = 0.50, short_threshold: float = 0.50,
                           stop_loss: float = 0.02, take_profit: float = 0.04,
                           mode: str = "fixed") -> dict:
    """3-sinifli model ile tam analiz (long + short)."""
    test_df, preds = _prepare_3class(symbol)

    if auto_optimize:
        with contextlib.redirect_stdout(io.StringIO()):
            lt, st, sl, tp = run_cma_optimization_3class(preds, test_df, window_size=WINDOW)
    else:
        lt = float(long_threshold); st = float(short_threshold)
        sl = float(stop_loss); tp = float(take_profit)

    with contextlib.redirect_stdout(io.StringIO()):
        rdf = run_backtest_3class(preds, test_df,
                                  long_threshold=lt, short_threshold=st,
                                  stop_loss=sl, take_profit=tp,
                                  window_size=WINDOW, mode=mode)

    # --- Metrikler ---
    equity = rdf["equity"].values
    sr = rdf["strategy_return"].values
    net_profit = float(equity[-1] - INITIAL_CAPITAL)
    rolling_max = np.maximum.accumulate(equity)
    dd = np.where(rolling_max > 0, (equity - rolling_max) / rolling_max * 100.0, 0.0)
    max_dd = float(abs(dd.min()))
    total_trades = int((sr != 0).sum())
    wins = int((sr > 0).sum())
    win_rate = float(wins / total_trades * 100.0) if total_trades > 0 else 0.0
    calmar = float(net_profit / (max_dd + 1.0))

    times = _ts(rdf.index)
    candles = [{"time": tt, "open": round(float(o), 4), "high": round(float(h), 4),
                "low": round(float(l), 4), "close": round(float(c), 4)}
               for tt, o, h, l, c in zip(times, rdf["open"], rdf["high"], rdf["low"], rdf["close"])]
    volume = [{"time": tt, "value": float(v)} for tt, v in zip(times, rdf["volume"])]
    kalman = [{"time": tt, "value": round(float(k), 4)} for tt, k in zip(times, rdf["kalman_close"])]
    equity_series = [{"time": tt, "value": round(float(e), 2)} for tt, e in zip(times, equity)]
    drawdown_series = [{"time": tt, "value": round(float(d), 3)} for tt, d in zip(times, dd)]

    sig = rdf["signal"].values
    long_markers = [{"time": tt, "price": round(float(p), 4)}
                    for tt, p, s in zip(times, rdf["low"], sig) if s == 1]
    short_markers = [{"time": tt, "price": round(float(p), 4)}
                     for tt, p, s in zip(times, rdf["high"], sig) if s == -1]
    exit_markers = [{"time": tt, "price": round(float(p), 4)}
                    for tt, p, r in zip(times, rdf["close"], sr) if r != 0]

    return {
        "symbol": symbol,
        "model_type": "3class",
        "params": {"long_threshold": round(lt, 4), "short_threshold": round(st, 4),
                   "stop_loss": round(sl, 4), "take_profit": round(tp, 4),
                   "mode": mode, "auto_optimize": auto_optimize},
        "metrics": {"net_profit": round(net_profit, 2), "final_equity": round(float(equity[-1]), 2),
                    "win_rate": round(win_rate, 2), "max_drawdown": round(max_dd, 2),
                    "calmar": round(calmar, 2), "total_trades": total_trades,
                    "long_count": int((sig == 1).sum()), "short_count": int((sig == -1).sum())},
        "candles": candles,
        "volume": volume,
        "kalman": kalman,
        "equity": equity_series,
        "drawdown": drawdown_series,
        "markers": {"long": long_markers, "short": short_markers, "exit": exit_markers},
        # 3-sinifli model icin tahminler [P_asagi, P_yatay, P_yukari]
        "predictions_3class": {
            "p_down": [round(float(p), 4) for p in preds[:, 0]],
            "p_flat": [round(float(p), 4) for p in preds[:, 1]],
            "p_up":   [round(float(p), 4) for p in preds[:, 2]],
        },
        "radar": _radar(rdf),
        "initial_capital": INITIAL_CAPITAL,
    }
