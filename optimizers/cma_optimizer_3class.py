"""
3-sinifli model icin 4-boyutlu CMA-ES optimizasyonu (short destegi).

Arama uzayi:
    [long_threshold, short_threshold, stop_loss, take_profit]

Adim 1 iyilestirmeleri (3-sinifli modelin dar olasilik dagilimi icin):
  - Alt sinir 0.35 -> 0.34 (rastgele tavaninin hemen ustu, ama anlamli)
  - min_trade_count 10 -> 3 (3-sinifli modelde islem uretmek dogal olarak zor)
  - Akilli baslangic: sembolun kendi olasilik dagiliminin p75'inden basla
    -> her sembol icin dogru bolgeye anida yakinsa
"""

import io
import contextlib
import cma
import numpy as np

from core.backtest_engine_3class import run_backtest_3class

# Adim 1: gevsetilmis sinirlar
THR_MIN = 0.34   # Rastgele tavaninin (0.333) hemen ustu
THR_MAX = 0.95
SL_MIN = 0.005
TP_MIN = 0.01
MIN_TRADES = 3   # Eski: 10. 3-sinifli modelde 3 yeterli istatistiksel sinyal


def fast_evaluator_3class(params, predictions, test_df, window_size=60, commission=0.002):
    long_thr, short_thr, sl, tp = params[0], params[1], params[2], params[3]

    # Genetik sinirlar
    if (long_thr < THR_MIN or long_thr > THR_MAX or
        short_thr < THR_MIN or short_thr > THR_MAX or
        sl < SL_MIN or tp < TP_MIN):
        return 999999.0

    # Sessiz backtest
    with contextlib.redirect_stdout(io.StringIO()):
        rdf = run_backtest_3class(predictions, test_df,
                                  long_threshold=long_thr, short_threshold=short_thr,
                                  stop_loss=sl, take_profit=tp, window_size=window_size)

    equity = rdf['equity'].values
    sr = rdf['strategy_return'].values
    trade_count = int((sr != 0).sum())

    # Cok az islem veya iflas
    if trade_count < MIN_TRADES or equity[-1] <= 0:
        return 999999.0

    net_profit = equity[-1] - 10000.0
    if net_profit <= 0:
        return abs(net_profit)

    rolling_max = np.maximum.accumulate(equity)
    drawdowns = np.where(rolling_max > 0, (equity - rolling_max) / rolling_max, 0)
    max_dd = abs(drawdowns.min()) * 100

    calmar = net_profit / (max_dd + 1.0)
    return -calmar


def _smart_initial(predictions):
    """
    Sembolun kendi olasilik dagilimina gore akilli baslangic noktasi.

    Eski yontem: hep [0.50, 0.50, 0.015, 0.04] -> dar dagilimli sembollerde
    CMA-ES bu noktadan hareket edemiyor cunku tum komsuluk 999999 cezasi.

    Yeni yontem: long_thr_init = clip(p_up_p75, THR_MIN, THR_MAX)
                 short_thr_init = clip(p_down_p75, THR_MIN, THR_MAX)
    -> Her sembolun "ust ceyrek" olasilik bolgesinden baslar; CMA-ES oradan
       lokal aramaya gecip uygun esikleri bulur.
    """
    p_down = predictions[:, 0]
    p_up = predictions[:, 2]
    long_init = float(np.clip(np.percentile(p_up, 75), THR_MIN, THR_MAX))
    short_init = float(np.clip(np.percentile(p_down, 75), THR_MIN, THR_MAX))
    return [long_init, short_init, 0.015, 0.04]


def run_cma_optimization_3class(predictions, test_df, window_size=60, seed=42):
    print("\n" + "=" * 60)
    print("4-BOYUTLU CMA-ES EVRIMI (long_thr / short_thr / SL / TP)")
    print("=" * 60)

    initial_params = _smart_initial(predictions)
    print(f"Akilli baslangic: long_thr=%{initial_params[0]*100:.2f} "
          f"short_thr=%{initial_params[1]*100:.2f}")
    sigma0 = 0.05

    es = cma.CMAEvolutionStrategy(initial_params, sigma0,
                                  {'popsize': 24, 'maxiter': 30, 'verbose': -9, 'seed': seed})

    generation = 1
    while not es.stop():
        solutions = es.ask()
        fitness = [fast_evaluator_3class(x, predictions, test_df, window_size) for x in solutions]
        es.tell(solutions, fitness)

        best_idx = int(np.argmin(fitness))
        best_p = solutions[best_idx]
        best_score = -fitness[best_idx]
        if best_score != -999999.0 and best_score > 0:
            print(f"  Nesil {generation:02d} | Calmar: {best_score:>7.2f} | "
                  f"long_thr=%{best_p[0]*100:.1f} short_thr=%{best_p[1]*100:.1f} "
                  f"SL=%{best_p[2]*100:.2f} TP=%{best_p[3]*100:.2f}")
        generation += 1

    best = es.result.xbest
    print("=" * 60)
    print(f"OPTIMIZE: long_thr=%{best[0]*100:.2f} short_thr=%{best[1]*100:.2f} "
          f"SL=%{best[2]*100:.2f} TP=%{best[3]*100:.2f}")
    return float(best[0]), float(best[1]), float(best[2]), float(best[3])
