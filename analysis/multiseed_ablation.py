"""
FAZ 1 - Cok-Seed Kalman Ablasyon Calismasi

Amac: "Kalman tabanli ozellikler modele gercekten katki sagliyor mu, yoksa
gozlenen fark egitim rastgeleliginden mi kaynaklaniyor?" sorusunu kesin
olarak yanitlamak.

Yontem:
  - 2 konfigurasyon: {kalman (5 ozellik), nokalman (3 ozellik)}
  - 3 farkli egitim seed'i: [42, 7, 123]
  - Toplam 6 model egitilir, her biri 10 sembolde degerlendirilir.
  - CMA-ES seed'i SABIT (42) tutulur ki sadece egitim rastgeleliginin etkisi olculsun.
  - Sonuclar seed'ler arasinda ortalama ± standart sapma olarak raporlanir.

Veri hazirligi seed'den bagimsiz oldugu icin her konfigurasyon icin BIR KEZ
yapilir ve tum seed'lerde yeniden kullanilir (hiz + bellek tasarrufu).

Cikti: analysis/multiseed_results.md
"""

import os
import sys
import io
import random
import contextlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score

from core.data_pipeline import load_and_fill_gaps
from core.signal_filters import apply_kalman_filter
from core.feature_engineering import create_features_and_target
from core.ai_prep import prepare_lstm_data
from core.ai_models import build_lstm_model
from core.backtest_engine import run_backtest
from optimizers.cma_optimizer import run_cma_optimization

TRAIN_SYMBOLS = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO', 'TUPRS', 'VAKBN', 'SASA', 'SISE']
EVAL_SYMBOLS = TRAIN_SYMBOLS + ['FROTO']  # FROTO = OOU testi
WINDOW = 60
SEEDS = [42, 7, 123]
CMA_SEED = 42

CONFIGS = {
    'kalman':   ['close', 'kalman_close', 'feature_kalman_diff', 'feature_return_1m', 'feature_volatility_15m'],
    'nokalman': ['close', 'feature_return_1m', 'feature_volatility_15m'],
}


def set_seeds(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)


def build_config_data(features):
    """Bir konfigurasyon icin mega egitim seti + per-sembol degerlendirme verisi (seed'den bagimsiz)."""
    per_symbol = {}
    Xtr_list, ytr_list, vX_list, vy_list = [], [], [], []
    for sym in EVAL_SYMBOLS:
        clean = load_and_fill_gaps(sym, timeframe='15m')
        clean['kalman_close'] = apply_kalman_filter(clean['close'])
        ai = create_features_and_target(clean, lookahead=15, threshold=0.001)
        with contextlib.redirect_stdout(io.StringIO()):
            Xtr, ytr, Xte, yte, sc, cw, tdf = prepare_lstm_data(
                df=ai, feature_cols=features, window_size=WINDOW)
        per_symbol[sym] = (Xte, yte, tdf)
        if sym in TRAIN_SYMBOLS:
            Xtr_list.append(Xtr); ytr_list.append(ytr)
            vX_list.append(Xte); vy_list.append(yte)
    mega_X = np.vstack(Xtr_list)
    mega_y = np.concatenate(ytr_list)
    val_X = np.vstack(vX_list)
    val_y = np.concatenate(vy_list)
    return mega_X, mega_y, val_X, val_y, per_symbol


def trading_metrics(df):
    eq = df['equity'].values
    sr = df['strategy_return'].values
    net = eq[-1] - 10000.0
    rmax = np.maximum.accumulate(eq)
    dd = np.where(rmax > 0, (eq - rmax) / rmax, 0)
    maxdd = abs(dd.min()) * 100
    return net, net / (maxdd + 1.0)


def quiet_cma(preds, tdf):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_cma_optimization(preds, tdf, window_size=WINDOW, seed=CMA_SEED)


def quiet_bt(preds, tdf, t, sl, tp):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_backtest(preds, tdf, threshold=t, stop_loss=sl, take_profit=tp, window_size=WINDOW)


def main():
    records = []
    for cfg_name, feats in CONFIGS.items():
        print(f"\n{'='*60}\nKONFIGURASYON: {cfg_name} ({len(feats)} ozellik)\n{'='*60}")
        print("Veri hazirlaniyor (bir kez)...")
        mega_X, mega_y, val_X, val_y, per_symbol = build_config_data(feats)
        print(f"  Mega train: {mega_X.shape} | Val: {val_X.shape}")

        classes = np.unique(mega_y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=mega_y)
        cw = dict(zip(classes, weights))

        for seed in SEEDS:
            print(f"\n--- {cfg_name} | seed={seed} ---")
            set_seeds(seed)
            model = build_lstm_model((WINDOW, len(feats)))
            es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            model.fit(mega_X, mega_y, epochs=30, batch_size=512,
                      validation_data=(val_X, val_y), class_weight=cw,
                      callbacks=[es], verbose=2, shuffle=True)

            f1s, accs, profits, calmars = [], [], [], []
            for sym in EVAL_SYMBOLS:
                Xte, yte, tdf = per_symbol[sym]
                preds = model.predict(Xte, verbose=0)
                yp = (preds.flatten() > 0.5).astype(int)
                f1s.append(f1_score(yte, yp, zero_division=0) * 100)
                accs.append(accuracy_score(yte, yp) * 100)
                t, sl, tp = quiet_cma(preds, tdf)
                net, calmar = trading_metrics(quiet_bt(preds, tdf, t, sl, tp))
                profits.append(net); calmars.append(calmar)

            rec = {
                'config': cfg_name, 'seed': seed,
                'avg_f1': np.mean(f1s), 'avg_acc': np.mean(accs),
                'total_profit': np.sum(profits), 'avg_calmar': np.mean(calmars),
            }
            records.append(rec)
            print(f"  -> F1={rec['avg_f1']:.2f}% Acc={rec['avg_acc']:.2f}% "
                  f"ToplamKar={rec['total_profit']:.0f}TL Calmar={rec['avg_calmar']:.1f}")

        del mega_X, mega_y, val_X, val_y, per_symbol  # bellegi serbest birak

    df = pd.DataFrame(records)
    print("\n\n" + "=" * 70)
    print("TUM KOSULAR")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    agg = df.groupby('config').agg(
        f1_mean=('avg_f1', 'mean'), f1_std=('avg_f1', 'std'),
        acc_mean=('avg_acc', 'mean'),
        profit_mean=('total_profit', 'mean'), profit_std=('total_profit', 'std'),
        calmar_mean=('avg_calmar', 'mean'), calmar_std=('avg_calmar', 'std'),
    ).reset_index()

    print("\n" + "=" * 70)
    print("SEED'LER ARASI OZET (ortalama +/- std)")
    print("=" * 70)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multiseed_results.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write("# Cok-Seed Kalman Ablasyon Sonuclari\n\n")
        f.write(f"Seed'ler: {SEEDS} | CMA seed (sabit): {CMA_SEED}\n\n")
        f.write("## Tum Kosular\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n\n## Seed'ler Arasi Ozet (ortalama +/- std)\n\n")
        f.write(agg.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n")
    print(f"\n[KAYDEDILDI] {out}")


if __name__ == "__main__":
    main()
