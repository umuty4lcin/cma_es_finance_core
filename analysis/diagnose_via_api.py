"""
3-sinifli model teshisi — calisan backend uzerinden (TF cakismasi yok).

Backend zaten modeli yuklu tutuyor; biz sadece her sembol icin
/api/analyze cagriyoruz ve donen veriden istatistik cikariyoruz.
"""

import json
import urllib.request
import numpy as np

SYMBOLS = ['ASELS', 'GARAN', 'HALKB', 'ISCTR', 'THYAO',
           'TUPRS', 'VAKBN', 'SASA', 'SISE', 'FROTO']
URL = "http://127.0.0.1:8000/api/analyze"


def analyze(symbol: str) -> dict:
    body = json.dumps({
        "symbol": symbol,
        "model_type": "3class",
        "auto_optimize": True,
        "mode": "fixed",
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main():
    print(f"{'Sembol':<7} | {'p_up max':>9} | {'p_up p90':>8} | {'p_dn max':>9} | "
          f"{'p_dn p90':>8} | {'CMA_lt':>7} | {'CMA_st':>7} | {'long':>5} | {'short':>5} | {'kar':>8}")
    print("-" * 110)
    results = []
    for sym in SYMBOLS:
        try:
            data = analyze(sym)
            p_up = np.array(data['predictions_3class']['p_up'])
            p_dn = np.array(data['predictions_3class']['p_down'])
            m = data['metrics']
            p = data['params']
            results.append({
                'sym': sym,
                'up_max': p_up.max(), 'up_p90': np.percentile(p_up, 90),
                'dn_max': p_dn.max(), 'dn_p90': np.percentile(p_dn, 90),
                'cma_lt': p['long_threshold'], 'cma_st': p['short_threshold'],
                'long': m['long_count'], 'short': m['short_count'],
                'kar': m['net_profit'],
            })
            print(f"{sym:<7} | {p_up.max():>9.3f} | {np.percentile(p_up, 90):>8.3f} | "
                  f"{p_dn.max():>9.3f} | {np.percentile(p_dn, 90):>8.3f} | "
                  f"{p['long_threshold']:>7.3f} | {p['short_threshold']:>7.3f} | "
                  f"{m['long_count']:>5} | {m['short_count']:>5} | {m['net_profit']:>8.0f}")
        except Exception as e:
            print(f"{sym:<7} | HATA: {e}")

    # Markdown ozet
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_3class.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write("# 3-Sinifli Model Teshis Sonuclari (API uzerinden)\n\n")
        f.write("| Sembol | p_up max | p_up p90 | p_dn max | p_dn p90 | CMA long_thr | CMA short_thr | Long | Short | Kar (TL) |\n")
        f.write("|--------|---------:|---------:|---------:|---------:|-------------:|--------------:|-----:|------:|---------:|\n")
        for r in results:
            f.write(f"| {r['sym']} | {r['up_max']:.3f} | {r['up_p90']:.3f} | "
                    f"{r['dn_max']:.3f} | {r['dn_p90']:.3f} | "
                    f"{r['cma_lt']:.3f} | {r['cma_st']:.3f} | "
                    f"{r['long']} | {r['short']} | {r['kar']:.0f} |\n")
    print(f"\n[KAYDEDILDI] {out}")


if __name__ == "__main__":
    main()
