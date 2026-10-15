from fvg_mtf_backtest import fetch_3m, backtest_mtf, CONFIG

df3 = fetch_3m("ETHUSD", days=365)
steps = [
    ("all rules",                 {}),
    ("3m BOS off",                dict(use_bos3=False)),
    ("3m BOS + trend off",        dict(use_bos3=False, trend_k3=None)),
    ("3m BOS + trend + size off", dict(use_bos3=False, trend_k3=None, min_range3=0, max_range3=1e9)),
]
for name, kw in steps:
    tr, f = backtest_mtf(df3, **{**CONFIG, **kw})
    print(f"{name:28s} zones {f['zones']:4d}  touched {f['touched']:4d}  orders {f['orders']:4d}  trades {len(tr):4d}")