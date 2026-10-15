from donchian_backtest import fetch_candles
from fvg_backtest import backtest, summarize

df = fetch_candles("ETHUSD", "15m", days=900)
df.to_csv("ethusd_15m_900d.csv")
old = df.loc[:"2026-04-11"]          # data never used for tuning
recent = df.loc["2026-04-12":]
scale = old.close.median() / recent.close.median()
print(len(old), old.index[0], old.index[-1], "price scale:", round(scale, 2))

for tk in (8, 10, 12, 15):
    tr, no = backtest(old, max_range=15 * scale, min_range=7 * scale,
                      target_r=2, trend_k=tk)
    print(tk, summarize(tr, no))