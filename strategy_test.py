import pandas as pd
from donchian_backtest import fetch_candles

df = fetch_candles("ETHUSD", "15m", days=180)
o, h, l, c = df.open, df.high, df.low, df.close

bull = (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2))
bear = (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2))
gap_b = bull & (h.shift(2) < l)       # candle1 high < candle3 low
gap_s = bear & (l.shift(2) > h)       # candle1 low  > candle3 high
setup = gap_b | gap_s

c1_range = (h - l).shift(2)           # candle 1 range
c1_price = c.shift(2)
ok = setup & (c1_range <= 10)

print("median ETH price:", round(c.median(), 1))
print("15m bar range quantiles:", (h - l).quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
print("raw setups:", int(setup.sum()), "| with candle1 <= 10 pts:", int(ok.sum()))
print("candle1 range quantiles (filtered):", c1_range[ok].quantile([.1, .5, .9]).round(2).to_dict())

cost_pts = 2 * (0.0005 + 0.0002) * c1_price[ok]   # taker-style round trip
print("median cost in R:", round((cost_pts / c1_range[ok]).median(), 2))