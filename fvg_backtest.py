"""
3-candle imbalance (FVG) + break-of-structure pullback strategy.
ETHUSD perps on Delta Exchange India, 15m. Results are in R-multiples.

Rules (bullish; bearish is the mirror image)
  - Candles 1,2,3 all bullish (close > open)
  - Gap: high[c1] < low[c3]
  - Candle-1 range within [min_range, max_range] points
  - BOS: one of the 3 closes breaks the latest CONFIRMED swing high
         (swing = k bars each side, known only k bars after it forms)
  - Limit buy at high[c1]; stop at low[c1]; target = entry + target_r * R
  - Order expires after `expiry` bars; one order/position at a time

Execution assumptions (conservative)
  - Setup known at close of candle 3; order live from the NEXT bar
  - fill="touch": price touching the level fills. fill="through": must trade
    one tick beyond it (applies to entry and target)
  - On the fill bar only the stop is checked (bar order is unknown)
  - Stop and target same bar -> stop assumed first
  - Entry/target = maker fee, stop/EOD exit = taker fee + slippage
  - Fees include 18% GST. Check your tier and the Scalper offer.
"""
import numpy as np
import pandas as pd

MAKER = 0.0002 * 1.18
TAKER = 0.0005 * 1.18


def swing_levels(h, l, k):
    """Latest confirmed swing high/low, forward-filled. A swing at bar j is
    only known at bar j+k, so index j+k carries it (no lookahead)."""
    n = len(h)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for j in range(k, n - k):
        if h[j] > h[j - k:j].max() and h[j] > h[j + 1:j + k + 1].max():
            sh[j + k] = h[j]
        if l[j] < l[j - k:j].min() and l[j] < l[j + 1:j + k + 1].min():
            sl[j + k] = l[j]
    return pd.Series(sh).ffill().to_numpy(), pd.Series(sl).ffill().to_numpy()


def trend_flags(h, l, tk):
    """Structure trend from the last two CONFIRMED swing highs and lows.
    up[t]   : latest swing high > previous one AND latest swing low > previous one
    down[t] : both lower. Values are as known at the close of bar t."""
    n = len(h)
    ev_h, ev_l = {}, {}
    for j in range(tk, n - tk):
        if h[j] > h[j - tk:j].max() and h[j] > h[j + 1:j + tk + 1].max():
            ev_h[j + tk] = h[j]
        if l[j] < l[j - tk:j].min() and l[j] < l[j + 1:j + tk + 1].min():
            ev_l[j + tk] = l[j]
    up = np.zeros(n, bool)
    down = np.zeros(n, bool)
    h1 = h2 = l1 = l2 = np.nan
    for t in range(n):
        if t in ev_h:
            h2, h1 = h1, ev_h[t]
        if t in ev_l:
            l2, l1 = l1, ev_l[t]
        if not (np.isnan(h2) or np.isnan(l2)):
            up[t] = h1 > h2 and l1 > l2
            down[t] = h1 < h2 and l1 < l2
    return up, down


def find_setups(o, h, l, c, max_range, min_range, k, fresh_bos,
                trend_k=None, use_bos=True):
    """Returns {bar_index_of_candle3: (side, entry, stop)}.
    trend_k: if set, longs need an HH/HL structure and shorts LH/LL, judged
    from swings (trend_k bars each side) confirmed before candle 1."""
    sw_h, sw_l = swing_levels(h, l, k)
    up, down = trend_flags(h, l, trend_k) if trend_k else (None, None)
    out = {}
    for t in range(3, len(c)):
        r1 = h[t - 2] - l[t - 2]
        if r1 <= 0 or r1 < min_range or r1 > max_range:
            continue
        if all(c[i] > o[i] for i in (t - 2, t - 1, t)) and h[t - 2] < l[t]:
            if up is not None and not up[t - 3]:
                continue
            L = sw_h[t - 3]  # level known before the impulse began
            if (not use_bos) or (not np.isnan(L) and c[t - 2:t + 1].max() > L
                                 and (not fresh_bos or c[t - 3] <= L)):
                out[t] = (1, h[t - 2], l[t - 2])
        elif all(c[i] < o[i] for i in (t - 2, t - 1, t)) and l[t - 2] > h[t]:
            if down is not None and not down[t - 3]:
                continue
            L = sw_l[t - 3]
            if (not use_bos) or (not np.isnan(L) and c[t - 2:t + 1].min() < L
                                 and (not fresh_bos or c[t - 3] >= L)):
                out[t] = (-1, l[t - 2], h[t - 2])
    return out


def backtest(df, max_range=10.0, min_range=0.0, k=3, expiry=20, target_r=2.0,
             fill="touch", tick=0.05, maker=MAKER, taker=TAKER,
             stop_slip=0.0002, fresh_bos=True, trend_k=None, use_bos=True):
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    idx = df.index
    n = len(df)
    setups = find_setups(o, h, l, c, max_range, min_range, k, fresh_bos, trend_k, use_bos)
    thr = tick if fill == "through" else 0.0

    trades, n_orders = [], 0
    state = 0            # 0 flat, 1 resting order, 2 in position
    s, E, S, TP, R = 1, 0.0, 0.0, 0.0, 0.0   # prices are in "long space" (x s)
    placed, entry_i, fpx = 0, 0, 0.0

    def record(exit_px, exit_fee, reason, t):
        gross = exit_px - fpx
        fees = maker * abs(fpx) + exit_fee * abs(exit_px)
        trades.append({
            "setup_time": idx[placed], "entry_time": idx[entry_i], "exit_time": idx[t],
            "side": s, "entry": s * fpx, "exit": s * exit_px, "R_pts": R,
            "reason": reason, "gross_R": gross / R, "net_R": (gross - fees) / R,
        })

    for t in range(n):
        if state:
            if s == 1:
                so, sh, sl = o[t], h[t], l[t]
            else:
                so, sh, sl = -o[t], -l[t], -h[t]

            if state == 1:                                  # resting limit order
                if t - placed > expiry or so <= S:          # expired / gapped past stop
                    state = 0
                elif sl <= E - thr:                         # filled
                    fpx, entry_i, state = min(so, E), t, 2
                    if sl <= S:                             # stopped on the fill bar
                        record(S - abs(S) * stop_slip, taker, "stop", t)
                        state = 0
            elif state == 2:
                if sl <= S:
                    record(min(so, S) - abs(S) * stop_slip, taker, "stop", t)
                    state = 0
                elif sh >= TP + thr:
                    record(max(so, TP), maker, "target", t)
                    state = 0

        if state == 0 and t in setups:
            s, e, st = setups[t]
            E, S = s * e, s * st
            R = E - S
            TP = E + target_r * R
            placed, state = t, 1
            n_orders += 1

    if state == 2:                                          # close leftover at last close
        record(s * c[-1], taker, "eod", n - 1)

    return pd.DataFrame(trades), n_orders


def summarize(tr, n_orders):
    if tr.empty:
        return {"orders": n_orders, "trades": 0}
    r = tr["net_R"]
    losses = -r[r <= 0].sum()
    cum = r.cumsum()
    return {
        "orders": n_orders,
        "trades": len(tr),
        "fill_%": round(100 * len(tr) / n_orders, 1),
        "win_%": round(100 * (r > 0).mean(), 1),
        "expect_R": round(r.mean(), 3),
        "PF": round(r[r > 0].sum() / losses, 2) if losses > 0 else np.inf,
        "total_R": round(r.sum(), 1),
        "maxDD_R": round((cum - cum.cummax()).min(), 1),
        "cost_R": round((tr["gross_R"] - r).mean(), 2),
    }


def sweep(df, **kw):
    rows = []
    for tgt in (2, 3):
        for fill in ("touch", "through"):
            for mn in (0, 3, 5):
                tr, n_orders = backtest(df, target_r=tgt, fill=fill, min_range=mn, **kw)
                rows.append({"tgt_R": tgt, "fill": fill, "min_pts": mn,
                             **summarize(tr, n_orders)})
    return pd.DataFrame(rows)


def by_size(tr, bins=(0, 3, 5, 7, 10)):
    b = pd.cut(tr["R_pts"], list(bins))
    g = tr.groupby(b, observed=True)
    return pd.DataFrame({"n": g.size(), "win_%": g["net_R"].apply(lambda x: round(100 * (x > 0).mean(), 1)),
                         "expect_R": g["net_R"].mean().round(3), "cost_R": (g["gross_R"].mean() - g["net_R"].mean()).round(2)})


if __name__ == "__main__":
    from donchian_backtest import fetch_candles   # reuse the data fetcher

    df = fetch_candles("ETHUSD", "15m", days=180)
    print(f"{len(df)} bars  {df.index[0]} -> {df.index[-1]}")

    # Pre-registered comparison: same rules, trend filter off vs on (3 swing sizes).
    # trend_k=10 means a swing = highest/lowest of 10 bars each side (2.5h).
    rows = []
    for use_bos in (True, False):
        for tk in (None, 5, 10, 20):
            for mx, mn in ((10, 0), (15, 7)):
                tr, no = backtest(df, max_range=mx, min_range=mn, target_r=2,
                                  trend_k=tk, use_bos=use_bos)
                rows.append({"bos": use_bos, "trend_k": tk, "cap": mx, "floor": mn,
                             **summarize(tr, no)})
    print(pd.DataFrame(rows).to_string(index=False))