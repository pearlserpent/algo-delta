"""
Two-timeframe FVG strategy (ETHUSD perps, Delta India). Results in R-multiples.

Flow
  1. 15m: same setup as before (3 bullish/bearish candles with a gap, size filter,
     BOS, HH/HL trend). Its candle-1 extreme = ZONE ENTRY, other end = ZONE STOP.
     No order is placed on 15m.
  2. 3m: wait for price to trade back to the zone entry ("touch").
  3. After the touch, wait for a FRESH 3m setup (same rules on 3m bars: 3 candles
     with a gap, size filter, BOS, HH/HL trend re-checked on 3m). Its candle 1
     may be the touch bar or any later bar.
  4. Limit order at the 3m candle-1 extreme, stop at the other end, target = target_r * R.
  5. Zone dies if the zone stop is broken first, or a timer runs out.

Execution assumptions (same conservative ones as before)
  - Signals known at bar close, orders live from the NEXT 3m bar
  - Fill bar: only the stop is checked. Stop and target in one bar -> stop first
  - Entry/target = maker fee, stop/EOD exit = taker fee + slippage (fees include 18% GST)
  - One zone/order/position at a time; new 15m setups arriving meanwhile are skipped
  - If a 3m order expires unfilled, the zone is dropped (no second attempt)
  - 15m candles are built from the 3m data, so both timeframes line up exactly

Protocol: fix CONFIG BEFORE looking at results, tune only on the development slice,
look at the test slice once. Every extra variation you try adds chances of fitting noise.
"""
import os
import time
import numpy as np
import pandas as pd
import requests

from fvg_backtest import find_setups, summarize, by_size, MAKER, TAKER

BASE = "https://api.india.delta.exchange"     # global: https://api.delta.exchange
BAR_SECONDS = 180


# ----------------------------------------------------------------- data
def fetch_3m(symbol="ETHUSD", days=365, chunk=1500, cache=True) -> pd.DataFrame:
    path = f"{symbol}_3m_{days}d.csv"
    if cache and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    end = int(time.time())
    t = end - days * 86400
    rows = []
    while t < end:
        t2 = min(t + chunk * BAR_SECONDS, end)
        r = requests.get(f"{BASE}/v2/history/candles",
                         params={"symbol": symbol, "resolution": "3m", "start": t, "end": t2},
                         timeout=20)
        r.raise_for_status()
        rows += r.json().get("result", [])
        t = t2
        time.sleep(0.25)
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float).iloc[:-1]
    if cache:
        df.to_csv(path)
    return df


def to_15m(df3: pd.DataFrame) -> pd.DataFrame:
    g = df3.resample("15min", label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out[g["close"].count() == 5]          # keep only complete 15m candles


# ------------------------------------------------------------- backtest
def backtest_mtf(df3, max_range15=10.0, min_range15=0.0, trend_k15=10, k15=3,
                 max_range3=10.0, min_range3=3.0, trend_k3=10, k3=3,
                 target_r=2.0, zone_expiry15=20, wait3=40, expiry3=20,
                 fill="touch", tick=0.05, maker=MAKER, taker=TAKER,
                 stop_slip=0.0002, fresh_bos=True, use_bos=True,
                 fresh_bos3=None, use_bos3=None):
    df15 = to_15m(df3)
    o15, h15, l15, c15 = (df15[x].to_numpy() for x in ("open", "high", "low", "close"))
    o, h, l, c = (df3[x].to_numpy() for x in ("open", "high", "low", "close"))
    idx3, idx15 = df3.index, df15.index
    n = len(df3)

    s15 = find_setups(o15, h15, l15, c15, max_range15, min_range15, k15, fresh_bos, trend_k15, use_bos)
    s3 = find_setups(o, h, l, c, max_range3, min_range3, k3,
                     fresh_bos if fresh_bos3 is None else fresh_bos3, trend_k3,
                     use_bos if use_bos3 is None else use_bos3)

    # 15m setup known at close of 15m candle 3 -> first 3m bar starting at/after that time
    zone_at = {}
    for t, v in s15.items():
        j = int(idx3.searchsorted(idx15[t] + pd.Timedelta("15min")))
        if j < n:
            zone_at[j] = v

    thr = tick if fill == "through" else 0.0
    zexp = zone_expiry15 * 5
    trades = []
    funnel = {"zones": 0, "touched": 0, "orders": 0, "filled": 0}
    state = 0          # 0 flat, 1 waiting touch, 2 waiting 3m pattern, 3 resting order, 4 in position
    s = 1
    Z = ZS = E = S = TP = R = fpx = 0.0
    zj = jt = placed = entry_i = 0

    def record(exit_px, exit_fee, reason, t):
        gross = exit_px - fpx
        fees = maker * abs(fpx) + exit_fee * abs(exit_px)
        trades.append({
            "zone_time": idx3[zj], "touch_time": idx3[jt], "entry_time": idx3[entry_i],
            "exit_time": idx3[t], "side": s, "entry": s * fpx, "exit": s * exit_px,
            "R_pts": R, "reason": reason, "gross_R": gross / R, "net_R": (gross - fees) / R,
        })

    for j in range(n):
        if state == 0 and j in zone_at:
            s, zr, zsr = zone_at[j]
            Z, ZS, zj, state = s * zr, s * zsr, j, 1
            funnel["zones"] += 1

        if state:
            if s == 1:
                so, sh, sl = o[j], h[j], l[j]
            else:
                so, sh, sl = -o[j], -l[j], -h[j]

            if state == 1:                                   # wait for return to zone entry
                if j - zj > zexp or sl <= ZS:
                    state = 0
                elif sl <= Z:
                    jt, state = j, 2
                    funnel["touched"] += 1
            elif state == 2:                                 # wait for fresh 3m impulse
                if sl <= ZS or j - jt > wait3:
                    state = 0
                elif j in s3 and s3[j][0] == s and j - 2 >= jt:
                    _, e3, st3 = s3[j]
                    E, S = s * e3, s * st3
                    R = E - S
                    TP = E + target_r * R
                    placed, state = j, 3
                    funnel["orders"] += 1
            elif state == 3:                                 # resting 3m limit order
                if j - placed > expiry3 or so <= S:
                    state = 0
                elif sl <= E - thr:
                    fpx, entry_i, state = min(so, E), j, 4
                    funnel["filled"] += 1
                    if sl <= S:
                        record(S - abs(S) * stop_slip, taker, "stop", j)
                        state = 0
            elif state == 4:
                if sl <= S:
                    record(min(so, S) - abs(S) * stop_slip, taker, "stop", j)
                    state = 0
                elif sh >= TP + thr:
                    record(max(so, TP), maker, "target", j)
                    state = 0

    if state == 4:
        record(s * c[-1], taker, "eod", n - 1)
    return pd.DataFrame(trades), funnel


def summarize_mtf(tr, funnel):
    return {**{k: funnel[k] for k in ("zones", "touched")}, **summarize(tr, funnel["orders"])}


def sweep(df3, **kw):
    rows = []
    for tgt in (2, 3):
        for mn in (0, 2, 3, 5):
            tr, f = backtest_mtf(df3, target_r=tgt, min_range3=mn, **kw)
            rows.append({"tgt_R": tgt, "min3": mn, "trend3": "on", **summarize_mtf(tr, f)})
    tr, f = backtest_mtf(df3, target_r=2, min_range3=3, trend_k3=None, **kw)
    rows.append({"tgt_R": 2, "min3": 3, "trend3": "off", **summarize_mtf(tr, f)})
    return pd.DataFrame(rows)


# Fixed in advance. Change only if you accept that the test slice is then no longer clean.
CONFIG = dict(max_range15=10.0, min_range15=0.0, trend_k15=10,
              max_range3=10.0, min_range3=3.0, trend_k3=10, target_r=2.0)

if __name__ == "__main__":
    df3 = fetch_3m("ETHUSD", days=365)
    split = df3.index[int(len(df3) * 0.6)].ceil("15min")
    dev, test = df3.loc[:split], df3.loc[split:]
    for name, d in (("development", dev), ("test", test)):
        print(f"{name}: {len(d)} bars  {d.index[0]} -> {d.index[-1]}  median price {d.close.median():.0f}")

    print("\nDevelopment slice, grid:")
    print(sweep(dev).to_string(index=False))

    print("\nCONFIG on development vs test (look at the test line once):")
    for name, d in (("dev", dev), ("test", test)):
        tr, f = backtest_mtf(d, **CONFIG)
        print(name, summarize_mtf(tr, f))
        if name == "dev" and not tr.empty:
            print(by_size(tr, bins=(0, 3, 5, 7, 10)).to_string())