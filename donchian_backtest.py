"""
Donchian breakout backtest for Delta Exchange (15m candles).

- Public candle data, no API key needed.
- Signal on bar close, fill at NEXT bar open (no lookahead).
- ATR stop checked intrabar, with gap-aware fills.
- Exit on shorter opposite channel (close-based), filled next open.
- Fees + slippage applied on every fill.

Verify against https://docs.delta.exchange before relying on it:
base URL (India vs global), symbol names, candle limit per request, fee tier.
"""
import time
import numpy as np
import pandas as pd
import requests

BASE = "https://api.india.delta.exchange"   # global: https://api.delta.exchange
RES_SECONDS = 900                            # 15m
MAX_BARS_PER_REQ = 1000                      # conservative chunk size


# ----------------------------------------------------------------- data
def fetch_candles(symbol="ETHUSD", resolution="15m", days=180) -> pd.DataFrame:
    end = int(time.time())
    t = end - days * 86400
    rows = []
    while t < end:
        t2 = min(t + MAX_BARS_PER_REQ * RES_SECONDS, end)
        r = requests.get(
            f"{BASE}/v2/history/candles",
            params={"symbol": symbol, "resolution": resolution, "start": t, "end": t2},
            timeout=15,
        )
        r.raise_for_status()
        rows += r.json().get("result", [])
        t = t2
        time.sleep(0.25)  # stay well under rate limits
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)
    return df.iloc[:-1]  # drop the possibly unfinished last candle


# ------------------------------------------------------------- backtest
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n).mean()


def backtest(df, n=96, m=48, atr_mult=3.0, fee=0.0005, slip=0.0002) -> pd.DataFrame:
    """fee and slip are per-side fractions of notional. Returns a trades frame."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = df.index
    up = df.high.rolling(n).max().shift(1).to_numpy()
    lo = df.low.rolling(n).min().shift(1).to_numpy()
    xhi = df.high.rolling(m).max().shift(1).to_numpy()
    xlo = df.low.rolling(m).min().shift(1).to_numpy()
    a = atr(df).to_numpy()

    trades = []
    pos, entry, stop, entry_t = 0, 0.0, 0.0, None
    pend_side, pend_atr, pend_exit = 0, 0.0, False

    def close_trade(px, t):
        nonlocal pos
        ret = pos * (px / entry - 1) - 2 * fee
        trades.append({"entry_time": entry_t, "exit_time": idx[t], "side": pos, "ret": ret})
        pos = 0

    for t in range(len(df)):
        # 1) fills at this bar's open
        if pend_exit and pos != 0:
            close_trade(o[t] * (1 - slip * pos), t)
        pend_exit = False
        if pend_side != 0 and pos == 0:
            pos = pend_side
            entry = o[t] * (1 + slip * pos)
            stop = entry - pos * atr_mult * pend_atr
            entry_t = idx[t]
        pend_side = 0

        # 2) intrabar stop (gap-aware)
        if pos == 1 and l[t] <= stop:
            close_trade(min(o[t], stop) * (1 - slip), t)
        elif pos == -1 and h[t] >= stop:
            close_trade(max(o[t], stop) * (1 + slip), t)

        # 3) decisions on this bar's close, executed next open
        if pos != 0:
            if (pos == 1 and c[t] < xlo[t]) or (pos == -1 and c[t] > xhi[t]):
                pend_exit = True
        elif not (np.isnan(up[t]) or np.isnan(a[t])):
            if c[t] > up[t]:
                pend_side, pend_atr = 1, a[t]
            elif c[t] < lo[t]:
                pend_side, pend_atr = -1, a[t]

    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0}
    eq = (1 + tr["ret"]).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    wins, losses = tr.ret[tr.ret > 0], tr.ret[tr.ret <= 0]
    return {
        "trades": len(tr),
        "win_rate": round((tr.ret > 0).mean(), 3),
        "avg_ret_%": round(tr.ret.mean() * 100, 3),
        "profit_factor": round(wins.sum() / abs(losses.sum()), 2) if len(losses) else np.inf,
        "total_ret_%": round((eq.iloc[-1] - 1) * 100, 1),
        "max_dd_%": round(dd * 100, 1),
    }


def sweep(df, ns=(24, 48, 96, 192, 288, 480, 672), **kw) -> pd.DataFrame:
    out = []
    for n in ns:
        m = max(n // 2, 5)
        out.append({"n": n, "m": m, **summarize(backtest(df, n=n, m=m, **kw))})
    return pd.DataFrame(out).set_index("n")



if __name__ == "__main__":
    """ data = fetch_candles("ETHUSD", "15m", days=180)
    data.to_csv("ETHusd_15m.csv")
    print(f"{len(data)} bars  {data.index[0]} -> {data.index[-1]}")
    print(sweep(data).to_string()) """

    for sym in ["BTCUSD", "ETHUSD", "SOLUSD"]:
        df = fetch_candles(sym, "15m", days=180)
        print(sym)
        print("gross:\n", sweep(df, fee=0, slip=0, ns=(288, 480, 672)))
        print("net:\n",   sweep(df, ns=(288, 480, 672)))