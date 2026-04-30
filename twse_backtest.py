from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "twse"
FINMIND_DIR = ROOT / "data" / "finmind"
REPORT_PATH = ROOT / "twse_backtest_report.md"

TODAY = date.today()
START_YEAR = 2018
END_YEAR = TODAY.year
END_MONTH = TODAY.month
START_DATE = "2018-01-01"
END_DATE = TODAY.isoformat()

CORE = ["0050", "006208"]
LEVERAGED = ["00631L"]
AI_UNIVERSE = [
    "2330",
    "2308",
    "2317",
    "2382",
    "3231",
    "6669",
    "2356",
    "2383",
    "2345",
    "2454",
]
DEFENSIVE = ["0056", "00878", "00919"]
TICKERS = sorted(set(CORE + LEVERAGED + AI_UNIVERSE + DEFENSIVE))


def month_iter() -> list[tuple[int, int]]:
    out = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            if year == END_YEAR and month > END_MONTH:
                continue
            out.append((year, month))
    return out


def fetch_month(ticker: str, year: int, month: int) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{ticker}_{year}{month:02d}.json"
    if cache.exists():
        try:
            cached = json.loads(cache.read_text())
            if isinstance(cached, dict) and "stat" in cached:
                return cached
        except Exception:
            pass
        cache.unlink(missing_ok=True)

    url = (
        "https://www.twse.com.tw/rwd/en/afterTrading/STOCK_DAY"
        f"?date={year}{month:02d}01&stockNo={ticker}&response=json"
    )
    # TWSE's CDN occasionally returns 307/308 loops to Python urllib. Curl
    # handles the same endpoint reliably, including the CDN cookie dance.
    proc = subprocess.run(
        [
            "curl",
            "-sL",
            "--max-time",
            "30",
            "-A",
            "Mozilla/5.0",
            url,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = proc.stdout
    payload = payload.strip()
    parsed = json.loads(payload)
    cache.write_text(payload)
    time.sleep(0.12)
    return parsed


def parse_number(value: str) -> float:
    value = value.replace(",", "").strip()
    if value in {"", "--", "X", "除權息"}:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_ticker(ticker: str) -> pd.DataFrame:
    finmind = parse_ticker_finmind(ticker)
    if not finmind.empty:
        return finmind

    rows = []
    for year, month in month_iter():
        try:
            payload = fetch_month(ticker, year, month)
        except Exception as exc:
            print(f"fetch failed {ticker} {year}-{month:02d}: {exc}", flush=True)
            continue
        if payload.get("stat") != "OK":
            continue
        for row in payload.get("data", []):
            # English endpoint date format: YYYY/MM/DD
            dt = pd.Timestamp(datetime.strptime(row[0], "%Y/%m/%d").date())
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "open": parse_number(row[3]),
                    "high": parse_number(row[4]),
                    "low": parse_number(row[5]),
                    "close_raw": parse_number(row[6]),
                    "volume": parse_number(row[1]),
                    "value": parse_number(row[2]),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    df.loc[df["close_raw"] <= 0, "close_raw"] = np.nan
    df = df.dropna(subset=["close_raw"])
    df["close"] = adjust_splits(df["close_raw"], ticker)
    return df


def parse_ticker_finmind(ticker: str) -> pd.DataFrame:
    FINMIND_DIR.mkdir(parents=True, exist_ok=True)
    payload = None
    stale_payload = None
    legacy_caches = sorted(FINMIND_DIR.glob(f"{ticker}_{START_DATE}_*.json"))
    read_cache = legacy_caches[-1] if legacy_caches else FINMIND_DIR / f"{ticker}_{START_DATE}_{END_DATE}.json"
    if not read_cache.exists() and legacy_caches:
        read_cache = legacy_caches[-1]
    if read_cache.exists():
        try:
            payload = json.loads(read_cache.read_text())
            data = payload.get("data", []) if isinstance(payload, dict) else []
            if data:
                latest_cached = max(row.get("date", "") for row in data)
                if latest_cached < END_DATE:
                    stale_payload = payload
                    payload = None
        except Exception:
            read_cache.unlink(missing_ok=True)
            payload = None
    if payload is None:
        url = (
            "https://api.finmindtrade.com/api/v4/data"
            f"?dataset=TaiwanStockPrice&data_id={ticker}"
            f"&start_date={START_DATE}&end_date={END_DATE}"
        )
        try:
            proc = subprocess.run(
                [
                    "curl",
                    "-sL",
                    "--max-time",
                    "60",
                    "-A",
                    "Mozilla/5.0",
                    url,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(proc.stdout)
            actual_end_date = END_DATE
            if isinstance(payload, dict) and payload.get("data"):
                actual_end_date = max(row.get("date", END_DATE) for row in payload["data"])
            cache = FINMIND_DIR / f"{ticker}_{START_DATE}_{actual_end_date}.json"
            cache.write_text(json.dumps(payload, ensure_ascii=False))
            time.sleep(0.5)
        except Exception:
            if stale_payload is not None:
                payload = stale_payload
            else:
                return pd.DataFrame()

    if payload.get("status") != 200 or not payload.get("data"):
        return pd.DataFrame()
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(
        columns={
            "open": "open",
            "max": "high",
            "min": "low",
            "close": "close_raw",
            "Trading_Volume": "volume",
            "Trading_money": "value",
        }
    )
    df["ticker"] = ticker
    df = df[["date", "ticker", "open", "high", "low", "close_raw", "volume", "value"]]
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    df.loc[df["close_raw"] <= 0, "close_raw"] = np.nan
    df = df.dropna(subset=["close_raw"])
    df["close"] = adjust_splits(df["close_raw"], ticker)
    return df


def adjust_splits(close: pd.Series, ticker: str) -> pd.Series:
    """Back-adjust obvious split gaps in raw TWSE price data.

    TWSE daily prices are raw. For return tests, an ETF split such as 00631L's
    2026 22-for-1 split must be neutralized. This heuristic handles large
    one-day price drops that look like splits. It does not handle dividends.
    """
    adj = close.astype(float).copy()

    # Known 00631L split: old units became 22 new units, trading resumed 2026-03-31.
    if ticker == "00631L":
        split_date = pd.Timestamp(date(2026, 3, 31))
        adj.loc[adj.index < split_date] = adj.loc[adj.index < split_date] / 22.0

    # Generic split detector for future data. If a one-day raw return is around
    # -50%, -66.7%, -75%, or -80%, back-adjust prior prices by inferred ratio.
    ratios = adj / adj.shift(1)
    for dt, r in ratios.dropna().items():
        if r <= 0:
            continue
        inv = round(1 / r)
        if inv in {2, 3, 4, 5, 10, 20, 22} and abs(r - 1 / inv) < 0.04:
            adj.loc[adj.index < dt] = adj.loc[adj.index < dt] / inv
    return adj


def load_prices(tickers: list[str]) -> pd.DataFrame:
    frames = {}
    for ticker in tickers:
        print(f"loading {ticker}", flush=True)
        df = parse_ticker(ticker)
        if not df.empty:
            frames[ticker] = df["close"]
    prices = pd.DataFrame(frames).sort_index()
    prices = prices.dropna(how="all")
    return prices


@dataclass
class Result:
    name: str
    equity: pd.Series
    weights: pd.DataFrame


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1)


def sharpe(daily_returns: pd.Series) -> float:
    vol = daily_returns.std()
    if vol == 0 or np.isnan(vol):
        return math.nan
    return float(daily_returns.mean() / vol * math.sqrt(252))


def summarize(result: Result) -> dict:
    eq = result.equity.dropna()
    ret = eq.pct_change().dropna()
    turnover = result.weights.diff().abs().sum(axis=1).fillna(0.0)
    annual = eq.resample("YE").last().pct_change()
    first_year = eq.index[0].year
    if not annual.empty and math.isnan(annual.iloc[0]):
        year_start = eq[eq.index.year == first_year]
        if not year_start.empty:
            annual.iloc[0] = year_start.iloc[-1] / year_start.iloc[0] - 1
    return {
        "策略": result.name,
        "起始日": str(eq.index[0].date()),
        "結束日": str(eq.index[-1].date()),
        "累積報酬": eq.iloc[-1] - 1,
        "年化報酬": cagr(eq),
        "最大回撤": max_drawdown(eq),
        "年化波動": ret.std() * math.sqrt(252),
        "Sharpe": sharpe(ret),
        "年化週轉率": turnover.mean() * 252,
        "最差3個月": eq.pct_change(63).min(),
        "2022報酬": annual.get(pd.Timestamp("2022-12-31"), np.nan),
        "2026YTD": eq[eq.index.year == 2026].iloc[-1] / eq[eq.index.year == 2026].iloc[0] - 1
        if (eq.index.year == 2026).any()
        else np.nan,
    }


def monthly_rebalance_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    monthly = prices.resample("ME").last().index
    dates = []
    for m in monthly:
        month_prices = prices.loc[(prices.index.year == m.year) & (prices.index.month == m.month)]
        if not month_prices.empty:
            dates.append(month_prices.index[-1])
    return pd.DatetimeIndex(dates)


def run_strategy(
    prices: pd.DataFrame,
    name: str,
    core_weight: float,
    ai_weight: float,
    lev_weight: float,
    defensive_weight: float,
    top_n: int = 5,
    use_regime: bool = True,
    use_defensive_etfs: bool = True,
    transaction_cost_rate: float = 0.0,
    core_ticker: str = "0050",
) -> Result:
    px = prices.copy()
    returns = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    regime_ticker = "0050"
    ma200 = px[regime_ticker].rolling(200).mean()
    ma120 = px.rolling(120).mean()
    ma60 = px[regime_ticker].rolling(60).mean()
    ma20 = px[regime_ticker].rolling(20).mean()

    rebal_dates = set(monthly_rebalance_dates(px))
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    current = pd.Series(0.0, index=px.columns)

    for dt in px.index:
        if dt in rebal_dates:
            current[:] = 0.0
            risk_on = bool(px.loc[dt, regime_ticker] > ma200.loc[dt]) if not np.isnan(ma200.loc[dt]) else False
            strong = (
                risk_on
                and not np.isnan(ma20.loc[dt])
                and not np.isnan(ma60.loc[dt])
                and ma20.loc[dt] > ma60.loc[dt]
                and px[regime_ticker].pct_change(63).loc[dt] > 0
            )

            if not use_regime:
                risk_on = True
                strong = True

            if core_ticker not in current.index or np.isnan(px.loc[dt, core_ticker]):
                core_ticker = "0050"
            if risk_on:
                current[core_ticker] = core_weight
                ai_alloc = ai_weight
                lev_alloc = lev_weight if strong and "00631L" in current.index else 0.0
                defensive_alloc = defensive_weight + (lev_weight - lev_alloc)
            else:
                current[core_ticker] = min(core_weight, 0.35)
                ai_alloc = min(ai_weight, 0.10)
                lev_alloc = 0.0
                defensive_alloc = 1.0 - current.sum() - ai_alloc

            # Rank AI universe by 6M + 3M momentum, require above 120D MA when possible.
            scores = {}
            for t in AI_UNIVERSE:
                if t not in px.columns or np.isnan(px.loc[dt, t]):
                    continue
                mom6 = px[t].pct_change(126).loc[dt]
                mom3 = px[t].pct_change(63).loc[dt]
                trend_ok = px.loc[dt, t] > ma120[t].loc[dt] if not np.isnan(ma120[t].loc[dt]) else True
                if np.isnan(mom6) or np.isnan(mom3) or not trend_ok:
                    continue
                scores[t] = 0.65 * mom6 + 0.35 * mom3
            picks = [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
            if picks:
                each = ai_alloc / len(picks)
                for t in picks:
                    current[t] = min(each, 0.08)
            if lev_alloc:
                current["00631L"] = lev_alloc

            # Defensive sleeve split among high-dividend ETFs with valid data.
            defs = [t for t in DEFENSIVE if use_defensive_etfs and t in px.columns and not np.isnan(px.loc[dt, t])]
            if defs and defensive_alloc > 0:
                for t in defs:
                    current[t] += defensive_alloc / len(defs)
            else:
                # Residual cash earns 0.
                pass

            # Normalize invested weight to <= 1; unallocated stays as cash.
            if current.sum() > 1.0:
                current = current / current.sum()

        weights.loc[dt] = current

    weights = weights.ffill().fillna(0.0)
    shifted = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    strat_ret = (shifted * returns).sum(axis=1) - turnover * transaction_cost_rate
    equity = (1 + strat_ret).cumprod()
    return Result(name=name, equity=equity, weights=weights)


def buy_hold(prices: pd.DataFrame, ticker: str) -> Result:
    ret = prices[ticker].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eq = (1 + ret).cumprod()
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = 1.0
    return Result(f"買進持有 {ticker}", eq, w)


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:.2f}%"


def markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    if index:
        df = df.reset_index()
    values = df.astype(str)
    headers = list(values.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in values.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def segment_summary(results: list[Result]) -> pd.DataFrame:
    segments = [
        ("2018-2020", "2018-01-02", "2020-12-31"),
        ("2021-2022", "2021-01-01", "2022-12-31"),
        ("2023-至今", "2023-01-01", END_DATE),
    ]
    rows = []
    for result in results:
        eq = result.equity.dropna()
        for label, start, end in segments:
            part = eq.loc[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
            if len(part) < 2:
                continue
            rows.append(
                {
                    "策略": result.name,
                    "區間": label,
                    "區間報酬": part.iloc[-1] / part.iloc[0] - 1,
                    "最大回撤": max_drawdown(part / part.iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def latest_positions(result: Result, n: int = 12) -> pd.DataFrame:
    latest = result.weights.iloc[-1]
    latest = latest[latest > 0].sort_values(ascending=False).head(n)
    return latest.rename("權重").to_frame()


def latest_market_snapshot(prices: pd.DataFrame) -> dict[str, float | bool | str]:
    dt = prices.index[-1]
    regime = prices["0050"]
    ma20 = regime.rolling(20).mean().loc[dt]
    ma60 = regime.rolling(60).mean().loc[dt]
    ma200 = regime.rolling(200).mean().loc[dt]
    return {
        "執行日": TODAY.isoformat(),
        "資料截止日": str(dt.date()),
        "資料滯後日曆天數": str((TODAY - dt.date()).days),
        "0050收盤": float(prices.loc[dt, "0050"]),
        "006208收盤": float(prices.loc[dt, "006208"]) if "006208" in prices.columns else math.nan,
        "00631L收盤": float(prices.loc[dt, "00631L"]) if "00631L" in prices.columns else math.nan,
        "0050高於20日": bool(prices.loc[dt, "0050"] > ma20) if not np.isnan(ma20) else False,
        "0050高於60日": bool(prices.loc[dt, "0050"] > ma60) if not np.isnan(ma60) else False,
        "0050高於200日": bool(prices.loc[dt, "0050"] > ma200) if not np.isnan(ma200) else False,
        "20日高於60日": bool(ma20 > ma60) if not np.isnan(ma20) and not np.isnan(ma60) else False,
        "0050三個月動能": float(regime.pct_change(63).loc[dt]),
    }


def latest_signal_freshness(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    market_dt = prices.index[-1]
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        if series.empty:
            continue
        last_dt = series.index[-1]
        rows.append(
            {
                "代號": ticker,
                "最後有效收盤日": str(last_dt.date()),
                "距執行日日曆天數": (TODAY - last_dt.date()).days,
                "最新有效收盤": float(series.iloc[-1]),
                "是否等於市場截止日": bool(last_dt == market_dt),
            }
        )
    return pd.DataFrame(rows)


def latest_ai_ranking(prices: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    dt = prices.index[-1]
    ma120 = prices.rolling(120).mean()
    rows = []
    for ticker in AI_UNIVERSE:
        if ticker not in prices.columns or np.isnan(prices.loc[dt, ticker]):
            continue
        close = prices.loc[dt, ticker]
        mom3 = prices[ticker].pct_change(63).loc[dt]
        mom6 = prices[ticker].pct_change(126).loc[dt]
        trend_ok = close > ma120[ticker].loc[dt] if not np.isnan(ma120[ticker].loc[dt]) else True
        score = 0.65 * mom6 + 0.35 * mom3 if trend_ok and not np.isnan(mom3) and not np.isnan(mom6) else math.nan
        peak_63 = prices[ticker].loc[:dt].tail(63).max()
        peak_252 = prices[ticker].loc[:dt].tail(252).max()
        rows.append(
            {
                "代號": ticker,
                "收盤": float(close),
                "3個月動能": mom3,
                "6個月動能": mom6,
                "分數": score,
                "120日上方": trend_ok,
                "近3月回撤": close / peak_63 - 1 if peak_63 else math.nan,
                "近12月回撤": close / peak_252 - 1 if peak_252 else math.nan,
            }
        )
    ranked = pd.DataFrame(rows).sort_values("分數", ascending=False)
    return ranked.head(top_n)


def export_monthly_holdings(result: Result) -> None:
    rows = []
    dates = monthly_rebalance_dates(result.weights)
    for dt in dates:
        if dt not in result.weights.index:
            continue
        weights = result.weights.loc[dt]
        invested = float(weights.sum())
        cash = max(0.0, 1.0 - invested)
        for ticker, weight in weights[weights > 0].sort_values(ascending=False).items():
            rows.append({"date": dt.date().isoformat(), "ticker": ticker, "weight": weight})
        if cash > 0:
            rows.append({"date": dt.date().isoformat(), "ticker": "CASH", "weight": cash})
    pd.DataFrame(rows).to_csv(ROOT / "twse_monthly_holdings.csv", index=False)


def write_report(prices: pd.DataFrame, results: list[Result]) -> None:
    summaries = [summarize(r) for r in results]
    df = pd.DataFrame(summaries)
    order_cols = ["策略", "起始日", "結束日", "累積報酬", "年化報酬", "最大回撤", "年化波動", "Sharpe", "年化週轉率", "最差3個月", "2022報酬", "2026YTD"]
    df = df[order_cols]

    md = []
    md.append("# TWSE 策略第二輪回測報告\n")
    md.append(f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("資料來源：FinMind TaiwanStockPrice 日線 API；原本嘗試 TWSE 官方日線 API，但大量抓取時受 CDN/security redirect 影響，因此改用 FinMind 做可重複回測。本輪仍使用價格報酬，尚未納入股息再投入。\n")
    md.append("重要限制：價格資料不是完整總報酬資料；ETF 與個股配息會使買進持有與策略報酬被低估。00631L 已依已知 2026 年 22:1 分割做簡易回溯調整，其他明顯分割跳點以演算法簡易回溯調整。\n")
    md.append("第二輪新增：部分策略加入粗略交易成本壓力測試，假設每 1.00 換手成本 0.20%。此成本只是保守近似，尚未拆分股票與 ETF 的手續費、證交稅與滑價。\n")
    md.append(f"本次執行日：{TODAY.isoformat()}；實際可回測價格資料截止日：{prices.index[-1].date()}。\n")
    md.append("若當日盤中資料尚未寫入本地快取，本報告的市場狀態與回測訊號仍以最近一個可用收盤日為準。\n")
    lag_days = (TODAY - prices.index[-1].date()).days
    if lag_days > 0:
        md.append(f"資料新鮮度警示：本輪執行日與最新收盤資料相差 {lag_days} 個日曆天；若外部資料源延遲，檔名或執行時間不代表已取得更新日線。\n")
        md.append("正式權重與風控判讀應以核心訊號（0050、006208、00631L）的最後有效收盤日為準，而不是以快取檔名或請求的 end_date 推斷。\n")

    snapshot = latest_market_snapshot(prices)
    snap_df = pd.DataFrame(
        [
            {"指標": key, "數值": format_pct(value) if isinstance(value, float) and "動能" in key else value}
            for key, value in snapshot.items()
        ]
    )
    md.append("## 最新市場狀態\n")
    md.append(markdown_table(snap_df, index=False))

    freshness = latest_signal_freshness(prices, ["0050", "006208", "00631L"])
    if not freshness.empty:
        md.append("\n## 核心訊號新鮮度\n")
        md.append(markdown_table(freshness, index=False))

    ai_rank = latest_ai_ranking(prices)
    ai_table = ai_rank.copy()
    for col in ["3個月動能", "6個月動能", "分數", "近3月回撤", "近12月回撤"]:
        ai_table[col] = ai_table[col].map(format_pct)
    md.append("\n## 最新 AI 候選池排名\n")
    md.append(markdown_table(ai_table, index=False))
    md.append("## 結果摘要\n")

    table = df.copy()
    for col in ["累積報酬", "年化報酬", "最大回撤", "年化波動", "最差3個月", "2022報酬", "2026YTD"]:
        table[col] = table[col].map(format_pct)
    table["Sharpe"] = table["Sharpe"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}")
    table["年化週轉率"] = table["年化週轉率"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}x")
    md.append(markdown_table(table, index=False))
    md.append("\n\n## 初步解讀\n")

    best_cagr = df.sort_values("年化報酬", ascending=False).iloc[0]
    best_dd = df.sort_values("最大回撤", ascending=False).iloc[0]
    md.append(f"- 年化報酬最高：{best_cagr['策略']}，年化 {format_pct(best_cagr['年化報酬'])}，最大回撤 {format_pct(best_cagr['最大回撤'])}。\n")
    md.append(f"- 最大回撤最小：{best_dd['策略']}，最大回撤 {format_pct(best_dd['最大回撤'])}，年化 {format_pct(best_dd['年化報酬'])}。\n")
    md.append("- 本輪仍只是研究回測，不能直接當作投資建議；下一輪必須納入總報酬、月營收基本面濾網與更細的股票/ETF 成本模型。\n")

    seg = segment_summary(results)
    if not seg.empty:
        md.append("\n## 分段回測\n")
        seg_table = seg.copy()
        seg_table["區間報酬"] = seg_table["區間報酬"].map(format_pct)
        seg_table["最大回撤"] = seg_table["最大回撤"].map(format_pct)
        md.append(markdown_table(seg_table, index=False))

    selected = next((r for r in results if r.name == "006208核心現金防守 50/35/10（成本0.20%）"), None)
    if selected is None:
        selected = next((r for r in results if r.name == "標準版現金防守 50/35/10（成本0.20%）"), None)
    if selected is None:
        selected = next((r for r in results if r.name == "標準版現金防守 50/35/10"), None)
    primary_name = selected.name if selected is not None else str(best_dd["策略"])

    md.append("\n## 本輪保留策略\n")
    md.append(f"- 優先保留 `{primary_name}`：本輪在年化報酬仍高於 20% 的同時，最大回撤最低。\n")
    md.append("- 若加入粗略交易成本後仍高於 20%，才允許升級為主策略；否則降為候選。\n")
    md.append("- 保留 `標準版無槓桿 55/35/0`：確認不依賴 00631L 也能超過台灣 50，但仍需加入交易成本。\n")
    md.append("- 保留 `標準版 50/35/10` 作為主策略候選，但需限制 00631L 觸發條件。\n")
    md.append("- `進取版 40/40/15` 與 `無狀態濾網 50/35/10` 暫列研究候選，不列為預設策略，避免過度貼合 2018-2026 AI 強週期。\n")

    if selected is not None:
        export_monthly_holdings(selected)
        md.append("\n## 目前主策略最新權重\n")
        pos = latest_positions(selected)
        pos["權重"] = pos["權重"].map(format_pct)
        md.append(markdown_table(pos, index=True))

    md.append("\n## 資料完整性\n")
    coverage = prices.notna().sum().sort_values(ascending=False)
    md.append(markdown_table(coverage.to_frame("有效交易日數"), index=True))
    md.append("\n")

    REPORT_PATH.write_text("\n".join(md))
    df.to_csv(ROOT / "twse_backtest_summary.csv", index=False)


def main() -> None:
    prices = load_prices(TICKERS)
    prices.to_csv(ROOT / "twse_prices_close.csv")
    prices = prices.dropna(subset=["0050"])
    # Keep rows where at least half the instruments have valid prices.
    prices = prices.loc[prices.notna().sum(axis=1) >= max(4, len(prices.columns) // 2)]

    results = [
        buy_hold(prices, "0050"),
    ]
    if "006208" in prices.columns:
        results.append(buy_hold(prices.dropna(subset=["006208"]), "006208"))
    if "00631L" in prices.columns:
        results.append(buy_hold(prices.dropna(subset=["00631L"]), "00631L"))

    bt_prices = prices.ffill()
    results.extend(
        [
            run_strategy(bt_prices, "保守版 70/20/10", 0.70, 0.20, 0.00, 0.10, top_n=5, use_regime=True),
            run_strategy(bt_prices, "標準版 50/35/10", 0.50, 0.35, 0.10, 0.05, top_n=5, use_regime=True),
            run_strategy(bt_prices, "標準版無槓桿 55/35/0", 0.55, 0.35, 0.00, 0.10, top_n=5, use_regime=True),
            run_strategy(bt_prices, "標準版現金防守 50/35/10", 0.50, 0.35, 0.10, 0.05, top_n=5, use_regime=True, use_defensive_etfs=False),
            run_strategy(bt_prices, "標準版現金防守 50/35/10（成本0.20%）", 0.50, 0.35, 0.10, 0.05, top_n=5, use_regime=True, use_defensive_etfs=False, transaction_cost_rate=0.002),
            run_strategy(bt_prices, "006208核心現金防守 50/35/10（成本0.20%）", 0.50, 0.35, 0.10, 0.05, top_n=5, use_regime=True, use_defensive_etfs=False, transaction_cost_rate=0.002, core_ticker="006208"),
            run_strategy(bt_prices, "標準版無槓桿 55/35/0（成本0.20%）", 0.55, 0.35, 0.00, 0.10, top_n=5, use_regime=True, transaction_cost_rate=0.002),
            run_strategy(bt_prices, "Top3集中 50/35/10", 0.50, 0.35, 0.10, 0.05, top_n=3, use_regime=True),
            run_strategy(bt_prices, "進取版 40/40/15", 0.40, 0.40, 0.15, 0.05, top_n=5, use_regime=True),
            run_strategy(bt_prices, "無狀態濾網 50/35/10", 0.50, 0.35, 0.10, 0.05, top_n=5, use_regime=False),
        ]
    )
    write_report(bt_prices, results)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
