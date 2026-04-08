# -*- coding: utf-8 -*-

"""
========================================================
Tushare版：A股日线 3 Bar Play 多日期扫描脚本（GitHub Action 版）
========================================================

本版说明：
1. 整理区改为：真正允许 1~3 根K线
   - 这里的“整理区K线”定义为：启动棒之后、target_date之前的休整K线
   - target_date 当天单独作为触发/判断日
2. 保留 A/B 类判断
   - A类：target_date 当天收盘接近整理区收盘高点，但未突破
   - B类：target_date 当天收盘突破整理区收盘高点
3. 删除“当日振幅 > 过去10日平均振幅 × 1.5”条件（沿用你当前版本）
4. 适配 GitHub Actions：
   - 使用仓库相对路径 data/ 与 output/
   - TUSHARE_TOKEN 从环境变量读取，不写死在代码里
   - 默认 target_date 为“北京时间今天”
   - 也支持环境变量 TARGET_DATES=20260407,20260408 这种覆盖
"""

import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tushare as ts


# =========================
# 清理代理
# =========================
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


# =========================
# 仓库路径
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CODE_FILE = os.path.join(DATA_DIR, "a_share_codes_for_akshare.csv")
BELOW_8B_FILE = os.path.join(DATA_DIR, "a_share_below_8b.csv")
ST_FILE = os.path.join(DATA_DIR, "st_stocks.csv")


# =========================
# 环境变量 / 运行参数
# =========================
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()

START_DATE = os.getenv("START_DATE", "20250101").strip()
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
TEST_LIMIT = None if not os.getenv("TEST_LIMIT") else int(os.getenv("TEST_LIMIT"))
BATCH_START = int(os.getenv("BATCH_START", "0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10000"))
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "0.10"))


# =========================
# 3 Bar Play 参数
# =========================
MA_PERIOD = 20

IGNITE_MIN_PCT = 0.04
BODY_RATIO_MIN = 0.60
CLOSE_NEAR_HIGH_MAX = 0.25
BREAKOUT_NEAR_30D_RATIO = 0.98

# 启动棒距离 target_date 的位置约束
# 由于现在整理区定义为“target_date之前的1~3根K线”，
# 所以 days_from_last = pullback_bars + 1，会自然落在 2~4
IGNITE_LOOKBACK_MIN = 2
IGNITE_LOOKBACK_MAX = 5

# 整理区根数：真正改为 1~3 根
PULLBACK_BARS_MIN = 1
PULLBACK_BARS_MAX = 3

PULLBACK_BAR_RANGE_RATIO_MAX = 0.60
BIG_BEAR_DROP_MAX = -0.03

A_CLASS_NEAR_HIGH_RATIO = 0.98


# =========================
# 日期处理
# =========================
def get_today_cn_str() -> str:
    return (datetime.utcnow() + timedelta(hours=8)).strftime("%Y%m%d")


def get_target_dates():
    """
    优先读取环境变量 TARGET_DATES，例如：
    TARGET_DATES=20260407,20260408
    否则默认使用北京时间今天。
    """
    raw = os.getenv("TARGET_DATES", "").strip()
    if raw:
        items = []
        for part in raw.replace(";", ",").replace("\n", ",").split(","):
            s = part.strip()
            if s:
                items.append(s)
    else:
        items = [get_today_cn_str()]

    target_dates = sorted({x for x in items if x.isdigit() and len(x) == 8})
    if not target_dates:
        raise ValueError("TARGET_DATES 为空，且未能生成默认日期。")
    return target_dates


def to_target_dt(target_date: str):
    return pd.to_datetime(target_date, format="%Y%m%d")


# =========================
# 工具函数
# =========================
def read_csv_safely(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except Exception as e:
            last_err = e
    raise last_err


def safe_ratio(a, b, default=np.nan):
    try:
        a = float(a)
        b = float(b)
        if pd.isna(a) or pd.isna(b) or b == 0:
            return default
        return a / b
    except Exception:
        return default


def normalize_to_6digits(x: str) -> str:
    s = str(x).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 6:
        return digits
    elif len(digits) > 6:
        return digits[-6:]
    elif len(digits) > 0:
        return digits.zfill(6)
    return ""


def code6_to_ts_code(code6: str) -> str:
    if code6.startswith(("600", "601", "603", "605", "688", "900")):
        return f"{code6}.SH"
    return f"{code6}.SZ"


def bool_or_none(x):
    if x is None:
        return None
    return bool(x)


# =========================
# 股票池过滤
# =========================
def load_st_codes() -> set:
    df = read_csv_safely(ST_FILE)
    if "ticker" not in df.columns:
        raise ValueError(f"ST 文件里没有 ticker 列，实际列名: {list(df.columns)}")
    return set(df["ticker"].astype(str).str.strip().str.zfill(6).dropna().tolist())


def load_below_8b_codes() -> set:
    df = read_csv_safely(BELOW_8B_FILE)
    if "ticker" not in df.columns:
        raise ValueError(f"80亿以下文件里没有 ticker 列，实际列名: {list(df.columns)}")
    return set(df["ticker"].astype(str).str.strip().str.zfill(6).dropna().tolist())


def load_universe_from_csv():
    df = read_csv_safely(CODE_FILE)

    required_cols = ["ticker", "secShortName"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"代码文件缺少列: {c}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).map(normalize_to_6digits)
    out["name"] = out["secShortName"].astype(str).str.strip()
    out = out[out["ticker"].str.len() == 6].copy()

    st_codes = load_st_codes()
    below_8b_codes = load_below_8b_codes()

    out["is_st"] = out["ticker"].isin(st_codes)
    out["is_below_8b"] = out["ticker"].isin(below_8b_codes)

    filtered_out = out[(out["is_st"]) | (out["is_below_8b"])].copy()
    universe = out[(~out["is_st"]) & (~out["is_below_8b"])].copy()

    out["ts_code"] = out["ticker"].map(code6_to_ts_code)
    universe["ts_code"] = universe["ticker"].map(code6_to_ts_code)
    filtered_out["ts_code"] = filtered_out["ticker"].map(code6_to_ts_code)

    universe = universe.drop_duplicates("ticker").reset_index(drop=True)
    filtered_out = filtered_out.drop_duplicates("ticker").reset_index(drop=True)

    if TEST_LIMIT is not None:
        universe = universe.head(TEST_LIMIT).copy()
    else:
        universe = universe.iloc[BATCH_START:BATCH_START + BATCH_SIZE].copy()

    return universe[["ticker", "ts_code", "name"]].reset_index(drop=True), filtered_out


# =========================
# Tushare 取数
# =========================
def fetch_tushare_daily_with_retry(pro, ts_code: str, start_date: str, end_date: str, max_retry: int = 3):
    last_err = None
    for attempt in range(max_retry):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                return df
            last_err = "empty dataframe"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        time.sleep(0.8 + attempt * 0.8)
    raise RuntimeError(str(last_err))


def standardize_tushare_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    required_cols = ["trade_date", "open", "high", "low", "close", "vol"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Tushare daily 缺少字段: {missing}")

    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["vol"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce") if "amount" in out.columns else np.nan
    out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce") / 100.0 if "pct_chg" in out.columns else np.nan

    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out[["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]


# =========================
# 指标构建
# =========================
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if "pct_chg" not in out.columns or out["pct_chg"].isna().all():
        out["pct_chg"] = out["close"].pct_change()

    out["daily_range_pct"] = (out["high"] - out["low"]) / out["close"].shift(1)
    out["bar_range_abs"] = (out["high"] - out["low"]).clip(lower=1e-8)

    out["body"] = (out["close"] - out["open"]).abs()
    out["body_ratio"] = out["body"] / out["bar_range_abs"]
    out["close_near_high"] = (out["high"] - out["close"]) / out["bar_range_abs"]

    out["ma20"] = out["close"].rolling(MA_PERIOD).mean()
    out["ma20_prev"] = out["ma20"].shift(1)

    out["vol_ma5"] = out["volume"].rolling(5).mean()
    out["vol_ma20"] = out["volume"].rolling(20).mean()

    # 仍然沿用“前30日最高 high”
    out["high_30_prev"] = out["high"].rolling(30).max().shift(1)

    return out


# =========================
# 数据切片
# =========================
def slice_df_to_target_date(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    target_dt = to_target_dt(target_date)
    sub = df[df["date"] <= target_dt].copy()
    sub = sub.sort_values("date").reset_index(drop=True)
    return sub


# =========================
# 因子级检查
# =========================
def check_last_day_trend(df: pd.DataFrame):
    last = df.iloc[-1]

    factor_flags = {
        "last_day_close_above_ma20_ok": None,
        "last_day_ma20_not_down_ok": None,
    }
    reasons = []

    if pd.isna(last["ma20"]) or pd.isna(last["ma20_prev"]):
        reasons.append("last_day_ma_missing")
        return False, reasons, factor_flags

    close_above = last["close"] > last["ma20"]
    ma20_not_down = last["ma20"] >= last["ma20_prev"]

    factor_flags["last_day_close_above_ma20_ok"] = bool(close_above)
    factor_flags["last_day_ma20_not_down_ok"] = bool(ma20_not_down)

    if not close_above:
        reasons.append("last_day_close_below_ma20")
    if not ma20_not_down:
        reasons.append("last_day_ma20_down")

    return len(reasons) == 0, reasons, factor_flags


def check_volume_available(df: pd.DataFrame):
    if "volume" not in df.columns:
        return False, "volume_column_missing"
    if df["volume"].notna().sum() < 20:
        return False, "volume_missing"
    return True, "ok"


def get_default_factor_flags():
    return {
        "last_day_close_above_ma20_ok": None,
        "last_day_ma20_not_down_ok": None,

        "ignite_window_match_ok": None,
        "ignite_pct_ge_4pct_ok": None,
        "ignite_body_ratio_ge_0p6_ok": None,
        "ignite_close_near_high_ok": None,
        "ignite_volume_above_ma5_ok": None,
        "ignite_volume_above_ma20_ok": None,
        "ignite_near_prev30_high_ok": None,

        "pullback_bars_1_to_3_ok": None,
        "pullback_low_above_body_mid_ok": None,
        "pullback_bar_range_small_ok": None,
        "pullback_avg_volume_lower_ok": None,
        "pullback_no_big_bear_ok": None,
        "pullback_close_above_body_mid_ok": None,

        "signal_is_a_class_ok": None,
        "signal_is_b_class_ok": None,
    }


def check_ignite_bar(df: pd.DataFrame, i: int):
    row = df.iloc[i]
    reasons = []
    factor_flags = get_default_factor_flags()

    need_cols = ["ma20", "ma20_prev", "vol_ma5", "vol_ma20", "high_30_prev", "volume"]
    for c in need_cols:
        if pd.isna(row[c]):
            reasons.append("ignite_ref_data_missing")
            return False, reasons, {}, factor_flags

    metrics = {
        "ignite_date": row["date"].strftime("%Y-%m-%d"),
        "ignite_pct_chg": round(float(row["pct_chg"]) * 100, 2),
        "ignite_body_ratio": round(float(row["body_ratio"]), 4),
        "ignite_close_near_high": round(float(row["close_near_high"]), 4),
        "ignite_range_pct": round(float(row["daily_range_pct"]) * 100, 2),
        "ignite_vol_vs_ma5": round(safe_ratio(row["volume"], row["vol_ma5"]), 2),
        "ignite_vol_vs_ma20": round(safe_ratio(row["volume"], row["vol_ma20"]), 2),
        "ignite_close_vs_prev30high": round(safe_ratio(row["close"], row["high_30_prev"]), 4),
        "ignite_open": round(float(row["open"]), 2),
        "ignite_close": round(float(row["close"]), 2),
        "ignite_high": round(float(row["high"]), 2),
        "ignite_low": round(float(row["low"]), 2),
        "ignite_volume": round(float(row["volume"]), 0),
    }

    pct_ok = row["pct_chg"] >= IGNITE_MIN_PCT
    body_ok = row["body_ratio"] >= BODY_RATIO_MIN
    close_near_high_ok = row["close_near_high"] <= CLOSE_NEAR_HIGH_MAX
    vol_ma5_ok = row["volume"] > row["vol_ma5"]
    vol_ma20_ok = row["volume"] > row["vol_ma20"]
    near_prev30_ok = row["close"] >= row["high_30_prev"] * BREAKOUT_NEAR_30D_RATIO

    factor_flags["ignite_pct_ge_4pct_ok"] = bool(pct_ok)
    factor_flags["ignite_body_ratio_ge_0p6_ok"] = bool(body_ok)
    factor_flags["ignite_close_near_high_ok"] = bool(close_near_high_ok)
    factor_flags["ignite_volume_above_ma5_ok"] = bool(vol_ma5_ok)
    factor_flags["ignite_volume_above_ma20_ok"] = bool(vol_ma20_ok)
    factor_flags["ignite_near_prev30_high_ok"] = bool(near_prev30_ok)

    if row["close"] <= row["ma20"]:
        reasons.append("ignite_close_below_ma20")
    if row["ma20"] < row["ma20_prev"]:
        reasons.append("ignite_ma20_down")
    if not pct_ok:
        reasons.append("ignite_pct_lt_4pct")
    if not body_ok:
        reasons.append("ignite_body_ratio_fail")
    if not close_near_high_ok:
        reasons.append("ignite_close_not_near_high")
    if not vol_ma5_ok:
        reasons.append("ignite_volume_not_above_ma5")
    if not vol_ma20_ok:
        reasons.append("ignite_volume_not_above_ma20")
    if not near_prev30_ok:
        reasons.append("ignite_not_near_prev30_high")

    return len(reasons) == 0, reasons, metrics, factor_flags


def check_pullback(df: pd.DataFrame, ignite_idx: int, target_idx: int):
    """
    整理区定义：
    启动棒之后，到 target_date 前一天为止。
    target_date 当天不计入整理区，而是单独作为触发/判断日。
    """
    ignite = df.iloc[ignite_idx]
    pb = df.iloc[ignite_idx + 1:target_idx].copy()

    reasons = []
    metrics = {}
    factor_flags = get_default_factor_flags()

    pullback_bars = len(pb)
    bars_ok = PULLBACK_BARS_MIN <= pullback_bars <= PULLBACK_BARS_MAX
    factor_flags["pullback_bars_1_to_3_ok"] = bool_or_none(bars_ok)

    if not bars_ok:
        reasons.append("pullback_bars_not_1_to_3")
        return False, reasons, metrics, factor_flags

    if ignite["close"] <= ignite["open"]:
        reasons.append("ignite_not_bullish")
        return False, reasons, metrics, factor_flags

    if pb["volume"].isna().any():
        reasons.append("pullback_volume_missing")
        return False, reasons, metrics, factor_flags

    ignite_body_mid = (ignite["open"] + ignite["close"]) / 2.0
    ignite_range_abs = ignite["bar_range_abs"]
    if ignite_range_abs <= 0:
        reasons.append("ignite_range_invalid")
        return False, reasons, metrics, factor_flags

    pb_range_abs = pb["bar_range_abs"]
    pb_pct = pb["close"].pct_change().fillna((pb.iloc[0]["close"] / ignite["close"]) - 1.0)

    pullback_low = pb["low"].min()

    low_above_mid_ok = pullback_low >= ignite_body_mid
    bar_range_small_ok = (pb_range_abs < ignite_range_abs * PULLBACK_BAR_RANGE_RATIO_MAX).all()
    avg_volume_lower_ok = pb["volume"].mean() < ignite["volume"]
    no_big_bear_ok = (pb_pct > BIG_BEAR_DROP_MAX).all()
    close_above_mid_ok = (pb["close"] >= ignite_body_mid).all()

    factor_flags["pullback_low_above_body_mid_ok"] = bool(low_above_mid_ok)
    factor_flags["pullback_bar_range_small_ok"] = bool(bar_range_small_ok)
    factor_flags["pullback_avg_volume_lower_ok"] = bool(avg_volume_lower_ok)
    factor_flags["pullback_no_big_bear_ok"] = bool(no_big_bear_ok)
    factor_flags["pullback_close_above_body_mid_ok"] = bool(close_above_mid_ok)

    metrics = {
        "pullback_bars": pullback_bars,
        "pullback_start_date": pb.iloc[0]["date"].strftime("%Y-%m-%d"),
        "pullback_end_date": pb.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "pullback_low": round(float(pullback_low), 2),
        "ignite_body_mid": round(float(ignite_body_mid), 2),
        "pullback_avg_volume": round(float(pb["volume"].mean()), 0),
        "pullback_max_bar_range_ratio": round(float((pb_range_abs / ignite_range_abs).max()), 4),
    }

    if not low_above_mid_ok:
        reasons.append("pullback_low_below_ignite_body_mid")
    if not bar_range_small_ok:
        reasons.append("pullback_bar_range_too_large")
    if not avg_volume_lower_ok:
        reasons.append("pullback_avg_volume_not_lower")
    if not no_big_bear_ok:
        reasons.append("pullback_big_bear_bar")
    if not close_above_mid_ok:
        reasons.append("pullback_close_below_ignite_body_mid")

    return len(reasons) == 0, reasons, metrics, factor_flags


def get_pullback_high_close_before_today(df: pd.DataFrame, ignite_idx: int, target_idx: int):
    """
    只取整理区（即 target_date 之前）的收盘高点。
    """
    pb_before_today = df.iloc[ignite_idx + 1:target_idx].copy()

    if pb_before_today.empty:
        return None, {
            "pullback_high_close": np.nan,
            "dist_to_pullback_high_close_pct": np.nan
        }

    pullback_high_close = float(pb_before_today["close"].max())
    return pullback_high_close, {}


def classify_signal(df: pd.DataFrame, pullback_high_close: float):
    last = df.iloc[-1]
    close_now = float(last["close"])

    metrics = {
        "signal_date": last["date"].strftime("%Y-%m-%d"),
        "close_now": round(close_now, 2),
        "pullback_high_close": round(float(pullback_high_close), 2),
        "dist_to_pullback_high_close_pct": round((close_now / float(pullback_high_close) - 1.0) * 100, 2),
    }

    factor_flags = get_default_factor_flags()

    a_ok = (close_now >= pullback_high_close * A_CLASS_NEAR_HIGH_RATIO) and (close_now <= pullback_high_close)
    b_ok = close_now > pullback_high_close

    factor_flags["signal_is_a_class_ok"] = bool(a_ok)
    factor_flags["signal_is_b_class_ok"] = bool(b_ok)

    if a_ok:
        return True, "A_待突破", [], metrics, factor_flags

    if b_ok:
        return True, "B_已突破", [], metrics, factor_flags

    return False, "", ["signal_not_near_pullback_high_and_not_breakout"], metrics, factor_flags


# =========================
# 总检查函数
# =========================
def check_3barplay_ab(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 40:
        return {
            "matched": False,
            "reason": "bars_not_enough",
            "reason_list": "bars_not_enough",
            **get_default_factor_flags()
        }

    vol_ok, vol_reason = check_volume_available(df)
    if not vol_ok:
        return {
            "matched": False,
            "reason": vol_reason,
            "reason_list": vol_reason,
            **get_default_factor_flags()
        }

    df = build_indicators(df)
    last_idx = len(df) - 1

    last_ok, last_reasons, last_flags = check_last_day_trend(df)
    if not last_ok:
        return {
            "matched": False,
            "reason": "last_day_trend_fail",
            "reason_list": "|".join(last_reasons),
            "close_now": round(float(df.iloc[-1]["close"]), 2),
            **get_default_factor_flags(),
            **last_flags
        }

    candidate_debugs = []

    for pullback_bars in range(PULLBACK_BARS_MIN, PULLBACK_BARS_MAX + 1):
        factor_flags = get_default_factor_flags()
        factor_flags.update(last_flags)

        # 关键修正：
        # pullback_bars 只统计 target_date 之前的整理K线，
        # target_date 当天单独作为触发/判断日
        ignite_idx = last_idx - pullback_bars - 1
        days_from_last = last_idx - ignite_idx

        window_match_ok = (
            ignite_idx >= 30 and
            IGNITE_LOOKBACK_MIN <= days_from_last <= IGNITE_LOOKBACK_MAX
        )
        factor_flags["ignite_window_match_ok"] = bool(window_match_ok)

        if not window_match_ok:
            candidate_debugs.append({
                "reason": "ignite_position_out_of_lookback_window",
                "reason_list": "ignite_position_out_of_lookback_window",
                "pullback_bars": pullback_bars,
                **factor_flags
            })
            continue

        ignite_ok, ignite_reasons, ignite_metrics, ignite_flags = check_ignite_bar(df, ignite_idx)
        factor_flags.update(ignite_flags)

        if not ignite_ok:
            candidate_debugs.append({
                "reason": "ignite_fail",
                "reason_list": "|".join(ignite_reasons),
                "pullback_bars": pullback_bars,
                **ignite_metrics,
                **factor_flags
            })
            continue

        pb_ok, pb_reasons, pb_metrics, pb_flags = check_pullback(df, ignite_idx, last_idx)
        factor_flags.update(pb_flags)

        if not pb_ok:
            candidate_debugs.append({
                "reason": "pullback_fail",
                "reason_list": "|".join(pb_reasons),
                **ignite_metrics,
                **pb_metrics,
                **factor_flags
            })
            continue

        pullback_high_close, _ = get_pullback_high_close_before_today(df, ignite_idx, last_idx)
        if pullback_high_close is None:
            candidate_debugs.append({
                "reason": "signal_state_fail",
                "reason_list": "no_pullback_before_today",
                **ignite_metrics,
                **pb_metrics,
                **factor_flags
            })
            continue

        signal_ok, signal_type, signal_reasons, signal_metrics, signal_flags = classify_signal(
            df, pullback_high_close
        )
        factor_flags.update(signal_flags)

        if not signal_ok:
            candidate_debugs.append({
                "reason": "signal_state_fail",
                "reason_list": "|".join(signal_reasons),
                **ignite_metrics,
                **pb_metrics,
                **signal_metrics,
                **factor_flags
            })
            continue

        return {
            "matched": True,
            "signal_type": signal_type,
            **ignite_metrics,
            **pb_metrics,
            **signal_metrics,
            **factor_flags
        }

    if candidate_debugs:
        best = candidate_debugs[0]
        return {
            "matched": False,
            "reason": best.get("reason", "not_matched"),
            "reason_list": best.get("reason_list", best.get("reason", "not_matched")),
            **best
        }

    return {
        "matched": False,
        "reason": "no_valid_3barplay_ab",
        "reason_list": "no_valid_3barplay_ab",
        **get_default_factor_flags()
    }


# =========================
# 单股多日期评估
# =========================
def evaluate_one_stock_multi_dates(pro, ts_code: str, ticker: str, name: str, target_dates, start_date, end_date):
    matched_list = []
    debug_list = []
    failed_list = []

    try:
        raw = fetch_tushare_daily_with_retry(pro, ts_code, start_date, end_date, max_retry=3)
    except Exception as e:
        failed_list.append({
            "ticker": ticker,
            "ts_code": ts_code,
            "name": name,
            "target_date": "",
            "error": f"daily_fetch_exception: {type(e).__name__}: {e}"
        })
        return matched_list, debug_list, failed_list

    try:
        df_full = standardize_tushare_daily(raw)
    except Exception as e:
        failed_list.append({
            "ticker": ticker,
            "ts_code": ts_code,
            "name": name,
            "target_date": "",
            "error": f"standardize_exception: {type(e).__name__}: {e}"
        })
        return matched_list, debug_list, failed_list

    if df_full.empty:
        failed_list.append({
            "ticker": ticker,
            "ts_code": ts_code,
            "name": name,
            "target_date": "",
            "error": "no_data"
        })
        return matched_list, debug_list, failed_list

    for target_date in target_dates:
        try:
            df = slice_df_to_target_date(df_full, target_date)

            if df.empty:
                debug_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    "error": "no_data_before_target_date",
                    "reason_list": "no_data_before_target_date",
                    **get_default_factor_flags()
                })
                continue

            if lenreason_list": "no_data_before_target_date",
                    **get_default_factor_flags()
                })
                continue

            if lenreason_list": "no_data_before_target_date",
                    **get_default_factor_flags()
                })
                continue

            if len(df) < 120:
                debug_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    "error": "not_enough_bars",
                    "reason_list": "not_enough_bars",
                    **get_default_factor_flags()
                })
                continue

            chk = check_3barplay_ab(df)

            if chk.get("matched", False):
                matched_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    **chk
                })
            else:
                debug_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    "error": chk.get("reason", "not_matched"),
                    "reason_list": chk.get("reason_list", ""),
                    **{k: v for k, v in chk.items() if k not in ["matched", "reason", "reason_list"]}
                })

            time.sleep(SLEEP_SEC)

        except Exception as e:
            failed_list.append({
                "ticker": ticker,
                "ts_code": ts_code,
                "name": name,
                "target_date": target_date,
                "error": f"check_exception: {type(e).__name__}: {e}"
            })

    return matched_list, debug_list, failed_list


# =========================
# 输出文件
# =========================
def make_output_paths(end_date: str):
    return {
        "output": os.path.join(OUTPUT_DIR, f"three_bar_play_ab_candidates_{end_date}.csv"),
        "debug": os.path.join(OUTPUT_DIR, f"three_bar_play_ab_debug_rejected_{end_date}.csv"),
        "failed": os.path.join(OUTPUT_DIR, f"three_bar_play_ab_failed_fetch_{end_date}.csv"),
        "filtered": os.path.join(OUTPUT_DIR, f"three_bar_play_ab_filtered_out_{end_date}.csv"),
    }


# =========================
# 主程序
# =========================
def main():
    if not TUSHARE_TOKEN:
        raise ValueError("未检测到环境变量 TUSHARE_TOKEN，请在 GitHub Secrets 中配置。")

    target_dates = get_target_dates()
    end_date = max(target_dates)
    output_paths = make_output_paths(end_date)

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    start_time = time.perf_counter()

    print("1) 读取本地全市场股票列表，并剔除 ST / 80亿以下 ...")
    universe, filtered_out = load_universe_from_csv()

    print("原始待扫描股票数:", len(universe))
    print("过滤掉数量(ST/80亿以下):", len(filtered_out))
    print("TARGET_DATES:", target_dates)
    print("END_DATE:", end_date)
    print("START_DATE:", START_DATE)

    for test_code in ["000004", "000638", "600608"]:
        sub = filtered_out[filtered_out["ticker"] == test_code]
        if not sub.empty:
            row = sub.iloc[0]
            print(f"检查 {test_code} | name={row['name']} | ts_code={row['ts_code']} | 已成功过滤")

    print("\n2) 开始 3 Bar Play 多日期扫描（Tushare）...")

    matched = []
    debug_rejected = []
    failed_fetch = []
    error_counter = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                evaluate_one_stock_multi_dates,
                pro,
                row["ts_code"],
                row["ticker"],
                row["name"],
                target_dates,
                START_DATE,
                end_date,
            ): row["ts_code"]
            for _, row in universe.iterrows()
        }

        total = len(future_map)

        for i, future in enumerate(as_completed(future_map), 1):
            matched_list, debug_list, failed_list = future.result()

            matched.extend(matched_list)
            debug_rejected.extend(debug_list)
            failed_fetch.extend(failed_list)

            for item in debug_list:
                err = item.get("error", "unknown")
                error_counter[err] = error_counter.get(err, 0) + 1

            for item in failed_list:
                err = item.get("error", "unknown")
                error_counter[err] = error_counter.get(err, 0) + 1

            if i % 100 == 0 or i == total:
                elapsed_now = time.perf_counter() - start_time
                avg_per_stock = elapsed_now / i
                est_total = avg_per_stock * total
                remain = est_total - elapsed_now

                rh = int(remain // 3600)
                rm = int((remain % 3600) // 60)
                rs = remain % 60

                print(
                    f"进度: {i}/{total} | 命中: {len(matched)} | 调试未过: {len(debug_rejected)} | 抓取失败: {len(failed_fetch)} "
                    f"| 预计剩余: {rh}小时 {rm}分钟 {rs:.1f}秒"
                )

    matched_df = pd.DataFrame(matched)
    debug_df = pd.DataFrame(debug_rejected)
    failed_df = pd.DataFrame(failed_fetch)
    filtered_df = filtered_out.copy()

    if not matched_df.empty:
        matched_df = matched_df.sort_values(
            by=["target_date", "signal_type", "dist_to_pullback_high_close_pct", "pullback_bars", "ignite_vol_vs_ma5"],
            ascending=[True, True, False, True, False]
        ).reset_index(drop=True)

    matched_df.to_csv(output_paths["output"], index=False, encoding="utf-8-sig")
    debug_df.to_csv(output_paths["debug"], index=False, encoding="utf-8-sig")
    failed_df.to_csv(output_paths["failed"], index=False, encoding="utf-8-sig")
    filtered_df.to_csv(output_paths["filtered"], index=False, encoding="utf-8-sig")

    print("\n候选结果已保存:", output_paths["output"])
    print("调试未通过已保存:", output_paths["debug"])
    print("抓取失败已保存:", output_paths["failed"])
    print("已过滤股票已保存:", output_paths["filtered"])

    print("\n扫描完成")
    print("候选数量:", len(matched_df))
    print("调试未通过数量:", len(debug_df))
    print("抓取失败数量:", len(failed_df))
    print("过滤掉数量(ST/80亿以下):", len(filtered_df))

    if not matched_df.empty and "signal_type" in matched_df.columns:
        print("\nA/B 类数量统计：")
        print(matched_df["signal_type"].value_counts())

    if not matched_df.empty and "target_date" in matched_df.columns:
        print("\n按日期统计候选数量：")
        print(matched_df.groupby("target_date").size())

    print("\n失败原因统计：")
    for k, v in sorted(error_counter.items(), key=lambda x: -x[1]):
        print(k, v)

    if not matched_df.empty:
        print("\n候选前20条：")
        cols = [
            "ticker", "name", "ts_code", "target_date", "signal_type", "signal_date",
            "ignite_date", "pullback_bars", "ignite_pct_chg", "ignite_vol_vs_ma5",
            "ignite_vol_vs_ma20", "pullback_high_close", "close_now", "dist_to_pullback_high_close_pct"
        ]
        cols = [c for c in cols if c in matched_df.columns]
        print(matched_df[cols].head(20).to_string(index=False))

    if not debug_df.empty:
        print("\n调试未通过前20条：")
        cols = [
            "ticker", "name", "ts_code", "target_date", "error", "reason_list", "close_now",
            "last_day_close_above_ma20_ok", "last_day_ma20_not_down_ok",
            "ignite_window_match_ok", "ignite_pct_ge_4pct_ok", "ignite_body_ratio_ge_0p6_ok",
            "ignite_close_near_high_ok",
            "ignite_volume_above_ma5_ok", "ignite_volume_above_ma20_ok",
            "ignite_near_prev30_high_ok",
            "pullback_bars_1_to_3_ok", "pullback_low_above_body_mid_ok",
            "pullback_bar_range_small_ok", "pullback_avg_volume_lower_ok",
            "pullback_no_big_bear_ok", "pullback_close_above_body_mid_ok",
            "signal_is_a_class_ok", "signal_is_b_class_ok"
        ]
        cols = [c for c in cols if c in debug_df.columns]
        print(debug_df[cols].head(20).to_string(index=False))

    if not failed_df.empty:
        print("\n抓取失败前20条：")
        cols = ["ticker", "name", "ts_code", "target_date", "error"]
        cols = [c for c in cols if c in failed_df.columns]
        print(failed_df[cols].head(20).to_string(index=False))

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"\n总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    if len(universe) > 0:
        print(f"平均每只股票耗时: {elapsed / len(universe):.2f} 秒")


if __name__ == "__main__":
    main()