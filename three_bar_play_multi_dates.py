# -*- coding: utf-8 -*-

"""
========================================================
Tushare版：A股日线 3 Bar Play 候补池扫描脚本
（C0 + A0/A1/A2/A3；target_day 属于当前结构；ABCS质量分级）
========================================================

【命名与时间轴】
1. C0_当日启动
   - target_day 当天本身就是启动棒

2. A0_第1根整理
   - target_day 当天是启动棒之后的第1根 resting bar

3. A1_第2根整理
   - target_day 当天是启动棒之后的第2根 resting bar

4. A2_第3根整理
   - target_day 当天是启动棒之后的第3根 resting bar

5. A3_第4根整理
   - target_day 当天是启动棒之后的第4根 resting bar

【使用定位】
- 本脚本用于“收盘后候补池筛选”
- 真正的突破/触发，留到下一个交易日盘中人工判断
- 因此 target_day 只判断“今天是否仍属于启动/整理结构”，不直接判断明日是否一定触发

【质量分级】
- 先按“启动棒质量 + resting结构质量 + equal-highs/inside标准度 + 趋势位置”计算总分
- 再映射成 S / A / B / C 四档
- inside/equal-highs 仍保留为单独备注与加分项，不做硬淘汰

【备注】
- inside/equal-highs 只做分级与排序，不参与硬淘汰
- 股票池先剔除 ST / 80亿以下
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

# GitHub Actions 可通过环境变量传入：
# TARGET_DATES=20260513 或 TARGET_DATES=20260512,20260513
# 未传入 TARGET_DATES 时，默认使用北京时间今天。
START_DATE = os.getenv("START_DATE", "20240301").strip()
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
TEST_LIMIT = None if not os.getenv("TEST_LIMIT") else int(os.getenv("TEST_LIMIT"))
BATCH_START = int(os.getenv("BATCH_START", "0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10000"))
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "0.10"))


# =========================
# 3 Bar Play 参数
# =========================
MA_PERIOD = 20

# 启动棒参数
IGNITE_MIN_PCT = 0.04
BODY_RATIO_MIN = 0.60
CLOSE_NEAR_HIGH_MAX = 0.35
BREAKOUT_NEAR_30D_RATIO = 0.98

# target_day 作为第 n 根 resting bar，n = 1~4
RESTING_BARS_MIN = 1
RESTING_BARS_MAX = 4
PULLBACK_BAR_RANGE_RATIO_MAX = 0.70
BIG_BEAR_DROP_MAX = -0.05

# inside / equal highs 备注参数（只做说明，不参与硬筛）
STRICT_EQUAL_HIGH_RATIO = 0.05   # 严格等高：启动棒振幅的 5%
LOOSE_EQUAL_HIGH_RATIO = 0.10    # 宽松等高：启动棒振幅的 10%
INSIDE_BAR_TICK_TOL = 0.01       # inside 严格容差
INSIDE_BAR_LOOSE_TOL = 0.03      # inside 宽松容差


# =========================
# 日期处理
# =========================
def get_today_cn_str() -> str:
    return (datetime.utcnow() + timedelta(hours=8)).strftime("%Y%m%d")


def get_target_dates():
    """
    优先读取环境变量 TARGET_DATES，例如：
    TARGET_DATES=20260513
    TARGET_DATES=20260512,20260513
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
    if len(digits) > 6:
        return digits[-6:]
    if len(digits) > 0:
        return digits.zfill(6)
    return ""


def code6_to_ts_code(code6: str) -> str:
    if code6.startswith(("600", "601", "603", "605", "688", "900")):
        return f"{code6}.SH"
    return f"{code6}.SZ"


def to_target_dt(target_date: str):
    return pd.to_datetime(target_date, format="%Y%m%d")


def bool_or_none(x):
    if x is None:
        return None
    return bool(x)


def build_signal_metrics(signal_date, close_now, ref_close=np.nan, ref_name="reference_close"):
    if pd.isna(ref_close):
        dist_pct = np.nan
        ref_close_val = np.nan
    else:
        ref_close_val = round(float(ref_close), 2)
        dist_pct = round((float(close_now) / float(ref_close) - 1.0) * 100, 2)

    return {
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "close_now": round(float(close_now), 2),
        ref_name: ref_close_val,
        f"dist_to_{ref_name}_pct": dist_pct,
    }



def get_default_grade_fields():
    return {
        "inside_bar_strict_ok": None,
        "inside_bar_loose_ok": None,
        "equal_highs_strict_ok": None,
        "equal_highs_loose_ok": None,
        "grade_tier": "",
        "grade_note": "",
        "grade_rank": np.nan,
        "total_score": np.nan,

        "score_ignite_total": np.nan,
        "score_ignite_body_ratio": np.nan,
        "score_ignite_close_quality": np.nan,
        "score_ignite_pct_chg": np.nan,
        "score_ignite_vol_vs_ma": np.nan,
        "score_ignite_vol_vs_prev": np.nan,

        "score_rest_total": np.nan,
        "score_rest_tightness": np.nan,
        "score_rest_volume_contraction": np.nan,
        "score_rest_high_drift": np.nan,
        "score_rest_pullback_depth": np.nan,
        "score_rest_bar_count_bias": np.nan,

        "score_pattern_total": np.nan,
        "score_equal_highs": np.nan,
        "score_inside_bars": np.nan,

        "score_trend_total": np.nan,
        "score_prev30_position": np.nan,
        "score_ma20_slope": np.nan,

        "resting_high_span": np.nan,
        "resting_high_vs_ignite_high_pct": np.nan,
    }


def _piecewise_score_desc(value, bands_scores, default=0.0):
    """
    value 越大越好，例如 body_ratio / pct_chg / vol ratio
    bands_scores: [(threshold, score), ...] 按 threshold 从高到低
    """
    try:
        x = float(value)
        if pd.isna(x):
            return default
    except Exception:
        return default

    for threshold, score in bands_scores:
        if x >= threshold:
            return float(score)
    return float(default)


def _piecewise_score_asc(value, bands_scores, default=0.0):
    """
    value 越小越好，例如 close_near_high / drift / span ratio
    bands_scores: [(threshold, score), ...] 按 threshold 从小到大
    """
    try:
        x = float(value)
        if pd.isna(x):
            return default
    except Exception:
        return default

    for threshold, score in bands_scores:
        if x <= threshold:
            return float(score)
    return float(default)


def _score_ignite_quality(df: pd.DataFrame, ignite_idx: int):
    row = df.iloc[ignite_idx]

    prev_volume = np.nan
    if ignite_idx - 1 >= 0:
        prev_volume = float(df.iloc[ignite_idx - 1]["volume"])

    body_ratio = float(row["body_ratio"])
    close_near_high = float(row["close_near_high"])
    pct_chg = float(row["pct_chg"])
    vol_vs_ma5 = safe_ratio(row["volume"], row["vol_ma5"])
    vol_vs_ma20 = safe_ratio(row["volume"], row["vol_ma20"])
    vol_vs_ma_min = np.nanmin([vol_vs_ma5, vol_vs_ma20]) if not (pd.isna(vol_vs_ma5) and pd.isna(vol_vs_ma20)) else np.nan
    vol_vs_prev = safe_ratio(row["volume"], prev_volume)

    s_body = _piecewise_score_desc(body_ratio, [
        (0.75, 10), (0.70, 8), (0.65, 6), (0.60, 4)
    ])
    s_close = _piecewise_score_asc(close_near_high, [
        (0.15, 10), (0.20, 8), (0.25, 6), (0.35, 4)
    ])
    s_pct = _piecewise_score_desc(pct_chg, [
        (0.07, 7), (0.06, 6), (0.05, 5), (0.04, 4)
    ])
    s_vol_ma = _piecewise_score_desc(vol_vs_ma_min, [
        (1.8, 5), (1.5, 4), (1.2, 3), (1.0, 2)
    ])
    s_vol_prev = _piecewise_score_desc(vol_vs_prev, [
        (2.0, 3), (1.5, 2), (1.2, 1)
    ])

    total = s_body + s_close + s_pct + s_vol_ma + s_vol_prev
    return {
        "score_ignite_total": round(total, 2),
        "score_ignite_body_ratio": round(s_body, 2),
        "score_ignite_close_quality": round(s_close, 2),
        "score_ignite_pct_chg": round(s_pct, 2),
        "score_ignite_vol_vs_ma": round(s_vol_ma, 2),
        "score_ignite_vol_vs_prev": round(s_vol_prev, 2),
    }


def _score_resting_quality(df: pd.DataFrame, ignite_idx: int, last_idx: int):
    """
    resting bars = ignite_idx + 1 ~ last_idx（含 target_day）
    总分 35
    """
    ignite = df.iloc[ignite_idx]
    pb = df.iloc[ignite_idx + 1:last_idx + 1].copy()
    if pb.empty:
        return {
            "score_rest_total": 0.0,
            "score_rest_tightness": 0.0,
            "score_rest_volume_contraction": 0.0,
            "score_rest_high_drift": 0.0,
            "score_rest_pullback_depth": 0.0,
            "score_rest_bar_count_bias": 0.0,
            "resting_high_span": np.nan,
            "resting_high_vs_ignite_high_pct": np.nan,
        }

    ignite_range = max(float(ignite["bar_range_abs"]), 1e-8)
    ignite_high = float(ignite["high"])
    ignite_volume = max(float(ignite["volume"]), 1e-8)
    ignite_body_mid = (float(ignite["open"]) + float(ignite["close"])) / 2.0

    pb_range_ratio_max = float((pb["bar_range_abs"] / ignite_range).max())
    pb_avg_volume_ratio = float(pb["volume"].mean() / ignite_volume)
    pb_high_max = float(pb["high"].max())
    pb_high_min = float(pb["high"].min())
    pb_high_span = pb_high_max - pb_high_min
    high_drift_ratio = max((pb_high_max / ignite_high) - 1.0, 0.0)
    low_safety_ratio = (float(pb["low"].min()) - ignite_body_mid) / ignite_range
    rest_bar_count = len(pb)

    s_tightness = _piecewise_score_asc(pb_range_ratio_max, [
        (0.25, 10), (0.35, 8), (0.45, 6), (0.60, 4)
    ])
    s_volume = _piecewise_score_asc(pb_avg_volume_ratio, [
        (0.35, 8), (0.50, 6), (0.70, 4), (1.00, 2)
    ])
    s_drift = _piecewise_score_asc(high_drift_ratio, [
        (0.01, 7), (0.03, 5), (0.05, 3), (0.10, 1)
    ])
    s_pullback = _piecewise_score_desc(low_safety_ratio, [
        (0.20, 5), (0.10, 4), (0.05, 3), (0.00, 2)
    ])
    if rest_bar_count == 1:
        s_count = 5.0
    elif rest_bar_count == 2:
        s_count = 4.0
    elif rest_bar_count == 3:
        s_count = 3.0
    else:
        s_count = 2.0

    total = s_tightness + s_volume + s_drift + s_pullback + s_count
    return {
        "score_rest_total": round(total, 2),
        "score_rest_tightness": round(s_tightness, 2),
        "score_rest_volume_contraction": round(s_volume, 2),
        "score_rest_high_drift": round(s_drift, 2),
        "score_rest_pullback_depth": round(s_pullback, 2),
        "score_rest_bar_count_bias": round(s_count, 2),
        "resting_high_span": round(pb_high_span, 4),
        "resting_high_vs_ignite_high_pct": round((pb_high_max / ignite_high - 1.0) * 100, 2),
    }


def _score_trend_position(df: pd.DataFrame, ignite_idx: int):
    row = df.iloc[ignite_idx]
    close_vs_prev30 = safe_ratio(row["close"], row["high_30_prev"])
    ma20 = float(row["ma20"])
    ma20_prev = float(row["ma20_prev"])
    ma20_slope = (ma20 / ma20_prev - 1.0) if ma20_prev not in [0, np.nan] and not pd.isna(ma20_prev) else np.nan

    s_pos = _piecewise_score_desc(close_vs_prev30, [
        (1.03, 5), (1.01, 4), (0.995, 3), (0.98, 2)
    ])
    s_ma = _piecewise_score_desc(ma20_slope, [
        (0.010, 5), (0.005, 4), (0.002, 3), (0.000, 2)
    ])

    total = s_pos + s_ma
    return {
        "score_trend_total": round(total, 2),
        "score_prev30_position": round(s_pos, 2),
        "score_ma20_slope": round(s_ma, 2),
    }


def _score_pattern_shape(df: pd.DataFrame, ignite_idx: int, last_idx: int):
    """
    评分 20 分：
    equal highs 12 分
    inside bars 8 分
    同时保留 strict/loose 标记
    """
    pb = df.iloc[ignite_idx + 1:last_idx + 1].copy()
    ignite = df.iloc[ignite_idx]

    if pb.empty:
        base = get_default_grade_fields()
        base.update({
            "grade_tier": "C",
            "grade_note": "empty_resting_bars",
            "grade_rank": 1,
            "score_pattern_total": 0.0,
            "score_equal_highs": 0.0,
            "score_inside_bars": 0.0,
        })
        return base

    ignite_range_abs = max(float(ignite["bar_range_abs"]), 1e-8)
    strict_equal_tol = max(ignite_range_abs * STRICT_EQUAL_HIGH_RATIO, 0.01)
    loose_equal_tol = max(ignite_range_abs * LOOSE_EQUAL_HIGH_RATIO, 0.01)

    prev_high = float(ignite["high"])
    prev_low = float(ignite["low"])
    nested_inside_strict_ok = True
    nested_inside_loose_ok = True

    for _, row in pb.iterrows():
        cur_high = float(row["high"])
        cur_low = float(row["low"])

        if not (cur_high <= prev_high + INSIDE_BAR_TICK_TOL and cur_low >= prev_low - INSIDE_BAR_TICK_TOL):
            nested_inside_strict_ok = False
        if not (cur_high <= prev_high + INSIDE_BAR_LOOSE_TOL and cur_low >= prev_low - INSIDE_BAR_LOOSE_TOL):
            nested_inside_loose_ok = False

        prev_high = cur_high
        prev_low = cur_low

    resting_high_span = float(pb["high"].max()) - float(pb["high"].min())
    equal_highs_strict_ok = resting_high_span <= strict_equal_tol
    equal_highs_loose_ok = resting_high_span <= loose_equal_tol
    resting_high_vs_ignite_high_pct = round((float(pb["high"].max()) / float(ignite["high"]) - 1.0) * 100, 2)

    if equal_highs_strict_ok:
        s_equal = 12.0
    elif equal_highs_loose_ok:
        s_equal = 8.0
    else:
        span_ratio = resting_high_span / ignite_range_abs
        s_equal = 4.0 if span_ratio <= 0.15 else 0.0

    if nested_inside_strict_ok:
        s_inside = 8.0
    elif nested_inside_loose_ok:
        s_inside = 5.0
    else:
        # 部分 inside：至少一半 resting bar 满足 loose inside
        prev_high = float(ignite["high"])
        prev_low = float(ignite["low"])
        loose_hits = 0
        for _, row in pb.iterrows():
            cur_high = float(row["high"])
            cur_low = float(row["low"])
            if cur_high <= prev_high + INSIDE_BAR_LOOSE_TOL and cur_low >= prev_low - INSIDE_BAR_LOOSE_TOL:
                loose_hits += 1
            prev_high = cur_high
            prev_low = cur_low
        s_inside = 2.0 if loose_hits >= max(1, len(pb) // 2) else 0.0

    pattern_total = s_equal + s_inside
    total_est = pattern_total  # placeholder if used outside

    # 临时先给等级，后面还会结合 total_score 做上限/修正
    if nested_inside_strict_ok and equal_highs_strict_ok:
        grade_tier = "S"
        grade_note = "inside_strict + equal_highs_strict"
        grade_rank = 4
    elif nested_inside_strict_ok or equal_highs_strict_ok:
        grade_tier = "A"
        grade_note = "inside_strict or equal_highs_strict"
        grade_rank = 3
    elif nested_inside_loose_ok or equal_highs_loose_ok:
        grade_tier = "B"
        grade_note = "inside_loose or equal_highs_loose"
        grade_rank = 2
    else:
        grade_tier = "C"
        grade_note = "no_clear_inside_or_equal_highs"
        grade_rank = 1

    return {
        "inside_bar_strict_ok": bool(nested_inside_strict_ok),
        "inside_bar_loose_ok": bool(nested_inside_loose_ok),
        "equal_highs_strict_ok": bool(equal_highs_strict_ok),
        "equal_highs_loose_ok": bool(equal_highs_loose_ok),
        "grade_tier": grade_tier,
        "grade_note": grade_note,
        "grade_rank": grade_rank,
        "score_pattern_total": round(pattern_total, 2),
        "score_equal_highs": round(s_equal, 2),
        "score_inside_bars": round(s_inside, 2),
        "resting_high_span": round(resting_high_span, 4),
        "resting_high_vs_ignite_high_pct": resting_high_vs_ignite_high_pct,
    }


def _finalize_grade_from_scores(grade_fields):
    total_score = (
        float(grade_fields.get("score_ignite_total", 0) or 0)
        + float(grade_fields.get("score_rest_total", 0) or 0)
        + float(grade_fields.get("score_pattern_total", 0) or 0)
        + float(grade_fields.get("score_trend_total", 0) or 0)
    )
    grade_fields["total_score"] = round(total_score, 2)

    base_grade = "C"
    base_rank = 1
    if total_score >= 85:
        base_grade, base_rank = "S", 4
    elif total_score >= 70:
        base_grade, base_rank = "A", 3
    elif total_score >= 55:
        base_grade, base_rank = "B", 2
    else:
        base_grade, base_rank = "C", 1

    # 若 inside/equal 都完全没有，则最高不超过 A
    if not grade_fields.get("inside_bar_loose_ok") and not grade_fields.get("equal_highs_loose_ok"):
        if base_grade == "S":
            base_grade, base_rank = "A", 3

    shape_grade = grade_fields.get("grade_tier", "C")
    shape_rank = grade_fields.get("grade_rank", 1) or 1

    # 最终等级取“总分等级”和“结构等级”中的较高者，但如果结构完全差，仅允许升到 A
    final_rank = max(int(base_rank), int(shape_rank))
    if shape_rank == 1 and final_rank > 3:
        final_rank = 3

    final_grade = {1: "C", 2: "B", 3: "A", 4: "S"}[final_rank]
    grade_fields["grade_tier"] = final_grade
    grade_fields["grade_rank"] = final_rank
    grade_fields["grade_note"] = f"{grade_fields.get('grade_note','')} | total_score={grade_fields['total_score']}"
    return grade_fields


def evaluate_pattern_grade_resting(df: pd.DataFrame, ignite_idx: int, last_idx: int):
    shape_fields = _score_pattern_shape(df, ignite_idx, last_idx)
    ignite_fields = _score_ignite_quality(df, ignite_idx)
    rest_fields = _score_resting_quality(df, ignite_idx, last_idx)
    trend_fields = _score_trend_position(df, ignite_idx)

    merged = get_default_grade_fields()
    merged.update(shape_fields)
    merged.update(ignite_fields)
    merged.update(rest_fields)
    merged.update(trend_fields)
    merged = _finalize_grade_from_scores(merged)
    return merged


def evaluate_pattern_grade_c0(df: pd.DataFrame, last_idx: int):
    """
    C0 没有 resting bars，主要看启动棒质量 + 趋势位置；
    结构分给一个中性基础分，但最高不给 S（因为还没形成 rest）
    """
    merged = get_default_grade_fields()
    ignite_fields = _score_ignite_quality(df, last_idx)
    trend_fields = _score_trend_position(df, last_idx)
    merged.update(ignite_fields)
    merged.update(trend_fields)

    merged["score_rest_total"] = 0.0
    merged["score_pattern_total"] = 0.0
    merged["score_equal_highs"] = 0.0
    merged["score_inside_bars"] = 0.0
    merged["grade_tier"] = "C"
    merged["grade_note"] = "C0_ignite_only"
    merged["grade_rank"] = 1

    merged = _finalize_grade_from_scores(merged)
    if merged["grade_rank"] > 3:
        merged["grade_rank"] = 3
        merged["grade_tier"] = "A"
    return merged
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

        "resting_bars_1_to_4_ok": None,
        "resting_low_above_body_mid_ok": None,
        "resting_bar_range_small_ok": None,
        "resting_avg_volume_lower_ok": None,
        "resting_no_big_bear_ok": None,
        "resting_close_above_body_mid_ok": None,

        "signal_is_a_class_ok": None,
        "signal_is_c_class_ok": None,
    }


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
        "ignite_range_pct": round(float(row["daily_range_pct"]) * 100, 2) if not pd.isna(row["daily_range_pct"]) else np.nan,
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
    close_near_high_ok = row["close_near_high"] < CLOSE_NEAR_HIGH_MAX
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


def check_resting_bars_including_today(df: pd.DataFrame, ignite_idx: int, last_idx: int):
    """
    A0~A3 统一逻辑：
    target_day 当天本身就是第 n 根 resting bar。
    resting bars = ignite_idx + 1 到 last_idx（含 target_day）
    """
    ignite = df.iloc[ignite_idx]
    pb = df.iloc[ignite_idx + 1:last_idx + 1].copy()

    reasons = []
    metrics = {}
    factor_flags = get_default_factor_flags()

    resting_bars = len(pb)
    bars_ok = RESTING_BARS_MIN <= resting_bars <= RESTING_BARS_MAX
    factor_flags["resting_bars_1_to_4_ok"] = bool_or_none(bars_ok)

    if not bars_ok:
        reasons.append("resting_bars_not_1_to_4")
        return False, reasons, metrics, factor_flags

    if ignite["close"] <= ignite["open"]:
        reasons.append("ignite_not_bullish")
        return False, reasons, metrics, factor_flags

    if pb["volume"].isna().any():
        reasons.append("resting_volume_missing")
        return False, reasons, metrics, factor_flags

    ignite_body_mid = (ignite["open"] + ignite["close"]) / 2.0
    ignite_range_abs = ignite["bar_range_abs"]
    if ignite_range_abs <= 0:
        reasons.append("ignite_range_invalid")
        return False, reasons, metrics, factor_flags

    pb_range_abs = pb["bar_range_abs"]
    pb_pct = pb["close"].pct_change().fillna((pb.iloc[0]["close"] / ignite["close"]) - 1.0)

    resting_low = pb["low"].min()
    resting_high_max = pb["high"].max()
    resting_high_min = pb["high"].min()

    low_above_mid_ok = resting_low >= ignite_body_mid
    bar_range_small_ok = (pb_range_abs < ignite_range_abs * PULLBACK_BAR_RANGE_RATIO_MAX).all()
    avg_volume_lower_ok = pb["volume"].mean() < ignite["volume"]
    no_big_bear_ok = (pb_pct > BIG_BEAR_DROP_MAX).all()
    close_above_mid_ok = (pb["close"] >= ignite_body_mid).all()

    # 调整：硬筛选不再要求 resting bars 的最低价都守住启动棒实体中点，
    # 只要求收盘不要跌破启动棒实体中点；最低价回踩深浅更多交给评分层体现。
    factor_flags["resting_low_above_body_mid_ok"] = bool(low_above_mid_ok)
    factor_flags["resting_bar_range_small_ok"] = bool(bar_range_small_ok)
    factor_flags["resting_avg_volume_lower_ok"] = bool(avg_volume_lower_ok)
    factor_flags["resting_no_big_bear_ok"] = bool(no_big_bear_ok)
    factor_flags["resting_close_above_body_mid_ok"] = bool(close_above_mid_ok)

    metrics = {
        "resting_bar_count": resting_bars,
        "pullback_bars": resting_bars,
        "resting_start_date": pb.iloc[0]["date"].strftime("%Y-%m-%d"),
        "resting_end_date": pb.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "pullback_start_date": pb.iloc[0]["date"].strftime("%Y-%m-%d"),
        "pullback_end_date": pb.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "resting_low": round(float(resting_low), 2),
        "resting_high_max": round(float(resting_high_max), 2),
        "resting_high_min": round(float(resting_high_min), 2),
        "ignite_body_mid": round(float(ignite_body_mid), 2),
        "resting_avg_volume": round(float(pb["volume"].mean()), 0),
        "resting_max_bar_range_ratio": round(float((pb_range_abs / ignite_range_abs).max()), 4),
        "resting_high_span": round(float(resting_high_max - resting_high_min), 4),
        "resting_high_vs_ignite_high_pct": round((float(resting_high_max) / float(ignite["high"]) - 1.0) * 100, 2),
    }

    if not bar_range_small_ok:
        reasons.append("resting_bar_range_too_large")
    if not avg_volume_lower_ok:
        reasons.append("resting_avg_volume_not_lower")
    if not no_big_bear_ok:
        reasons.append("resting_big_bear_bar")
    if not close_above_mid_ok:
        reasons.append("resting_close_below_ignite_body_mid")

    return len(reasons) == 0, reasons, metrics, factor_flags


# =========================
# 总检查函数
# =========================
def check_3barplay_resting(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 40:
        return {
            "matched": False,
            "reason": "bars_not_enough",
            "reason_list": "bars_not_enough",
            **get_default_factor_flags(),
            **get_default_grade_fields(),
        }

    vol_ok, vol_reason = check_volume_available(df)
    if not vol_ok:
        return {
            "matched": False,
            "reason": vol_reason,
            "reason_list": vol_reason,
            **get_default_factor_flags(),
            **get_default_grade_fields(),
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
            **get_default_grade_fields(),
            **last_flags,
        }

    # -------------------------------------------------
    # 1) 先判 C0：今天本身就是启动棒
    # -------------------------------------------------
    c_ok, c_reasons, c_metrics, c_flags = check_ignite_bar(df, last_idx)
    if c_ok:
        factor_flags = get_default_factor_flags()
        factor_flags.update(last_flags)
        factor_flags.update(c_flags)
        factor_flags["signal_is_c_class_ok"] = True

        grade_fields = evaluate_pattern_grade_c0(df, last_idx)

        return {
            "matched": True,
            "signal_type": "C_当日启动",
            "setup_type": "C0_当日启动",
            "resting_bar_count": 0,
            **build_signal_metrics(df.iloc[-1]["date"], df.iloc[-1]["close"], np.nan, "ignite_reference_close"),
            **grade_fields,
            **c_metrics,
            **factor_flags,
        }

    candidate_debugs = []

    # -------------------------------------------------
    # 2) 判 A0~A3：target_day 当天就是第 1~4 根 resting bar
    # -------------------------------------------------
    for resting_bar_count in range(RESTING_BARS_MIN, RESTING_BARS_MAX + 1):
        factor_flags = get_default_factor_flags()
        factor_flags.update(last_flags)

        ignite_idx = last_idx - resting_bar_count
        window_match_ok = ignite_idx >= 30
        factor_flags["ignite_window_match_ok"] = bool(window_match_ok)

        if not window_match_ok:
            candidate_debugs.append({
                "reason": "ignite_position_out_of_lookback_window",
                "reason_list": "ignite_position_out_of_lookback_window",
                "resting_bar_count": resting_bar_count,
                **get_default_grade_fields(),
                **factor_flags,
            })
            continue

        ignite_ok, ignite_reasons, ignite_metrics, ignite_flags = check_ignite_bar(df, ignite_idx)
        factor_flags.update(ignite_flags)

        if not ignite_ok:
            candidate_debugs.append({
                "reason": "ignite_fail",
                "reason_list": "|".join(ignite_reasons),
                "resting_bar_count": resting_bar_count,
                **get_default_grade_fields(),
                **ignite_metrics,
                **factor_flags,
            })
            continue

        rest_ok, rest_reasons, rest_metrics, rest_flags = check_resting_bars_including_today(df, ignite_idx, last_idx)
        factor_flags.update(rest_flags)

        if not rest_ok:
            candidate_debugs.append({
                "reason": "resting_fail",
                "reason_list": "|".join(rest_reasons),
                **get_default_grade_fields(),
                **ignite_metrics,
                **rest_metrics,
                **factor_flags,
            })
            continue

        grade_fields = evaluate_pattern_grade_resting(df, ignite_idx, last_idx)
        factor_flags["signal_is_a_class_ok"] = True

        setup_num = resting_bar_count - 1
        return {
            "matched": True,
            "signal_type": "A_整理候补",
            "setup_type": f"A{setup_num}_第{resting_bar_count}根整理",
            **build_signal_metrics(df.iloc[-1]["date"], df.iloc[-1]["close"], df.iloc[ignite_idx]["close"], "ignite_reference_close"),
            **grade_fields,
            **ignite_metrics,
            **rest_metrics,
            **factor_flags,
        }

    if candidate_debugs:
        best = candidate_debugs[0]
        return {
            "matched": False,
            "reason": best.get("reason", "not_matched"),
            "reason_list": best.get("reason_list", best.get("reason", "not_matched")),
            **best,
        }

    return {
        "matched": False,
        "reason": "no_valid_3barplay_resting",
        "reason_list": "no_valid_3barplay_resting",
        **get_default_factor_flags(),
        **get_default_grade_fields(),
    }


# =========================
# 单股多日期评估
# =========================
def evaluate_one_stock_multi_dates(pro, ts_code: str, ticker: str, name: str, target_dates, start_date: str, end_date: str):
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
            "error": f"daily_fetch_exception: {type(e).__name__}: {e}",
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
            "error": f"standardize_exception: {type(e).__name__}: {e}",
        })
        return matched_list, debug_list, failed_list

    if df_full.empty:
        failed_list.append({
            "ticker": ticker,
            "ts_code": ts_code,
            "name": name,
            "target_date": "",
            "error": "no_data",
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
                    **get_default_factor_flags(),
                    **get_default_grade_fields(),
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
                    **get_default_factor_flags(),
                    **get_default_grade_fields(),
                })
                continue

            chk = check_3barplay_resting(df)

            if chk.get("matched", False):
                matched_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    **chk,
                })
            else:
                debug_list.append({
                    "ticker": ticker,
                    "ts_code": ts_code,
                    "name": name,
                    "target_date": target_date,
                    "error": chk.get("reason", "not_matched"),
                    "reason_list": chk.get("reason_list", ""),
                    **{k: v for k, v in chk.items() if k not in ["matched", "reason", "reason_list"]},
                })

            time.sleep(SLEEP_SEC)

        except Exception as e:
            failed_list.append({
                "ticker": ticker,
                "ts_code": ts_code,
                "name": name,
                "target_date": target_date,
                "error": f"check_exception: {type(e).__name__}: {e}",
            })

    return matched_list, debug_list, failed_list


# =========================
# 输出路径
# =========================
def make_output_paths(end_date: str):
    return {
        "output": os.path.join(OUTPUT_DIR, f"three_bar_play_resting_candidates_{end_date}.csv"),
        "debug": os.path.join(OUTPUT_DIR, f"three_bar_play_resting_debug_rejected_{end_date}.csv"),
        "failed": os.path.join(OUTPUT_DIR, f"three_bar_play_resting_failed_fetch_{end_date}.csv"),
        "filtered": os.path.join(OUTPUT_DIR, f"three_bar_play_resting_filtered_out_{end_date}.csv"),
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

    print("1) 读取仓库 data 目录股票列表，并剔除 ST / 80亿以下 ...")
    universe, filtered_out = load_universe_from_csv()

    print("原始待扫描股票数:", len(universe))
    print("过滤掉数量(ST/80亿以下):", len(filtered_out))
    print("TARGET_DATES:", target_dates)
    print("END_DATE:", end_date)
    print("START_DATE:", START_DATE)

    print("\n2) 开始 3 Bar Play 日线候补池扫描（resting bar + ABCS质量分级版）...")

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
        sort_cols = [c for c in [
            "target_date", "signal_type", "grade_rank", "total_score", "setup_type", "resting_bar_count", "ignite_vol_vs_ma5"
        ] if c in matched_df.columns]

        ascending_flags = []
        for c in sort_cols:
            if c in ["target_date", "signal_type", "setup_type", "resting_bar_count"]:
                ascending_flags.append(True)
            else:
                ascending_flags.append(False)

        matched_df = matched_df.sort_values(by=sort_cols, ascending=ascending_flags).reset_index(drop=True)

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
        print("\nA/C 类数量统计：")
        print(matched_df["signal_type"].value_counts())

    if not matched_df.empty and "setup_type" in matched_df.columns:
        print("\n形态子类型统计：")
        print(matched_df["setup_type"].value_counts())

    if not matched_df.empty and "grade_tier" in matched_df.columns:
        print("\n图形质量等级统计：")
        print(matched_df["grade_tier"].value_counts())

    if not matched_df.empty and "target_date" in matched_df.columns:
        print("\n按日期统计候选数量：")
        print(matched_df.groupby("target_date").size())

    print("\n失败原因统计：")
    for k, v in sorted(error_counter.items(), key=lambda x: -x[1]):
        print(k, v)

    if not matched_df.empty:
        print("\n候选前20条：")
        cols = [
            "ticker", "name", "ts_code", "target_date", "signal_type", "setup_type", "signal_date",
            "grade_tier", "total_score", "grade_note", "inside_bar_strict_ok", "equal_highs_strict_ok",
            "ignite_date", "resting_bar_count", "ignite_pct_chg", "ignite_vol_vs_ma5",
            "score_ignite_total", "score_rest_total", "score_pattern_total", "score_trend_total",
            "resting_high_span", "resting_high_vs_ignite_high_pct", "close_now"
        ]
        cols = [c for c in cols if c in matched_df.columns]
        print(matched_df[cols].head(20).to_string(index=False))

    if not debug_df.empty:
        print("\n调试未通过前20条：")
        cols = [
            "ticker", "name", "ts_code", "target_date", "error", "reason_list", "close_now",
            "setup_type", "grade_tier", "total_score", "grade_note",
            "last_day_close_above_ma20_ok", "last_day_ma20_not_down_ok",
            "ignite_window_match_ok", "ignite_pct_ge_4pct_ok", "ignite_body_ratio_ge_0p6_ok",
            "ignite_close_near_high_ok", "ignite_volume_above_ma5_ok", "ignite_volume_above_ma20_ok",
            "ignite_near_prev30_high_ok",
            "resting_bars_1_to_4_ok", "resting_low_above_body_mid_ok",
            "resting_bar_range_small_ok", "resting_avg_volume_lower_ok",
            "resting_no_big_bear_ok", "resting_close_above_body_mid_ok",
            "signal_is_a_class_ok", "signal_is_c_class_ok"
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
