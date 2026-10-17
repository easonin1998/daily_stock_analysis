# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""Independent risk overlay for shortlisted picks."""

from __future__ import annotations

from src.services.screening.models import Pick

_DEFAULT_RISK_PROFILE = {
    "chase_change_pct": 5.0,
    "chase_points": 4.0,
    "breakdown_change_pct": -4.0,
    "breakdown_points": 3.5,
    "low_nominal_price": 4.0,
    "low_nominal_price_points": 0.75,
    "near_par_price": 2.0,
    "near_par_price_points": 2.0,
    "abnormal_volume_ratio": 6.0,
    "abnormal_volume_ratio_points": 3.0,
    "high_turnover_rate": 15.0,
    "high_turnover_points": 3.0,
    "high_adr_pct": 7.0,
    "high_adr_points": 2.0,
    "extended_ema21_pct": 6.0,
    "extended_ema21_points": 2.5,
    "weak_sector_resonance_score": 35.0,
    "weak_sector_resonance_points": 2.0,
    "early_trigger_distance_pct": -2.5,
    "early_trigger_points": 1.5,
    "low_llm_confidence": 0.35,
    "low_llm_confidence_points": 1.5,
    "llm_risk_points": 1.2,
    "llm_risk_points_cap": 4.0,
    "deep_risk_points": 1.5,
    "deep_risk_points_cap": 4.5,
    "low_daily_quality_score": 70.0,
    "low_daily_quality_points": 2.0,
    "bad_daily_quality_flag_points": 3.0,
    "stale_daily_cache_points": 2.5,
    "fallback_daily_errors_points": 1.5,
    "fetch_failed_daily_points": 6.0,
}
_DEFAULT_PORTFOLIO_BUCKETS = {
    "金融": ("券商", "银行", "保险", "金融"),
    "地产链": ("地产", "房地产", "建材", "家居", "物业"),
    "新能源": ("新能源", "光伏", "锂电", "电池", "储能"),
    "AI算力": ("AI算力", "算力", "数据中心", "服务器", "光模块"),
    "消费": ("白酒", "食品", "家电", "零售", "消费"),
    "医药": ("医药", "医疗", "创新药"),
    "半导体": ("半导体", "芯片"),
}


def apply_risk_overlay(
    picks: list[Pick],
    *,
    max_penalty: float = 12.0,
    veto_high_risk: bool = False,
    profile: dict[str, object] | None = None,
) -> tuple[list[Pick], list[str]]:
    """Attach risk flags and subtract a bounded penalty from final_score."""
    if not picks:
        return picks, []

    max_penalty = max(float(max_penalty), 0.0)
    degradation: list[str] = []
    kept: list[Pick] = []
    risk_profile = _risk_profile(profile)

    for pick in picks:
        points, flags = assess_pick_risk(pick, profile=risk_profile)
        penalty = min(points, max_penalty)
        pick.risk_penalty = round(penalty, 4)
        pick.risk_score = round(0.0 if max_penalty == 0 else min(points / max_penalty * 100, 100), 4)
        pick.risk_level = _risk_level(points, max_penalty)
        pick.risk_flags = _unique([*pick.risk_flags, *flags])
        pick.final_score = round(float(pick.final_score) - penalty, 4)
        pick.excluded_by_risk = veto_high_risk and pick.risk_level == "high"
        if pick.excluded_by_risk:
            degradation.append(f"Risk veto excluded {pick.code}: {', '.join(pick.risk_flags)}")
            continue
        kept.append(pick)

    kept.sort(key=lambda item: item.final_score, reverse=True)
    for i, pick in enumerate(kept, start=1):
        pick.rank = i
    return kept, degradation


def apply_portfolio_overlay(
    picks: list[Pick],
    *,
    max_same_sector: int = 1,
    concentration_penalty: float = 4.0,
    profile: dict[str, object] | None = None,
) -> tuple[list[Pick], list[str]]:
    """Penalize repeated LLM sectors so Top N is not only one crowded trade."""
    if not picks:
        return picks, []

    portfolio_profile = profile or {}
    max_same_sector = max(int(portfolio_profile.get("max_same_bucket", max_same_sector)), 1)
    penalty_step = max(float(portfolio_profile.get("concentration_penalty", concentration_penalty)), 0.0)
    if penalty_step == 0:
        return picks, []

    ordered = sorted(picks, key=lambda item: item.final_score, reverse=True)
    if not any(_canonical_sector(_pick_sector(pick)) for pick in ordered):
        return ordered, []

    bucket_counts: dict[str, int] = {}
    bucket_penalties: dict[str, list[tuple[str, str, float]]] = {}
    for pick in ordered:
        sector = _canonical_sector(_pick_sector(pick))
        if not sector:
            continue
        bucket = _portfolio_bucket(sector, pick.llm_theme, buckets=portfolio_profile.get("buckets"))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        excess = bucket_counts[bucket] - max_same_sector
        if excess <= 0:
            continue

        penalty = min(penalty_step * excess, penalty_step * 3)
        flag = f"portfolio_sector_concentration:{bucket}"
        pick.portfolio_penalty = round(float(pick.portfolio_penalty) + penalty, 4)
        pick.portfolio_flags = _unique([*pick.portfolio_flags, flag])
        pick.risk_flags = _unique([*pick.risk_flags, flag])
        pick.final_score = round(float(pick.final_score) - penalty, 4)
        bucket_penalties.setdefault(bucket, []).append((pick.code, sector, penalty))

    ordered.sort(key=lambda item: item.final_score, reverse=True)
    for i, pick in enumerate(ordered, start=1):
        pick.rank = i
    notes = [
        (
            f"Portfolio concentration bucket={bucket}: "
            f"penalized={len(items)}, "
            f"codes={_format_penalty_codes(items)}"
        )
        for bucket, items in bucket_penalties.items()
    ]
    return ordered, notes


def apply_sector_quota(
    picks: list[Pick],
    *,
    max_output: int,
    max_same_sector: int = 2,
    max_score_gap: float = 3.0,
    profile: dict[str, object] | None = None,
) -> tuple[list[Pick], list[str], int]:
    """Diversify the final set without replacing materially stronger picks.

    Only candidates below the initial output boundary are used as replacements.
    If no different-sector candidate is within ``max_score_gap``, the original
    over-quota pick is kept so diversification never reduces output quality or
    result count merely to satisfy a cosmetic quota.
    """
    output_count = max(int(max_output), 0)
    if not picks or output_count == 0:
        return [], [], 0

    portfolio_profile = profile or {}
    max_same_sector = max(int(portfolio_profile.get("max_same_bucket", max_same_sector)), 1)
    max_score_gap = max(float(portfolio_profile.get("quota_max_score_gap", max_score_gap)), 0.0)
    baseline = list(picks[:output_count])
    reserve = list(picks[output_count:])
    if not any(_canonical_sector(_pick_sector(pick)) for pick in baseline):
        return _rerank(baseline), [], 0

    bucket_counts: dict[str, int] = {}
    kept: list[Pick] = []
    overflow: list[Pick] = []
    for pick in baseline:
        bucket = _pick_bucket(pick, portfolio_profile)
        if not bucket:
            kept.append(pick)
            continue
        if bucket_counts.get(bucket, 0) < max_same_sector:
            kept.append(pick)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        else:
            overflow.append(pick)

    if not overflow:
        return _rerank(baseline), [], 0

    notes: list[str] = []
    replacements = 0
    used_codes = {pick.code for pick in kept}
    for displaced in overflow:
        replacement = next(
            (
                candidate
                for candidate in reserve
                if candidate.code not in used_codes
                and _quota_candidate_allowed(
                    candidate,
                    displaced=displaced,
                    bucket_counts=bucket_counts,
                    max_same_sector=max_same_sector,
                    max_score_gap=max_score_gap,
                    profile=portfolio_profile,
                )
            ),
            None,
        )
        if replacement is None:
            kept.append(displaced)
            used_codes.add(displaced.code)
            bucket = _pick_bucket(displaced, portfolio_profile)
            if bucket:
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            notes.append(
                f"Sector quota retained {displaced.code}: no comparable cross-sector reserve "
                f"within {max_score_gap:.1f} points"
            )
            continue

        kept.append(replacement)
        used_codes.add(replacement.code)
        bucket = _pick_bucket(replacement, portfolio_profile)
        if bucket:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        replacement.portfolio_flags = _unique(
            [*replacement.portfolio_flags, "portfolio_sector_quota_backfill"]
        )
        replacements += 1
        notes.append(
            f"Sector quota replaced {displaced.code} with {replacement.code} "
            f"(score_gap={float(displaced.final_score) - float(replacement.final_score):.2f})"
        )

    return _rerank(kept[:output_count]), notes, replacements


def _quota_candidate_allowed(
    candidate: Pick,
    *,
    displaced: Pick,
    bucket_counts: dict[str, int],
    max_same_sector: int,
    max_score_gap: float,
    profile: dict[str, object],
) -> bool:
    bucket = _pick_bucket(candidate, profile)
    if not bucket or bucket_counts.get(bucket, 0) >= max_same_sector:
        return False
    return float(candidate.final_score) >= float(displaced.final_score) - max_score_gap


def _pick_bucket(pick: Pick, profile: dict[str, object]) -> str:
    sector = _canonical_sector(_pick_sector(pick))
    if not sector:
        return ""
    return _portfolio_bucket(sector, pick.llm_theme, buckets=profile.get("buckets"))


def _rerank(picks: list[Pick]) -> list[Pick]:
    ordered = sorted(picks, key=lambda item: (-float(item.final_score), item.code))
    for index, pick in enumerate(ordered, start=1):
        pick.rank = index
    return ordered


def assess_pick_risk(
    pick: Pick,
    *,
    profile: dict[str, float] | None = None,
) -> tuple[float, list[str]]:
    """Return risk points and human-readable flags for one pick."""
    profile = _risk_profile(profile)
    points = 0.0
    flags: list[str] = []

    if pick.change_pct >= profile["chase_change_pct"]:
        points += profile["chase_points"]
        flags.append("single_day_chase_risk")
    elif pick.change_pct <= profile["breakdown_change_pct"]:
        points += profile["breakdown_points"]
        flags.append("single_day_breakdown_risk")

    if 0 < pick.price < profile["near_par_price"]:
        points += profile["near_par_price_points"]
        flags.append("near_par_price_risk")
    elif 0 < pick.price < profile["low_nominal_price"]:
        points += profile["low_nominal_price_points"]
        flags.append("low_nominal_price")

    if pick.volume_ratio is not None and pick.volume_ratio >= profile["abnormal_volume_ratio"]:
        points += profile["abnormal_volume_ratio_points"]
        flags.append("abnormal_volume_ratio")
    if pick.turnover_rate is not None and pick.turnover_rate >= profile["high_turnover_rate"]:
        points += profile["high_turnover_points"]
        flags.append("high_turnover")
    if pick.adr_20_pct is not None and pick.adr_20_pct >= profile["high_adr_pct"]:
        points += profile["high_adr_points"]
        flags.append("high_adr_execution_risk")
    if pick.ema21 is not None and pick.ema21 > 0:
        distance_to_ema21 = (pick.price / pick.ema21 - 1.0) * 100.0
        if distance_to_ema21 >= profile["extended_ema21_pct"]:
            points += profile["extended_ema21_points"]
            flags.append("extended_from_ema21")
    sector_score = float((pick.factor_scores or {}).get("sector_resonance", 50.0))
    if sector_score < profile["weak_sector_resonance_score"]:
        points += profile["weak_sector_resonance_points"]
        flags.append("weak_sector_resonance")
    if (
        pick.trigger_distance_pct is not None
        and pick.trigger_distance_pct < profile["early_trigger_distance_pct"]
    ):
        points += profile["early_trigger_points"]
        flags.append("trigger_not_confirmed")
    if pick.llm_confidence is not None and pick.llm_confidence < profile["low_llm_confidence"]:
        points += profile["low_llm_confidence_points"]
        flags.append("low_llm_confidence")

    daily_points, daily_flags = _assess_daily_data_risk(pick, profile)
    points += daily_points
    flags.extend(daily_flags)

    llm_risks = [risk for risk in pick.llm_risks if risk]
    if llm_risks:
        points += min(len(llm_risks) * profile["llm_risk_points"], profile["llm_risk_points_cap"])
        flags.extend(llm_risks)

    if pick.deep_analysis_risk_flags:
        points += min(
            len(pick.deep_analysis_risk_flags) * profile["deep_risk_points"],
            profile["deep_risk_points_cap"],
        )
        flags.extend(pick.deep_analysis_risk_flags)

    return points, _unique(flags)


def _assess_daily_data_risk(pick: Pick, profile: dict[str, float]) -> tuple[float, list[str]]:
    """Convert row-level daily-history quality metadata into risk points."""
    points = 0.0
    flags: list[str] = []

    if pick.daily_quality_score is not None and pick.daily_quality_score < profile["low_daily_quality_score"]:
        points += profile["low_daily_quality_points"]
        flags.append("low_daily_quality")

    quality_flags = _daily_quality_flag_set(pick.daily_quality_flags)
    if "fetch_failed" in quality_flags:
        points += profile["fetch_failed_daily_points"]
        flags.append("daily_fetch_failed")
    if "stale_cache" in quality_flags:
        points += profile["stale_daily_cache_points"]
        flags.append("daily_stale_cache")
    if "fallback_errors" in quality_flags:
        points += profile["fallback_daily_errors_points"]
        flags.append("daily_source_fallback_errors")

    severe_quality_flags = {"invalid_ohlc", "non_positive_price", "negative_volume"}
    if quality_flags & severe_quality_flags:
        points += profile["bad_daily_quality_flag_points"]
        flags.append("bad_daily_quality_flags")

    return points, flags


def _daily_quality_flag_set(value: str) -> set[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return set()
    normalized = text
    for separator in (",", "，", "|", " "):
        normalized = normalized.replace(separator, ";")
    return {item.strip() for item in normalized.split(";") if item.strip()}


def _risk_level(points: float, max_penalty: float) -> str:
    if max_penalty <= 0:
        return "low"
    if points >= max_penalty * 0.66:
        return "high"
    if points >= max_penalty * 0.33:
        return "medium"
    return "low"


def _canonical_sector(label: str) -> str:
    cleaned = str(label or "").strip()[:40]
    if not cleaned:
        return ""
    aliases = {
        "券商": ("券商", "证券"),
        "银行": ("银行",),
        "保险": ("保险",),
        "地产": ("地产", "房地产"),
        "医药": ("医药", "医疗", "创新药"),
        "白酒": ("白酒", "酿酒"),
        "半导体": ("半导体", "芯片"),
        "AI算力": ("AI算力", "算力", "数据中心"),
        "新能源": ("新能源", "光伏", "锂电", "电池"),
    }
    for canonical, needles in aliases.items():
        if any(needle in cleaned for needle in needles):
            return canonical
    return cleaned


def _pick_sector(pick: Pick) -> str:
    return pick.llm_sector or pick.industry


def _risk_profile(profile: dict[str, object] | None) -> dict[str, float]:
    result = dict(_DEFAULT_RISK_PROFILE)
    for key, value in (profile or {}).items():
        if key in result:
            result[key] = float(value)
    return result


def _portfolio_bucket(
    sector: str,
    theme: str = "",
    *,
    buckets: object = None,
) -> str:
    text = f"{sector} {theme or ''}"
    bucket_map = _portfolio_buckets(buckets)
    for bucket, needles in bucket_map.items():
        if any(needle in text for needle in needles):
            return bucket
    return sector


def _portfolio_buckets(custom_buckets: object = None) -> dict[str, tuple[str, ...]]:
    bucket_map = {key: tuple(value) for key, value in _DEFAULT_PORTFOLIO_BUCKETS.items()}
    if not isinstance(custom_buckets, dict):
        return bucket_map
    for bucket, needles in custom_buckets.items():
        if isinstance(needles, str):
            items = [needles]
        elif isinstance(needles, list):
            items = [str(item) for item in needles]
        else:
            continue
        if items:
            bucket_map[str(bucket)] = tuple(items)
    return bucket_map


def _format_penalty_codes(items: list[tuple[str, str, float]], limit: int = 5) -> str:
    shown = [
        f"{code}:{sector}(-{penalty:.1f})"
        for code, sector, penalty in items[:limit]
    ]
    if len(items) > limit:
        shown.append(f"+{len(items) - limit} more")
    return ",".join(shown)


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result
