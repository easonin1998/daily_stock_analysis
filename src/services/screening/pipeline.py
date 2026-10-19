# -*- coding: utf-8 -*-
# Derived from AlphaSift revision 9f522747caafd3c0b1ddb7e14d5cf44c8580b6cf.
# Licensed under Apache-2.0 and modified for daily_stock_analysis.
"""Main pipeline — orchestrates L1 → L2 → result."""

import copy
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.services.screening.config import Config
from src.services.screening.candidate_context import collect_candidate_context
from src.services.screening.context import build_llm_context
from src.services.screening.daily import enrich_daily_features
from src.services.screening.daily_library import (
    completed_snapshot_bar_date,
    latest_daily_feature_date,
    merge_daily_feature_library,
    refresh_daily_feature_library_from_cache,
    upsert_daily_feature_library,
)
from src.services.screening.dsa_provider import apply_dsa_provider_context
from src.services.screening.filter import (
    apply_hard_filters,
    hard_filter_rejection_summary,
    hard_filter_waterfall,
    requires_daily_features,
    without_daily_filters,
)
from src.services.screening.industry import enrich_industry_concepts
from src.services.screening.martin_setups import (
    MULTI_SETUP_MODE,
    attach_martin_entry_setups,
    common_martin_hard_filters,
)
from src.services.screening.sector_strength import (
    attach_sector_strength_features,
    compute_sector_strength_snapshot,
    load_sector_strength_history,
    select_sector_strength_snapshot,
    summarize_sector_strength_snapshot,
    upsert_sector_strength_history,
)
from src.services.screening.market_regime import (
    MarketRegimePolicy,
    REGIME_RISK_OFF,
    load_market_regime_history,
    resolve_market_regime_limits,
    unavailable_market_regime,
    update_market_regime_history,
)
from src.services.screening.models import Pick, ScreenResult
from src.services.screening.normalize import (
    normalize_code,
    safe_bool as _safe_bool,
    safe_float as _safe_float,
    safe_int as _safe_int,
    safe_text,
)
from src.services.screening.post_analysis import normalize_post_analyzers, run_post_analyzers
from src.services.screening.ranker import rank_candidates_with_metadata
from src.services.screening.risk import apply_portfolio_overlay, apply_risk_overlay, apply_sector_quota
from src.services.screening.scorer import compute_screen_scores, factor_score_columns
from src.services.screening.trade_plan import MartinTradePlanConfig, build_martin_trade_plan
from src.services.screening.selection_variant import apply_seeded_selection_variant
from src.services.screening.snapshot import fetch_snapshot_with_fallback
from src.services.screening.strategy import load_all_strategies
from src.services.screening.universe import build_seed_pool

logger = logging.getLogger(__name__)


def screen(
    strategy: str,
    *,
    market: str = "cn",
    max_output: int | None = None,
    use_llm: bool = True,
    llm_context: str | None = None,
    llm_context_files: list[str | Path] | None = None,
    candidate_context_files: list[str | Path] | None = None,
    collect_llm_candidate_context: bool | None = None,
    candidate_context_max_candidates: int | None = None,
    candidate_context_providers: list[str] | None = None,
    industry_map_files: list[str | Path] | None = None,
    industry_provider: str | None = None,
    post_analyzers: list[str] | None = None,
    post_analysis_max_picks: int | None = None,
    daily_enrich: bool | None = None,
    daily_enrich_max_candidates: int | None = None,
    explain_filters: bool = False,
    deep_analysis: bool = False,
    deep_analysis_max_picks: int | None = None,
    selection_seed: str = "",
    context: dict[str, object] | None = None,
    config: Config | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    daily_history_fetcher: Callable[..., pd.DataFrame] | None = None,
) -> ScreenResult:
    """Execute stock screening with the given strategy.

    Args:
        strategy: Strategy name (matches a YAML file in strategies/).
        market: Market scope, currently only "cn".
        max_output: Override max output count from strategy.
        use_llm: Whether to use LLM for L2 ranking.
        llm_context: Optional market/news/theme context supplied to the LLM ranker.
        llm_context_files: Optional text files appended to LLM context.
        candidate_context_files: Optional CSV/JSON/JSONL files keyed by code with candidate-level context.
        collect_llm_candidate_context: Whether to fetch Top-K candidate news/fund-flow context for LLM.
        candidate_context_max_candidates: Max candidates to fetch external context for.
        candidate_context_providers: Optional provider names: news, fund_flow, announcement.
        industry_map_files: Optional code->industry/concepts files used before L1/L2.
        industry_provider: Optional provider for board mapping, e.g. "akshare".
        post_analyzers: Optional L3 analyzers, e.g. ["scorecard", "dsa"].
        post_analysis_max_picks: Override the remote post-analyzer candidate
            cap. The local scorecard always evaluates the shortlisted pool.
        daily_enrich: Whether to enrich shortlisted candidates with daily K-line features.
        daily_enrich_max_candidates: Max candidates to enrich after snapshot filtering.
        explain_filters: Whether to include sequential hard-filter waterfall diagnostics.
        deep_analysis: Backward-compatible alias for post_analyzers=["dsa"].
        deep_analysis_max_picks: Backward-compatible max-picks alias for DSA.
        selection_seed: Optional opaque client seed used for bounded per-run
            sampling among near-score candidates. The seed is never persisted.
        context: Optional host runtime context. DSA may provide LLM settings and
            callable data providers under context["dsa"].
        config: Runtime config. Defaults to Config.from_env().
        daily_history_fetcher: Optional request-scoped daily-history provider.
            It is tried in place of the bundled fetcher without global patching.

    Returns:
        ScreenResult with ranked picks.
    """
    if config is None:
        config = Config.from_env()

    if market not in ("cn", "us"):
        raise ValueError(f"Unsupported market: {market!r} (supported: cn, us)")

    run_id = uuid.uuid4().hex[:12]
    degradation: list[str] = []

    # 1. Load strategy
    strategies = load_all_strategies(config.strategies_dir)
    if strategy not in strategies:
        available = ", ".join(strategies.keys()) or "(none)"
        raise ValueError(f"Strategy '{strategy}' not found. Available: {available}")

    strat = strategies[strategy]
    screening = strat.screening
    execution_profile = screening.execution_profile or {}
    if market not in screening.market_scope:
        raise ValueError(
            f"Strategy '{strategy}' does not support market '{market}'. "
            f"Supported: {', '.join(screening.market_scope)}"
        )
    output_count = max_output or screening.max_output
    analyzer_names = normalize_post_analyzers(
        post_analyzers if post_analyzers is not None else config.post_analyzers
    )
    if deep_analysis and "dsa" not in analyzer_names:
        analyzer_names.append("dsa")
    analyzer_max_picks = (
        post_analysis_max_picks
        or deep_analysis_max_picks
    )
    daily_needed = requires_daily_features(screening.hard_filters)
    daily_requested = config.daily_enrich_enabled if daily_enrich is None else daily_enrich
    daily_limit = daily_enrich_max_candidates or config.daily_enrich_max_candidates
    snapshot_filters = without_daily_filters(screening.hard_filters) if daily_needed else screening.hard_filters

    # 2. Fetch snapshot
    _emit_progress(progress_callback, 25, "正在读取全市场快照")
    snapshot_df = fetch_snapshot_with_fallback(
        config.snapshot_source_priority,
        required_columns=_required_snapshot_columns(snapshot_filters),
        fallback_snapshot_path=config.fallback_snapshot_path,
        fallback_max_age_hours=config.snapshot_fallback_max_age_hours,
        cache_ttl_seconds=config.snapshot_cache_ttl_seconds,
        market=market,
    )
    effective_industry_map_files = (
        list(industry_map_files)
        if industry_map_files is not None
        else list(config.industry_map_files)
    )
    effective_industry_provider = (
        industry_provider
        if industry_provider is not None
        else config.industry_provider
    )
    effective_industry_provider = str(effective_industry_provider or "none").strip().lower()
    if effective_industry_map_files or effective_industry_provider not in {"", "none", "off", "false"}:
        snapshot_df, industry_notes = enrich_industry_concepts(
            snapshot_df,
            map_files=effective_industry_map_files,
            provider=effective_industry_provider,
            max_boards=config.industry_provider_max_boards,
            provider_cache_dir=config.industry_provider_cache_dir,
            provider_cache_ttl_hours=config.industry_provider_cache_ttl_hours,
            include_fund_flow=config.sector_fund_flow_enabled,
        )
        degradation.extend(f"Industry/concepts enrichment: {item}" for item in industry_notes)
    snapshot_df = _attach_market_regime_features(snapshot_df)
    snapshot_count = len(snapshot_df)
    snapshot_source = str(snapshot_df.attrs.get("snapshot_source", ""))
    source_errors = [str(item) for item in snapshot_df.attrs.get("source_errors", [])]
    degradation.extend(f"Snapshot source fallback: {item}" for item in source_errors)
    if bool(snapshot_df.attrs.get("fallback_used")):
        stale_age = snapshot_df.attrs.get("stale_age_hours")
        if stale_age is None:
            degradation.append("Snapshot source fallback: last_good_cache stale")
        else:
            degradation.append(
                f"Snapshot source fallback: last_good_cache stale_age_hours={stale_age}"
            )

    completed_bar_date = None
    library_latest_date = None
    if (
        market == "cn"
        and config.daily_feature_library_enabled
        and config.daily_feature_library_path is not None
    ):
        completed_bar_date = completed_snapshot_bar_date(snapshot_df, market=market)
        library_latest_date = latest_daily_feature_date(config.daily_feature_library_path)
        needs_library_refresh = library_latest_date is None or (
            completed_bar_date is not None
            and library_latest_date < completed_bar_date
        )
        if needs_library_refresh and config.daily_history_cache_dir is not None:
            library_refresh = refresh_daily_feature_library_from_cache(
                snapshot_df,
                cache_dir=config.daily_history_cache_dir,
                library_path=config.daily_feature_library_path,
                source=config.daily_source,
                lookback_days=config.daily_lookback_days,
                price_adjustment="qfq",
                append_snapshot_bar=completed_bar_date is not None,
                snapshot_bar_date=completed_bar_date,
            )
            degradation.append(
                "Local daily-feature library maintenance: "
                f"cached={library_refresh.get('cache_count', 0)}, "
                f"updated={library_refresh.get('updated_count', 0)}, "
                f"appended={library_refresh.get('appended_count', 0)}, "
                f"errors={library_refresh.get('error_count', 0)}"
            )
            library_latest_date = latest_daily_feature_date(
                config.daily_feature_library_path
            )

    effective_bar_date = completed_bar_date or library_latest_date
    if completed_bar_date is None and library_latest_date is not None:
        degradation.append(
            "Premarket analysis date: using latest completed local daily bar "
            f"{library_latest_date.isoformat()}"
        )

    full_market_feature_universe = pd.DataFrame()
    full_market_library_diagnostics: dict[str, object] = {}
    if (
        market == "cn"
        and effective_bar_date is not None
        and config.daily_feature_library_enabled
        and config.daily_feature_library_path is not None
    ):
        try:
            full_market_feature_universe, full_market_library_diagnostics = (
                merge_daily_feature_library(
                    snapshot_df,
                    library_path=config.daily_feature_library_path,
                    max_age_days=config.daily_feature_library_max_age_days,
                )
            )
            full_market_feature_universe = _attach_martin_cross_sectional_features(
                full_market_feature_universe
            )
        except Exception as exc:  # Local overlays must not break stock selection.
            degradation.append(f"Full-market local feature overlay failed: {exc}")
            full_market_feature_universe = pd.DataFrame()

    market_regime = unavailable_market_regime(
        "市场状态调节未启用"
        if not config.market_regime_enabled
        else "尚未建立可用的全市场状态"
    )
    if market == "cn" and config.market_regime_enabled:
        if (
            config.daily_feature_library_enabled
            and config.daily_feature_library_path is not None
        ):
            regime_policy = MarketRegimePolicy(
                min_universe_size=config.market_regime_min_universe_size,
                upgrade_confirmation_days=(
                    config.market_regime_upgrade_confirmation_days
                ),
            )
            regime_history_path = config.data_dir / "market_regime_history.json"
            try:
                if effective_bar_date is not None and not full_market_feature_universe.empty:
                    market_regime = update_market_regime_history(
                        full_market_feature_universe,
                        history_path=regime_history_path,
                        as_of=effective_bar_date,
                        policy=regime_policy,
                    )
                    degradation.append(
                        "Market regime technical coverage: "
                        f"matched={full_market_library_diagnostics.get('matched_count', 0)} of "
                        f"{len(full_market_feature_universe)}"
                    )
                else:
                    market_regime = load_market_regime_history(
                        regime_history_path,
                        policy=regime_policy,
                    )
                degradation.append(
                    "Market regime: "
                    f"state={market_regime.get('state')}, "
                    f"score={market_regime.get('score')}, "
                    f"date={market_regime.get('date') or '--'}"
                )
            except Exception as exc:  # The overlay must not break stock selection.
                market_regime = unavailable_market_regime(
                    f"市场状态计算失败：{exc}"
                )
                degradation.append(str(market_regime["reason"]))
        else:
            market_regime = unavailable_market_regime(
                "未启用本地日线指标库，无法计算多日市场状态"
            )
            degradation.append(str(market_regime["reason"]))

    sector_snapshot = pd.DataFrame()
    sector_overview = summarize_sector_strength_snapshot(sector_snapshot)
    if market == "cn" and bool(execution_profile.get("sector_leadership_enabled", False)):
        persisted_snapshot = pd.DataFrame()
        if config.sector_strength_history_path is not None:
            persisted_snapshot = select_sector_strength_snapshot(
                load_sector_strength_history(config.sector_strength_history_path),
                as_of=effective_bar_date,
            )
        effective_date_text = (
            effective_bar_date.isoformat() if effective_bar_date is not None else ""
        )
        persisted_date = str(
            persisted_snapshot.attrs.get("selected_as_of") or ""
        )
        if not persisted_snapshot.empty and persisted_date == effective_date_text:
            sector_snapshot = persisted_snapshot
            degradation.append(
                "Sector strength contract: reused full-market point-in-time snapshot "
                f"as_of={persisted_date}, sectors={len(sector_snapshot)}"
            )
        elif not full_market_feature_universe.empty:
            sector_snapshot = compute_sector_strength_snapshot(
                full_market_feature_universe,
                as_of=effective_bar_date,
                source="local_daily_feature_library",
            )
            degradation.append(
                "Sector strength contract: computed from full-market local features "
                f"as_of={effective_date_text or '--'}, sectors={len(sector_snapshot)}"
            )
            if (
                effective_bar_date is not None
                and config.sector_strength_history_path is not None
                and not sector_snapshot.empty
            ):
                sector_update = upsert_sector_strength_history(
                    sector_snapshot,
                    full_market_feature_universe,
                    path=config.sector_strength_history_path,
                    as_of=effective_bar_date,
                )
                degradation.append(
                    "Sector strength history updated: "
                    f"rows={sector_update.get('updated_count', 0)}, "
                    f"total={sector_update.get('record_count', 0)}"
                )
        elif not persisted_snapshot.empty:
            sector_snapshot = persisted_snapshot
            degradation.append(
                "Sector strength contract: using latest prior point-in-time snapshot "
                f"as_of={persisted_date or '--'}"
            )
        sector_overview = summarize_sector_strength_snapshot(sector_snapshot)

    # 3. L1 hard filter. If a strategy needs daily features, first apply only
    # snapshot-safe filters, then enrich a narrowed candidate pool.
    if explain_filters:
        snapshot_waterfall = hard_filter_waterfall(snapshot_df, snapshot_filters)
        if snapshot_waterfall:
            degradation.append(
                "Snapshot hard-filter waterfall: "
                + _format_filter_waterfall(snapshot_waterfall)
            )
    df = apply_hard_filters(snapshot_df, snapshot_filters)
    if (
        market == "cn"
        and config.daily_feature_library_enabled
        and config.daily_feature_library_path is not None
    ):
        df, library_diagnostics = merge_daily_feature_library(
            df,
            library_path=config.daily_feature_library_path,
            max_age_days=config.daily_feature_library_max_age_days,
        )
        degradation.append(
            "Local daily-feature library: "
            f"matched {library_diagnostics.get('matched_count', 0)} of {len(df)} candidates; "
            f"fresh={library_diagnostics.get('fresh_count', 0)}, "
            f"stale={library_diagnostics.get('stale_count', 0)}"
        )
    snapshot_filtered_count = len(df)
    after_filter_count = snapshot_filtered_count
    seed_pool_count = 0
    seed_pool_diagnostics: dict[str, object] = {}
    _emit_progress(
        progress_callback,
        42,
        f"快照筛选完成，保留 {snapshot_filtered_count} 条候选",
    )

    if df.empty:
        return ScreenResult(
            strategy=strategy,
            market=market,
            snapshot_count=snapshot_count,
            snapshot_filtered_count=0,
            after_filter_count=0,
            run_id=run_id,
            degradation=[*degradation, "No candidates after hard filter"],
            snapshot_source=snapshot_source,
            source_errors=source_errors,
            strategy_version=strat.version,
            strategy_category=strat.category,
            post_analyzers=analyzer_names,
            daily_enriched=False,
            risk_enabled=config.risk_enabled,
            portfolio_diversity_enabled=config.portfolio_diversity_enabled,
            market_regime=market_regime,
            sector_overview=sector_overview,
            analysis_date=(
                effective_bar_date.isoformat() if effective_bar_date is not None else ""
            ),
        )

    daily_enriched = False
    daily_enrich_count = 0
    if daily_needed or daily_requested:
        seed_selection = build_seed_pool(
            df,
            limit=daily_limit,
            period=f"{datetime.now(timezone.utc).date().isoformat()}:{market}:{strategy}",
            core_ratio=config.seed_pool_core_ratio,
            strength_ratio=config.seed_pool_strength_ratio,
            activity_ratio=config.seed_pool_activity_ratio,
            max_sector_ratio=config.seed_pool_max_sector_ratio,
        )
        daily_candidates = seed_selection.dataframe
        seed_pool_diagnostics = seed_selection.diagnostics
        seed_pool_count = len(daily_candidates)
        enrich_count = seed_pool_count
        degradation.append(
            "Seed pool selected "
            f"{seed_pool_count} of {snapshot_filtered_count} snapshot-filtered candidates; "
            f"sleeves={seed_pool_diagnostics.get('sleeve_counts', {})}"
        )
        _emit_progress(
            progress_callback,
            48,
            f"种子池已构建，正在计算 {seed_pool_count} 只股票的日线特征",
        )
        try:
            enriched = enrich_daily_features(
                daily_candidates,
                max_rows=enrich_count,
                lookback_days=config.daily_lookback_days,
                source=config.daily_source,
                fetch_retries=config.daily_fetch_retries,
                cache_dir=config.daily_history_cache_dir,
                cache_ttl_seconds=config.daily_history_cache_ttl_hours * 3600,
                max_workers=config.daily_fetch_max_workers,
                history_fetcher=daily_history_fetcher,
                price_adjustment="qfq" if market == "cn" else None,
            )
            if (
                market == "cn"
                and config.daily_feature_library_enabled
                and config.daily_feature_library_path is not None
            ):
                library_update = upsert_daily_feature_library(
                    enriched,
                    library_path=config.daily_feature_library_path,
                )
                degradation.append(
                    "Local daily-feature library updated: "
                    f"rows={library_update.get('updated_count', 0)}, "
                    f"total={library_update.get('entry_count', 0)}"
                )
            enriched = _attach_martin_cross_sectional_features_from_universe(
                enriched,
                universe=df,
            )
            setup_mode = str(
                execution_profile.get("entry_setup_mode") or "legacy_pullback"
            )
            if setup_mode == MULTI_SETUP_MODE:
                enriched = attach_martin_entry_setups(
                    enriched,
                    mode=setup_mode,
                    tradable_setup_types=execution_profile.get(
                        "tradable_setup_types"
                    ),
                )
            if bool(execution_profile.get("sector_leadership_enabled", False)):
                if sector_snapshot.empty:
                    sector_universe = _attach_martin_cross_sectional_features(df)
                    sector_snapshot = compute_sector_strength_snapshot(
                        sector_universe,
                        as_of=effective_bar_date,
                        source="screening_candidate_subset",
                    )
                    sector_overview = summarize_sector_strength_snapshot(
                        sector_snapshot
                    )
                enriched = attach_sector_strength_features(
                    enriched,
                    sector_snapshot,
                    unavailable_source="live_sector_unavailable",
                )
                degradation.append(
                    "Sector strength contract: "
                    f"available={len(sector_snapshot)} sectors, "
                    f"as_of={effective_bar_date or '--'}"
                )
            daily_enriched = True
            daily_errors = [str(item) for item in enriched.attrs.get("daily_errors", [])]
            daily_enrich_count = int(enriched.attrs.get("daily_success_count", len(enriched)))
            daily_source_counts = dict(enriched.attrs.get("daily_source_counts", {}) or {})
            daily_quality_flag_counts = dict(enriched.attrs.get("daily_quality_flag_counts", {}) or {})
            daily_source_order_notes = [str(item) for item in enriched.attrs.get("daily_source_order_notes", [])]
            daily_source_health_notes = _daily_source_health_notes(
                dict(enriched.attrs.get("daily_source_health", {}) or {})
            )
            degradation.append(
                f"Daily K-line enrichment attempted {enrich_count} candidates, "
                f"succeeded {daily_enrich_count} of {snapshot_filtered_count} snapshot-filtered candidates"
            )
            if daily_source_counts:
                source_summary = ", ".join(
                    f"{name}={count}" for name, count in sorted(daily_source_counts.items())
                )
                degradation.append(f"Daily K-line sources: {source_summary}")
            if daily_quality_flag_counts:
                flag_summary = ", ".join(
                    f"{name}={count}" for name, count in sorted(daily_quality_flag_counts.items())
                )
                degradation.append(f"Daily K-line quality flags: {flag_summary}")
            if daily_source_order_notes:
                degradation.append("Daily K-line source ordering: " + " | ".join(daily_source_order_notes[:3]))
            if daily_source_health_notes:
                degradation.append("Daily K-line source health: " + "; ".join(daily_source_health_notes))
            if daily_errors:
                sample = " | ".join(daily_errors[:5])
                suffix = f" | +{len(daily_errors) - 5} more" if len(daily_errors) > 5 else ""
                degradation.append(f"Daily K-line enrichment row errors: {sample}{suffix}")
            if daily_needed:
                effective_filters = (
                    common_martin_hard_filters(screening.hard_filters)
                    if setup_mode == MULTI_SETUP_MODE
                    else screening.hard_filters
                )
                if explain_filters:
                    daily_waterfall = hard_filter_waterfall(enriched, effective_filters)
                    if daily_waterfall:
                        degradation.append(
                            "Daily hard-filter waterfall: "
                            + _format_filter_waterfall(daily_waterfall)
                        )
                daily_filter_rejections = hard_filter_rejection_summary(
                    enriched,
                    effective_filters,
                    limit=6,
                )
                df = apply_hard_filters(enriched, effective_filters)
                if setup_mode == MULTI_SETUP_MODE and "setup_confirmed" in df.columns:
                    df = df[df["setup_confirmed"] == True].copy()  # noqa: E712
                after_filter_count = len(df)
                if daily_filter_rejections:
                    degradation.append(
                        "Daily hard-filter rejections: "
                        + "; ".join(daily_filter_rejections)
                    )
            else:
                df = enriched
        except Exception as exc:
            if daily_needed:
                raise RuntimeError(
                    "Daily K-line enrichment is required by this strategy but failed: "
                    f"{exc}"
                ) from exc
            degradation.append(f"Daily K-line enrichment skipped: {exc}")

    if df.empty:
        return ScreenResult(
            strategy=strategy,
            market=market,
            strategy_version=strat.version,
            strategy_category=strat.category,
            snapshot_count=snapshot_count,
            snapshot_filtered_count=snapshot_filtered_count,
            seed_pool_count=seed_pool_count,
            seed_pool_limit=daily_limit,
            seed_pool_diagnostics=seed_pool_diagnostics,
            after_filter_count=0,
            run_id=run_id,
            degradation=[*degradation, "No candidates after daily hard filter"],
            snapshot_source=snapshot_source,
            source_errors=source_errors,
            post_analyzers=analyzer_names,
            daily_enriched=daily_enriched,
            daily_enrich_count=daily_enrich_count,
            risk_enabled=config.risk_enabled,
            portfolio_diversity_enabled=config.portfolio_diversity_enabled,
            market_regime=market_regime,
            sector_overview=sector_overview,
            analysis_date=(
                effective_bar_date.isoformat() if effective_bar_date is not None else ""
            ),
        )

    # 4. Compute screen_score
    df = _sort_screened_candidates(compute_screen_scores(df, screening), screening)

    # 5. Take Top K for LLM ranking
    top_k = min(
        max(output_count * config.llm_candidate_multiplier, output_count),
        config.llm_max_candidates,
        len(df),
    )
    df_top = df.head(top_k)

    # 6. Build Pick list
    picks = _df_to_picks(df_top)

    # 6.5. Host-provided candidate context, e.g. DSA realtime quote,
    # fundamentals, and news. This runs before LLM ranking so L2 can use it.
    _emit_progress(progress_callback, 52, "正在补充候选行情与基本面")
    degradation.extend(apply_dsa_provider_context(picks, context))
    llm_fallback_picks = [copy.deepcopy(pick) for pick in picks]

    # 7. L2 LLM ranking
    llm_ranked = False
    llm_market_view = ""
    llm_selection_logic = ""
    llm_portfolio_risk = ""
    llm_coverage: float | None = None
    llm_parse_errors: list[str] = []
    llm_model_used = ""
    llm_attempted_models: list[str] = []
    llm_failure_reason = ""
    if use_llm and config.has_llm_config():
        _emit_progress(progress_callback, 66, "正在执行 LLM 候选重排")
        candidate_context_rows: list[dict[str, object]] = []
        event_source_weights = _event_source_weights(screening.event_profile)
        should_collect_candidate_context = (
            config.llm_candidate_context_enabled
            if collect_llm_candidate_context is None
            else collect_llm_candidate_context
        )
        if should_collect_candidate_context:
            candidate_context_rows, candidate_context_errors = collect_candidate_context(
                df_top,
                max_rows=(
                    candidate_context_max_candidates
                    or config.llm_candidate_context_max_candidates
                ),
                providers=(
                    candidate_context_providers
                    if candidate_context_providers is not None
                    else config.llm_candidate_context_providers
                ),
                news_limit=config.llm_candidate_context_news_limit,
                announcement_limit=config.llm_candidate_context_announcement_limit,
                cache_dir=(
                    config.data_dir / "candidate_context"
                    if config.llm_candidate_context_cache_enabled
                    else None
                ),
                cache_ttl_hours=config.llm_candidate_context_cache_ttl_hours,
                source_weights=event_source_weights,
            )
            degradation.append(
                f"Candidate context collected rows={len(candidate_context_rows)}"
            )
            if candidate_context_errors:
                sample = " | ".join(candidate_context_errors[:5])
                suffix = (
                    f" | +{len(candidate_context_errors) - 5} more"
                    if len(candidate_context_errors) > 5
                    else ""
                )
                degradation.append(f"Candidate context row errors: {sample}{suffix}")
        llm_context_degradation: list[str] = []
        effective_context = build_llm_context(
            base_context=llm_context if llm_context is not None else config.llm_context,
            context_files=llm_context_files,
            candidate_context_files=candidate_context_files,
            candidate_context_rows=candidate_context_rows,
            snapshot_df=snapshot_df,
            candidate_df=df_top,
            event_profile=screening.event_profile,
            max_chars=config.llm_context_max_chars,
            degradation=llm_context_degradation,
        )
        degradation.extend(llm_context_degradation)
        llm_prompt_degradation: list[str] = []
        llm_result = rank_candidates_with_metadata(
            picks,
            screening.ranking_hints,
            config.llm_api_key,
            config.llm_model,
            config.llm_base_url,
            context=effective_context,
            rank_weight=config.llm_rank_weight,
            max_retries=config.llm_max_retries,
            min_coverage=config.llm_min_coverage,
            fallback_models=config.llm_fallback_models,
            temperature=config.llm_temperature,
            json_mode=config.llm_json_mode,
            silent=config.llm_silent,
            channels=config.llm_channels,
            config_path=str(config.llm_config_path or ""),
            timeout_sec=config.llm_timeout_sec,
            max_tokens=config.llm_max_tokens,
            degradation=llm_prompt_degradation,
        )
        degradation.extend(llm_prompt_degradation)
        picks = llm_result.picks
        llm_market_view = llm_result.market_view
        llm_selection_logic = llm_result.selection_logic
        llm_portfolio_risk = llm_result.portfolio_risk
        llm_coverage = llm_result.coverage
        llm_parse_errors = llm_result.errors
        llm_model_used = getattr(llm_result, "model_used", "")
        llm_attempted_models = list(getattr(llm_result, "attempted_models", None) or [])
        llm_failure_reason = getattr(llm_result, "failure_reason", "")
        llm_ranked = llm_result.ranked
        if not llm_ranked:
            picks = llm_fallback_picks
            degradation.append("LLM ranking failed: fell back to screen_score")
            for i, p in enumerate(picks):
                p.rank = i + 1
                p.final_score = p.screen_score
    else:
        if use_llm and not config.has_llm_config():
            degradation.append("LLM ranking skipped: no LLM config")
        for i, p in enumerate(picks):
            p.rank = i + 1
            p.final_score = p.screen_score

    # 8. Independent risk overlay
    if config.risk_enabled:
        picks, risk_degradation = apply_risk_overlay(
            picks,
            max_penalty=config.risk_max_penalty,
            veto_high_risk=config.risk_veto_high,
            profile=screening.risk_profile,
        )
        degradation.extend(risk_degradation)

    # 9. LLM-driven portfolio overlay. This runs before trimming so an
    # over-crowded sector can make room for a comparable candidate elsewhere.
    portfolio_concentration_notes: list[str] = []
    if config.portfolio_diversity_enabled:
        picks, portfolio_concentration_notes = apply_portfolio_overlay(
            picks,
            max_same_sector=config.portfolio_max_same_llm_sector,
            concentration_penalty=config.portfolio_concentration_penalty,
            profile=screening.portfolio_profile,
        )

    # 10. Selection variant application moved below so rotation occurs after
    # final post-analysis scoring. See PR feedback: preserve final-ranking invariants.

    # 11. Optional L3 post-analysis, DSA is only one possible analyzer.
    if analyzer_names:
        _emit_progress(progress_callback, 82, "正在执行最终评分与风险校验")
        # The local scorecard covers the complete shortlist. Remote analyzers
        # retain their configured operational cap; the variant stage below
        # admits only candidates that completed every configured analyzer.
        post_max_picks = (
            analyzer_max_picks
            if analyzer_max_picks is not None
            else config.post_analysis_max_picks
        )
        # Do not silently raise the remote cap to output_count. If it is lower
        # than the requested page size, record why rotation may be constrained.
        try:
            post_max_picks = int(post_max_picks)
        except Exception:
            post_max_picks = int(config.post_analysis_max_picks or 3)
        capped_analyzers = [
            name for name in analyzer_names if name in {"dsa", "external_http"}
        ]
        if capped_analyzers and int(post_max_picks) < int(output_count):
            degradation.append(
                f"Remote post-analysis cap {post_max_picks} < requested output {output_count}; "
                "rotation eligibility constrained to remotely analyzed picks"
            )
        picks, post_degradation = run_post_analyzers(
            picks,
            analyzer_names=analyzer_names,
            run_id=run_id,
            config=config,
            max_picks=post_max_picks,
            scorecard_profile=screening.scorecard_profile,
        )
        degradation.extend(post_degradation)

    # Apply selection variant after post-analyzers to ensure rotation respects
    # final_score adjustments made by L3 analyzers (e.g. scorecard/dsa).
    ranked_pool = list(picks)
    selection_variant = apply_seeded_selection_variant(
        picks,
        max_output=output_count,
        seed=selection_seed,
        period=(
            f"{datetime.now(timezone.utc).date().isoformat()}"
            f":{market}:{strategy}:{run_id}"
        ),
        analyzer_names=analyzer_names,
    )
    selected_codes = {pick.code for pick in selection_variant.picks}
    preferred_pool = [
        *selection_variant.picks,
        *(pick for pick in ranked_pool if pick.code not in selected_codes),
    ]
    sector_quota_replacements = 0
    sector_quota_max_per_bucket = 0
    if config.portfolio_diversity_enabled:
        sector_quota_max_per_bucket = max(
            int(screening.portfolio_profile.get("max_same_bucket", 2)),
            1,
        )
        picks, quota_notes, sector_quota_replacements = apply_sector_quota(
            preferred_pool,
            max_output=output_count,
            max_same_sector=sector_quota_max_per_bucket,
            profile=screening.portfolio_profile,
        )
        portfolio_concentration_notes.extend(quota_notes)
    else:
        picks = selection_variant.picks
    for pick in picks:
        pick.trade_plan = _build_market_aware_trade_plan(
            pick,
            market_regime=market_regime,
            enabled=config.market_regime_enabled,
            execution_profile=screening.execution_profile,
        )
    _emit_progress(progress_callback, 88, "选股核心流程完成")

    return ScreenResult(
        strategy=strategy,
        market=market,
        strategy_version=strat.version,
        strategy_category=strat.category,
        snapshot_count=snapshot_count,
        snapshot_filtered_count=snapshot_filtered_count,
        seed_pool_count=seed_pool_count,
        seed_pool_limit=daily_limit if (daily_needed or daily_requested) else 0,
        seed_pool_diagnostics=seed_pool_diagnostics,
        after_filter_count=after_filter_count,
        picks=picks,
        run_id=run_id,
        llm_ranked=llm_ranked,
        llm_market_view=llm_market_view,
        llm_selection_logic=llm_selection_logic,
        llm_portfolio_risk=llm_portfolio_risk,
        llm_coverage=llm_coverage,
        llm_parse_errors=llm_parse_errors,
        llm_model_used=llm_model_used,
        llm_attempted_models=llm_attempted_models,
        llm_failure_reason=llm_failure_reason,
        ranking_mode="llm" if llm_ranked else "factor",
        degradation=degradation,
        snapshot_source=snapshot_source,
        source_errors=source_errors,
        deep_analysis_requested=("dsa" in analyzer_names),
        post_analyzers=analyzer_names,
        daily_enriched=daily_enriched,
        daily_enrich_count=daily_enrich_count,
        risk_enabled=config.risk_enabled,
        portfolio_diversity_enabled=config.portfolio_diversity_enabled,
        portfolio_concentration_notes=portfolio_concentration_notes,
        sector_quota_applied=sector_quota_replacements > 0,
        sector_quota_replacements=sector_quota_replacements,
        sector_quota_max_per_bucket=sector_quota_max_per_bucket,
        result_variant_applied=selection_variant.applied,
        result_variant_pool_size=selection_variant.pool_size,
        result_variant_rotated_slots=selection_variant.rotated_slots,
        market_regime=market_regime,
        sector_overview=sector_overview,
        analysis_date=(
            effective_bar_date.isoformat() if effective_bar_date is not None else ""
        ),
    )


def _emit_progress(
    callback: Callable[[int, str], None] | None,
    progress: int,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        callback(progress, message)
    except Exception as exc:  # noqa: BLE001 - progress reporting must not fail screening.
        logger.debug("Screening progress callback failed: %s", exc)


def _df_to_picks(df: pd.DataFrame) -> list[Pick]:
    """Convert DataFrame rows to Pick objects."""
    picks = []
    factor_cols = factor_score_columns()
    for i, (_, row) in enumerate(df.iterrows()):
        factor_scores = {
            factor: _safe_float(row.get(col)) or 0.0
            for factor, col in factor_cols.items()
            if col in df.columns
        }
        picks.append(Pick(
            rank=i + 1,
            code=normalize_code(row.get("code", row.get("代码", "")), allow_ticker=True),
            name=str(row.get("name", row.get("名称", row.get("股票名称", "")))),
            screen_score=float(row.get("screen_score", 0)),
            final_score=float(row.get("screen_score", 0)),
            price=float(row.get("price", row.get("最新价", 0)) or 0),
            change_pct=float(row.get("change_pct", row.get("涨跌幅", 0)) or 0),
            amount=float(row.get("amount", row.get("成交额", 0)) or 0),
            total_mv=_safe_float(row.get("total_mv", row.get("总市值"))),
            turnover_rate=_safe_float(row.get("turnover_rate", row.get("换手率"))),
            volume_ratio=_safe_float(row.get("volume_ratio", row.get("量比"))),
            pe_ratio=_safe_float(row.get("pe_ratio", row.get("市盈率"))),
            pb_ratio=_safe_float(row.get("pb_ratio", row.get("市净率"))),
            seed_pool_score=_safe_float(row.get("seed_pool_score")),
            seed_pool_reason=_safe_text(row.get("seed_pool_reason")),
            industry=_safe_text(row.get("industry", row.get("行业", row.get("所属行业", "")))),
            concepts=_safe_text(row.get("concepts", row.get("概念", row.get("概念题材", "")))),
            industry_rank=_safe_int(row.get("industry_rank")),
            industry_change_pct=_safe_float(row.get("industry_change_pct")),
            industry_heat_score=_safe_float(row.get("industry_heat_score")),
            concept_heat_score=_safe_float(row.get("concept_heat_score")),
            board_heat_score=_safe_float(row.get("board_heat_score")),
            board_heat_latest_score=_safe_float(row.get("board_heat_latest_score")),
            board_heat_trend_score=_safe_float(row.get("board_heat_trend_score")),
            board_heat_persistence_score=_safe_float(row.get("board_heat_persistence_score")),
            board_heat_cooling_score=_safe_float(row.get("board_heat_cooling_score")),
            board_heat_observations=_safe_int(row.get("board_heat_observations")),
            board_heat_state=_safe_text(row.get("board_heat_state")),
            board_heat_summary=_safe_text(row.get("board_heat_summary")),
            change_60d=_safe_float(row.get("change_60d")),
            change_20d=_safe_float(row.get("change_20d")),
            relative_strength_20d_pctile=_safe_float(row.get("relative_strength_20d_pctile")),
            relative_strength_60d_pctile=_safe_float(row.get("relative_strength_60d_pctile")),
            adr_20_pct=_safe_float(row.get("adr_20_pct")),
            ema9=_safe_float(row.get("ema9")),
            ema21=_safe_float(row.get("ema21")),
            ema50=_safe_float(row.get("ema50")),
            ema_stack_bullish=_safe_bool(row.get("ema_stack_bullish")),
            ema21_rising=_safe_bool(row.get("ema21_rising")),
            ema50_rising=_safe_bool(row.get("ema50_rising")),
            distance_to_120d_high_pct=_safe_float(row.get("distance_to_120d_high_pct")),
            pullback_depth_10d_pct=_safe_float(row.get("pullback_depth_10d_pct")),
            range_contraction_ratio=_safe_float(row.get("range_contraction_ratio")),
            volume_contraction_ratio=_safe_float(row.get("volume_contraction_ratio")),
            anchored_vwap_60d_high=_safe_float(row.get("anchored_vwap_60d_high")),
            support_confluence_count=_safe_int(row.get("support_confluence_count")),
            trigger_distance_pct=_safe_float(row.get("trigger_distance_pct")),
            one_word_limit_up=_safe_bool(row.get("one_word_limit_up")),
            avg_amount_20d=_safe_float(row.get("avg_amount_20d")),
            sector_breadth_count=_safe_int(row.get("sector_breadth_count")),
            sector_strength_score=_safe_float(row.get("sector_strength_score")),
            sector_strength_rank=_safe_int(row.get("sector_strength_rank")),
            sector_member_count=_safe_int(row.get("sector_member_count")),
            sector_breadth_ratio=_safe_float(row.get("sector_breadth_ratio")),
            sector_leader_count=_safe_int(row.get("sector_leader_count")),
            sector_high_adr_leader_count=_safe_int(row.get("sector_high_adr_leader_count")),
            sector_trend_state=_safe_text(row.get("sector_trend_state")),
            sector_leadership_priority=_safe_bool(row.get("sector_leadership_priority")),
            sector_data_status=_safe_text(row.get("sector_data_status")),
            sector_data_source=_safe_text(row.get("sector_data_source")),
            sector_data_as_of=_safe_text(row.get("sector_data_as_of")),
            setup_type=_safe_text(row.get("setup_type")),
            setup_trigger_price=_safe_float(row.get("setup_trigger_price")),
            setup_confirmed=_safe_bool(row.get("setup_confirmed")),
            setup_trade_eligible=_safe_bool(row.get("setup_trade_eligible")),
            setup_execution_status=_safe_text(row.get("setup_execution_status")),
            signal_score=_safe_float(row.get("signal_score")),
            ma_bullish=_safe_bool(row.get("ma_bullish")),
            price_above_ma20=_safe_bool(row.get("price_above_ma20")),
            macd_status=str(row.get("macd_status", "") or ""),
            rsi_status=str(row.get("rsi_status", "") or ""),
            breakout_20d_pct=_safe_float(row.get("breakout_20d_pct")),
            range_20d_pct=_safe_float(row.get("range_20d_pct")),
            volume_ratio_20d=_safe_float(row.get("volume_ratio_20d")),
            body_pct=_safe_float(row.get("body_pct")),
            pullback_to_ma20_pct=_safe_float(row.get("pullback_to_ma20_pct")),
            consolidation_days_20d=_safe_int(row.get("consolidation_days_20d")),
            volatility_20d_pct=_safe_float(row.get("volatility_20d_pct")),
            max_drawdown_20d_pct=_safe_float(row.get("max_drawdown_20d_pct")),
            atr_20_pct=_safe_float(row.get("atr_20_pct")),
            daily_quality_score=_safe_float(row.get("daily_quality_score")),
            daily_quality_flags=_safe_text(row.get("daily_quality_flags")),
            daily_source=_safe_text(row.get("daily_source")),
            factor_scores=factor_scores,
        ))
    return picks


def _build_market_aware_trade_plan(
    pick: Pick,
    *,
    market_regime: dict[str, object],
    enabled: bool,
    execution_profile: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one trade plan with market-state limits applied to new entries."""
    profile = execution_profile or {}
    base = MartinTradePlanConfig(
        pyramiding_enabled=bool(profile.get("pyramiding_enabled", False)),
        adaptive_exit_enabled=bool(profile.get("adaptive_exit_enabled", False)),
        initial_position_weight_pct=float(
            profile.get("initial_position_weight_pct", 20.0)
        ),
        add_on_position_weight_pct=float(
            profile.get("add_on_position_weight_pct", 5.0)
        ),
        max_add_ons=int(profile.get("max_add_ons", 0)),
        first_add_on_r_multiple=float(profile.get("first_add_on_r_multiple", 1.0)),
        second_add_on_r_multiple=float(profile.get("second_add_on_r_multiple", 2.0)),
        first_partial_r_multiple=float(
            profile.get("first_partial_r_multiple", 3.0)
        ),
        second_partial_r_multiple=float(
            profile.get("second_partial_r_multiple", 5.0)
        ),
        partial_exit_fraction=float(profile.get("partial_exit_fraction", 0.20)),
        adaptive_trailing_drawdown_pct=float(
            profile.get("adaptive_trailing_drawdown_pct", 6.0)
        ),
        mature_trailing_drawdown_pct=float(
            profile.get("mature_trailing_drawdown_pct", 5.0)
        ),
        adaptive_exit_model=str(
            profile.get("adaptive_exit_model", "legacy_tightening")
        ),
        developing_exit_activation_r=float(
            profile.get("developing_exit_activation_r", 1.0)
        ),
        developing_trailing_adr_multiple=float(
            profile.get("developing_trailing_adr_multiple", 1.0)
        ),
        developing_trailing_min_pct=float(
            profile.get("developing_trailing_min_pct", 5.0)
        ),
        developing_trailing_max_pct=float(
            profile.get("developing_trailing_max_pct", 8.0)
        ),
        established_trailing_adr_multiple=float(
            profile.get("established_trailing_adr_multiple", 1.5)
        ),
        established_trailing_min_pct=float(
            profile.get("established_trailing_min_pct", 7.0)
        ),
        established_trailing_max_pct=float(
            profile.get("established_trailing_max_pct", 12.0)
        ),
        mature_trailing_adr_multiple=float(
            profile.get("mature_trailing_adr_multiple", 1.75)
        ),
        mature_trailing_min_pct=float(
            profile.get("mature_trailing_min_pct", 9.0)
        ),
        mature_trailing_max_pct=float(
            profile.get("mature_trailing_max_pct", 15.0)
        ),
        established_profit_floor_r=float(
            profile.get("established_profit_floor_r", 1.0)
        ),
        mature_profit_floor_r=float(profile.get("mature_profit_floor_r", 2.5)),
        first_partial_exit_fraction=float(
            profile.get("first_partial_exit_fraction", 0.0)
        ),
        second_partial_exit_fraction=float(
            profile.get("second_partial_exit_fraction", 0.15)
        ),
        partial_exit_requires_exhaustion=bool(
            profile.get("partial_exit_requires_exhaustion", True)
        ),
        ema9_confirmation_days=int(profile.get("ema9_confirmation_days", 2)),
        risk_off_ema9_confirmation_days=int(
            profile.get("risk_off_ema9_confirmation_days", 1)
        ),
        min_position_weight_pct=(
            8.0 if bool(profile.get("pyramiding_enabled", False)) else 12.0
        ),
    )
    limits = resolve_market_regime_limits(
        market_regime if enabled else unavailable_market_regime("市场状态调节已关闭"),
        base_max_positions=5,
        base_risk_per_trade_pct=base.max_portfolio_risk_per_trade_pct,
        base_max_portfolio_heat_pct=base.max_portfolio_heat_pct,
        base_max_position_weight_pct=base.max_position_weight_pct,
        base_min_position_weight_pct=base.min_position_weight_pct,
    )
    plan_rules = base
    if limits["new_entries_allowed"]:
        plan_rules = MartinTradePlanConfig(
            max_portfolio_risk_per_trade_pct=float(limits["risk_per_trade_pct"]),
            max_portfolio_heat_pct=float(limits["max_portfolio_heat_pct"]),
            max_position_weight_pct=float(limits["max_position_weight_pct"]),
            min_position_weight_pct=float(limits["min_position_weight_pct"]),
            pyramiding_enabled=base.pyramiding_enabled,
            adaptive_exit_enabled=base.adaptive_exit_enabled,
            initial_position_weight_pct=base.initial_position_weight_pct,
            add_on_position_weight_pct=base.add_on_position_weight_pct,
            max_add_ons=base.max_add_ons,
            first_add_on_r_multiple=base.first_add_on_r_multiple,
            second_add_on_r_multiple=base.second_add_on_r_multiple,
            first_partial_r_multiple=base.first_partial_r_multiple,
            second_partial_r_multiple=base.second_partial_r_multiple,
            partial_exit_fraction=base.partial_exit_fraction,
            adaptive_trailing_drawdown_pct=base.adaptive_trailing_drawdown_pct,
            mature_trailing_drawdown_pct=base.mature_trailing_drawdown_pct,
            adaptive_exit_model=base.adaptive_exit_model,
            developing_exit_activation_r=base.developing_exit_activation_r,
            developing_trailing_adr_multiple=base.developing_trailing_adr_multiple,
            developing_trailing_min_pct=base.developing_trailing_min_pct,
            developing_trailing_max_pct=base.developing_trailing_max_pct,
            established_trailing_adr_multiple=base.established_trailing_adr_multiple,
            established_trailing_min_pct=base.established_trailing_min_pct,
            established_trailing_max_pct=base.established_trailing_max_pct,
            mature_trailing_adr_multiple=base.mature_trailing_adr_multiple,
            mature_trailing_min_pct=base.mature_trailing_min_pct,
            mature_trailing_max_pct=base.mature_trailing_max_pct,
            established_profit_floor_r=base.established_profit_floor_r,
            mature_profit_floor_r=base.mature_profit_floor_r,
            first_partial_exit_fraction=base.first_partial_exit_fraction,
            second_partial_exit_fraction=base.second_partial_exit_fraction,
            partial_exit_requires_exhaustion=(
                base.partial_exit_requires_exhaustion
            ),
            ema9_confirmation_days=base.ema9_confirmation_days,
            risk_off_ema9_confirmation_days=(
                base.risk_off_ema9_confirmation_days
            ),
        )
    plan = build_martin_trade_plan(
        price=pick.price,
        trigger_distance_pct=pick.trigger_distance_pct,
        adr_20_pct=pick.adr_20_pct,
        ema21=pick.ema21,
        atr_20_pct=pick.atr_20_pct,
        anchored_vwap=pick.anchored_vwap_60d_high,
        config=plan_rules,
    )
    plan["market_regime"] = {
        "enabled": enabled,
        "date": market_regime.get("date"),
        "state": market_regime.get("state"),
        "score": market_regime.get("score"),
        "reason": market_regime.get("reason"),
    }
    plan["portfolio_limits"] = {
        key: value for key, value in limits.items() if key != "profile"
    }
    plan["setup_permission"] = {
        "setup_type": pick.setup_type,
        "confirmed": pick.setup_confirmed,
        "trade_eligible": pick.setup_trade_eligible,
        "status": pick.setup_execution_status,
    }
    if pick.setup_confirmed is True and pick.setup_trade_eligible is False:
        plan["signal_status"] = "watch_only_setup"
        plan["action"] = "no_order"
        plan["setup_watch_reason"] = (
            f"{pick.setup_type or '当前形态'} 已完成识别，但当前策略版本仅允许"
            "首次回踩形态进入交易；本候选保留观察，不设置买入条件单。"
        )
        condition_orders = plan.get("condition_orders")
        if isinstance(condition_orders, dict):
            entry_order = condition_orders.get("entry")
            if isinstance(entry_order, dict):
                entry_order["status"] = "not_armed"
    if (
        enabled
        and str(market_regime.get("state") or "") == REGIME_RISK_OFF
        and plan.get("signal_status") == "buy_signal"
    ):
        plan["signal_status"] = "market_blocked"
        plan["action"] = "no_order"
        plan["market_block_reason"] = (
            str(market_regime.get("reason") or "")
            or "多日市场状态处于风险关闭，暂停新开仓。"
        )
        condition_orders = plan.get("condition_orders")
        if isinstance(condition_orders, dict):
            entry_order = condition_orders.get("entry")
            if isinstance(entry_order, dict):
                entry_order["status"] = "not_armed"
    return plan


def _sort_screened_candidates(df: pd.DataFrame, screening=None) -> pd.DataFrame:
    """Sort scored candidates deterministically with factor-aware tie breakers."""
    factor_order = [
        "trend_quality",
        "relative_strength",
        "sector_resonance",
        "contraction",
        "support_confluence",
        "trigger_quality",
        "liquidity",
    ]
    if screening is not None and screening.factor_weights:
        factor_order = [
            factor
            for factor, _weight in sorted(
                screening.factor_weights.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )
        ]
    sort_columns = [
        column
        for column in [
            "setup_trade_eligible",
            "sector_leadership_priority",
            "sector_strength_score",
            "screen_score",
        ] + [f"factor_{factor}_score" for factor in factor_order]
        if column in df.columns
    ]
    ascending = [False] * len(sort_columns)
    if "code" in df.columns:
        sort_columns.append("code")
        ascending.append(True)
    if not sort_columns:
        return df
    return df.sort_values(sort_columns, ascending=ascending, kind="mergesort")


def _attach_martin_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rank momentum within the enriched pool and measure sector participation."""
    result = df.copy()
    for periods in (20, 60):
        source = f"change_{periods}d"
        target = f"relative_strength_{periods}d_pctile"
        raw_values = result[source] if source in result.columns else pd.Series(pd.NA, index=result.index)
        values = pd.to_numeric(raw_values, errors="coerce")
        if values.notna().any():
            result[target] = values.rank(method="average", pct=True) * 100.0
        else:
            result[target] = pd.NA

    raw_change_20d = (
        result["change_20d"]
        if "change_20d" in result.columns
        else pd.Series(pd.NA, index=result.index)
    )
    change_20d = pd.to_numeric(raw_change_20d, errors="coerce")
    result["market_median_change_20d_pct"] = (
        round(float(change_20d.median()), 4) if change_20d.notna().any() else pd.NA
    )

    industry_col = next(
        (name for name in ("industry", "行业", "所属行业") if name in result.columns),
        None,
    )
    if industry_col is None:
        result["sector_breadth_count"] = 0
        return result

    industries = result[industry_col].fillna("").astype(str).str.strip()
    ema_ok = result.get("ema_stack_bullish", pd.Series(False, index=result.index)) == True  # noqa: E712
    rs60 = result.get(
        "relative_strength_60d_pctile", pd.Series(pd.NA, index=result.index)
    )
    strong = ema_ok & pd.to_numeric(rs60, errors="coerce").ge(70)
    breadth = strong.groupby(industries).transform("sum")
    result["sector_breadth_count"] = breadth.where(industries.ne(""), 0).astype(int)
    return result


def _attach_martin_cross_sectional_features_from_universe(
    enriched: pd.DataFrame,
    *,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Map full-universe relative strength back to the enriched seed pool."""
    if enriched.empty or universe.empty or "code" not in enriched.columns or "code" not in universe.columns:
        return _attach_martin_cross_sectional_features(enriched)

    full = universe.copy()
    full["__cross_code"] = full["code"].map(
        lambda value: normalize_code(value, allow_ticker=True)
    )
    selected = enriched.copy()
    selected["__cross_code"] = selected["code"].map(
        lambda value: normalize_code(value, allow_ticker=True)
    )
    selected = selected.drop_duplicates("__cross_code", keep="last").set_index("__cross_code")
    for column in ("change_20d", "change_60d", "ema_stack_bullish"):
        if column not in selected.columns:
            continue
        mapped = full["__cross_code"].map(selected[column])
        if column in full.columns:
            full[column] = mapped.where(mapped.notna(), full[column])
        else:
            full[column] = mapped

    full = _attach_martin_cross_sectional_features(full)
    full = full.drop_duplicates("__cross_code", keep="last").set_index("__cross_code")
    result = enriched.copy()
    result_codes = result["code"].map(lambda value: normalize_code(value, allow_ticker=True))
    for column in (
        "relative_strength_20d_pctile",
        "relative_strength_60d_pctile",
        "sector_breadth_count",
        "market_median_change_20d_pct",
    ):
        if column in full.columns:
            result[column] = result_codes.map(full[column])
    return result


def _attach_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic one-day breadth gates to every snapshot row."""
    result = df.copy()
    change_col = next(
        (name for name in ("change_pct", "涨跌幅") if name in result.columns),
        None,
    )
    if change_col is None:
        result["market_positive_ratio"] = pd.NA
        result["market_median_change_pct"] = pd.NA
        return result
    changes = pd.to_numeric(result[change_col], errors="coerce").dropna()
    if changes.empty:
        result["market_positive_ratio"] = pd.NA
        result["market_median_change_pct"] = pd.NA
        return result
    result["market_positive_ratio"] = round(float(changes.gt(0).mean() * 100.0), 4)
    result["market_median_change_pct"] = round(float(changes.median()), 4)
    return result


def _required_snapshot_columns(filters) -> list[str]:
    columns: list[str] = []
    if filters.exclude_st:
        columns.append("name")
    if filters.amount_min is not None:
        columns.append("amount")
    if filters.price_min is not None or filters.price_max is not None:
        columns.append("price")
    if filters.market_cap_min is not None or filters.market_cap_max is not None:
        columns.append("total_mv")
    if filters.pe_ttm_min is not None or filters.pe_ttm_max is not None:
        columns.append("pe_ratio")
    if filters.pb_min is not None or filters.pb_max is not None:
        columns.append("pb_ratio")
    if filters.volume_ratio_min is not None:
        columns.append("volume_ratio")
    if filters.turnover_rate_min is not None:
        columns.append("turnover_rate")
    if any([
        filters.change_pct_min is not None,
        filters.change_pct_max is not None,
        filters.market_positive_ratio_min is not None,
        filters.market_median_change_pct_min is not None,
    ]):
        columns.append("change_pct")
    return list(dict.fromkeys(columns))


def _event_source_weights(event_profile: dict[str, object]) -> dict[str, float] | None:
    value = (event_profile or {}).get("source_weights")
    if not isinstance(value, dict):
        return None
    result: dict[str, float] = {}
    for key, raw in value.items():
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result or None


def _daily_source_health_notes(health: dict[str, object], *, limit: int = 4) -> list[str]:
    source_states: list[tuple[tuple[int, float, float, str], str, dict[object, object]]] = []
    for source, raw_state in health.items():
        if not isinstance(raw_state, dict):
            continue
        failures = _safe_float(raw_state.get("failures")) or 0.0
        total_failures = _safe_float(raw_state.get("total_failures")) or 0.0
        disabled = bool(raw_state.get("disabled"))
        if not disabled and failures <= 0 and total_failures <= 0:
            continue
        severity_key = (
            0 if disabled else 1 if failures > 0 else 2,
            -failures,
            -total_failures,
            str(source),
        )
        source_states.append((severity_key, str(source), raw_state))

    notes: list[str] = []
    for _severity_key, source, raw_state in sorted(source_states):
        failures = _safe_float(raw_state.get("failures")) or 0.0
        total_failures = _safe_float(raw_state.get("total_failures")) or 0.0
        disabled = bool(raw_state.get("disabled"))
        last_rows = _safe_float(raw_state.get("last_rows")) or 0.0
        parts: list[str] = []
        if disabled:
            parts.append("disabled")
        if failures > 0:
            parts.append(f"failures={failures:g}")
        elif total_failures > 0:
            parts.append(f"total_failures={total_failures:g}")
        if last_rows > 0:
            parts.append(f"last_rows={last_rows:g}")
        if parts:
            notes.append(f"{source} " + ",".join(parts))
        if len(notes) >= limit:
            break
    hidden_count = len(source_states) - len(notes)
    if hidden_count > 0:
        notes.append(f"+{hidden_count} more")
    return notes


def _format_filter_waterfall(steps: list[dict[str, object]], *, limit: int = 8) -> str:
    parts: list[str] = []
    for step in steps[:limit]:
        text = (
            f"{step.get('filter')} {step.get('before')}->{step.get('after')} "
            f"removed={step.get('removed')}"
        )
        samples = step.get("samples")
        if isinstance(samples, list) and samples:
            sample_names = [
                str(item.get("name") or item.get("code") or item.get("value") or "")
                for item in samples
                if isinstance(item, dict)
            ]
            sample_names = [item for item in sample_names if item]
            if sample_names:
                text += f" samples={','.join(sample_names[:3])}"
        if step.get("suggestion"):
            text += f" next={step.get('suggestion')}"
        parts.append(text)
    hidden = len(steps) - len(parts)
    if hidden > 0:
        parts.append(f"+{hidden} more")
    return "; ".join(parts)


def _safe_text(v: object) -> str:
    return safe_text(v, max_len=120)
