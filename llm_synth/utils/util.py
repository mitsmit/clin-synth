import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _gower_encode(
    df_query: pd.DataFrame,
    df_ref:   pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared Gower encoding of two DataFrames for distance computation.

    Returns (enc_query, enc_ref, col_ranges, is_categorical).
    Categorical columns are integer label-encoded using a shared vocabulary built
    from the union of both frames so distances are comparable across splits.
    Numeric columns are passed as-is; missing values filled with the combined median.
    col_ranges[j] == 1.0 for categoricals; == max-min for numerics (floored to 1.0).
    """
    combined = pd.concat([df_query, df_ref], ignore_index=True)
    q_parts, r_parts, col_ranges, is_cat_flags = [], [], [], []

    for col in cat_cols:
        def _get(frame: pd.DataFrame) -> pd.Series:
            return frame[col].astype(str).fillna("__NaN__") if col in frame.columns \
                   else pd.Series(["__NaN__"] * len(frame))
        shared  = _get(combined)
        _, uniq = pd.factorize(shared)
        mapping = {v: i for i, v in enumerate(uniq)}
        q_parts.append(_get(df_query).map(mapping).fillna(-1).astype(np.float32).values.reshape(-1, 1))
        r_parts.append(_get(df_ref  ).map(mapping).fillna(-1).astype(np.float32).values.reshape(-1, 1))
        col_ranges.append(1.0)
        is_cat_flags.append(True)

    for col in num_cols:
        def _getnum(frame: pd.DataFrame) -> pd.Series:
            return pd.to_numeric(frame[col], errors="coerce") if col in frame.columns \
                   else pd.Series([np.nan] * len(frame))
        combined_num = _getnum(combined)
        median = float(combined_num.median()) if not combined_num.isna().all() else 0.0
        rng    = float(combined_num.max() - combined_num.min())
        rng    = rng if rng > 0 else 1.0
        q_parts.append(_getnum(df_query).fillna(median).astype(np.float32).values.reshape(-1, 1))
        r_parts.append(_getnum(df_ref  ).fillna(median).astype(np.float32).values.reshape(-1, 1))
        col_ranges.append(rng)
        is_cat_flags.append(False)

    enc_q = np.hstack(q_parts) if q_parts else np.zeros((len(df_query), 0), dtype=np.float32)
    enc_r = np.hstack(r_parts) if r_parts else np.zeros((len(df_ref),   0), dtype=np.float32)
    return enc_q, enc_r, np.array(col_ranges, dtype=np.float32), np.array(is_cat_flags, dtype=bool)


def _gower_nn_dist(
    query:      np.ndarray,
    ref:        np.ndarray,
    col_ranges: np.ndarray,
    is_cat:     np.ndarray,
    chunk_size: int = 400,
) -> np.ndarray:
    """Nearest-neighbor Gower distance from each row in query to the ref set.

    Processes query in chunks to bound peak memory.
    Distance for each column j:
      categorical → 0/1 (equal/not)
      numeric     → |a-b| / col_ranges[j]
    Final distance = mean across all columns.
    """
    n_cols    = query.shape[1]
    min_dists = np.full(len(query), np.inf, dtype=np.float32)

    for start in range(0, len(query), chunk_size):
        end     = min(start + chunk_size, len(query))
        q_chunk = query[start:end]
        dists   = np.zeros((end - start, len(ref)), dtype=np.float32)
        for j in range(n_cols):
            if is_cat[j]:
                dists += (q_chunk[:, j:j+1] != ref[:, j].reshape(1, -1)).astype(np.float32)
            else:
                dists += np.abs(q_chunk[:, j:j+1] - ref[:, j].reshape(1, -1)) / col_ranges[j]
        dists /= n_cols
        min_dists[start:end] = dists.min(axis=1)

    return min_dists


def _is_hash_column(series: pd.Series) -> bool:
    """Return True if a column appears to contain hash IDs rather than real values."""
    sample = series.dropna().head(20).astype(str)
    return bool(sample.str.match(r"^[0-9a-f]{32}$").mean() > 0.8)


# ── PSI constants ─────────────────────────────────────────────────────────────
_PSI_EPS   = 1e-4
_Q_KEYS    = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
_Q_PROBS   = [0.01, 0.05, 0.10,  0.25,  0.50,  0.75,  0.90,  0.95,  0.99]
_PSI_STABLE  = 0.10
_PSI_MONITOR = 0.20


def _psi_status(psi: float) -> str:
    """Map a PSI value to a human-readable stability label: stable / monitor / unstable."""
    if psi < _PSI_STABLE:
        return "stable"
    if psi < _PSI_MONITOR:
        return "monitor"
    return "unstable"


def _psi_categorical(seed_freq: dict, synth_series: pd.Series) -> tuple[float, int]:
    """PSI for a categorical column — each category is one bin.

    seed_freq is the frequency_table dict from seed_stats:
        { "value": {"count": int, "proportion": float}, ... }

    The seed_stats frequency_table stores only the top-N categories.  Any
    categories beyond that are rolled into an implicit "other" bucket so
    the expected proportions still sum to 1.
    """
    synth_props = synth_series.astype(str).value_counts(normalize=True)

    known_seed_total = sum(v["proportion"] for v in seed_freq.values())
    other_expected   = max(1.0 - known_seed_total, 0.0)

    novel_cats   = set(synth_props.index) - set(seed_freq.keys())
    other_actual = sum(float(synth_props.get(c, 0.0)) for c in novel_cats)

    psi    = 0.0
    n_bins = 0

    for cat, stats in seed_freq.items():
        expected = max(stats["proportion"], _PSI_EPS)
        actual   = max(float(synth_props.get(cat, 0.0)), _PSI_EPS)
        psi     += (actual - expected) * np.log(actual / expected)
        n_bins  += 1

    if other_expected > 0.0 or other_actual > 0.0:
        expected = max(other_expected, _PSI_EPS)
        actual   = max(other_actual,   _PSI_EPS)
        psi     += (actual - expected) * np.log(actual / expected)
        n_bins  += 1

    return round(float(psi), 6), n_bins


def _psi_numeric(seed_col_stats: dict, synth_series: pd.Series) -> tuple[float, int]:
    """PSI for a numeric column using seed quantile boundaries as bins.

    Bins are defined by the seed quantile values (p1…p99).  Each bin's
    expected proportion equals the probability mass between adjacent
    quantile levels, so the expected proportions are fully determined by
    the seed statistics without needing the raw seed data.

    Bin layout (side='left' → upper-inclusive, lower-exclusive after bin 0):
        bin 0 : (-inf,  p1]  → 1 %
        bin 1 : ( p1,   p5]  → 4 %
        …
        bin 9 : ( p99, +inf) → 1 %
    """
    quantiles = seed_col_stats.get("quantiles", {})

    seen: dict[float, float] = {}
    for key, prob in zip(_Q_KEYS, _Q_PROBS):
        val = quantiles.get(key)
        if val is not None:
            v = float(val)
            if v not in seen:
                seen[v] = prob

    if len(seen) < 2:
        return 0.0, 0

    sorted_pts  = sorted(seen.items())
    inner_edges = np.array([v for v, _ in sorted_pts])
    cum_probs   = np.array([0.0] + [p for _, p in sorted_pts] + [1.0])
    expected    = np.diff(cum_probs)

    clean = pd.to_numeric(synth_series, errors="coerce").dropna().values
    if len(clean) == 0:
        return 0.0, 0

    bin_idx = np.searchsorted(inner_edges, clean, side="left")
    n_bins  = len(inner_edges) + 1
    counts  = np.bincount(bin_idx, minlength=n_bins)
    actual  = counts / len(clean)

    psi = float(np.sum(
        (actual - expected) * np.log(
            np.maximum(actual,   _PSI_EPS) /
            np.maximum(expected, _PSI_EPS)
        )
    ))
    return round(psi, 6), n_bins


def compute_psi(synth_df: pd.DataFrame, seed_stats: dict) -> dict:
    """Compute Population Stability Index for every column present in seed_stats.

    PSI quantifies how much the distribution of the synthetic data has
    shifted relative to the seed (reference) data.

    Thresholds
    ----------
    PSI < 0.10  : stable   — no significant distribution shift
    PSI < 0.20  : monitor  — minor shift, worth watching
    PSI >= 0.20 : unstable — major shift; generation quality at risk

    Parameters
    ----------
    synth_df   : Synthetic DataFrame produced by generate.py.
    seed_stats : Statistics dict produced by statistics.extract_seed_stats().

    Returns
    -------
    dict
        columns : {col: {psi, status, type, n_bins}}
        summary : {n_checked, n_stable, n_monitor, n_unstable,
                   mean_psi, worst_column, worst_psi}
    """
    results: dict[str, dict] = {}

    for col in seed_stats["meta"].get("categorical_columns", []):
        if col not in synth_df.columns:
            continue
        seed_freq = seed_stats["columns"].get(col, {}).get("frequency_table", {})
        if not seed_freq:
            continue
        psi, n_bins = _psi_categorical(seed_freq, synth_df[col])
        results[col] = {
            "psi": psi, "status": _psi_status(psi),
            "type": "categorical", "n_bins": n_bins,
        }

    for col in seed_stats["meta"].get("numeric_columns", []):
        if col not in synth_df.columns:
            continue
        seed_col = seed_stats["columns"].get(col, {})
        if not seed_col.get("quantiles"):
            continue
        psi, n_bins = _psi_numeric(seed_col, synth_df[col])
        results[col] = {
            "psi": psi, "status": _psi_status(psi),
            "type": "numeric", "n_bins": n_bins,
        }

    psi_vals = [v["psi"] for v in results.values()]
    statuses = [v["status"] for v in results.values()]
    worst    = max(results, key=lambda c: results[c]["psi"]) if results else None

    summary = {
        "n_checked":    len(results),
        "n_stable":     statuses.count("stable"),
        "n_monitor":    statuses.count("monitor"),
        "n_unstable":   statuses.count("unstable"),
        "mean_psi":     round(float(np.mean(psi_vals)), 6) if psi_vals else 0.0,
        "worst_column": worst,
        "worst_psi":    results[worst]["psi"] if worst else 0.0,
    }

    return {"columns": results, "summary": summary}


# ── Report rendering ──────────────────────────────────────────────────────────

_SEV_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
_SEV_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

_DEFAULT_THRESHOLDS = {
    "mia_auc":        0.60,
    "linkage_risk":   0.05,
    "dcr_ratio":      0.80,
    "nndr":           0.20,
}


def _print_report(report: dict, thresholds: dict | None = None) -> None:
    """Print a formatted table-based summary of the validation report to stdout."""
    th  = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    s   = report["summary"]
    llm = report.get("llm_analysis", {})

    W = 100
    logger.info("\n" + "═" * W)
    logger.info("  SYNTHETIC DATA VALIDATION REPORT")
    logger.info("═" * W)

    # ── Header: verdict + score + summary ────────────────────────────────────
    verdict      = s.get("llm_verdict", "N/A")
    score        = s.get("llm_score",   "N/A")
    verdict_icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(verdict, "ℹ️ ")
    logger.info(f"\n  Verdict : {verdict_icon} {verdict}    Score : {score}/100")

    if llm.get("executive_summary"):
        summary = llm["executive_summary"]
        words, line = summary.split(), ""
        lines_out = ["\n  Summary : "]
        for w in words:
            if len(line) + len(w) + 1 > W - 12:
                lines_out.append(line)
                lines_out.append("            ")
                line = w + " "
            else:
                line += w + " "
        lines_out.append(line.strip())
        logger.info("".join(lines_out))

    # ── Table 1: Issue severity counts ───────────────────────────────────────
    logger.info(f"\n  ── Severity Summary ({s['total']} total issues) ──\n")
    sev_rows = [
        [_SEV_ICON.get(sev, "•") + " " + sev, str(s.get(sev, 0))]
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    ]
    logger.info("  " + _table(["Severity", "Count"], sev_rows, [20, 8]).replace("\n", "\n  "))

    # ── Table 2: Programmatic issues ─────────────────────────────────────────
    if report["programmatic_issues"]:
        logger.info("\n  ── Programmatic Issues ──\n")
        issue_rows = [
            [
                _SEV_ICON.get(i["severity"], "•") + " " + i["severity"],
                i["check"],
                i["detail"],
            ]
            for i in sorted(report["programmatic_issues"],
                            key=lambda x: _SEV_RANK.get(x["severity"], 99))
        ]
        logger.info("  " + _table(
            ["Severity", "Check", "Detail"],
            issue_rows,
            [12, 38, 44],
        ).replace("\n", "\n  "))

    # ── Table 3: PSI per column ───────────────────────────────────────────────
    psi = report.get("psi", {})
    if psi.get("columns"):
        ps = psi["summary"]
        status_icon = {"stable": "🟢", "monitor": "🟡", "unstable": "🔴"}
        logger.info(
            f"\n  ── Population Stability Index "
            f"(mean={ps['mean_psi']:.4f}  "
            f"stable={ps['n_stable']}  monitor={ps['n_monitor']}  unstable={ps['n_unstable']}) ──\n"
        )
        psi_rows = [
            [
                status_icon.get(info["status"], "•") + " " + info["status"].upper(),
                col,
                info["type"],
                f"{info['psi']:.4f}",
                str(info["n_bins"]),
            ]
            for col, info in sorted(psi["columns"].items(),
                                    key=lambda kv: kv[1]["psi"], reverse=True)
        ]
        logger.info("  " + _table(
            ["Status", "Column", "Type", "PSI", "Bins"],
            psi_rows,
            [12, 30, 12, 8, 5],
        ).replace("\n", "\n  "))

    # ── Table 4: LLM additional issues ───────────────────────────────────────
    if llm.get("additional_issues"):
        logger.info(f"\n  ── LLM-Detected Issues ({len(llm['additional_issues'])}) ──\n")
        llm_rows = [
            [
                _SEV_ICON.get(i["severity"], "•") + " " + i["severity"],
                i.get("category", ""),
                i.get("issue", ""),
                i.get("detail", ""),
            ]
            for i in llm["additional_issues"]
        ]
        logger.info("  " + _table(
            ["Severity", "Category", "Issue", "Detail"],
            llm_rows,
            [12, 14, 24, 44],
        ).replace("\n", "\n  "))

    # ── Table 5: Improvement strategies ──────────────────────────────────────
    if llm.get("improvement_strategies"):
        effort_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        logger.info("\n  ── Improvement Strategies ──\n")
        strat_rows = [
            [
                f"#{s['priority']}",
                effort_icon.get(s.get("effort", ""), "•") + " " + s.get("effort", "").upper(),
                s.get("strategy", ""),
                s.get("implementation", ""),
            ]
            for s in sorted(llm["improvement_strategies"], key=lambda x: x["priority"])
        ]
        logger.info("  " + _table(
            ["#", "Effort", "Strategy", "How to fix"],
            strat_rows,
            [3, 10, 28, 52],
        ).replace("\n", "\n  "))

    # ── Table 6: Privacy risk assessment ─────────────────────────────────────
    if llm.get("privacy_risk_assessment"):
        pra = llm["privacy_risk_assessment"]
        risk_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        icon = risk_icon.get(pra.get("overall_risk", "").lower(), "•")
        logger.info("\n  ── Privacy Risk Assessment ──\n")
        priv_rows = [
            ["Overall risk",      f"{icon} {pra.get('overall_risk','').upper()}"],
            ["Re-identification", pra.get("re_identification_risk", "")],
            ["Data leakage",      pra.get("data_leakage_risk", "")],
        ]
        for rec in pra.get("recommendations", []):
            priv_rows.append(["Recommendation", rec])
        logger.info("  " + _table(
            ["Dimension", "Assessment"],
            priv_rows,
            [20, 72],
        ).replace("\n", "\n  "))

    # ── Table 7: Privacy test metrics ────────────────────────────────────────
    pm = report.get("privacy_metrics", {})
    if pm:
        logger.info("\n  ── Privacy Test Metrics ──\n")
        priv_metric_rows = []

        mia = pm.get("membership_inference", {})
        if mia and "auc" in mia:
            auc_icon = "🟢" if mia["auc"] <= th["mia_auc"] else ("🔴" if mia["auc"] > 0.75 else "🟠")
            priv_metric_rows += [
                ["MIA", "AUC",               f"{auc_icon} {mia['auc']}"],
                ["MIA", "Train mean dist",   str(mia.get("train_mean_dist",   ""))],
                ["MIA", "Holdout mean dist", str(mia.get("holdout_mean_dist", ""))],
                ["MIA", "Sample size",       f"train={mia.get('n_train_sample','')}  holdout={mia.get('n_holdout_sample','')}"],
            ]

        rl = pm.get("record_linkage", {})
        if rl and "unique_linkage_rate" in rl:
            ul      = rl["unique_linkage_rate"]
            ul_icon = "🟢" if ul <= th["linkage_risk"] else ("🔴" if ul > 0.10 else "🟠")
            priv_metric_rows += [
                ["Record linkage", "QI cols",             ", ".join(rl.get("qi_cols", []))],
                ["Record linkage", "Unique linkage rate", f"{ul_icon} {ul*100:.1f}%"],
                ["Record linkage", "No-match rate",       f"{rl.get('no_match_rate', 0)*100:.1f}% (novel QI combos)"],
                ["Record linkage", "Mean seed group",     str(rl.get("mean_seed_group_size", ""))],
            ]

        nn = pm.get("nn_similarity", {})
        if nn and "dcr_ratio" in nn:
            dcr_icon  = "🟢" if nn["dcr_ratio"] >= th["dcr_ratio"]  else ("🔴" if nn["dcr_ratio"] < 0.5 else "🟠")
            nndr_icon = "🟢" if nn["nndr_mean"] >= th["nndr"]        else "🔴"
            priv_metric_rows += [
                ["NN similarity", "DCR ratio",     f"{dcr_icon} {nn['dcr_ratio']} (synth={nn['dcr_synth_mean']} vs holdout={nn['dcr_holdout_mean']})"],
                ["NN similarity", "DCR p5 / p50",  f"{nn.get('dcr_p5','')} / {nn.get('dcr_p50','')}"],
                ["NN similarity", "% DCR < 0.1",   f"{nn.get('pct_dcr_below_0_1', 0)*100:.1f}%"],
                ["NN similarity", "NNDR mean",     f"{nndr_icon} {nn['nndr_mean']}"],
                ["NN similarity", "NNDR p5 / p50", f"{nn.get('nndr_p5','')} / {nn.get('nndr_p50','')}"],
            ]

        if priv_metric_rows:
            logger.info("  " + _table(
                ["Test", "Metric", "Value"],
                priv_metric_rows,
                [16, 22, 52],
            ).replace("\n", "\n  "))

    logger.info("\n" + "═" * W + "\n")


# ── Text rendering ─────────────────────────────────────────────────────────────
def _wrap(text: str, width: int) -> list[str]:
    """Wrap text to lines of at most `width` characters."""
    text = str(text)
    if len(text) <= width:
        return [text]
    lines = []
    while text:
        lines.append(text[:width])
        text = text[width:]
    return lines


def _table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> str:
    """Render a list of rows as a Unicode box-drawing table. col_widths caps each column width."""
    n = len(headers)
    if col_widths is None:
        col_widths = [60] * n

    widths = [min(max(len(h), max((len(str(r[i])) for r in rows), default=0)), col_widths[i])
              for i, h in enumerate(headers)]

    def _row_line(cells: list[str]) -> list[str]:
        wrapped = [_wrap(str(c), widths[i]) for i, c in enumerate(cells)]
        height  = max(len(w) for w in wrapped)
        lines   = []
        for ln in range(height):
            parts = [w[ln] if ln < len(w) else "" for w in wrapped]
            lines.append("│ " + " │ ".join(p.ljust(widths[i]) for i, p in enumerate(parts)) + " │")
        return lines

    def _bar(left, mid, right):
        return left + (mid).join("─" * (w + 2) for w in widths) + right

    out = [
        _bar("┌", "┬", "┐"),
        *_row_line(headers),
        _bar("├", "┼", "┤"),
    ]
    for row in rows:
        out.extend(_row_line([str(c) for c in row]))
    out.append(_bar("└", "┴", "┘"))
    return "\n".join(out)
