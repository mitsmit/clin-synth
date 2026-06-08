import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

def _infer_col_type(series: pd.Series) -> str:
    """Return a high-level semantic type for a column."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Handle both object dtype (pandas 1.x) and StringDtype (pandas 2.x)
    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype):
        sample = series.dropna().head(50)
        try:
            with warnings.catch_warnings():
                # Non-date columns (the common case) raise/warn here; both are
                # expected probing noise, not something the caller should see.
                warnings.simplefilter("ignore", UserWarning)
                pd.to_datetime(sample)
            return "datetime"
        except Exception:
            pass
        n_unique = series.nunique()
        n_total = series.count()
        if n_unique / max(n_total, 1) < 0.10 or n_unique <= 20:
            return "categorical"
        return "text"
    return "unknown"

def _fit_best_distribution(series: pd.Series) -> dict:
    """
    Try to fit several common continuous distributions and return
    the best fit by AIC (Akaike Information Criterion).
    """
    candidates = ["norm", "lognorm", "expon", "gamma", "beta", "uniform"]
    data = series.dropna().values.astype(float)
    if len(data) < 10:
        return {}

    best = {"distribution": None, "params": {}, "aic": np.inf}

    for dist_name in candidates:
        dist = getattr(scipy_stats, dist_name)
        try:
            params = dist.fit(data)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            if aic < best["aic"]:
                param_names = (
                    list(dist.shapes.split(",")) if dist.shapes else []
                ) + ["loc", "scale"]
                best = {
                    "distribution": dist_name,
                    "params": dict(zip(param_names, params)),
                    "aic": round(aic, 4),
                }
        except Exception:
            continue

    return best

def _numeric_stats(series: pd.Series, col_type: str) -> dict:
    """Detailed stats for numeric (integer/float) columns."""
    clean = series.dropna()
    if clean.empty:
        return {}

    q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantiles = {f"p{int(p*100)}": round(clean.quantile(p), 6) for p in q}

    skewness = round(float(scipy_stats.skew(clean)), 6)
    kurtosis = round(float(scipy_stats.kurtosis(clean)), 6)

    # Normality test (only meaningful with enough data)
    normality = {}
    if 8 <= len(clean) <= 5000:
        stat, p_value = scipy_stats.shapiro(clean.sample(min(len(clean), 5000), random_state=42))
        normality = {"shapiro_stat": round(float(stat), 6), "shapiro_p": round(float(p_value), 6)}

    best_dist = _fit_best_distribution(clean) if col_type == "float" else {}

    return {
        "min": round(float(clean.min()), 6),
        "max": round(float(clean.max()), 6),
        "mean": round(float(clean.mean()), 6),
        "median": round(float(clean.median()), 6),
        "std": round(float(clean.std()), 6),
        "variance": round(float(clean.var()), 6),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "quantiles": quantiles,
        "normality_test": normality,
        "best_fit_distribution": best_dist,
    }

def _categorical_stats(series: pd.Series) -> dict:
    """Frequency and entropy stats for categorical columns."""
    clean = series.dropna().astype(str)
    counts = clean.value_counts()
    probs = counts / counts.sum()

    entropy = round(float(scipy_stats.entropy(probs)), 6)

    top_n = 50
    freq_table = {
        str(k): {"count": int(v), "proportion": round(float(probs[k]), 6)}
        for k, v in counts.head(top_n).items()
    }

    return {
        "n_unique": int(series.nunique()),
        "mode": str(counts.index[0]) if not counts.empty else None,
        "mode_frequency": round(float(probs.iloc[0]), 6) if not counts.empty else None,
        "entropy_bits": entropy,
        "frequency_table": freq_table,
    }

def _datetime_stats(series: pd.Series) -> dict:
    """Stats for datetime columns."""
    dt = pd.to_datetime(series, infer_datetime_format=True, errors="coerce").dropna()
    if dt.empty:
        return {}

    deltas = dt.sort_values().diff().dropna().dt.total_seconds()
    return {
        "min": str(dt.min()),
        "max": str(dt.max()),
        "range_days": round((dt.max() - dt.min()).total_seconds() / 86400, 2),
        "median": str(dt.median()),
        "mean_interval_seconds": round(float(deltas.mean()), 2) if not deltas.empty else None,
        "std_interval_seconds": round(float(deltas.std()), 2) if not deltas.empty else None,
    }

def _boolean_stats(series: pd.Series) -> dict:
    clean = series.dropna()
    true_rate = round(float(clean.mean()), 6)
    return {"true_rate": true_rate, "false_rate": round(1 - true_rate, 6)}

# ─────────────────────────────────────────────
# Correlation analysis
# ─────────────────────────────────────────────

def _correlation_stats(df: pd.DataFrame, numeric_cols: list, cat_cols: list) -> dict:
    result = {}

    # ── Pearson / Spearman for numeric pairs ──────────────────────────────
    if len(numeric_cols) >= 2:
        num_df = df[numeric_cols].dropna()
        pearson = num_df.corr(method="pearson").round(4)
        spearman = num_df.corr(method="spearman").round(4)

        # Highlight strongly correlated pairs (|r| > 0.5)
        strong_pairs = []
        cols = list(pearson.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = pearson.iloc[i, j]
                if abs(r) > 0.5:
                    strong_pairs.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "pearson_r": round(float(r), 4),
                        "spearman_r": round(float(spearman.iloc[i, j]), 4),
                    })

        result["numeric_pearson"] = pearson.to_dict()
        result["numeric_spearman"] = spearman.to_dict()
        result["strong_numeric_pairs"] = strong_pairs

    # ── Cramér's V for categorical pairs ──────────────────────────────────
    if len(cat_cols) >= 2:
        cramers_matrix = {}
        for c1 in cat_cols:
            cramers_matrix[c1] = {}
            for c2 in cat_cols:
                if c1 == c2:
                    cramers_matrix[c1][c2] = 1.0
                    continue
                ct = pd.crosstab(df[c1], df[c2])
                chi2 = scipy_stats.chi2_contingency(ct, correction=False)[0]
                n = ct.sum().sum()
                phi2 = chi2 / n
                r, k = ct.shape
                v = float(np.sqrt(phi2 / min(k - 1, r - 1))) if min(k - 1, r - 1) > 0 else 0.0
                cramers_matrix[c1][c2] = round(v, 4)
        result["categorical_cramers_v"] = cramers_matrix

    # ── Point-biserial: numeric vs categorical (binary) ───────────────────
    binary_cats = [c for c in cat_cols if df[c].nunique() == 2]
    if binary_cats and numeric_cols:
        pb_corr = {}
        for bc in binary_cats:
            codes = pd.Categorical(df[bc].dropna()).codes
            pb_corr[bc] = {}
            for nc in numeric_cols:
                valid = df[[bc, nc]].dropna()
                if len(valid) < 4:
                    continue
                codes_valid = pd.Categorical(valid[bc]).codes
                r, p = scipy_stats.pointbiserialr(codes_valid, valid[nc].values)
                pb_corr[bc][nc] = {"r": round(float(r), 4), "p_value": round(float(p), 4)}
        result["point_biserial"] = pb_corr

    return result

# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def extract_seed_stats(
    input_df: pd.DataFrame | None = None,
    csv_path: str | None = None,
    output_json: str | None = None,
    sample_size: int | None = None,
    datetime_cols: list[str] | None = None,
    
) -> dict:
    """
    Extract comprehensive statistics from a seed CSV for synthetic data generation.

    Parameters
    ----------
    csv_path      : Path to the seed CSV file.
    output_json   : Optional path to save the stats as a JSON file.
    sample_size   : If set, sample this many rows (useful for very large files).
    datetime_cols : Column names to force-parse as datetime.

    Returns
    -------
    dict with keys:
        - meta            : Dataset-level metadata
        - columns         : Per-column statistics keyed by column name
        - missing         : Missingness summary
        - correlations    : Correlation matrices and strong pairs
        - multivariate    : Variance explained (PCA) for numeric columns
    """

    if input_df is not None:
        df = input_df
    elif csv_path is not None:
        # Use index_col=0 only when the first column is unnamed (a saved RangeIndex).
        # Named first columns (e.g. "age") must be kept as data columns.
        _peek = pd.read_csv(csv_path, nrows=0)
        _first = str(_peek.columns[0]) if len(_peek.columns) else ""
        _idx = 0 if (_first == "" or _first == "Unnamed: 0") else None
        df = pd.read_csv(csv_path, index_col=_idx)
    else:
        raise ValueError("Either input_df or csv_path must be provided")

    # Force datetime parsing where requested
    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    n_rows, n_cols = df.shape

    # ── Per-column type inference ──────────────────────────────────────────
    col_types = {col: _infer_col_type(df[col]) for col in df.columns}
    numeric_cols  = [c for c, t in col_types.items() if t in ("integer", "float")]
    cat_cols      = [c for c, t in col_types.items() if t == "categorical"]
    datetime_cols_inferred = [c for c, t in col_types.items() if t == "datetime"]
    bool_cols     = [c for c, t in col_types.items() if t == "boolean"]

    # ── Missing values ─────────────────────────────────────────────────────
    missing = {
        col: {
            "count": int(df[col].isna().sum()),
            "rate": round(float(df[col].isna().mean()), 6),
        }
        for col in df.columns
        if df[col].isna().any()
    }

    # ── Per-column statistics ──────────────────────────────────────────────
    columns_stats: dict = {}
    for col in df.columns:
        ctype = col_types[col]
        base = {
            "dtype": str(df[col].dtype),
            "semantic_type": ctype,
            "n_total": n_rows,
            "n_non_null": int(df[col].count()),
            "n_unique": int(df[col].nunique()),
        }

        if ctype in ("integer", "float"):
            base.update(_numeric_stats(df[col], ctype))
        elif ctype == "categorical":
            base.update(_categorical_stats(df[col]))
        elif ctype == "datetime":
            base.update(_datetime_stats(df[col]))
        elif ctype == "boolean":
            base.update(_boolean_stats(df[col]))

        columns_stats[col] = base

    # ── Correlations ──────────────────────────────────────────────────────
    # correlations = _correlation_stats(df, numeric_cols, cat_cols)
    correlations = {
    "numeric": df[numeric_cols].corr().to_dict() if numeric_cols else {},
    "categorical": {},  # Optionally implement Cramér's V or leave empty
}

    # ── Multivariate: PCA variance explained ──────────────────────────────
    multivariate: dict = {}
    if len(numeric_cols) >= 2:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            num_df = df[numeric_cols].dropna()
            scaled = StandardScaler().fit_transform(num_df)
            pca = PCA()
            pca.fit(scaled)
            explained = pca.explained_variance_ratio_
            multivariate["pca"] = {
                "n_components": len(explained),
                "explained_variance_ratio": [round(float(v), 4) for v in explained],
                "cumulative_variance_ratio": [
                    round(float(v), 4) for v in np.cumsum(explained)
                ],
                "components_for_90pct_variance": int(
                    np.searchsorted(np.cumsum(explained), 0.90) + 1
                ),
            }
        except ImportError:
            multivariate["pca"] = {"error": "scikit-learn not installed"}

    # ── Assemble final result ──────────────────────────────────────────────
    result = {
        "meta": {
            # "source_file": str(path.name),
            "n_rows": n_rows,
            "n_columns": n_cols,
            "column_names": list(df.columns),
            "column_types": col_types,
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
            "datetime_columns": datetime_cols_inferred,
            "boolean_columns": bool_cols,
        },
        "columns": columns_stats,
        "missing": missing,
        "correlations": correlations,
        "multivariate": multivariate,
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Stats saved to %s", output_json)

    return result

def plot_seed_data(df):
    """Optional: Plot distributions for numeric columns and frequency tables for categorical columns.
    This can be useful for a quick visual check of the data."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.sample(frac=1.0)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    for col in cat_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index[:20])
        plt.title(f"Top categories in {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.show()

if __name__ == "__main__":
    import sys
    import pprint

    csv_file = sys.argv[1] if len(sys.argv) > 1 else "./data/diab_seed.csv"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "./data/diab_stats.json"

    stats = extract_seed_stats(csv_path=csv_file, output_json=out_file)

    logger.info("\n=== METADATA ===")
    logger.debug("%s", pprint.pformat(stats["meta"]))

    logger.info("\n=== MISSING VALUES ===")
    logger.debug("%s", pprint.pformat(stats["missing"] or "None"))

    logger.info("\n=== STRONG NUMERIC CORRELATIONS ===")
    logger.debug("%s", pprint.pformat(stats["correlations"].get("strong_numeric_pairs", [])))

    # print("\n=== PCA SUMMARY ===")
    logger.debug("%s", pprint.pformat(stats["multivariate"].get("pca", {})))

    # plot some distributions if desired
    plot_seed_data(df=pd.read_csv(csv_file, index_col=0))

   