import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

FPR_GRID = np.linspace(0, 1, 1000)

# ── Vivid Bold palette ────────────────────────────────────────────────────────
_PALETTE   = ["#2196F3", "#FF4757", "#FF9F00", "#00C853", "#A29BFE", "#FD79A8"]
_C_REAL    = "#2196F3"   # vivid blue        — real / seed data
_C_SYNTH   = "#FF4757"   # vivid coral-red   — synthetic data
_C_AMBER   = "#FF9F00"   # vivid amber       — 3rd series / accent
_C_GREEN   = "#00C853"   # vivid green       — faithful / positive fill
_C_DIVERGE = "#FF6348"   # vivid orange-red  — divergence fill
_C_R_SHIFT = "#74B9FF"   # blue tint         — real after shift
_C_S_SHIFT = "#FF9F9F"   # red tint          — synth after shift

plt.rcParams.update({
    "axes.prop_cycle":   plt.cycler("color", _PALETTE),
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#2d3148",
    "grid.color":        "#2d3148",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "text.color":        "#e0e0e0",
    "axes.labelcolor":   "#aaa",
    "xtick.color":       "#aaa",
    "ytick.color":       "#aaa",
    "legend.facecolor":  "#1e2130",
    "legend.edgecolor":  "#2d3148",
    "legend.labelcolor": "#e0e0e0",
})


def find_divergence_regions(fpr_r, tpr_r, fpr_s, tpr_s, threshold):
    tpr_r_i = np.interp(FPR_GRID, fpr_r, tpr_r)
    tpr_s_i = np.interp(FPR_GRID, fpr_s, tpr_s)
    delta = tpr_r_i - tpr_s_i

    regions = []
    above = np.abs(delta) > threshold
    idx = np.where(np.diff(above.astype(int)))[0] + 1
    boundaries = [0] + idx.tolist() + [len(above)]
    for start, end in zip(boundaries, boundaries[1:]):
        if above[start]:
            seg = delta[start:end]
            direction = "real > synth" if seg.mean() > 0 else "synth > real"
            regions.append({
                "fpr_start": round(float(FPR_GRID[start]), 4),
                "fpr_end":   round(float(FPR_GRID[end - 1]), 4),
                "max_delta": round(float(np.abs(seg).max()), 4),
                "direction": direction,
            })
    return regions, tpr_r_i, tpr_s_i


def plot_roc_divergence_panel(models, real_probas, synth_probas, y_te, threshold):
    """3-panel ROC divergence plot (real vs synth per model). Returns max_divergence dict."""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    max_divergence = {}

    logger.info("\nROC divergence report  (threshold = %s)\n%s", threshold, '='*60)

    for ax, (name, _), color in zip(axes, models.items(), _PALETTE):
        fpr_r, tpr_r, _ = roc_curve(y_te, real_probas[name])
        fpr_s, tpr_s, _ = roc_curve(y_te, synth_probas[name])
        regions, tpr_r_i, tpr_s_i = find_divergence_regions(
            fpr_r, tpr_r, fpr_s, tpr_s, threshold
        )
        max_divergence[name] = max((r['max_delta'] for r in regions), default=0.0)

        ax.plot(fpr_r, tpr_r, label="real",  linewidth=2.0, color=_C_REAL)
        ax.plot(fpr_s, tpr_s, label="synth", linewidth=2.0, linestyle="--", color=_C_SYNTH)
        for r in regions:
            mask = (FPR_GRID >= r["fpr_start"]) & (FPR_GRID <= r["fpr_end"])
            ax.fill_between(FPR_GRID[mask], tpr_r_i[mask], tpr_s_i[mask],
                            alpha=0.30, color=_C_DIVERGE, label="_nolegend_")
        ax.plot([0, 1], [0, 1], linewidth=0.8, linestyle=":", color="#555")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        logger.info("\n%s", name)
        if not regions:
            logger.info("  No region exceeds threshold %s", threshold)
        else:
            logger.info("  %-20s %-14s direction", "FPR range", "max |ΔTPR|")
            logger.info("  %s", '-'*50)
            for r in regions:
                fpr_range = f"[{r['fpr_start']:.3f}, {r['fpr_end']:.3f}]"
                logger.info("  %-20s %-14.4f %s", fpr_range, r['max_delta'], r['direction'])

    logger.info("")
    plt.suptitle(f"ROC curves — shaded = |ΔTPR| > {threshold}", fontsize=11)
    plt.tight_layout()
    plt.show()
    return max_divergence


def plot_tstr_trtr_auroc(models, real_clfs, synth_clfs, X_te, y_te):
    """Single combined ROC plot: all models, TRTR solid vs TSTR dashed."""
    _, ax = plt.subplots(figsize=(7, 6))

    for (name, _), color in zip(models.items(), _PALETTE):
        fpr_trtr, tpr_trtr, _ = roc_curve(y_te, real_clfs[name].predict_proba(X_te)[:, 1])
        fpr_tstr, tpr_tstr, _ = roc_curve(y_te, synth_clfs[name].predict_proba(X_te)[:, 1])
        auc_trtr = roc_auc_score(y_te, real_clfs[name].predict_proba(X_te)[:, 1])
        auc_tstr = roc_auc_score(y_te, synth_clfs[name].predict_proba(X_te)[:, 1])

        ax.plot(fpr_trtr, tpr_trtr, color=color, linewidth=2.0,
                label=f"{name} TRTR  AUC={auc_trtr:.3f}")
        ax.plot(fpr_tstr, tpr_tstr, color=color, linewidth=2.0, linestyle="--",
                label=f"{name} TSTR  AUC={auc_tstr:.3f}")

    ax.plot([0, 1], [0, 1], linewidth=0.9, linestyle=":", color="#555", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("TSTR vs TRTR — ROC curves\nsolid = Train on Real  |  dashed = Train on Synthetic")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_tstr_panel(models, real_clfs, synth_clfs, X_te, y_te):
    """3-panel TSTR ROC plot: one subplot per model, TRTR vs TSTR with shaded gap."""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)

    for ax, (name, _), color in zip(axes, models.items(), _PALETTE):
        fpr_base, tpr_base, _ = roc_curve(y_te, real_clfs[name].predict_proba(X_te)[:, 1])
        fpr_tstr, tpr_tstr, _ = roc_curve(y_te, synth_clfs[name].predict_proba(X_te)[:, 1])
        auc_base = roc_auc_score(y_te, real_clfs[name].predict_proba(X_te)[:, 1])
        auc_tstr = roc_auc_score(y_te, synth_clfs[name].predict_proba(X_te)[:, 1])
        gap = auc_base - auc_tstr
        verdict = "faithful" if abs(gap) < 0.05 else ("synth > real" if gap < 0 else "diverges")

        ax.plot(fpr_base, tpr_base, color=_C_REAL,  linewidth=2.0, label=f"TRTR  AUC={auc_base:.3f}")
        ax.plot(fpr_tstr, tpr_tstr, color=_C_SYNTH, linewidth=2.0, linestyle="--",
                label=f"TSTR  AUC={auc_tstr:.3f}")
        ax.plot([0, 1], [0, 1], linewidth=0.8, linestyle=":", color="#555")

        tpr_b_i = np.interp(FPR_GRID, fpr_base, tpr_base)
        tpr_t_i = np.interp(FPR_GRID, fpr_tstr, tpr_tstr)
        fill_color = _C_GREEN if verdict == "faithful" else (_C_AMBER if verdict == "synth > real" else _C_DIVERGE)
        ax.fill_between(FPR_GRID, tpr_b_i, tpr_t_i, alpha=0.20, color=fill_color)

        ax.set_title(f"{name}\nΔAUC={gap:+.3f}  [{verdict}]", fontsize=9)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle("TSTR: Train on Synthetic, Test on Real\n"
                 "green = faithful  |  orange-red = diverges  |  amber = synth > real", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_trts_panel(models, real_clfs, trts_rows, X_te, y_te, X_te_synth, y_te_synth):
    """3-panel TRTS ROC plot: one subplot per model, real-test vs synth-test with shaded gap."""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)

    for ax, (name, auc_base, _, auc_trts, gap_trts, _, verdict), color in zip(
        axes, trts_rows, _PALETTE
    ):
        fpr_base, tpr_base, _ = roc_curve(y_te, real_clfs[name].predict_proba(X_te)[:, 1])
        fpr_trts, tpr_trts, _ = roc_curve(y_te_synth, real_clfs[name].predict_proba(X_te_synth)[:, 1])

        ax.plot(fpr_base, tpr_base, color=_C_REAL,  linewidth=2.0,
                label=f"real test  AUC={auc_base:.3f}")
        ax.plot(fpr_trts, tpr_trts, color=_C_SYNTH, linewidth=2.0, linestyle="--",
                label=f"synth test AUC={auc_trts:.3f}")
        ax.plot([0, 1], [0, 1], linewidth=0.8, linestyle=":", color="#555")

        tpr_b_i = np.interp(FPR_GRID, fpr_base, tpr_base)
        tpr_t_i = np.interp(FPR_GRID, fpr_trts, tpr_trts)
        ax.fill_between(FPR_GRID, tpr_b_i, tpr_t_i,
                        alpha=0.20, color=_C_DIVERGE if verdict == "diverges" else _C_GREEN)

        ax.set_title(f"{name}\nΔAUC={gap_trts:+.3f}  [{verdict}]", fontsize=9)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle("TRTS: Train on Real, Test on Synthetic\n"
                 "green fill = faithful  |  orange-red fill = synthetic diverges", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_shift_robustness(rows, shift_pct):
    """Two-panel chart: (top) AUC before vs after shift; (bottom) ΔAUC real vs synth."""
    model_names = [r[0] for r in rows]
    x, w = np.arange(len(model_names)), 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9))

    # ── Panel 1: absolute AUC before vs after shift ──────────────────────────
    ax1.bar(x - 1.5*w, [r[1] for r in rows], w, label="real → orig",   color=_C_REAL)
    ax1.bar(x - 0.5*w, [r[2] for r in rows], w, label="real → shift",  color=_C_R_SHIFT)
    ax1.bar(x + 0.5*w, [r[4] for r in rows], w, label="synth → orig",  color=_C_SYNTH)
    ax1.bar(x + 1.5*w, [r[5] for r in rows], w, label="synth → shift", color=_C_S_SHIFT)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha="right")
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.5, 1.0)
    ax1.set_title(f"AUC before vs after {int(shift_pct*100)}% distribution shift")
    ax1.legend(fontsize=8)

    # ── Panel 2: ΔAUC (shift − orig) for real vs synth ───────────────────────
    d_real  = [r[3] for r in rows]
    d_synth = [r[6] for r in rows]
    bars_r = ax2.bar(x - 0.5*w, d_real,  w, label="ΔAUC real",  color=_C_REAL)
    bars_s = ax2.bar(x + 0.5*w, d_synth, w, label="ΔAUC synth", color=_C_SYNTH)
    ax2.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha="right")
    ax2.set_ylabel("ΔAUC (shifted − original)")
    ax2.set_title(f"Change in AUC under {int(shift_pct*100)}% distribution shift")
    ax2.legend(fontsize=8)

    # annotate each bar with its value
    for bar in list(bars_r) + list(bars_s):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.0005 if h >= 0 else -0.002),
            f"{h:+.4f}",
            ha="center", va="bottom" if h >= 0 else "top",
            fontsize=7,
        )

    plt.tight_layout()
    plt.show()


def plot_subgroup_heatmap(df_p, col):
    """Heatmap of subgroup AUROC across Baseline / TSTR / TRTS regimes."""
    n_groups = df_p[col].nunique()
    _, axes_h = plt.subplots(1, 3, figsize=(14, max(3, n_groups * 0.6 + 1.5)), sharey=True)
    for ax_h, regime in zip(axes_h, ('Baseline', 'TSTR', 'TRTS')):
        piv = df_p.pivot_table(index=col, columns='model', values=regime)
        im  = ax_h.imshow(piv.values, aspect='auto', cmap='RdYlBu', vmin=0.5, vmax=0.75)
        ax_h.set_xticks(range(len(piv.columns)))
        ax_h.set_xticklabels(piv.columns, rotation=30, ha='right', fontsize=8)
        ax_h.set_yticks(range(len(piv.index)))
        ax_h.set_yticklabels(piv.index, fontsize=8)
        ax_h.set_title(regime, fontsize=10)
        for i in range(piv.values.shape[0]):
            for j in range(piv.values.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax_h.text(j, i, f"{v:.3f}", ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax_h, shrink=0.8)
    plt.suptitle(f"Subgroup AUROC — {col}  (green=higher, red=lower)", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_calibration_panel(models, regimes, n_bins, ece_fn):
    """Reliability curves grid (regime × model). Returns cal_results dict."""
    n_models  = len(models)
    n_regimes = len(regimes)
    fig, axes = plt.subplots(
        n_regimes, n_models,
        figsize=(5 * n_models, 4.5 * n_regimes),
        sharex=True, sharey=True,
    )
    if n_regimes == 1:
        axes = axes[np.newaxis, :]

    logger.info("\nCalibration Summary  (%d bins)\n%s", n_bins, '='*72)
    logger.info("%-25s %-12s %8s  %8s", "Model", "Regime", "Brier", "ECE")
    logger.info("%s", '-'*55)

    cal_results = {}
    for row_idx, (regime, (clfs, X_eval, y_eval)) in enumerate(regimes.items()):
        for col_idx, name in enumerate(models):
            ax = axes[row_idx, col_idx]
            proba = clfs[name].predict_proba(X_eval)[:, 1]

            frac_pos, mean_pred = calibration_curve(y_eval, proba, n_bins=n_bins, strategy='uniform')
            bs      = brier_score_loss(y_eval, proba)
            ece_val = ece_fn(y_eval, proba)
            cal_results[(name, regime)] = {'brier': bs, 'ece': ece_val}

            logger.info("%-25s %-12s %8.4f  %8.4f", name, regime, bs, ece_val)

            ax.plot(mean_pred, frac_pos, 's-', linewidth=2.0, markersize=5,
                    color=_C_REAL, label=f"model  (ECE={ece_val:.3f})")
            ax.plot([0, 1], [0, 1], linewidth=1, linestyle='--', color='#aaa', label='perfect')

            ax2 = ax.twinx()
            ax2.hist(proba, bins=n_bins, range=(0, 1), alpha=0.25, color=_C_AMBER, edgecolor='none')
            ax2.set_ylabel("count", fontsize=7, color=_C_AMBER)
            ax2.tick_params(axis='y', labelcolor=_C_AMBER, labelsize=7)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted probability", fontsize=8)
            ax.set_ylabel("Fraction of positives", fontsize=8)
            ax.set_title(f"{name}\n{regime}  |  Brier={bs:.4f}  ECE={ece_val:.4f}", fontsize=8)
            ax.legend(fontsize=7)

    logger.info("")
    plt.suptitle("Reliability curves — closer to diagonal = better calibration", fontsize=11)
    plt.tight_layout()
    plt.show()
    return cal_results


def plot_synthetic_data(df, seed_stats: dict) -> None:
    """Distribution plots for each categorical and numeric column in the synthetic data."""
    for col in seed_stats["meta"]["categorical_columns"]:
        if col not in df.columns:
            continue
        plt.figure(figsize=(10, 5))
        sns.countplot(x=col, data=df, order=df[col].value_counts().index, color=_C_REAL)
        plt.title(f"Synthetic Distribution of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    for col in seed_stats["meta"]["numeric_columns"]:
        if col not in df.columns:
            continue
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col].dropna(), kde=True, color=_C_SYNTH, line_kws={"linewidth": 2})
        plt.title(f"Synthetic Distribution of {col}")
        plt.tight_layout()
        plt.show()
