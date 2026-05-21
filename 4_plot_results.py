"""
Figure Generator — Publication-quality plots for ablation study.

Produces:
  fig1_ablation_bar.png       — Accuracy + F1 bar chart (M0–M5)
  fig2_fnr_fpr.png            — Safety–Efficiency tradeoff scatter
  fig3_scenario_heatmap.png   — Per-scenario accuracy heatmap
  fig4_confidence_band.png    — M5 performance by confidence band
  fig5_latency.png            — Inference latency comparison

Usage:
  python 4_plot_results.py            # all variants
  python 4_plot_results.py --paper    # main-text subset (M0, M1, M4, M5)
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Visual constants ─────────────────────────────────────────────────────────

COLORS = {
    "M0": "#B4B2A9", "M1": "#5DCAA5", "M2": "#F0997B",
    "M3": "#FAC775", "M4": "#AFA9EC", "M5": "#639922",
}
SHORT_LABELS = {
    "M0": "M0\nBaseline", "M1": "M1\n+Audio", "M2": "M2\n+SafeCritic",
    "M3": "M3\n+EffCritic", "M4": "M4\nDualCritic", "M5": "M5\nFull (Ours)",
}
SCENARIO_SHORT = {
    "S01_Blind_Spot": "S01\nBlind Spot", "S02_Occlusion": "S02\nOcclusion",
    "S03_False_Positive": "S03\nFalse Pos.", "S04_Multi_Hazard": "S04\nMulti-Haz.",
    "S05_Night_Rain": "S05\nNight/Rain",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,

})


def _load_summary() -> dict:
    with open(config.METRICS_OUTPUT, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_variants(summary: dict, variant_filter: list[str] | None = None) -> list[str]:
    order = variant_filter or config.ABLATION_VARIANTS
    return [v for v in order if v in summary]


def _save_fig(fig, name: str) -> None:
    path = os.path.join(config.FIGURES_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")


# ── Fig 1: Accuracy + F1 bar chart ──────────────────────────────────────────

def fig1_ablation_bar(summary: dict, vf=None) -> None:
    variants = _get_variants(summary, vf)
    x = np.arange(len(variants))
    acc = [summary[v]["overall"]["accuracy"] for v in variants]
    f1m = [summary[v]["overall"]["f1_macro"] for v in variants]
    f1w = [summary[v]["overall"]["f1_weighted"] for v in variants]

    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.25
    b1 = ax.bar(x - w, acc, w, label="Accuracy", color=[COLORS[v] for v in variants], alpha=0.95)
    b2 = ax.bar(x, f1m, w, label="F1 (Macro)", color=[COLORS[v] for v in variants], alpha=0.65)
    b3 = ax.bar(x + w, f1w, w, label="F1 (Weighted)", color=[COLORS[v] for v in variants], alpha=0.40)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[v] for v in variants], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Overall Performance", fontsize=13, pad=12)
    ax.legend(loc="lower right")
    ax.axhline(y=max(acc), color="#639922", lw=0.8, ls="--", alpha=0.5)
    fig.tight_layout()
    _save_fig(fig, "fig1_ablation_bar.png")


# ── Fig 2: FNR vs FPR scatter ───────────────────────────────────────────────

def fig2_fnr_fpr(summary: dict, vf=None) -> None:
    variants = _get_variants(summary, vf)
    fnrs = [summary[v]["overall"]["fnr_hazard"] for v in variants]
    fprs = [summary[v]["overall"]["fpr_hazard"] for v in variants]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for v, fnr, fpr in zip(variants, fnrs, fprs):
        ax.scatter(fpr, fnr, s=160, color=COLORS[v], zorder=5, edgecolors="white", lw=1.5)
        ax.annotate(v, (fpr, fnr), textcoords="offset points", xytext=(8, 6), fontsize=10, fontweight="bold")

    ax.annotate("← Ideal", xy=(0.02, 0.02), fontsize=9, color="gray", style="italic")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("False Positive Rate (phantom braking risk)")
    ax.set_ylabel("False Negative Rate (missed HAZARD — safety risk)")
    ax.set_title("Safety–Efficiency Tradeoff\n(lower-left = better)", fontsize=12, pad=10)
    ax.set_xlim(-0.03, max(fprs) * 1.25 + 0.05)
    ax.set_ylim(-0.03, max(fnrs) * 1.25 + 0.05)

    patches = [mpatches.Patch(color=COLORS[v], label=f"{v}: {config.VARIANT_LABELS[v][:28]}")
               for v in variants]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    fig.tight_layout()
    _save_fig(fig, "fig2_fnr_fpr.png")


# ── Fig 3: Scenario heatmap ─────────────────────────────────────────────────

def fig3_scenario_heatmap(summary: dict, vf=None) -> None:
    variants = _get_variants(summary, vf)
    scenarios = list(SCENARIO_SHORT.keys())

    matrix = np.zeros((len(variants), len(scenarios)))
    for i, v in enumerate(variants):
        for j, s in enumerate(scenarios):
            acc = summary[v].get("by_scenario", {}).get(s, {}).get("accuracy", 0.0)
            matrix[i, j] = acc if isinstance(acc, (int, float)) else 0.0

    fig, ax = plt.subplots(figsize=(9, max(4, len(variants) * 0.8 + 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios], fontsize=9)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([SHORT_LABELS[v].replace("\n", " ") for v in variants], fontsize=9)

    for i in range(len(variants)):
        for j in range(len(scenarios)):
            val = matrix[i, j]
            color = "white" if val < 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    ax.set_title("Per-Scenario Accuracy Heatmap", fontsize=12, pad=12)
    fig.tight_layout()
    _save_fig(fig, "fig3_scenario_heatmap.png")


# ── Fig 4: Confidence band (M5) ─────────────────────────────────────────────

def fig4_confidence_band(summary: dict) -> None:
    if "M5" not in summary or "by_confidence" not in summary["M5"]:
        print("  [skip] M5 data not available for confidence band plot.")
        return

    bands = ["low", "mid", "high"]
    metrics = ["accuracy", "fnr_hazard", "fpr_hazard"]
    labels = {"accuracy": "Accuracy", "fnr_hazard": "FNR (HAZARD)", "fpr_hazard": "FPR (HAZARD)"}
    colors = {"accuracy": "#639922", "fnr_hazard": "#D85A30", "fpr_hazard": "#378ADD"}
    band_labels = ["Low conf\n(0.20–0.44)", "Mid conf\n(0.45–0.69)", "High conf\n(0.70–0.96)"]

    x = np.arange(len(bands))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for k, metric in enumerate(metrics):
        vals = [summary["M5"]["by_confidence"].get(b, {}).get(metric, 0.0) for b in bands]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=labels[metric], color=colors[metric], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title("M5 (Full System) — Performance by Confidence Band", fontsize=12, pad=10)
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "fig4_confidence_band.png")


# ── Fig 5: Latency ──────────────────────────────────────────────────────────

def fig5_latency(summary: dict, vf=None) -> None:
    variants = _get_variants(summary, vf)
    latencies = [summary[v]["overall"]["avg_latency_s"] for v in variants]

    fig, ax = plt.subplots(figsize=(8, max(3, len(variants) * 0.6 + 1)))
    labels = [SHORT_LABELS[v].replace("\n", " ") for v in variants]
    bars = ax.barh(labels, latencies, color=[COLORS[v] for v in variants], alpha=0.9)

    for bar, lat in zip(bars, latencies):
        ax.text(lat + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{lat:.1f}s", va="center", fontsize=9)

    ax.set_xlabel("Average inference latency (seconds)")
    ax.set_title("Inference Latency per Variant", fontsize=12, pad=10)
    ax.invert_yaxis()
    fig.tight_layout()
    _save_fig(fig, "fig5_latency.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(variant_filter: list[str] | None = None) -> None:
    if not os.path.exists(config.METRICS_OUTPUT):
        print(f"[ERROR] Metrics JSON not found: {config.METRICS_OUTPUT}")
        print("  → Run: python 3_evaluate_metrics.py")
        return

    summary = _load_summary()
    print(f"[plot] Generating figures → {config.FIGURES_DIR}")

    fig1_ablation_bar(summary, variant_filter)
    fig2_fnr_fpr(summary, variant_filter)
    fig3_scenario_heatmap(summary, variant_filter)
    fig4_confidence_band(summary)
    fig5_latency(summary, variant_filter)

    print(f"\n[plot] All figures saved to: {config.FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--paper", action="store_true", help="Main-text variants only (M0, M1, M4, M5)")
    args = parser.parse_args()
    vf = config.PAPER_VARIANTS if args.paper else None
    main(variant_filter=vf)
