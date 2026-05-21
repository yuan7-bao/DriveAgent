"""
Evaluation Script — Comprehensive metrics for ablation study.

Computes per-variant, per-scenario, per-confidence-level:
  - Accuracy, Macro F1, Weighted F1
  - Precision / Recall for HAZARD class
  - False Negative Rate (FNR) — safety-critical: missed real hazards
  - False Positive Rate (FPR) — efficiency: phantom braking risk
  - Average inference latency
  - Optional: Bootstrap 95% CI

Outputs:
  - Console tables (ablation summary, per-scenario, insights)
  - results/metrics_summary.json (for plot script)

Usage:
  python 3_evaluate_metrics.py                 # basic evaluation
  python 3_evaluate_metrics.py --bootstrap 1000 # with 95% CI (recommended for paper)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

LABEL_ORDER = ["SAFE", "CAUTION", "HAZARD"]
_INVALID_PREDICTIONS = frozenset({"ERROR", "UNKNOWN", "PARSE_ERROR"})


def _safe_divide(num: float, den: float) -> float:
    return round(num / den, 4) if den > 0 else 0.0


def compute_hazard_fnr_fpr(y_true: list[str], y_pred: list[str]) -> tuple[float, float]:
    """Compute FNR and FPR specifically for the HAZARD class.

    FNR = missed HAZARDs / total real HAZARDs  (safety-critical metric)
    FPR = false HAZARD alarms / total real non-HAZARDs  (efficiency metric)
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "HAZARD" and p == "HAZARD")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "HAZARD" and p != "HAZARD")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "HAZARD" and p == "HAZARD")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != "HAZARD" and p != "HAZARD")
    return _safe_divide(fn, tp + fn), _safe_divide(fp, fp + tn)


def bootstrap_ci_95(
    y_true: list[str],
    y_pred: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict | None:
    """Bootstrap 95% confidence intervals for accuracy, FNR, FPR."""
    n = len(y_true)
    if n < 5 or n_bootstrap <= 0:
        return None

    rng = np.random.default_rng(seed)
    accs, fnrs, fprs = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        accs.append(accuracy_score(yt, yp))
        fnr, fpr = compute_hazard_fnr_fpr(yt, yp)
        fnrs.append(fnr)
        fprs.append(fpr)

    def _ci(arr):
        return [round(float(np.quantile(arr, 0.025)), 4), round(float(np.quantile(arr, 0.975)), 4)]

    return {
        "accuracy": _ci(accs),
        "fnr_hazard": _ci(fnrs),
        "fpr_hazard": _ci(fprs),
        "n_bootstrap": n_bootstrap,
    }


def evaluate_variant(
    variant_df: pd.DataFrame,
    bootstrap_n: int = 0,
    bootstrap_seed: int = 42,
) -> dict:
    """Compute all metrics for one variant's predictions."""
    y_true = [str(v).strip().upper() for v in variant_df["ground_truth"]]
    y_pred = [str(v).strip().upper() for v in variant_df["final_decision"]]

    # Filter invalid predictions
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p not in _INVALID_PREDICTIONS]
    if not valid:
        return {"error": "no valid predictions", "n_total": len(y_true), "n_valid": 0}

    yt, yp = zip(*valid)
    yt, yp = list(yt), list(yp)

    fnr, fpr = compute_hazard_fnr_fpr(yt, yp)
    avg_latency = variant_df["latency_s"].mean() if "latency_s" in variant_df.columns else 0.0

    result = {
        "n_total": len(y_true),
        "n_valid": len(valid),
        "accuracy": round(accuracy_score(yt, yp), 4),
        "f1_macro": round(f1_score(yt, yp, labels=LABEL_ORDER, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(yt, yp, labels=LABEL_ORDER, average="weighted", zero_division=0), 4),
        "precision_hazard": round(precision_score(yt, yp, labels=["HAZARD"], average="macro", zero_division=0), 4),
        "recall_hazard": round(recall_score(yt, yp, labels=["HAZARD"], average="macro", zero_division=0), 4),
        "fnr_hazard": round(fnr, 4),
        "fpr_hazard": round(fpr, 4),
        "avg_latency_s": round(avg_latency, 3),
    }

    if bootstrap_n > 0:
        ci = bootstrap_ci_95(yt, yp, n_bootstrap=bootstrap_n, seed=bootstrap_seed)
        if ci:
            result["bootstrap_ci_95"] = ci

    return result


def evaluate_by_group(variant_df: pd.DataFrame, group_col: str) -> dict:
    """Evaluate metrics grouped by a column (scenario_type or confidence_level)."""
    results = {}
    for group_val in sorted(variant_df[group_col].unique()):
        sub = variant_df[variant_df[group_col] == group_val]
        if len(sub) > 0:
            results[group_val] = evaluate_variant(sub)
    return results


def print_summary(summary: dict) -> None:
    """Print formatted ablation summary table."""
    variants = [v for v in config.ABLATION_VARIANTS if v in summary]
    if not variants:
        print("[WARN] No variant results to display.")
        return

    print(f"\n{'=' * 116}")
    print(f"{'ABLATION STUDY — OVERALL RESULTS':^116}")
    print(f"{'=' * 116}")
    header = (
        f"{'Var':<5} {'Label':<36} {'Acc':>6} {'F1-M':>6} {'F1-W':>6} "
        f"{'P(H)':>6} {'R(H)':>6} {'FNR':>6} {'FPR':>6} {'Lat':>6}"
    )
    print(header)
    print("-" * 116)

    for vid in variants:
        m = summary[vid]["overall"]
        if "error" in m:
            print(f"{vid:<5} {config.VARIANT_LABELS[vid]:<36} — no valid predictions —")
            continue
        label = config.VARIANT_LABELS[vid][:34]
        print(
            f"{vid:<5} {label:<36} "
            f"{m['accuracy']:>6.3f} {m['f1_macro']:>6.3f} {m['f1_weighted']:>6.3f} "
            f"{m['precision_hazard']:>6.3f} {m['recall_hazard']:>6.3f} "
            f"{m['fnr_hazard']:>6.3f} {m['fpr_hazard']:>6.3f} "
            f"{m['avg_latency_s']:>6.1f}"
        )
    print(f"{'=' * 116}")

    # Safety analysis
    print("\n[Safety — FNR for HAZARD class (lower = safer)]")
    for vid in variants:
        m = summary[vid]["overall"]
        if "error" in m:
            continue
        fnr = m["fnr_hazard"]
        bar = "█" * int(fnr * 40) + "░" * (40 - int(fnr * 40))
        print(f"  {vid}  FNR={fnr:.3f}  {bar}")

    # Per-scenario detail
    scenarios = sorted(
        set(s for vid in variants for s in summary[vid].get("by_scenario", {}))
    )
    if scenarios:
        print(f"\n[Per-Scenario Accuracy]")
        header = f"  {'Scenario':<24} " + "  ".join(f"{v:>6}" for v in variants)
        print(header)
        print("  " + "-" * (24 + 8 * len(variants)))
        for scen in scenarios:
            vals = []
            for vid in variants:
                acc = summary[vid].get("by_scenario", {}).get(scen, {}).get("accuracy")
                vals.append(f"{acc:>6.3f}" if acc is not None else "   N/A")
            print(f"  {scen:<24} {'  '.join(vals)}")

    # Ablation insights
    print("\n[Ablation Insights]")
    pairs = [
        ("M0", "M1", "Audio Agent contribution"),
        ("M1", "M4", "Dual Critic contribution"),
        ("M4", "M5", "Self-Consistency contribution"),
        ("M0", "M5", "Total system improvement"),
    ]
    for v_from, v_to, label in pairs:
        if v_from in summary and v_to in summary:
            a1 = summary[v_from]["overall"].get("accuracy", 0)
            a2 = summary[v_to]["overall"].get("accuracy", 0)
            print(f"  {label} ({v_to} vs {v_from}):  Δacc = {a2 - a1:+.3f}")

    # Bootstrap CIs if available
    if any(
        "bootstrap_ci_95" in summary.get(v, {}).get("overall", {})
        for v in variants
    ):
        print("\n[Bootstrap 95% CI]")
        for vid in variants:
            ci = summary[vid]["overall"].get("bootstrap_ci_95")
            if ci:
                print(
                    f"  {vid}  acc={ci['accuracy']}  "
                    f"FNR={ci['fnr_hazard']}  FPR={ci['fpr_hazard']}"
                )


def evaluate(bootstrap_n: int = 0, bootstrap_seed: int = 42) -> None:
    """Main evaluation entry point."""
    print(f"[eval] Loading: {config.ABLATION_RESULTS}")

    try:
        df = pd.read_csv(config.ABLATION_RESULTS)
    except FileNotFoundError:
        print("[ERROR] Results file not found. Run 2_run_ablation.py first.")
        return

    print(f"[eval] {len(df)} total records, variants: {sorted(df['variant'].unique())}")

    summary = {}
    for vid in config.ABLATION_VARIANTS:
        sub = df[df["variant"] == vid]
        if len(sub) == 0:
            print(f"[WARN] No results for variant {vid}, skipping.")
            continue

        summary[vid] = {
            "overall": evaluate_variant(sub, bootstrap_n, bootstrap_seed),
            "by_scenario": evaluate_by_group(sub, "scenario_type"),
            "by_confidence": evaluate_by_group(sub, "confidence_level"),
        }

    print_summary(summary)

    # Save JSON
    with open(config.METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] Saved: {config.METRICS_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ablation metrics")
    parser.add_argument(
        "--bootstrap", type=int, default=0,
        help="Bootstrap iterations (0=none, 1000 recommended for paper)",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(bootstrap_n=args.bootstrap, bootstrap_seed=args.bootstrap_seed)
