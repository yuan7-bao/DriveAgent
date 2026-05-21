"""
Ablation Runner — iterates over dataset × variants (M0–M5).

Usage:
  python 2_run_ablation.py                        # full 300 × 6 = 1800 calls
  python 2_run_ablation.py --limit 30             # first 30 rows only (quick test)
  python 2_run_ablation.py --test30               # shortcut for --limit 30
  python 2_run_ablation.py --variants M4 M5       # specific variants only
  python 2_run_ablation.py --resume               # skip already-completed rows
  python 2_run_ablation.py --resume --variants M5 # resume, only run M5
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from agents.fusion_agents import FusionPipeline

logger = logging.getLogger("driveagent")

PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fusion_prompts.json")

# Variants that require the teacher (GPT-4o) API key
_NEEDS_TEACHER = frozenset({"M2", "M3", "M4", "M5"})


def _check_api_keys(variants: list[str]) -> None:
    """Validate required API keys are set before starting."""
    if not config.DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set. Configure config_local.py or environment variable.")
        sys.exit(1)

    needs_teacher = _NEEDS_TEACHER & set(variants)
    if needs_teacher and not config.GPT4O_API_KEY:
        print(f"[ERROR] GPT4O_API_KEY required for variants: {sorted(needs_teacher)}")
        print("        Configure config_local.py or environment variable GPT4O_API_KEY.")
        sys.exit(1)


def run_ablation(
    variants: list[str] | None = None,
    resume: bool = False,
    limit: int | None = None,
) -> None:
    """Run ablation experiments."""

    # Load dataset
    if not os.path.exists(config.DATASET_CSV):
        print(f"[ERROR] Dataset not found: {config.DATASET_CSV}")
        print("  → Run: python 1_generate_dataset.py")
        return

    df = pd.read_csv(config.DATASET_CSV)
    if limit is not None and limit > 0:
        df = df.head(limit)

    variants = variants or config.ABLATION_VARIANTS
    _check_api_keys(variants)

    n_total = len(df) * len(variants)
    print(f"[ablation] Variants: {variants}")
    print(f"[ablation] Samples: {len(df)} per variant, {n_total} total API calls")
    print(f"[ablation] Results file: {config.ABLATION_RESULTS}")

    pipeline = FusionPipeline(PROMPTS_PATH)

    # Load existing results for resumption
    all_results: list[dict] = []
    existing: set[tuple[int, str]] = set()
    if resume and os.path.exists(config.ABLATION_RESULTS):
        existing_df = pd.read_csv(config.ABLATION_RESULTS)
        existing = set(zip(existing_df["row_idx"], existing_df["variant"]))
        all_results = existing_df.to_dict("records")
        print(f"[ablation] Resuming: {len(existing)} existing results found, skipping those.")

    t_start = time.time()

    for variant_id in variants:
        print(f"\n{'=' * 70}")
        print(f"  Variant {variant_id}: {config.VARIANT_LABELS[variant_id]}")
        print(f"{'=' * 70}")

        n_skipped = 0
        n_errors = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=variant_id, unit="sample"):
            if (idx, variant_id) in existing:
                n_skipped += 1
                continue

            try:
                result = pipeline.run_variant(variant_id, row.to_dict())
            except Exception as e:
                n_errors += 1
                logger.error("Row %d, Variant %s: %s", idx, variant_id, e)
                result = {
                    "variant": variant_id,
                    "final_decision": "ERROR",
                    "decision_confidence": 0.0,
                    "final_reasoning": f"Exception: {str(e)[:200]}",
                    "audio_description": "ERROR",
                    "draft_decision": "ERROR",
                    "critic_safety": "ERROR",
                    "critic_efficiency": "ERROR",
                    "latency_s": 0.0,
                }

            record = {
                "row_idx": idx,
                "scenario_type": row["scenario_type"],
                "confidence_level": row["confidence_level"],
                "audio_class": row["audio_class"],
                "audio_angle": row["audio_angle"],
                "audio_confidence": row["audio_confidence"],
                "ground_truth": row["ground_truth"],
                **result,
            }
            all_results.append(record)

        # Save after each variant
        pd.DataFrame(all_results).to_csv(config.ABLATION_RESULTS, index=False)
        print(f"  [saved] {config.ABLATION_RESULTS}  (skipped={n_skipped}, errors={n_errors})")

    elapsed = time.time() - t_start
    res_df = pd.DataFrame(all_results)
    print(f"\n[ablation] Complete. {len(res_df)} total records in {elapsed:.0f}s")
    print(f"[ablation] Saved: {config.ABLATION_RESULTS}")

    # Quick accuracy summary
    for vid in variants:
        sub = res_df[res_df["variant"] == vid]
        valid = sub[~sub["final_decision"].isin(["ERROR", "UNKNOWN", "PARSE_ERROR"])]
        if len(valid) > 0:
            acc = (valid["final_decision"] == valid["ground_truth"]).mean()
            print(f"  {vid}: acc={acc:.3f} ({len(valid)} valid / {len(sub)} total)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--variants", nargs="+", choices=config.ABLATION_VARIANTS,
        default=None, help="Which variants to run (default: all)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-completed rows")
    parser.add_argument("--limit", type=int, default=None, help="Only first N rows")
    parser.add_argument("--test30", action="store_true", help="Shortcut: --limit 30")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("driveagent").setLevel(logging.DEBUG)

    limit = 30 if args.test30 else args.limit
    run_ablation(variants=args.variants, resume=args.resume, limit=limit)
