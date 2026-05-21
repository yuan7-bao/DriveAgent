"""
Dataset Generator — Controlled golden dataset for ablation evaluation.

Structure:
  5 scenario types × 3 confidence levels × 20 samples = 300 total

Scenarios:
  S01  Blind spot       — emergency vehicle in blind spot, vision clear    → varies by conf
  S02  Occlusion        — crash/squeal behind large vehicle, LiDAR blocked → varies by conf
  S03  False positive   — benign audio (music/alarm), all sensors clear    → always SAFE
  S04  Multi-hazard     — two simultaneous audio threats, complex scene    → varies by conf
  S05  Night / rain     — degraded sensors, moderate audio signal          → varies by conf

Confidence bands:
  low   0.20–0.42   system should not over-react to uncertain signals
  mid   0.45–0.69   ambiguous boundary zone
  high  0.70–0.96   system should act decisively on strong signals

Usage:
  python 1_generate_dataset.py              # default seed=42
  python 1_generate_dataset.py --seed 123   # reproducibility / multi-seed comparison
"""

import argparse
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

DEFAULT_SEED = 42
SAMPLES_PER_BAND = 20

# ── Scenario Specifications ─────────────────────────────────────────────────

SCENARIOS = {
    "S01_Blind_Spot": {
        "vehicle_expert_outputs": [
            "LiDAR detects fixed installations and vegetation. No dynamic objects detected in the immediate forward vicinity.",
            "Forward point cloud clear. Static environment only. No moving obstacles identified.",
            "Camera: clear road ahead. LiDAR: no dynamic targets within 30m.",
        ],
        "environments": [
            "Multi-lane urban road. Forward visibility unobstructed. No pedestrians or vehicles in front camera.",
            "Suburban road with light traffic. Ego vehicle at traffic light. Forward sensor range fully clear.",
            "Highway on-ramp. Clear ahead. No merge conflicts detected by primary sensors.",
        ],
        "audio_classes": ["ambulance_siren", "police_siren", "fire_truck_horn"],
        "audio_angles_low": [135, 150, -135, -150],
        "audio_angles_high": [135, 150, -135, -150],
        "conf_bands": {"low": (0.20, 0.39), "mid": (0.50, 0.65), "high": (0.80, 0.96)},
        "ground_truth_by_conf": {"low": "SAFE", "mid": "CAUTION", "high": "HAZARD"},
    },
    "S02_Occlusion": {
        "vehicle_expert_outputs": [
            "Large four-wheel vehicle detected at x=4m. Forward point cloud heavily occluded beyond 4m.",
            "Bus at x=3.5m blocking forward LiDAR. No data available beyond obstruction.",
            "Large truck detected close ahead. Full forward occlusion confirmed by sensor fusion.",
        ],
        "environments": [
            "Urban intersection. Forward visibility 100% blocked by large truck.",
            "City road. Ego vehicle queued behind a bus. No camera visibility beyond first vehicle.",
            "Narrow street. Large delivery vehicle immediately ahead. Sensor occlusion zone active.",
        ],
        "audio_classes": ["heavy_brake_squeal_and_crash", "collision_impact", "emergency_brake_squeal"],
        "audio_angles_low": [0, 10, -10],
        "audio_angles_high": [0, 5, -5],
        "conf_bands": {"low": (0.22, 0.40), "mid": (0.52, 0.68), "high": (0.82, 0.95)},
        "ground_truth_by_conf": {"low": "CAUTION", "mid": "HAZARD", "high": "HAZARD"},
    },
    "S03_False_Positive": {
        "vehicle_expert_outputs": [
            "Civilian vehicle detected at x=8m, left lane. Stable trajectory. No threat.",
            "Multi-lane road. All vehicles maintaining lane discipline. No anomalies.",
            "Clear road. Pedestrians on sidewalk only. All sensor channels nominal.",
        ],
        "environments": [
            "Multi-lane road with normal traffic flow. Weather clear.",
            "Suburban road. Light traffic. No intersections nearby.",
            "Highway. Sparse traffic. All vehicles in correct lanes.",
        ],
        "audio_classes": [
            "loud_music_with_siren_like_frequencies",
            "car_alarm_stationary",
            "tv_broadcast_siren_sound",
        ],
        "audio_angles_low": [-90, 90, 180],
        "audio_angles_high": [-90, 90, 180],
        "conf_bands": {"low": (0.20, 0.35), "mid": (0.38, 0.49), "high": (0.50, 0.62)},
        "ground_truth_by_conf": {"low": "SAFE", "mid": "SAFE", "high": "SAFE"},
    },
    "S04_Multi_Hazard": {
        "vehicle_expert_outputs": [
            "LiDAR clear forward. Possible moving object at far range, right side, partially occluded.",
            "Camera: vehicles merging from right. LiDAR confidence low due to rain artifacts.",
            "Multiple vehicles detected at various distances. Forward path nominally clear.",
        ],
        "environments": [
            "Complex urban intersection. Multiple lanes merging. Heavy pedestrian activity.",
            "Rush hour highway. Dense traffic. Frequent lane changes detected.",
            "Urban road with construction zone. Reduced lanes. Workers possibly present.",
        ],
        "audio_classes": [
            "ambulance_siren+horn_blast",
            "police_siren+brake_squeal",
            "fire_truck+collision_impact",
        ],
        "audio_angles_low": [45, -45],
        "audio_angles_high": [45, -45],
        "conf_bands": {"low": (0.45, 0.60), "mid": (0.65, 0.78), "high": (0.82, 0.96)},
        "ground_truth_by_conf": {"low": "CAUTION", "mid": "HAZARD", "high": "HAZARD"},
    },
    "S05_Night_Rain": {
        "vehicle_expert_outputs": [
            "Camera degraded due to low light and rain droplets. LiDAR range reduced to 15m.",
            "Night conditions. Headlight glare causing false detections. LiDAR range 12m.",
            "Heavy rain. Camera confidence low. LiDAR point cloud noisy. Sensor fusion uncertainty high.",
        ],
        "environments": [
            "Night driving. Road wet. Reduced visibility. Streetlights sparse.",
            "Rainy highway. Spray from other vehicles reducing camera clarity.",
            "Urban night scene. Puddles causing LiDAR ground return anomalies.",
        ],
        "audio_classes": ["distant_ambulance_siren", "faint_horn_blast", "distant_police_siren"],
        "audio_angles_low": [0, 45, -45, 90],
        "audio_angles_high": [0, 45, -45, 90],
        "conf_bands": {"low": (0.25, 0.42), "mid": (0.48, 0.65), "high": (0.70, 0.88)},
        "ground_truth_by_conf": {"low": "SAFE", "mid": "CAUTION", "high": "CAUTION"},
    },
}


def _rand_conf(band: tuple[float, float]) -> float:
    return round(random.uniform(band[0], band[1]), 2)


def generate_golden_dataset(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate the 300-sample golden dataset and save to CSV."""
    random.seed(seed)
    print(f"[dataset] Generating golden dataset (seed={seed})...")

    rows = []
    for scenario_key, spec in SCENARIOS.items():
        for conf_level, conf_band in spec["conf_bands"].items():
            gt = spec["ground_truth_by_conf"][conf_level]
            angle_pool = spec["audio_angles_high"] if conf_level == "high" else spec["audio_angles_low"]

            for _ in range(SAMPLES_PER_BAND):
                rows.append({
                    "scenario_type": scenario_key,
                    "confidence_level": conf_level,
                    "vehicle_expert_output": random.choice(spec["vehicle_expert_outputs"]),
                    "aggregated_driving_environment": random.choice(spec["environments"]),
                    "audio_class": random.choice(spec["audio_classes"]),
                    "audio_angle": f"{random.choice(angle_pool)} degrees",
                    "audio_confidence": _rand_conf(conf_band),
                    "ground_truth": gt,
                })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(config.DATASET_CSV, index=False)

    # Print summary
    print(f"[dataset] Saved: {config.DATASET_CSV}")
    print(f"[dataset] Total samples: {len(df)}")
    for skey in SCENARIOS:
        sub = df[df["scenario_type"] == skey]
        counts = sub["ground_truth"].value_counts()
        print(
            f"  {skey:<28} n={len(sub):>3}  "
            f"HAZARD={counts.get('HAZARD', 0):>2}  "
            f"CAUTION={counts.get('CAUTION', 0):>2}  "
            f"SAFE={counts.get('SAFE', 0):>2}"
        )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate golden evaluation dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    args = parser.parse_args()
    generate_golden_dataset(seed=args.seed)
