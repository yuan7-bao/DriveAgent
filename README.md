# DriveAgent: Audio-Grounded Multi-Agent Fusion for Blind-Spot Aware Autonomous Driving

> 基于音频增强的多智能体融合盲区感知推理方法

## Overview

DriveAgent is a multi-agent LLM framework that enhances autonomous vehicle decision-making by incorporating **audio perception** as a complementary modality to vision and LiDAR — specifically targeting **blind spot** and **sensor occlusion** scenarios where visual sensors alone are insufficient.

### Core Insight

Autonomous vehicles over-rely on visual/LiDAR sensors, creating a "visual dominance bias." Emergency sirens, brake squeals, and collision sounds carry critical safety information that arrives **before** visual confirmation — especially from blind spots, behind occluding vehicles, and in degraded weather conditions.

### Architecture

```
Raw Audio Features ──→ [Audio Agent] ──→ Spatial-auditory description
                                               │
Vision + LiDAR ──────→ [Draft Agent] ←─────────┘
                           │
                    Draft Decision
                      ┌────┴────┐
                      ▼         ▼
              [Safety Critic] [Efficiency Critic]    ← GPT-4o (Teacher)
                      │         │
                      └────┬────┘
                           ▼
                    [Refine Agent]  × K samples
                           │
                    Self-Consistency Voting
                           │
                      Final Decision
```

### Ablation Variants

| Variant | Components | Purpose |
|---------|-----------|---------|
| **M0** | Vision + LiDAR only | Baseline |
| **M1** | M0 + Audio Agent | Audio contribution |
| **M2** | M1 + Safety Critic | Safety-only critique |
| **M3** | M1 + Efficiency Critic | Efficiency-only critique |
| **M4** | M1 + Dual Critic | Balanced critique |
| **M5** | M4 + Self-Consistency | Full system |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp config_local_example.py config_local.py
# Edit config_local.py with your API keys
```

### 3. Generate dataset

```bash
python 1_generate_dataset.py
```

### 4. Run ablation (test with 30 samples first)

```bash
python 2_run_ablation.py --test30          # quick test
python 2_run_ablation.py                   # full 300 samples
python 2_run_ablation.py --resume --variants M5  # resume specific variant
```

### 5. Evaluate metrics

```bash
python 3_evaluate_metrics.py
python 3_evaluate_metrics.py --bootstrap 1000  # with confidence intervals
```

### 6. Generate figures

```bash
python 4_plot_results.py           # all variants
python 4_plot_results.py --paper   # main-text subset
```

## Project Structure

```
driveagent/
├── config.py                  # Global configuration (paths, models, parameters)
├── config_local.py            # API keys (git-ignored)
├── config_local_example.py    # Template for config_local.py
├── fusion_prompts.json        # All agent prompt templates
├── 1_generate_dataset.py      # Golden dataset generator (300 samples)
├── 2_run_ablation.py          # Ablation experiment runner
├── 3_evaluate_metrics.py      # Metrics computation and analysis
├── 4_plot_results.py          # Publication figure generator
├── requirements.txt           # Python dependencies
├── agents/
│   ├── __init__.py
│   └── fusion_agents.py       # LLMClient + FusionPipeline (M0–M5)
├── data/
│   └── golden_dataset.csv     # Generated evaluation dataset
├── results/
│   ├── ablation_results.csv   # Raw experiment results
│   └── metrics_summary.json   # Computed metrics
└── figures/
    ├── fig1_ablation_bar.png
    ├── fig2_fnr_fpr.png
    ├── fig3_scenario_heatmap.png
    ├── fig4_confidence_band.png
    └── fig5_latency.png
```

## Evaluation Metrics

- **Accuracy**: Overall classification correctness
- **F1 (Macro/Weighted)**: Class-balanced performance
- **FNR (HAZARD)**: False Negative Rate — missed real hazards (safety-critical)
- **FPR (HAZARD)**: False Positive Rate — phantom braking (efficiency)
- **Latency**: Average inference time per decision

## Models

- **Student Model**: DeepSeek-V3 (Audio Agent, Draft Agent, Refine Agent)
- **Teacher Model**: GPT-4o (Safety Critic, Efficiency Critic)

## License

Academic use only. Part of undergraduate thesis research.
