# Training Arguments Reference

Complete documentation for all command-line arguments in `train.py`.

## Table of Contents

1. [Core Training Arguments](#core-training-arguments)
2. [Reward Parameters](#reward-parameters)
3. [Termination Tuning (H1)](#termination-tuning-h1)
4. [Entropy Control](#entropy-control)
5. [Advanced Learning Parameters](#advanced-learning-parameters)
6. [SMDP Coverage Parameters](#smdp-coverage-parameters)
7. [Simulator Selection](#simulator-selection)
8. [Hypothesis-Specific Arguments](#hypothesis-specific-arguments)
9. [Evaluation & Visualization](#evaluation--visualization)
10. [Debugging & Testing](#debugging--testing)

---

## Core Training Arguments

### `--mode`
Training mode selection.

| Option | Description |
|--------|-------------|
| `hrl` (default) | Hierarchical RL with Option-Critic |
| `flat` | Flat RL without hierarchy |

```bash
python train.py --mode hrl
```

### `--variant`
Model variant shortcut. Overrides `--mode` and sets appropriate defaults.

| Option | Description |
|--------|-------------|
| `baseline` | SMDP hierarchical (same as `--mode hrl`) |
| `h1` | Flat MDP baseline |

```bash
python train.py --variant h1 --episodes 500
```

### `--episodes`
Number of training episodes. Default: `500`

```bash
python train.py --episodes 1000
```

### `--turns`
Maximum turns per episode. Default: `50`

```bash
python train.py --turns 30
```

### `--lr`
Learning rate for all networks. Default: `1e-4`

```bash
python train.py --lr 5e-5
```

### `--gamma`
Discount factor for future rewards. Default: `0.99`

```bash
python train.py --gamma 0.95
```

### `--device`
Compute device for training.

| Option | Description |
|--------|-------------|
| `cpu` (default) | CPU training |
| `cuda` | GPU training (if available) |

```bash
python train.py --device cuda
```

### `--seed`
Random seed for reproducibility. Default: `1`

Use different seeds (1, 2, 3) for statistical significance testing.

```bash
python train.py --seed 42
```

### `--name`
Experiment name for folder naming. Auto-generated if not specified.

```bash
python train.py --name my_experiment
```

### `--experiment-name`
Display name for experiment (shown in logs).

```bash
python train.py --experiment-name "Exp1_LR5e5_Baseline"
```

### `--experiment-type`
Experiment classification.

| Option | Description |
|--------|-------------|
| `minor` (default) | Regular experiments |
| `major` | Significant changes (auto-set for variants) |

### `--compare`
Run both HRL (SMDP) and flat (MDP) back-to-back with identical settings.

```bash
python train.py --compare --episodes 500
```

---

## Reward Parameters

Per paper Section 4.7: R_t = r^eng + r^nov + (augmented components)

### `--reward_mode`
Reward function mode.

| Option | Description |
|--------|-------------|
| `baseline` (default) | R_t = r^eng + r^nov only |
| `augmented` | + responsiveness, transition, conclude, ask |

```bash
python train.py --reward_mode augmented
```

### `--w-engagement`
Engagement reward weight. Default: `1.0`

Formula: r^eng_t = dwell_t × w_engagement

```bash
python train.py --w-engagement 1.5
```

### `--novelty-per-fact`
Novelty reward scale. Default: `1.0`

Formula: r^nov_t = novelty_per_fact × |new_facts|

```bash
python train.py --novelty-per-fact 0.5
```

### `--w-responsiveness`
Responsiveness reward weight. Default: `0.5`

- +w_responsiveness for answering questions
- -0.6×w_responsiveness for deflecting

```bash
python train.py --w-responsiveness 0.8
```

### `--w-conclude`
Conclude bonus weight. Default: `0.4`

Formula: w_conclude × |exhibits_covered|

```bash
python train.py --w-conclude 0.6
```

### `--w-ask`
Question-asking incentive weight. Default: `0.5`

Hybrid reward considering spacing, engagement impact, and response quality.

```bash
python train.py --w-ask 0.3
```

---

## Termination Tuning (H1)

Parameters for Option-Critic termination learning.

### `--termination-reg`
Termination regularization coefficient. Default: `0.01`

Higher values (0.05-0.1) encourage more option switching.

```bash
python train.py --termination-reg 0.05
```

### `--intra-option-threshold`
Threshold for intra-option advantage termination signal. Default: `0.1`

```bash
python train.py --intra-option-threshold 0.2
```

### `--intra-option-weight`
Weight for intra-option termination signal. Default: `0.5`

```bash
python train.py --intra-option-weight 0.7
```

### `--deliberation-cost`
Per-step cost for staying in option (Harb et al. 2018). Default: `0.0`

Try 0.01-0.05 to encourage option switching.

```bash
python train.py --deliberation-cost 0.02
```

### `--max-option-duration`
Maximum steps in an option before forced termination. Default: `None` (disabled)

Try 8 for museum dialogue.

```bash
python train.py --max-option-duration 8
```

---

## Entropy Control

Parameters for exploration via entropy regularization.

### `--entropy-coef`
Initial entropy coefficient. Default: `0.08`

Higher values (0.15-0.25) increase exploration.

```bash
python train.py --entropy-coef 0.15
```

### `--entropy-floor`
Minimum entropy coefficient. Default: `0.02`

```bash
python train.py --entropy-floor 0.05
```

### `--entropy-decay-start`
Episode to start entropy decay. Default: `0`

Try 100 to allow initial exploration.

```bash
python train.py --entropy-decay-start 100
```

### `--entropy-decay-end`
Episode to finish entropy decay. Default: same as `--episodes`

```bash
python train.py --entropy-decay-end 400
```

### `--adaptive-entropy`
Enable OCI-aware adaptive entropy boost when collapse detected.

```bash
python train.py --adaptive-entropy
```

### `--adaptive-entropy-threshold`
OCI threshold for adaptive entropy boost. Default: `2.5`

```bash
python train.py --adaptive-entropy --adaptive-entropy-threshold 3.0
```

### `--adaptive-entropy-multiplier`
Multiplier for entropy boost when collapse detected. Default: `1.5`

```bash
python train.py --adaptive-entropy --adaptive-entropy-multiplier 2.0
```

---

## Advanced Learning Parameters

### `--lr-intra-option`
Separate learning rate for intra-option policies. Default: same as `--lr`

Allows faster learning for subactions.

```bash
python train.py --lr-intra-option 2e-4
```

### `--entropy-coef-option`
Entropy coefficient for option policy only. Default: same as `--entropy-coef`

```bash
python train.py --entropy-coef-option 0.1
```

### `--entropy-coef-intra`
Entropy coefficient for intra-option policies. Default: same as `--entropy-coef`

```bash
python train.py --entropy-coef-intra 0.05
```

### `--diversity-reward-coef`
Diversity reward coefficient (Kamat & Precup 2020). Default: `0.0`

```bash
python train.py --diversity-reward-coef 0.1
```

### `--value-loss-coef`
Value loss coefficient. Default: `0.5`

Balances policy vs value learning.

```bash
python train.py --value-loss-coef 0.3
```

### `--target-update-interval`
Episodes between target network updates. Default: `20`

```bash
python train.py --target-update-interval 10
```

### `--value-clip`
Value clipping range: clip values to [-value-clip, value-clip]. Default: `10.0`

Use 50.0 for high-return episodes.

```bash
python train.py --value-clip 50.0
```

---

## SMDP Coverage Parameters

Reward structure parameters for exhibit coverage.

### `--exhaustion-penalty`
Penalty for Explain actions at exhausted exhibits. Default: `-0.5`

Try -2.0 for stronger signal.

```bash
python train.py --exhaustion-penalty -2.0
```

### `--transition-bonus`
Immediate bonus for successful exhibit transitions. Default: `0.0`

Try 1.5 for transition incentive.

```bash
python train.py --transition-bonus 1.5
```

### `--zero-engagement-exhausted`
Zero engagement reward for Explain at exhausted exhibits.

Creates Q-value separation between options.

```bash
python train.py --zero-engagement-exhausted
```

### `--beta-supervision-weight`
Beta supervision weight for termination guidance. Default: `0.0`

- 0.0 = pure Option-Critic
- 0.5-1.0 = heuristic guidance

```bash
python train.py --beta-supervision-weight 0.5
```

---

## Simulator Selection

### `--simulator`
Visitor simulator backend.

| Option | Description |
|--------|-------------|
| `sim8` (default) | Lightweight, template-based |
| `sim8_original` | Full neural (T5 + VAE) |
| `state_machine` | Literature-grounded state machine |

```bash
python train.py --simulator state_machine
```

---

## Hypothesis-Specific Arguments

### `--termination`
Termination strategy (H5).

| Option | Description |
|--------|-------------|
| `learned` (default) | Option-Critic learned termination |
| `fixed-3` | Always terminate after 3 turns |
| `threshold` | Terminate when dwell < 0.5 |

```bash
python train.py --termination fixed-3
```

### `--state-representation`
State representation variant (H4).

| Option | Description |
|--------|-------------|
| `dialoguebert` (default) | 149-d full representation |
| `dialogue_act` | 23-d compact representation |

```bash
python train.py --state-representation dialogue_act
```

### `--option-granularity`
Option granularity (H6).

| Option | Description |
|--------|-------------|
| `medium` (default) | 4 options (function-based) |
| `coarse` | 2 options (Explain/Engage) |
| `coarse_3opt` | 3 options (deprecated) |
| `coarse_4opt` | 4 options (reward-aligned) |

```bash
python train.py --option-granularity coarse
```

---

## Evaluation & Visualization

### `--map-interval`
Save map visualization every N episodes. Default: `50`

Set to 0 to disable.

```bash
python train.py --map-interval 100
```

### `--save-map-frames`
Save map snapshots at EVERY turn (all episodes).

```bash
python train.py --save-map-frames
```

### `--live-map-display`
Show live map windows during training.

```bash
python train.py --live-map-display
```

---

## Debugging & Testing

### `--verbose`
Enable verbose output mode.

```bash
python train.py --verbose
```

### `--show-prompts`
Show LLM prompts during training.

```bash
python train.py --show-prompts
```

### `--force-option`
Force agent to always choose this option (testing only).

| Options | `Explain`, `AskQuestion`, `OfferTransition`, `Conclude` |

```bash
python train.py --force-option Explain --episodes 50
```

### `--force-subaction`
Force agent to always choose this subaction (testing only).

```bash
python train.py --force-subaction ExplainNewFact --episodes 50
```

### `--enable-live-monitor`
Enable live training monitor with turn-by-turn visualization.

```bash
python train.py --enable-live-monitor
```

---

## Example Configurations

### Quick Development Test

```bash
python train.py --episodes 20 --name quick_test --verbose
```

### Standard Training Run

```bash
python train.py --episodes 500 --seed 1 --name standard_run
```

### High-Quality Research Run

```bash
python train.py --episodes 1000 --seed 1 --device cuda \
    --simulator state_machine --name research_run
```

### Hyperparameter Sweep

```bash
# Learning rate sweep
for lr in 1e-5 5e-5 1e-4 5e-4; do
    python train.py --episodes 500 --lr $lr --name "lr_${lr}"
done
```

### Entropy Exploration

```bash
python train.py --episodes 500 \
    --entropy-coef 0.2 \
    --entropy-decay-start 100 \
    --entropy-floor 0.05 \
    --name entropy_exploration
```

### Termination Tuning

```bash
python train.py --episodes 500 \
    --termination-reg 0.05 \
    --deliberation-cost 0.02 \
    --max-option-duration 8 \
    --name termination_tuning
```

### Multiple Seeds for Significance

```bash
for seed in 1 2 3 4 5; do
    python train.py --episodes 1000 --seed $seed --name "exp_seed${seed}"
done
```
