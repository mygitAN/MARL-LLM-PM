# Multi-Agent Reinforcement Learning with LLM for Portfolio Management

A framework for building intelligent portfolio management systems that allocate across **strategy sleeves** rather than individual assets, combining:

- **Multi-Agent RL** — one agent per strategy sleeve, each emitting a preference signal α ∈ [0,1]
- **Meta-Allocator** — converts preference signals to portfolio weights via temperature-scaled softmax with mandate-style caps
- **LLM Regime Interpreter** — closed-label market regime classifier (deterministic rules + optional Claude)
- **Walk-Forward Evaluation** — proportional train/val/test/holdout splits with sealed holdout

---

## Architecture

The primary pipeline operates on **strategy sleeves** (pre-computed return streams such as factor indices or backtested strategies), not on individual securities.

```
Sleeve Returns CSV (MOMENTUM, VALUE, QUALITY, ...)
    ↓
StrategySleeveEnv  — PnL / cost accounting only
    ↓
Rolling window metrics  →  RegimeInterpreter  →  regime label (closed set)
    ↓
┌── StrategyPreferenceAgent (MOMENTUM) ──┐
├── StrategyPreferenceAgent (VALUE)    ──┤  →  MetaAllocator  →  w_t ∈ simplex (capped)
└── StrategyPreferenceAgent (QUALITY)  ──┘
    ↓
StrategySleeveEnv.step(w_t)  →  reward, NAV
    ↓
Walk-Forward Evaluation  →  Performance Metrics
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **StrategySleeveEnv** | Simulate PnL and transaction costs; no feature engineering |
| **StrategyPreferenceAgent** | Emit α ∈ [0,1] from sleeve-level signal; drop-in RL target |
| **MetaAllocator** | Temperature-scaled softmax → simplex projection → per-sleeve cap |
| **RegimeInterpreter** | Deterministic 5-label classifier; optional LLM at temperature=0 |
| **proportional_walk_forward** | Sealed holdout + rolling test windows |

---

## Project Structure

```
MARL-LLM-PM/
├── src/marl_llm_pm/
│   ├── strategy_allocator/        # Primary pipeline
│   │   ├── environment/           # StrategySleeveEnv
│   │   ├── agents/                # StrategyPreferenceAgent, collect_preferences
│   │   ├── orchestration/         # MetaAllocator
│   │   ├── llm/                   # RegimeInterpreter
│   │   └── evaluation/            # proportional_walk_forward
│   ├── thesis/                    # Reference implementation (same structure)
│   ├── environment/               # Legacy: PortfolioEnv (Gymnasium, asset-level)
│   ├── agents/                    # Legacy: BaseAgent, DummyAgent, AgentCoordinator
│   ├── llm/                       # Legacy: SentimentAnalyzer
│   ├── backtesting/               # Legacy: Backtester, BacktestResults
│   ├── config/                    # ConfigManager
│   └── constants.py               # Shared constants
├── configs/
│   ├── strategy_allocator.yaml    # Primary config
│   └── default.yaml               # Legacy asset pipeline config
├── data/
│   └── sleeve_returns.csv         # Weekly sleeve returns (MOMENTUM, VALUE, QUALITY)
├── tests/
├── main.py                        # CLI entry point
└── requirements.txt
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/mygitAN/MARL-LLM-PM.git
cd MARL-LLM-PM
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set the Anthropic API key only if using the LLM regime interpreter:

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Quick Start

**Run strategy-sleeve backtest (primary):**
```bash
python main.py sleeve-backtest --config configs/strategy_allocator.yaml
```

**Run with walk-forward splits (train/val/test/holdout):**
```bash
python main.py sleeve-backtest --config configs/strategy_allocator.yaml --walk-forward
```

**Run legacy asset-level backtest:**
```bash
python main.py backtest --assets AAPL GOOGL MSFT --start-date 2023-01-01 --end-date 2024-12-31
```

**Run test suite:**
```bash
python main.py test --coverage
```

---

## Usage Examples

### Strategy-Sleeve Backtest

```python
import pandas as pd
from marl_llm_pm.strategy_allocator.environment import StrategySleeveEnv
from marl_llm_pm.strategy_allocator.agents import StrategyPreferenceAgent, collect_preferences
from marl_llm_pm.strategy_allocator.orchestration import MetaAllocator
from marl_llm_pm.strategy_allocator.llm import RegimeInterpreter

sleeves = ['MOMENTUM', 'VALUE', 'QUALITY']

# Load pre-computed sleeve returns
sleeve_returns = pd.read_csv('data/sleeve_returns.csv', index_col=0, parse_dates=True)

# Environment: simulates PnL and transaction costs
env = StrategySleeveEnv(
    sleeve_names=sleeves,
    transaction_cost=0.001,
    max_weight_per_sleeve=0.70,   # mandate cap per sleeve
    initial_value=100_000.0,
)
env.set_sleeve_returns(sleeve_returns)

# One preference agent per sleeve
agents = [StrategyPreferenceAgent(s) for s in sleeves]

# Meta-allocator: alphas → constrained weights via temperature-scaled softmax
allocator = MetaAllocator(sleeves, cap=0.70, temperature=1.0)

# Regime interpreter: deterministic 5-label classifier (optional LLM at temperature=0)
regime = RegimeInterpreter(cache_dir='.cache/regimes', use_cache=True)

env.reset()
for t in range(len(sleeve_returns)):
    window = sleeve_returns.iloc[max(0, t - 12): t]

    avg = window.mean(axis=1)
    metrics = {
        'vol':      float(avg.std())            if len(avg) > 2 else 0.0,
        'trend':    float(avg.mean())           if len(avg) > 2 else 0.0,
        'drawdown': float(avg.cumsum().min())   if len(avg) > 2 else 0.0,
        'corr':     float(window.corr().mean().mean()) if len(window) > 3 else 0.0,
    }
    reg = regime.classify(key=str(sleeve_returns.index[t].date()), metrics=metrics)

    obs = {
        'regime_label': reg.label,
        'sleeve_features': {
            s: {'signal': float(window[s].mean()) if len(window) > 2 else 0.0}
            for s in sleeves
        },
        'global_features': metrics,
        'prev_weights': env.w.copy(),
    }

    alphas = collect_preferences(agents, obs)   # {sleeve: alpha ∈ [0,1]}
    w = allocator.allocate(alphas, obs)          # weights on simplex, capped
    reward, info, done = env.step(w)

    if done:
        break

print(f"Final NAV: £{info.portfolio_value:,.2f}")
```

### Walk-Forward Evaluation

```python
from marl_llm_pm.strategy_allocator.evaluation import proportional_walk_forward

train_df, val_df, test_windows, holdout_df = proportional_walk_forward(
    sleeve_returns,
    split_train=0.60,
    split_val=0.20,
    split_test=0.20,
    test_interval_months=6,
    holdout_months=12,      # sealed — do not touch until final evaluation
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | "
      f"Test windows: {len(test_windows)} | Holdout: {len(holdout_df)}")
```

### Implementing a Custom Preference Agent

The `StrategyPreferenceAgent` interface is designed as a drop-in target for a learned RL policy:

```python
from marl_llm_pm.strategy_allocator.agents import PreferenceOutput
from typing import Dict

class MyRLAgent:
    def __init__(self, sleeve_name: str):
        self.sleeve_name = sleeve_name

    def get_preference(self, obs: Dict) -> PreferenceOutput:
        feats = obs.get('sleeve_features', {}).get(self.sleeve_name, {})
        alpha = my_policy(feats)   # learned policy returning float in [0, 1]
        return PreferenceOutput(sleeve=self.sleeve_name, alpha=alpha)
```

---

## Regime Labels

The `RegimeInterpreter` uses a fixed closed vocabulary of 5 labels, classified from rolling numeric metrics (vol, drawdown, trend, correlation). The LLM path is optional and temperature-locked to 0.

| Label | Trigger conditions |
|-------|--------------------|
| `TRENDING-LOWVOL` | trend > 8%, vol < 15% |
| `STRESS-DRAWDOWN` | drawdown < −12%, vol > 20% |
| `RECOVERY` | trend > 0%, drawdown > −5%, vol < 20% |
| `SIDEWAYS-HIGHCORR` | corr > 60%, abs(trend) < 3% |
| `RISK-OFF-DEFENSIVE` | default (none of the above) |

---

## Configuration

`configs/strategy_allocator.yaml`:

```yaml
environment:
  sleeves: [MOMENTUM, VALUE, QUALITY]
  transaction_cost: 0.001
  max_weight_per_sleeve: 0.70
  steps_per_year: 52          # weekly rebalancing

regime:
  labels:
    - TRENDING-LOWVOL
    - STRESS-DRAWDOWN
    - RECOVERY
    - SIDEWAYS-HIGHCORR
    - RISK-OFF-DEFENSIVE
  lookback_steps: 12

evaluation:
  split_train: 0.60
  split_val:   0.20
  split_test:  0.20
  test_interval_months: 6
  holdout_months: 12

llm:
  use_llm: false              # true = Claude at temperature=0
  cache_enabled: true
  cache_dir: .cache/regimes

data:
  sleeve_returns_csv: data/sleeve_returns.csv
```

---

## Performance Metrics

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | `(Ann. Return − rf) / Ann. Vol`; rf = 0 by default |
| **Sortino Ratio** | Like Sharpe, but denominator uses downside volatility only |
| **Max Drawdown** | Maximum peak-to-trough decline |
| **Calmar Ratio** | `Ann. Return / abs(Max Drawdown)` |
| **Alpha / Beta** | Excess return and market correlation vs. benchmark |

---

## Testing

```bash
python main.py test
python main.py test --coverage
pytest tests/ -v
```

---

## Legacy Asset Pipeline

The asset-level pipeline (`PortfolioEnv`, `AgentCoordinator`, `SentimentAnalyzer`, `Backtester`) remains available for individual-security experiments:

```bash
python main.py backtest --assets AAPL GOOGL MSFT
python main.py analyze --assets AAPL GOOGL --date 2024-01-15
```

---

## Requirements

- Python 3.9+
- numpy >= 1.21.0
- pandas >= 1.3.0
- gymnasium >= 0.27.0
- anthropic >= 0.25.0
- yfinance >= 0.2.0
- pyyaml >= 6.0

## License

MIT License
