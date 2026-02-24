# MARL-LLM Portfolio Management: Equity Strategy Rotation

A thesis-oriented framework for **factor / strategy sleeve rotation** using Multi-Agent
Reinforcement Learning (MARL) and LLM-based market regime classification.

The system allocates capital across broad equity *strategy sleeves* — such as Momentum,
Value, and Quality — rather than individual stocks. Each sleeve is represented by a
factor index (e.g. MSCI or FTSE factor indices) and the agents learn when to tilt
toward each factor based on market regime.

---

## Research Context

Traditional factor investing rotates between equity strategies (Momentum, Value, Quality,
Low-Vol, etc.) using simple heuristics or static rules. This project investigates whether
a **MARL framework with LLM-assisted regime classification** can produce a more adaptive
and explainable factor rotation policy.

Key research questions:
- Can per-sleeve MARL agents learn robust allocation signals from factor return history?
- Does LLM-based regime labelling add signal beyond purely numeric classifiers?
- How does the allocator generalize out-of-sample under walk-forward evaluation?

---

## Architecture

```
Factor Index Levels (CSV)
        │
        ▼
sleeve_returns_builder  ──► data/sleeve_returns.csv   (daily/weekly returns)
        │
        ▼
strategy_feature_builder ──► data/strategy_features.csv
  (rolling returns, vol, drawdown, cross-sleeve correlation)
        │
        ▼
  StrategySleeveEnv  (Gymnasium-style environment)
        │
        ├── StrategyPreferenceAgent (MOMENTUM)
        ├── StrategyPreferenceAgent (VALUE)         ──► alphas ──► MetaAllocator
        └── StrategyPreferenceAgent (QUALITY)
                                                          │
                          RegimeInterpreter ─────────────►│ (regime context)
                          (closed 5-label classifier)     │
                                                          ▼
                                                   sleeve weights w_t
                                                   (simplex + mandate cap)
                                                          │
                                                          ▼
                                               Portfolio NAV / PnL
```

### Components

| Component | Role |
|-----------|------|
| `StrategySleeveEnv` | Simulates sleeve returns with transaction costs and mandate caps |
| `StrategyPreferenceAgent` | Emits a preference signal α ∈ [0,1] per sleeve from numeric features |
| `MetaAllocator` | Converts per-sleeve alphas to valid portfolio weights via softmax + cap |
| `RegimeInterpreter` | Maps rolling numeric metrics to one of 5 closed regime labels |
| `proportional_walk_forward` | Generates train / val / test / holdout splits for out-of-sample evaluation |

---

## Strategy Sleeves

The default sleeves track broad equity factor strategies:

| Sleeve | Description | Example Index |
|--------|-------------|---------------|
| `MOMENTUM` | Recent price-trend following | MSCI World Momentum, FTSE RAFI Momentum |
| `VALUE` | Cheap vs. expensive by fundamentals | MSCI World Value, FTSE RAFI Value |
| `QUALITY` | High-profitability, low-leverage firms | MSCI World Quality, FTSE RAFI Quality |

Additional sleeves (e.g. `LOW_VOL`, `MIN_VAR`, `GROWTH`) can be added by extending the
input CSV and updating `configs/strategy_allocator.yaml`.

---

## Market Regime Labels

The `RegimeInterpreter` classifies each period into one of five closed labels using
rolling numeric metrics (volatility, drawdown, trend, cross-sleeve correlation):

| Label | Interpretation |
|-------|---------------|
| `TRENDING-LOWVOL` | Sustained uptrend with low realized volatility — momentum-friendly |
| `RECOVERY` | Positive trend after a drawdown period — quality / momentum tilt |
| `SIDEWAYS-HIGHCORR` | Low trend, high cross-strategy correlation — diversify equally |
| `STRESS-DRAWDOWN` | Deep drawdown + high volatility — defensive / reduce risk |
| `RISK-OFF-DEFENSIVE` | Residual / ambiguous — default to defensive allocation |

LLM classification (Claude) is optional and off by default (`llm.use_llm: false`). The
deterministic numeric classifier is used for all reproducible research runs.

---

## Project Structure

```
MARL-LLM-PM/
├── src/marl_llm_pm/
│   ├── strategy_allocator/          # Primary thesis pipeline
│   │   ├── environment/             # StrategySleeveEnv
│   │   ├── agents/                  # StrategyPreferenceAgent, collect_preferences
│   │   ├── orchestration/           # MetaAllocator
│   │   ├── llm/                     # RegimeInterpreter (+ LABELS)
│   │   ├── evaluation/              # proportional_walk_forward, SplitWindow
│   │   ├── data/                    # sleeve_returns_builder, SleeveReturnsBuildSpec
│   │   └── features/               # strategy_feature_builder, StrategyFeatureSpec
│   ├── environment/                 # Legacy: PortfolioEnv
│   ├── agents/                      # Legacy: BaseAgent, DummyAgent, AgentCoordinator
│   ├── llm/                         # Legacy: SentimentAnalyzer
│   ├── backtesting/                 # Legacy: Backtester, BacktestResults
│   └── config/                      # ConfigManager
├── configs/
│   ├── strategy_allocator.yaml      # Primary config (sleeves, regime, evaluation)
│   └── default.yaml                 # Legacy asset-pipeline config
├── data/
│   ├── factor_index_levels_example.csv   # Example input (index levels)
│   └── sleeve_returns_test.csv           # Minimal test fixture
├── tests/
├── main.py                          # CLI entry point
└── requirements.txt
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/mygitAN/MARL-LLM-PM
cd MARL-LLM-PM
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation

**Step 1 — Obtain factor index levels.**
Export daily or weekly index levels for each strategy sleeve to a wide CSV:

```
date,MOMENTUM,VALUE,QUALITY
2020-01-06,100.000,100.000,100.000
2020-01-07,100.215,99.841,100.103
...
```

Suitable sources: MSCI factor indices, FTSE RAFI indices, Bloomberg tickers such as
`M1WO000V Index` (MSCI World Value), `M1WO000M Index` (MSCI World Momentum),
`M1WO000Q Index` (MSCI World Quality).

**Step 2 — Convert levels to returns:**

```bash
python main.py build-sleeve-returns \
  --prices-csv data/factor_index_levels_example.csv \
  --out-csv data/sleeve_returns.csv \
  --method simple
```

**Step 3 — Build deterministic features (optional):**

```bash
python main.py build-strategy-features \
  --sleeve-returns-csv data/sleeve_returns.csv \
  --out-csv data/strategy_features.csv
```

Features include rolling returns (5d / 21d / 63d / 126d), rolling volatility (21d / 63d),
rolling max drawdown (252d), and mean pairwise cross-sleeve correlation (63d).

### Running the Strategy Backtest

```bash
# Full history run
python main.py sleeve-backtest --config configs/strategy_allocator.yaml

# Walk-forward evaluation (train / val / test / holdout)
python main.py sleeve-backtest --config configs/strategy_allocator.yaml --walk-forward
```

Results are written to `results/sleeve_backtest_<timestamp>.csv` with per-period NAV,
returns, regime label, and sleeve weights.

---

## Configuration

Edit `configs/strategy_allocator.yaml` to configure the pipeline:

```yaml
environment:
  sleeves: [MOMENTUM, VALUE, QUALITY]   # sleeve names must match CSV columns
  transaction_cost: 0.001               # one-way cost (10 bps)
  max_weight_per_sleeve: 0.70           # hard mandate cap per sleeve
  rebalance_frequency: "weekly"
  steps_per_year: 52

regime:
  labels:
    - "TRENDING-LOWVOL"
    - "STRESS-DRAWDOWN"
    - "RECOVERY"
    - "SIDEWAYS-HIGHCORR"
    - "RISK-OFF-DEFENSIVE"
  lookback_steps: 12                    # rolling window for regime metrics

llm:
  use_llm: false                        # true = Claude; false = deterministic rules
  model: "claude-3-5-sonnet-20241022"
  cache_enabled: true
  cache_dir: ".cache/regimes"

evaluation:
  split_train: 0.60
  split_val: 0.20
  split_test: 0.20
  holdout_months: 12                    # sealed holdout — do not use for tuning

data:
  sleeve_returns_csv: "data/sleeve_returns.csv"
```

---

## Walk-Forward Evaluation

The `proportional_walk_forward` function splits the full history into four non-overlapping
segments to prevent look-ahead bias:

```
|<──── 60% train ────>|<── 20% val ──>|<── 20% test ──>|<── 12mo holdout ──>|
```

- **Train**: used for agent calibration / RL policy training (future work)
- **Val**: hyperparameter tuning (transaction cost, temperature, lookback)
- **Test**: rolling 6-month windows for out-of-sample evaluation
- **Holdout**: sealed until thesis submission — never used for tuning

---

## Key Metrics Reported

| Metric | Description |
|--------|-------------|
| Total return | Cumulative NAV growth over the evaluation period |
| Annualised return | Geometric mean annual return |
| Annualised volatility | Std of period returns × √steps_per_year |
| Sharpe ratio | Annualised return / annualised vol (rf = 0) |
| Max drawdown | Peak-to-trough NAV decline |
| Average sleeve weights | Mean allocation to each strategy sleeve |

---

## Requirements

- Python 3.9+
- pandas >= 1.3
- numpy >= 1.21
- gymnasium >= 0.27
- anthropic >= 0.25 *(only required if `llm.use_llm: true`)*
- pyyaml >= 6.0

---

## Legacy Pipeline

The repository also contains a legacy asset-level pipeline (`PortfolioEnv`,
`AgentCoordinator`, `SentimentAnalyzer`, `Backtester`) that manages individual equity
tickers. This pipeline is preserved for reference but is not the focus of this research.

```bash
# Legacy asset backtest (kept for reference only)
python main.py backtest --assets AAPL GOOGL MSFT --config configs/default.yaml
```

---

## License

MIT
