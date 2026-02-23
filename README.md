# Multi-Agent Reinforcement Learning with LLM for Portfolio Management

A sophisticated framework for building intelligent portfolio management systems combining:
- **Multi-Agent Reinforcement Learning** for distributed decision-making across assets
- **Large Language Models** (Claude) for market sentiment analysis with daily caching
- **Gymnasium-compatible Environment** for realistic portfolio operations
- **Advanced Backtesting** with comprehensive performance metrics (Sharpe, Drawdown, etc.)

## 🎯 Features

### Environment: `PortfolioEnv`
- **Gymnasium-compliant** interface for standard RL workflows
- Multi-asset portfolio management (customizable assets)
- Realistic transaction costs and rebalancing mechanics
- Observation space includes price returns, sentiment signals, current weights, and portfolio metrics

### Agents: `BaseAgent` & `AgentCoordinator`
- **Abstract base agent interface** for implementing custom decision-making strategies
- **DummyAgent** for baseline equal-weight comparison
- **AgentCoordinator** aggregates decisions from multiple agents using:
  - Mean aggregation
  - Median aggregation
  - Weighted aggregation with configurable agent weights
- Ensemble diversity metrics for analyzing agent disagreement

### LLM Layer: `SentimentAnalyzer`
- **Claude-powered sentiment analysis** using Anthropic API
- **Daily caching** to minimize API calls and costs
- Analyzes market sentiment on configurable 1-N scale
- Batch analysis for multiple assets
- Automatic cache cleanup for old entries

### Backtesting: `Backtester` & `BacktestResults`
Performance metrics include:
- **Return metrics**: Total return, annualized return
- **Risk metrics**: Volatility, maximum drawdown, Calmar ratio
- **Risk-adjusted returns**: Sharpe ratio, Sortino ratio
- **Relative performance**: Alpha and Beta vs. benchmark
- **Trade metrics**: Win rate, transaction cost analysis

### Configuration: `ConfigManager`
- Centralized YAML configuration management
- Pre-defined config sections for environment, agents, LLM, backtesting, and training
- Easy override of hyperparameters
- Default configuration in `configs/default.yaml`

## 📦 Project Structure

```
MARL-LLM-PM/
├── src/marl_llm_pm/           # Main package
│   ├── environment/           # PortfolioEnv implementation
│   ├── agents/               # Agent base classes and coordinator
│   ├── llm/                  # Claude sentiment analyzer
│   ├── backtesting/          # Backtesting engine
│   └── config/               # Configuration management
├── tests/                     # Pytest test suite
├── configs/                   # Configuration files
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🚀 Getting Started

### Installation

1. **Clone the repository:**
   ```bash
   cd MARL-LLM-PM
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key:**
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

### Quick Start

**Run a backtest with default configuration:**
```bash
python main.py backtest
```

**Backtest with custom assets and dates:**
```bash
python main.py backtest --assets AAPL GOOGL MSFT NVDA --start-date 2023-06-01 --end-date 2024-12-31
```

**Analyze market sentiment (requires API key):**
```bash
python main.py analyze --assets AAPL GOOGL --date 2024-01-15
```

**Run test suite:**
```bash
python main.py test --coverage
```

## 🎓 Thesis: Strategy Sleeve Allocator (Factor Rotation)

If you're using this repo for **strategy sleeve rotation** (e.g., Momentum / Value / Quality sleeves),
the `strategy_allocator` package expects a **wide CSV of sleeve returns** like:

- `data/sleeve_returns.csv`

### 1) Get sleeve index *levels*

Start with provider **factor/strategy indices** (recommended for the proposal stage).
Export daily index levels to a wide CSV shaped like `data/factor_index_levels_example.csv`:

```text
date,MOMENTUM,VALUE,QUALITY
2024-01-02,100.0,100.0,100.0
...
```

### 2) Convert levels → returns

```bash
python main.py build-sleeve-returns \
  --prices-csv data/factor_index_levels_example.csv \
  --out-csv data/sleeve_returns.csv \
  --method simple
```

### 3) Optional: build deterministic features

```bash
python main.py build-strategy-features \
  --sleeve-returns-csv data/sleeve_returns.csv \
  --out-csv data/strategy_features.csv
```

### 4) Run sleeve backtest

```bash
python main.py sleeve-backtest --config configs/strategy_allocator.yaml
```

## 📖 Usage Examples

### Using PortfolioEnv Directly

```python
import pandas as pd
from marl_llm_pm import PortfolioEnv

# Create environment
env = PortfolioEnv(
    asset_names=['AAPL', 'GOOGL', 'MSFT'],
    initial_portfolio_value=100000.0,
    max_steps=252,  # 1 trading year
)

# Set market data
price_data = pd.DataFrame(...)  # Your price data
env.set_market_data(price_data)

# Reset and interact
obs, info = env.reset()
for step in range(252):
    action = agent.get_action(obs)  # Your agent decision
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Using AgentCoordinator

```python
from marl_llm_pm import DummyAgent, AgentCoordinator

# Create agents
agents = [
    DummyAgent(f"agent_{i}", n_assets=3)
    for i in range(3)
]

# Create coordinator with mean aggregation
coordinator = AgentCoordinator(
    agents,
    aggregation_method="mean",
    weights=[0.33, 0.33, 0.34]  # Agent importance weights
)

# Get aggregated decision
observation = env.reset()[0]
portfolio_weights = coordinator.get_actions(observation)
```

### Using SentimentAnalyzer

```python
from marl_llm_pm import SentimentAnalyzer

analyzer = SentimentAnalyzer(
    cache_dir=".cache/sentiment",
    model="claude-3-5-sonnet-20241022",
)

# Analyze sentiment (cached daily)
sentiments = analyzer.analyze_sentiment(
    assets=['AAPL', 'GOOGL', 'MSFT'],
    date='2024-01-15'
)
# Returns: {'AAPL': 0.7, 'GOOGL': 0.5, 'MSFT': 0.8} (0-1 scale)
```

### Running Backtests

```python
from marl_llm_pm import Backtester
import pandas as pd

backtester = Backtester(
    initial_capital=100000.0,
    transaction_cost=0.001,
)

# Price data with datetime index
price_data = pd.DataFrame(...)

# Weight calculator function
def get_weights(observation):
    return coordinator.get_actions(observation)

# Run backtest
results = backtester.run(price_data, get_weights)

# Print summary
print(results.summary())

# Access metrics
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
environment:
  assets: [AAPL, GOOGL, MSFT, NVDA, TSLA]
  initial_portfolio_value: 100000.0
  transaction_cost: 0.001

agents:
  n_agents: 3
  aggregation_method: "mean"

llm:
  enabled: true
  model: "claude-3-5-sonnet-20241022"
  cache_enabled: true
  
backtesting:
  initial_capital: 100000.0
  transaction_cost: 0.001
```

## 🧪 Testing

Run the test suite:

```bash
# Basic tests
python main.py test

# With coverage report
python main.py test --coverage

# Direct pytest
pytest tests/ -v
```

Tests cover:
- Environment correctness (reset, step, observation space)
- Action normalization and constraint satisfaction
- Transaction cost application
- Portfolio value evolution
- Agent interface and coordination
- Weight aggregation methods

## 🏗️ Architecture Overview

### Information Flow

```
Market Data (yfinance)
    ↓
PortfolioEnv (Observation)
    ↓
┌── Agent 1 ──┐
├── Agent 2 ──┤→ AgentCoordinator → Portfolio Weights
└── Agent 3 ──┘
    ↓
[Optional: SentimentAnalyzer → Sentiment Scores]
    ↓
PortfolioEnv (Step) → Reward Calculation
    ↓
Backtester → Performance Metrics
```

### Component Responsibilities

| Component | Responsibility |
|-----------|-----------------|
| **PortfolioEnv** | Simulate portfolio operations with realistic constraints |
| **BaseAgent** | Define decision-making interface |
| **AgentCoordinator** | Aggregate multi-agent decisions |
| **SentimentAnalyzer** | Provide LLM-based market insights |
| **Backtester** | Evaluate strategy performance |
| **ConfigManager** | Centralize configuration |

## 📊 Performance Metrics Explained

- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
  - Formula: (Annual Return - Risk-Free Rate) / Annual Volatility
  
- **Max Drawdown**: Maximum peak-to-trough decline
  - Indicates worst-case loss scenario
  
- **Sortino Ratio**: Like Sharpe but penalizes only downside volatility
  - Useful for strategies with frequent small losses
  
- **Calmar Ratio**: Annual return / Max drawdown
  - Combines returns and risk management
  
- **Alpha**: Excess return vs. benchmark
  - Positive alpha = outperformance

- **Beta**: Correlation with benchmark
  - <1: Less volatile than benchmark
  - >1: More volatile than benchmark

## 🔄 Extending the Framework

### Implement Custom Agent

```python
from marl_llm_pm import BaseAgent
import numpy as np

class MyCustomAgent(BaseAgent):
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        # Implement your decision logic
        return np.ones(self.n_assets) / self.n_assets
    
    def update(self, observation, action, reward, next_observation, done):
        # Implement learning logic here
        self.step_count += 1
```

### Integrate Sentiment Signals

```python
sentiment_scores = analyzer.analyze_sentiment(assets, date)

# Adjust weights based on sentiment
for i, asset in enumerate(assets):
    weights[i] *= (1 + sentiment_scores[asset])

weights /= weights.sum()  # Re-normalize
```

## 📋 Requirements

- Python 3.9+
- gymnasium >= 0.27.0
- anthropic >= 0.25.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- yfinance >= 0.2.0
- pyyaml >= 6.0

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional agent implementations (DQN, PPO, A3C)
- More sophisticated weight aggregation methods
- Real-time sentiment analysis
- Multi-objective optimization
- Additional backtesting metrics

## 📧 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Happy portfolio optimization! 📈**
