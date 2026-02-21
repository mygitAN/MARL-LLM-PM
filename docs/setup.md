# Setup & Installation

## Requirements

- Python 3.10+
- pip or conda

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/mygitAN/MARL-LLM-PM.git
cd MARL-LLM-PM

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy and edit the default config:

```bash
cp configs/default.yaml configs/my_config.yaml
# Edit my_config.yaml with your API keys, tickers, and hyperparameters
```

## Running

```bash
# Train agents
python scripts/train.py --config configs/my_config.yaml

# Backtest
python scripts/backtest.py --config configs/my_config.yaml \
    --start 2020-01-01 --end 2024-12-31 \
    --tickers AAPL MSFT GOOGL AMZN NVDA

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint outputs/runs/checkpoint.pt \
    --start 2024-01-01 --end 2024-12-31
```

## Running Tests

```bash
pytest tests/ -v
```
