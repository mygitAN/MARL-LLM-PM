# System Architecture

## Overview

MARL-LLM-PM is a **Multi-Agent Reinforcement Learning** framework for portfolio management,
augmented with **Large Language Model** signals for regime detection and sentiment analysis.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                    Orchestration Layer                       │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  ┌─────────────┐  ┌──────────┐  ┌────────────┐
  │  Agents     │  │   LLM    │  │ Environment│
  │  (MARL)     │  │ Signals  │  │ (Gym-like) │
  └──────┬──────┘  └────┬─────┘  └─────┬──────┘
         │              │               │
         └──────────────┼───────────────┘
                        ▼
                ┌───────────────┐
                │  Backtesting  │
                │  & Evaluation │
                └───────────────┘
```

## Agents

| Agent | Strategy | Action Space |
|---|---|---|
| `MomentumAgent` | Trend-following (12-1 month momentum) | Long/short weights |
| `ValueAgent` | Mean-reversion on fundamental ratios | Long/short weights |
| `QualityAgent` | High-ROE, low-debt factor tilt | Long/short weights |
| `Coordinator` | Meta-allocator — combines agent signals | Final portfolio weights |

## LLM Integration

The `LLM` module calls an external language model (e.g. Claude / GPT-4) to:
- Detect macroeconomic regimes from news and earnings transcripts
- Adjust agent risk budgets based on sentiment signals

## Data Flow

```
Raw prices / news
      │
      ▼
DataPreprocessor  →  feature matrix
      │
      ▼
Portfolio Environment  →  observations
      │
      ├──► MomentumAgent  ─┐
      ├──► ValueAgent     ─┼──► Coordinator ──► Portfolio weights
      └──► QualityAgent   ─┘         │
                                     ▼
                              Backtester / WalkForwardValidator
```

## Configuration

All hyper-parameters are managed via `configs/default.yaml` and loaded through
`src/marl_llm_pm/config/config_manager.py`.
