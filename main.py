#!/usr/bin/env python3
"""
CLI entry point for MARL-LLM-PM portfolio management system.

Strategy-allocator pipeline (primary):
    python main.py sleeve-backtest --config configs/strategy_allocator.yaml
    python main.py sleeve-backtest --config configs/strategy_allocator.yaml --walk-forward

Legacy asset pipeline:
    python main.py backtest --config configs/default.yaml --assets AAPL GOOGL MSFT
    python main.py analyze --assets AAPL GOOGL --date 2024-01-15
    python main.py test --coverage
"""

import argparse
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from marl_llm_pm import (
    PortfolioEnv,
    DummyAgent,
    AgentCoordinator,
    SentimentAnalyzer,
    Backtester,
    BacktestResults,
    ConfigManager,
)
from marl_llm_pm.strategy_allocator.environment import StrategySleeveEnv
from marl_llm_pm.strategy_allocator.agents import StrategyPreferenceAgent, collect_preferences
from marl_llm_pm.strategy_allocator.orchestration import MetaAllocator
from marl_llm_pm.strategy_allocator.llm import RegimeInterpreter
from marl_llm_pm.strategy_allocator.evaluation import proportional_walk_forward


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
_TICKER_RE = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')


def validate_config_path(config_arg: str) -> Path:
    """Resolve config path and ensure it stays within the project and is a YAML file."""
    path = Path(config_arg).resolve()
    if path.suffix.lower() not in {'.yaml', '.yml'}:
        raise ValueError(f"Config file must be a .yaml/.yml file, got: {path.suffix}")
    if not str(path).startswith(str(_PROJECT_ROOT)):
        raise ValueError(f"Config path must be inside the project directory: {path}")
    return path


def validate_ticker(ticker: str) -> str:
    """Validate a single ticker symbol."""
    if not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker symbol '{ticker}'. Expected 1-5 uppercase letters.")
    return ticker


def validate_date(date_str: str) -> str:
    """Validate a date string in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD")
    return date_str


def load_market_data(assets: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close prices via yfinance."""
    logger.info(f"Downloading data for {len(assets)} asset(s)...")
    data = yf.download(assets, start=start_date, end=end_date, progress=False)
    if len(assets) == 1:
        data = data[['Close']].rename(columns={'Close': assets[0]})
    else:
        data = data['Close']
    logger.info(f"Loaded {len(data)} days of data")
    return data


def cmd_backtest(args, config: ConfigManager) -> None:
    """Run the legacy asset-level backtest."""
    logger.info("Starting backtest...")

    raw_assets = args.assets or config.get('environment', 'assets')
    try:
        assets = [validate_ticker(t) for t in raw_assets]
    except ValueError as e:
        logger.error(f"Invalid ticker: {e}")
        return

    raw_start = args.start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    raw_end = args.end_date or datetime.now().strftime('%Y-%m-%d')
    try:
        start_date = validate_date(raw_start)
        end_date = validate_date(raw_end)
    except ValueError as e:
        logger.error(f"Invalid date: {e}")
        return

    try:
        price_data = load_market_data(assets, start_date, end_date)
    except (IOError, KeyError, ValueError) as e:
        logger.error(f"Failed to load market data: {e}")
        return

    env = PortfolioEnv(
        asset_names=assets,
        initial_portfolio_value=config.get('backtesting', 'initial_portfolio_value'),
        max_steps=len(price_data),
        transaction_cost=config.get('backtesting', 'transaction_cost'),
    )
    env.set_market_data(price_data)

    agents = [
        DummyAgent(f"agent_{i}", n_assets=len(assets))
        for i in range(config.get('agents', 'n_agents'))
    ]
    coordinator = AgentCoordinator(
        agents,
        aggregation_method=config.get('agents', 'aggregation_method')
    )

    def get_weights(observation):
        return coordinator.get_actions(observation)

    backtester = Backtester(
        initial_capital=config.get('backtesting', 'initial_capital'),
        transaction_cost=config.get('backtesting', 'transaction_cost'),
    )

    try:
        results = backtester.run(price_data, get_weights)
        logger.info("\n" + results.summary())

        results_path = Path(config.get('logging', 'results_dir', './results'))
        results_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_path / f"backtest_{timestamp}.csv"
        pd.DataFrame({
            'portfolio_value': results.portfolio_values,
            'returns': np.concatenate([[0], results.returns]),
        }).to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

    except (ValueError, RuntimeError) as e:
        logger.error(f"Backtest failed: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error during backtest: {e}", exc_info=True)
        raise


def cmd_backtest_sleeves(args, config: ConfigManager) -> None:
    """Run strategy-sleeve backtest.

    Pipeline: Data (CSV) → Env → Agents (preferences) → Meta-allocator → Weights → PnL
    """
    logger.info("Starting strategy-sleeve backtest...")

    sleeves = config.get('environment', 'sleeves')
    tc = config.get('environment', 'transaction_cost')
    cap = config.get('environment', 'max_weight_per_sleeve')
    initial_capital = config.get('backtesting', 'initial_capital')
    sleeve_csv = config.get('data', 'sleeve_returns_csv')

    if not Path(sleeve_csv).exists():
        logger.error(f"Sleeve returns CSV not found: {sleeve_csv}")
        return

    try:
        sleeve_returns = pd.read_csv(sleeve_csv, index_col=0, parse_dates=True)
        sleeve_returns = sleeve_returns.sort_index()
        logger.info(f"Loaded sleeve returns: {sleeve_returns.shape}")
    except Exception as e:
        logger.error(f"Failed to load sleeve returns: {e}")
        return

    try:
        env = StrategySleeveEnv(
            sleeve_names=sleeves,
            transaction_cost=tc,
            max_weight_per_sleeve=cap,
            initial_value=initial_capital,
        )
        env.set_sleeve_returns(sleeve_returns)
        logger.info(f"Environment initialized with sleeves: {sleeves}")
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return

    agents = [StrategyPreferenceAgent(sleeve_name) for sleeve_name in sleeves]
    allocator = MetaAllocator(sleeves, cap=cap, temperature=1.0)
    regime = RegimeInterpreter(
        cache_dir=config.get('llm', 'cache_dir'),
        use_cache=config.get('llm', 'cache_enabled'),
        labels=config.get('llm', 'labels'),
    )

    values = []
    allocations = []
    regimes = []

    env.reset()

    try:
        for t in range(len(sleeve_returns)):
            window = sleeve_returns.iloc[max(0, t - 12): t]

            if len(window) > 2:
                vol = float(window.mean(axis=1).std())
                drawdown_cumsum = (1.0 + window.mean(axis=1)).cumprod() - 1.0
                drawdown = float(drawdown_cumsum.min()) if len(drawdown_cumsum) > 0 else 0.0
                trend = float(window.mean().mean())
                corr_vals = window.corr()
                corr = float(corr_vals.mean().mean()) if len(corr_vals) > 1 else 0.0
            else:
                vol, drawdown, trend, corr = 0.0, 0.0, 0.0, 0.0

            metrics = {"vol": vol, "drawdown": drawdown, "trend": trend, "corr": corr}
            regime_out = regime.classify(key=str(sleeve_returns.index[t].date()), metrics=metrics)
            regimes.append(regime_out.label)

            obs = {
                "regime_label": regime_out.label,
                "sleeve_features": {
                    s: {"signal": float(window[s].mean()) if len(window) > 2 else 0.0}
                    for s in sleeves
                },
                "global_features": metrics,
                "prev_weights": env.w.copy(),
            }

            alphas = collect_preferences(agents, obs)
            w = allocator.allocate(alphas, obs)
            reward, info, done = env.step(w)

            values.append(info.portfolio_value)
            allocations.append(w.copy())

            if t % 10 == 0:
                logger.info(
                    f"[t={t:3d}] regime={regime_out.label:20s} "
                    f"value=${info.portfolio_value:12.2f} "
                    f"weights={[f'{x:.2f}' for x in w]} "
                    f"turnover={info.turnover:.3f}"
                )

            if done:
                break

        returns = np.array([
            (values[i] / values[i - 1]) - 1.0 if i > 0 else 0.0
            for i in range(len(values))
        ])
        final_value = values[-1] if values else initial_capital
        total_return = (final_value / initial_capital) - 1.0
        annual_return = total_return * (config.get('environment', 'steps_per_year') / len(values)) if len(values) > 0 else 0.0
        annual_vol = float(returns.std() * np.sqrt(config.get('environment', 'steps_per_year'))) if len(returns) > 0 else 0.0

        logger.info("\n" + "=" * 70)
        logger.info("STRATEGY-SLEEVE BACKTEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"Initial Capital:       ${initial_capital:,.2f}")
        logger.info(f"Final Value:           ${final_value:,.2f}")
        logger.info(f"Total Return:          {total_return:>8.2%}")
        logger.info(f"Annualized Return:     {annual_return:>8.2%}")
        logger.info(f"Annualized Volatility: {annual_vol:>8.2%}")
        if annual_vol > 0:
            logger.info(f"Sharpe Ratio (rf=0):   {annual_return / annual_vol:>8.2f}")
        logger.info("=" * 70)

        results_path = Path(config.get('logging', 'results_dir', './results'))
        results_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_path / f"sleeve_backtest_{timestamp}.csv"

        results_df = pd.DataFrame({
            'date': sleeve_returns.index[:len(values)],
            'portfolio_value': values,
            'returns': returns,
            'regime': regimes,
        })
        for i, sleeve in enumerate(sleeves):
            results_df[f'weight_{sleeve}'] = [alloc[i] for alloc in allocations]
        results_df.to_csv(results_file, index=False)
        logger.info(f"\nResults saved to {results_file}")

    except Exception as e:
        logger.error(f"Strategy-sleeve backtest failed: {e}", exc_info=True)
        raise


def cmd_train(args, config: ConfigManager) -> None:
    """Train agents on historical data."""
    logger.info("Training not yet implemented")
    logger.info(
        "To implement training, extend agents to inherit from BaseAgent "
        "and implement learning logic in the update() method"
    )


def cmd_analyze(args, config: ConfigManager) -> None:
    """Analyze sentiment for given assets."""
    if not config.get('llm', 'enabled'):
        logger.error("LLM is disabled in configuration")
        return

    raw_assets = args.assets or config.get('environment', 'assets')
    try:
        assets = [validate_ticker(t) for t in raw_assets]
    except ValueError as e:
        logger.error(f"Invalid ticker: {e}")
        return

    raw_date = args.date or datetime.now().strftime('%Y-%m-%d')
    try:
        date = validate_date(raw_date)
    except ValueError as e:
        logger.error(f"Invalid date: {e}")
        return

    logger.info(f"Analyzing sentiment for {len(assets)} asset(s) on {date}...")
    analyzer = SentimentAnalyzer(
        cache_dir=config.get('llm', 'cache_dir'),
        model=config.get('llm', 'model'),
        use_cache=config.get('llm', 'cache_enabled'),
    )

    try:
        sentiments = analyzer.analyze_sentiment(assets, date)
        logger.info("Sentiment Analysis Results:")
        logger.info("-" * 40)
        for asset, score in sentiments.items():
            sentiment_text = "Bullish" if score > 0.5 else "Bearish" if score < 0.5 else "Neutral"
            logger.info(f"{asset:8s}: {score:.2f} ({sentiment_text})")

    except (ValueError, RuntimeError) as e:
        logger.error(f"Sentiment analysis failed: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error during sentiment analysis: {e}", exc_info=True)
        raise


def _build_metrics(window: pd.DataFrame, lookback: int = 12) -> dict:
    """Compute rolling numeric metrics for regime classification."""
    avg = window.mean(axis=1)
    vol      = float(avg.std())          if len(avg) > 2 else 0.0
    trend    = float(avg.mean())         if len(avg) > 2 else 0.0
    drawdown = float(avg.cumsum().min()) if len(avg) > 2 else 0.0
    if window.shape[1] > 1 and len(window) > 3:
        corr = float(window.corr().values[window.corr().values != 1].mean())
    else:
        corr = 0.0
    return {"vol": vol, "trend": trend, "drawdown": drawdown, "corr": corr}


def _run_sleeve_episode(
    sleeve_returns: pd.DataFrame,
    sleeves: list,
    tc: float,
    cap: float,
    initial_capital: float,
    lookback: int,
    config: ConfigManager,
) -> tuple:
    """Run a single sleeve-backtest episode. Returns (values, returns, weight_history)."""
    env = StrategySleeveEnv(
        sleeve_names=sleeves,
        transaction_cost=tc,
        max_weight_per_sleeve=cap,
        initial_value=initial_capital,
    )
    env.set_sleeve_returns(sleeve_returns)

    agents = [
        StrategyPreferenceAgent(s, bias=config.get('agents', f'{s.lower()}_bias') or 0.0)
        for s in sleeves
    ]
    allocator = MetaAllocator(
        sleeve_names=sleeves,
        cap=cap,
        temperature=config.get('agents', 'temperature') or 1.0,
    )
    regime = RegimeInterpreter(
        cache_dir=config.get('llm', 'cache_dir') or '.cache/regimes',
        use_cache=config.get('llm', 'cache_enabled') if config.get('llm', 'cache_enabled') is not None else True,
        use_llm=config.get('llm', 'use_llm') or False,
    )

    env.reset()
    values, rets, weight_history = [], [], []
    prev_value = env.value
    index = sleeve_returns.index

    for t in range(len(sleeve_returns)):
        window = sleeve_returns.iloc[max(0, t - lookback): t]
        metrics = _build_metrics(window, lookback)

        date_key = str(index[t].date()) if hasattr(index[t], 'date') else str(index[t])
        reg_out = regime.classify(key=date_key, metrics=metrics)

        obs = {
            "regime_label": reg_out.label,
            "sleeve_features": {
                s: {"signal": float(window[s].mean()) if len(window) > 2 else 0.0}
                for s in sleeves
            },
            "global_features": metrics,
            "prev_weights": env.w.copy(),
        }

        alphas = collect_preferences(agents, obs)
        w = allocator.allocate(alphas, obs)
        _, info, done = env.step(w)

        values.append(info.portfolio_value)
        rets.append((info.portfolio_value / prev_value) - 1.0)
        weight_history.append({s: w[i] for i, s in enumerate(sleeves)})
        prev_value = info.portfolio_value

        if done:
            break

    return values, rets, weight_history


def cmd_sleeve_backtest(args, config: ConfigManager) -> None:
    """Run the strategy-allocator sleeve backtest pipeline."""
    logger.info("Starting strategy-allocator sleeve backtest...")

    csv_path = Path(
        args.sleeve_returns or config.get('data', 'sleeve_returns_csv') or 'data/sleeve_returns.csv'
    ).resolve()

    if not csv_path.exists():
        logger.error(f"Sleeve returns CSV not found: {csv_path}")
        return

    try:
        sleeve_returns = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except (OSError, ValueError) as e:
        logger.error(f"Failed to read sleeve returns: {e}")
        return

    sleeves     = config.get('environment', 'sleeves') or ['MOMENTUM', 'VALUE', 'QUALITY']
    tc          = config.get('environment', 'transaction_cost') or 0.001
    cap         = config.get('environment', 'max_weight_per_sleeve') or 0.70
    initial_cap = config.get('backtesting', 'initial_capital') or 100_000.0
    lookback    = config.get('regime', 'lookback_steps') or 12

    missing = [s for s in sleeves if s not in sleeve_returns.columns]
    if missing:
        logger.error(f"Missing sleeve columns in CSV: {missing}")
        return

    sleeve_returns = sleeve_returns[sleeves].dropna()
    logger.info(f"Loaded {len(sleeve_returns)} periods | sleeves: {sleeves}")

    if args.walk_forward:
        eval_cfg = config.get('evaluation') or {}
        train_df, val_df, test_windows, holdout_df = proportional_walk_forward(
            sleeve_returns,
            split_train=eval_cfg.get('split_train', 0.60),
            split_val=eval_cfg.get('split_val', 0.20),
            split_test=eval_cfg.get('split_test', 0.20),
            test_interval_months=eval_cfg.get('test_interval_months', 6),
            holdout_months=eval_cfg.get('holdout_months', 12),
        )
        logger.info(
            f"Walk-forward splits — train: {len(train_df)}, val: {len(val_df)}, "
            f"test windows: {len(test_windows)}, holdout: {len(holdout_df)} (sealed)"
        )
        logger.info("Running on validation set...")
        data_to_run = val_df
    else:
        data_to_run = sleeve_returns

    values, rets, weight_history = _run_sleeve_episode(
        data_to_run, sleeves, tc, cap, initial_cap, lookback, config
    )

    if not values:
        logger.error("No results produced.")
        return

    returns_arr    = np.array(rets)
    steps_per_year = config.get('environment', 'steps_per_year') or 52
    ann_factor     = steps_per_year ** 0.5
    total_ret      = (values[-1] / initial_cap) - 1.0
    ann_vol        = float(np.std(returns_arr)) * ann_factor
    ann_ret        = float(np.mean(returns_arr)) * steps_per_year
    sharpe         = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
    cum            = np.cumprod(1 + returns_arr)
    peak           = np.maximum.accumulate(cum)
    max_dd         = float(((cum - peak) / np.where(peak > 0, peak, 1)).min())
    avg_w          = {s: float(np.mean([wh[s] for wh in weight_history])) for s in sleeves}

    logger.info("=" * 52)
    logger.info("STRATEGY-ALLOCATOR SLEEVE BACKTEST — RESULTS")
    logger.info("=" * 52)
    logger.info(f"  Periods run:        {len(values)}")
    logger.info(f"  Final NAV:          £{values[-1]:>12,.2f}")
    logger.info(f"  Total return:       {total_ret:>10.2%}")
    logger.info(f"  Annualised return:  {ann_ret:>10.2%}")
    logger.info(f"  Annualised vol:     {ann_vol:>10.2%}")
    logger.info(f"  Sharpe ratio:       {sharpe:>10.2f}")
    logger.info(f"  Max drawdown:       {max_dd:>10.2%}")
    logger.info(f"  Avg sleeve weights: { {k: f'{v:.1%}' for k, v in avg_w.items()} }")
    logger.info("=" * 52)

    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = results_dir / f"sleeve_backtest_{ts}.csv"
    rows = [
        {"period": i, "portfolio_value": v, "return": r, **wh}
        for i, (v, r, wh) in enumerate(zip(values, rets, weight_history))
    ]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(f"Results saved to {out_csv}")


def cmd_test(args, config: ConfigManager) -> None:
    """Run pytest test suite."""
    import subprocess
    logger.info("Running test suite...")
    test_dir = Path(__file__).parent / 'tests'
    cmd = ['python', '-m', 'pytest', str(test_dir), '-v']
    if args.coverage:
        cmd.append('--cov=src/marl_llm_pm')
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, shell=False)
        sys.exit(result.returncode)
    except FileNotFoundError as e:
        logger.error(f"Test runner not found (is pytest installed?): {e}")
    except OSError as e:
        logger.error(f"Test execution failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Agent Reinforcement Learning with LLM for Portfolio Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  python main.py sleeve-backtest --config configs/strategy_allocator.yaml
  python main.py sleeve-backtest --config configs/strategy_allocator.yaml --walk-forward
  python main.py backtest --assets AAPL GOOGL MSFT
  python main.py analyze --assets AAPL GOOGL --date 2024-01-15
  python main.py test --coverage
        '''
    )

    parser.add_argument(
        '--config',
        default='configs/strategy_allocator.yaml',
        help='Path to configuration file (default: configs/strategy_allocator.yaml)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    sleeve_parser = subparsers.add_parser('sleeve-backtest', help='Run strategy-sleeve allocator backtest')
    sleeve_parser.add_argument('--sleeve-returns', help='Path to sleeve returns CSV (overrides config)')
    sleeve_parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward splits (train/val/test/holdout)')
    sleeve_parser.set_defaults(func=cmd_sleeve_backtest)

    sleeves_parser = subparsers.add_parser('backtest-sleeves', help='Run strategy-sleeve backtest (verbose)')
    sleeves_parser.set_defaults(func=cmd_backtest_sleeves)

    backtest_parser = subparsers.add_parser('backtest', help='Run legacy asset backtest')
    backtest_parser.add_argument('--assets', nargs='+', help='Asset tickers')
    backtest_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    backtest_parser.set_defaults(func=cmd_backtest)

    train_parser = subparsers.add_parser('train', help='Train agents')
    train_parser.add_argument('--episodes', type=int, help='Number of episodes')
    train_parser.add_argument('--assets', nargs='+', help='Asset tickers')
    train_parser.set_defaults(func=cmd_train)

    analyze_parser = subparsers.add_parser('analyze', help='Analyze sentiment')
    analyze_parser.add_argument('--assets', nargs='+', help='Asset tickers')
    analyze_parser.add_argument('--date', help='Analysis date (YYYY-MM-DD)')
    analyze_parser.set_defaults(func=cmd_analyze)

    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--coverage', action='store_true', help='Run with coverage')
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()

    try:
        config_path = validate_config_path(args.config)
    except ValueError as e:
        parser.error(str(e))
    config = ConfigManager(str(config_path))

    if args.command is None:
        parser.print_help()
    else:
        args.func(args, config)


if __name__ == '__main__':
    main()
