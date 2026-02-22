"""Configuration management module."""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Portfolio environment configuration."""
    initial_portfolio_value: float = 100000.0
    max_steps: int = 252
    transaction_cost: float = 0.001
    assets: list = None
    
    def __post_init__(self):
        if self.assets is None:
            self.assets = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']


@dataclass
class AgentConfig:
    """Agent configuration."""
    n_agents: int = 3
    agent_type: str = "dummy"  # can be: dummy, rl, llm
    aggregation_method: str = "mean"
    update_frequency: str = "daily"


@dataclass
class LLMConfig:
    """LLM sentiment analyzer configuration."""
    enabled: bool = True
    model: str = "claude-3-5-sonnet-20241022"
    sentiment_scale: int = 10
    cache_enabled: bool = True
    cache_dir: str = ".cache/sentiment"
    batch_analysis: bool = True
    api_key: Optional[str] = None


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    rebalance_frequency: str = "daily"
    benchmark: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_episodes: int = 100
    max_steps_per_episode: int = 252
    learning_rate: float = 0.001
    batch_size: int = 32
    update_frequency: int = 10
    checkpoint_dir: str = "./checkpoints"


class ConfigManager:
    """Load and manage configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = {}
        
        if self.config_path and self.config_path.exists():
            self._load_from_file()
        else:
            self._load_defaults()
    
    def _load_from_file(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.error(f"Error loading config from {self.config_path}: {e}")
            self._load_defaults()
    
    def _load_defaults(self) -> None:
        """Load default configuration."""
        self.config = {
            'environment': asdict(EnvironmentConfig()),
            'agents': asdict(AgentConfig()),
            'llm': asdict(LLMConfig()),
            'backtesting': asdict(BacktestConfig()),
            'training': asdict(TrainingConfig()),
        }
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Config section (e.g., 'environment', 'agents')
            key: Optional specific key within section
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if section not in self.config:
            return default
        
        section_config = self.config[section]
        
        if key is None:
            return section_config
        
        return section_config.get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except OSError as e:
            logger.error(f"Error saving config to {output_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config


__all__ = [
    'ConfigManager',
    'EnvironmentConfig',
    'AgentConfig',
    'LLMConfig',
    'BacktestConfig',
    'TrainingConfig',
]
