"""
Asset Allocation & Configuration Engine.
Manages asset definitions, single tickers vs constituent baskets, hierarchical weighting,
currency metadata, and strict portfolio configuration validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
import logging

from ..data.currency import infer_currency_from_ticker, get_required_fx_tickers

logger = logging.getLogger(__name__)


@dataclass
class AssetConfig:
    """
    Configuration model for an individual asset or an internal basket in the portfolio.
    """
    name: str
    type: str  # 'equity_etf', 'crypto', 'equity', 'commodity_or_gold_proxy', 'basket', 'bond'
    allocation: float  # Weight in portfolio [0.0, 1.0]
    ticker: Optional[str] = None
    currency: Optional[str] = None  # Quote currency: 'USD', 'IDR', 'GBP', etc.
    constituents: Optional[Dict[str, float]] = None  # For basket assets: {ticker: internal_weight}
    constituent_currencies: Optional[Dict[str, str]] = None
    description: Optional[str] = None

    def __post_init__(self):
        # Auto-infer currency if not explicitly provided
        if self.type == "basket" and self.constituents:
            if not self.constituent_currencies:
                self.constituent_currencies = {
                    t: infer_currency_from_ticker(t) for t in self.constituents.keys()
                }
            if not self.currency:
                # If all constituents share same currency, use that; else mark mixed
                first_curr = list(self.constituent_currencies.values())[0]
                if all(c == first_curr for c in self.constituent_currencies.values()):
                    self.currency = first_curr
                else:
                    self.currency = "MIXED"
        elif self.ticker and not self.currency:
            self.currency = infer_currency_from_ticker(self.ticker)

    def validate(self) -> None:
        """Validate individual asset configuration rules."""
        if not (0.0 <= self.allocation <= 1.0):
            raise ValueError(f"Asset '{self.name}' allocation {self.allocation} must be between 0.0 and 1.0.")

        if self.type == "basket":
            if not self.constituents:
                raise ValueError(f"Basket asset '{self.name}' must define a non-empty 'constituents' dictionary.")
            total_internal_weight = sum(self.constituents.values())
            if abs(total_internal_weight - 1.0) > 1e-4:
                raise ValueError(
                    f"Basket asset '{self.name}' internal constituent weights sum to {total_internal_weight:.4f}, expected 1.0."
                )
            for sub_ticker, sub_weight in self.constituents.items():
                if sub_weight <= 0.0:
                    raise ValueError(f"Constituent '{sub_ticker}' in basket '{self.name}' has non-positive weight {sub_weight}.")
        else:
            if not self.ticker:
                raise ValueError(f"Asset '{self.name}' with type '{self.type}' must define a 'ticker' symbol.")


@dataclass
class PortfolioConfig:
    """
    Top-level configuration holding all assets, metadata, data settings, and backtest parameters.
    """
    portfolio_name: str
    base_currency: str = "USD"
    assets: Dict[str, AssetConfig] = field(default_factory=dict)
    market_data_settings: Dict[str, Any] = field(default_factory=dict)
    backtest_settings: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Ensure portfolio allocations sum to 100% and assets are well-formed."""
        if not self.assets:
            raise ValueError("Portfolio must contain at least one asset.")

        total_alloc = sum(asset.allocation for asset in self.assets.values())
        if abs(total_alloc - 1.0) > 1e-4:
            raise ValueError(
                f"Portfolio asset allocations sum to {total_alloc:.4f} ({total_alloc*100:.2f}%), "
                f"which violates the requirement of exactly 1.0 (100%)."
            )

        for asset in self.assets.values():
            asset.validate()

    def get_top_level_allocations(self) -> Dict[str, float]:
        """Return top-level asset name -> allocation mapping."""
        return {name: asset.allocation for name, asset in self.assets.items()}

    def get_effective_ticker_weights(self) -> Dict[str, float]:
        """
        Calculate effective portfolio weight for each underlying market ticker.
        Expands baskets into their proportional portfolio weights:
        effective_weight = asset_allocation * constituent_weight
        """
        weights: Dict[str, float] = {}
        for asset in self.assets.values():
            if asset.type == "basket" and asset.constituents:
                for sub_ticker, sub_w in asset.constituents.items():
                    effective_w = asset.allocation * sub_w
                    weights[sub_ticker] = weights.get(sub_ticker, 0.0) + effective_w
            else:
                if asset.ticker:
                    weights[asset.ticker] = weights.get(asset.ticker, 0.0) + asset.allocation
        return weights

    def get_required_tickers(self) -> List[str]:
        """Return list of all unique market ticker symbols that need to be fetched."""
        tickers: List[str] = []
        for asset in self.assets.values():
            if asset.type == "basket" and asset.constituents:
                tickers.extend(list(asset.constituents.keys()))
            elif asset.ticker:
                tickers.append(asset.ticker)
        return sorted(list(set(tickers)))

    def get_effective_ticker_currencies(self) -> Dict[str, str]:
        """Return mapping of underlying market ticker to its quote currency."""
        curr_map: Dict[str, str] = {}
        for asset in self.assets.values():
            if asset.type == "basket" and asset.constituents:
                for sub_t in asset.constituents:
                    curr_map[sub_t] = (
                        asset.constituent_currencies.get(sub_t)
                        if asset.constituent_currencies
                        else infer_currency_from_ticker(sub_t)
                    )
            elif asset.ticker:
                curr_map[asset.ticker] = asset.currency or infer_currency_from_ticker(asset.ticker)
        return curr_map

    def get_required_fx_tickers(self) -> List[str]:
        """Return list of FX tickers needed to convert all assets to portfolio base currency."""
        ticker_currencies = self.get_effective_ticker_currencies()
        return get_required_fx_tickers(ticker_currencies, self.base_currency)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioConfig":
        """Construct PortfolioConfig from a dictionary."""
        portfolio_name = data.get("portfolio_name", "Custom Portfolio")
        base_currency = data.get("base_currency", "USD")
        raw_assets = data.get("assets", {})
        market_data_settings = data.get("market_data", {})
        backtest_settings = data.get("backtest", {})

        parsed_assets: Dict[str, AssetConfig] = {}
        for asset_name, asset_info in raw_assets.items():
            parsed_assets[asset_name] = AssetConfig(
                name=asset_name,
                type=asset_info.get("type", "equity"),
                allocation=float(asset_info.get("allocation", 0.0)),
                ticker=asset_info.get("ticker"),
                currency=asset_info.get("currency"),
                constituents={k: float(v) for k, v in asset_info.get("constituents", {}).items()} if asset_info.get("constituents") else None,
                constituent_currencies=asset_info.get("constituent_currencies"),
                description=asset_info.get("description")
            )

        config = cls(
            portfolio_name=portfolio_name,
            base_currency=base_currency,
            assets=parsed_assets,
            market_data_settings=market_data_settings,
            backtest_settings=backtest_settings
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PortfolioConfig":
        """Load and validate PortfolioConfig from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        return cls.from_dict(raw_data)
