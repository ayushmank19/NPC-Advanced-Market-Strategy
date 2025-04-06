# NPC-Advanced-Market-Strategy

# Advanced Market Making Strategy for Hummingbot

This repository contains an advanced market making strategy for Hummingbot that incorporates volatility indicators, trend analysis, and risk management.

## Strategy Overview

The `AdvancedMarketMaker` strategy enhances the basic Pure Market Making (PMM) strategy with sophisticated features:

1. **Multi-timeframe Analysis**: Analyzes market data across three different timeframes (5m, 15m, 1h) to make more informed trading decisions.

2. **Dynamic Spread Adjustment**: Automatically adjusts bid and ask spreads based on market volatility measured by the Normalized Average True Range (NATR) indicator.

3. **Price Positioning Based on Market Trends**: Uses multiple technical indicators (RSI, MACD, Bollinger Bands) to detect market trends and position orders accordingly.

4. **Inventory Management**: Maintains a target base/quote asset ratio (50/50 by default) and adjusts order sizes and prices to rebalance the portfolio.

5. **Risk Management Framework**: Implements position limits, volatility-based exposure controls, and confidence-based order sizing.

## Key Features

### Volatility-based Spread Adjustment
- Uses NATR to measure market volatility
- Dynamically widens spreads during high volatility periods
- Narrows spreads during low volatility to capture more trades

### Trend Analysis
- Multiple indicators: RSI, MACD, and Bollinger Bands
- Weighted analysis across three timeframes for reliable trend detection
- Counter-trend positioning to avoid adverse selection (configurable via trend_scalar parameter)

### Inventory Management
- Targets a specific base/quote asset ratio (50/50 by default)
- Adjusts spreads asymmetrically to favor accumulation of the underweight asset
- Sets tighter spreads on buy side when base asset is below target, and vice versa

### Risk Controls
- Maximum position limit (80% of portfolio) and minimum position limit (20% of portfolio)
- Order size adjustment based on market confidence score
- Performance tracking of portfolio value and filled orders

## Configuration Parameters

The strategy offers many configurable parameters:

```python
# Basic parameters
bid_spread = 0.0005                   # Default bid spread (0.05%)
ask_spread = 0.0005                   # Default ask spread (0.05%)
order_refresh_time = 15               # Order refresh time in seconds
order_amount = 0.01                   # Base order amount

# Timeframe settings
primary_interval = "5m"               # Primary candle timeframe
secondary_interval = "15m"            # Secondary candle timeframe
tertiary_interval = "1h"              # Tertiary candle timeframe

# Volatility parameters
natr_window = 14                      # NATR calculation window
vol_max_spread_scalar = 150           # Maximum volatility multiplier
vol_min_spread_scalar = 50            # Minimum volatility multiplier

# Trend analysis parameters
rsi_window = 14                       # RSI calculation window
rsi_overbought = 70                   # RSI overbought threshold
rsi_oversold = 30                     # RSI oversold threshold
macd_fast = 12                        # MACD fast length
macd_slow = 26                        # MACD slow length
macd_signal = 9                       # MACD signal length
trend_strength_scalar = 2.0           # Trend strength impact multiplier
trend_scalar = -1                     # Negative for counter-trend, positive for trend-following

# Inventory management
target_base_pct = 0.5                 # Target base asset percentage (50%)
inventory_range_multiplier = 2.0      # Inventory imbalance impact multiplier

# Risk management
max_position_pct = 0.8                # Maximum position as % of portfolio
min_position_pct = 0.2                # Minimum position as % of portfolio
max_order_size_multiplier = 3.0       # Maximum order size multiplier
min_order_size_multiplier = 0.3       # Minimum order size multiplier
```

## Running the Strategy

1. Install Hummingbot following the [official documentation](https://docs.hummingbot.org/installation/)
2. Place the `advanced_market_maker.py` script in the Hummingbot scripts directory
3. Start Hummingbot and load the script using the command `import_script advanced_market_maker.py`
4. Start the script with `start --script advanced_market_maker.py`

## Performance Monitoring

The strategy includes detailed logging and status display:
- Current portfolio value and performance metrics
- Active orders and balance information
- Technical indicator values from multiple timeframes
- Strategy parameter settings (spreads, reference price)
- Inventory position relative to target

## Strategy Justification

This market making strategy is designed to adapt to changing market conditions and maintain profitability across various environments:

1. In ranging markets, it captures the bid-ask spread while managing inventory effectively.
2. During trends, it adjusts the reference price to avoid adverse selection.
3. In volatile conditions, it widens spreads to mitigate risk while still participating in the market.
4. The multi-timeframe approach reduces false signals and improves decision quality.
5. Risk management features protect capital during extreme market conditions. 
