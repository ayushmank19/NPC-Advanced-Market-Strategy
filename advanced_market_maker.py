import logging
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Optional

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

class AdvancedMarketMaker(ScriptStrategyBase):
    """
    Advanced Market Making Strategy
    
    This strategy combines:
    1. Dynamic spread adjustment based on volatility (NATR)
    2. Trend-following price positioning based on multiple indicators (RSI, MACD, Bollinger Bands)
    3. Inventory management with position-based order sizing
    4. Risk management including position limits and volatility-based exposure controls
    5. Multiple time frame analysis for more robust market predictions
    """
    # Basic parameters
    bid_spread = 0.0005
    ask_spread = 0.0005
    order_refresh_time = 15  # seconds
    order_amount = 0.01
    max_order_age = 60  # Cancel orders older than this many seconds
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    base, quote = trading_pair.split('-')
    
    # Candles parameters for multiple timeframes
    candle_exchange = "binance"
    
    # Primary timeframe for main strategy decisions
    primary_interval = "5m"
    primary_length = 30
    
    # Secondary timeframe for trend confirmation
    secondary_interval = "15m"
    secondary_length = 20
    
    # Tertiary timeframe for longer-term context
    tertiary_interval = "1h"
    tertiary_length = 24
    
    max_records = 1000
    
    # Volatility parameters
    natr_window = 14  # Window for NATR calculation
    vol_max_spread_scalar = 150  # Maximum volatility-based spread multiplier
    vol_min_spread_scalar = 50   # Minimum volatility-based spread multiplier
    bb_std_dev = 2.0  # Standard deviations for Bollinger Bands
    
    # Trend analysis parameters
    rsi_window = 14
    rsi_overbought = 70
    rsi_oversold = 30
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    trend_strength_scalar = 2.0  # Multiplier for trend strength impact
    
    # Price shifting parameters
    max_shift_pct = 0.001  # Maximum price shift (0.1%)
    trend_scalar = -1  # Negative to counter-trend, positive to trend-follow
    
    # Inventory management parameters
    target_base_pct = 0.5  # Target base asset percentage (50% base, 50% quote)
    inventory_range_multiplier = 2.0  # Multiplier for inventory imbalance impact
    
    # Risk management parameters
    max_position_pct = 0.8  # Maximum position as percentage of portfolio
    min_position_pct = 0.2  # Minimum position as percentage of portfolio
    max_order_size_multiplier = 3.0  # Maximum order size multiplier for high confidence situations
    min_order_size_multiplier = 0.3  # Minimum order size multiplier for low confidence situations
    
    # Performance tracking
    total_filled_orders = 0
    total_profit_loss = 0
    start_portfolio_value = 0
    current_portfolio_value = 0
    
    # Initialize candles for multiple timeframes
    primary_candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=primary_interval,
        max_records=max_records
    ))
    
    secondary_candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=secondary_interval,
        max_records=max_records
    ))
    
    tertiary_candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=tertiary_interval,
        max_records=max_records
    ))
    
    # Define markets
    markets = {exchange: {trading_pair}}
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        # Start candles feeds
        self.primary_candles.start()
        self.secondary_candles.start()
        self.tertiary_candles.start()
        
        # Initialize strategy state variables
        self.bid_spread_adjusted = self.bid_spread
        self.ask_spread_adjusted = self.ask_spread
        self.reference_price = Decimal("0")
        self.last_executed_price = Decimal("0")
        self.market_mid_price = Decimal("0")
        
        # Calculate initial portfolio value
        self.update_portfolio_value()
        self.start_portfolio_value = self.current_portfolio_value
        
        # Log strategy initialization
        self.log_with_clock(logging.INFO, "Advanced Market Making Strategy initialized")
        
    def on_stop(self):
        # Stop candles feeds
        self.primary_candles.stop()
        self.secondary_candles.stop()
        self.tertiary_candles.stop()
        
        # Log final performance metrics
        self.update_portfolio_value()
        pnl_pct = ((self.current_portfolio_value / self.start_portfolio_value) - 1) * 100
        self.log_with_clock(
            logging.INFO,
            f"Strategy stopped. Performance: PnL: {pnl_pct:.2f}%, Orders filled: {self.total_filled_orders}"
        )
    
    def update_portfolio_value(self):
        """Calculate current portfolio value in quote currency"""
        try:
            mid_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            base_balance = self.connectors[self.exchange].get_balance(self.base)
            quote_balance = self.connectors[self.exchange].get_balance(self.quote)
            base_value_in_quote = base_balance * mid_price
            self.current_portfolio_value = float(base_value_in_quote + quote_balance)
        except Exception as e:
            self.log_with_clock(logging.ERROR, f"Error updating portfolio value: {str(e)}")
    
    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            # Cancel old orders
            self.cancel_all_orders()
            
            # Get market and technical analysis data
            self.analyze_market_conditions()
            
            # Create and place orders
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            
            # Log current strategy state
            self.log_with_clock(
                logging.INFO,
                f"Creating orders with spreads - Bid: {self.bid_spread_adjusted:.6f}, Ask: {self.ask_spread_adjusted:.6f}"
            )
            
            # Place orders
            self.place_orders(proposal_adjusted)
            
            # Update timestamp for next order refresh
            self.create_timestamp = self.order_refresh_time + self.current_timestamp
    
    def analyze_market_conditions(self):
        """
        Analyze market conditions using technical indicators across multiple timeframes
        and update strategy parameters accordingly
        """
        # Get technical indicators from all timeframes
        primary_df = self.get_primary_indicators()
        secondary_df = self.get_secondary_indicators()
        tertiary_df = self.get_tertiary_indicators()
        
        if primary_df.empty or secondary_df.empty or tertiary_df.empty:
            self.log_with_clock(logging.WARNING, "Insufficient data for analysis, using default parameters")
            return
        
        # Extract latest values
        try:
            # Convert all DataFrame values to Python floats before using them
            # This ensures we won't have type problems when mixing with Decimal
            
            # Update market mid price
            self.market_mid_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, self.price_source
            )
            
            # ---- Calculate volatility-based spread adjustment ----
            primary_natr = float(primary_df[f"NATR_{self.natr_window}"].iloc[-1])
            secondary_natr = float(secondary_df[f"NATR_{self.natr_window}"].iloc[-1])
            
            # Weighted average of NATR from different timeframes
            volatility = (primary_natr * 0.6) + (secondary_natr * 0.4)
            
            # Scale volatility to spread adjustment
            vol_scalar_bid = min(self.vol_max_spread_scalar, max(self.vol_min_spread_scalar, 
                                self.vol_min_spread_scalar + (volatility * 100)))
            vol_scalar_ask = min(self.vol_max_spread_scalar, max(self.vol_min_spread_scalar, 
                                self.vol_min_spread_scalar + (volatility * 100)))
            
            self.bid_spread_adjusted = float(volatility * vol_scalar_bid)
            self.ask_spread_adjusted = float(volatility * vol_scalar_ask)
            
            # ---- Calculate trend signals ----
            # RSI signals from different timeframes
            primary_rsi = float(primary_df[f"RSI_{self.rsi_window}"].iloc[-1])
            secondary_rsi = float(secondary_df[f"RSI_{self.rsi_window}"].iloc[-1])
            tertiary_rsi = float(tertiary_df[f"RSI_{self.rsi_window}"].iloc[-1])
            
            # Weighted RSI signal
            weighted_rsi = (primary_rsi * 0.5) + (secondary_rsi * 0.3) + (tertiary_rsi * 0.2)
            
            # MACD signals
            macd_hist = float(primary_df[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"].iloc[-1])
            macd_hist_prev = float(primary_df[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"].iloc[-2])
            macd_signal = 1 if macd_hist > 0 and macd_hist > macd_hist_prev else -1 if macd_hist < 0 and macd_hist < macd_hist_prev else 0
            
            # Bollinger Band position - Convert all values to Decimal first
            bb_upper = Decimal(str(float(primary_df[f"BBU_{self.natr_window}_{self.bb_std_dev}"].iloc[-1])))
            bb_lower = Decimal(str(float(primary_df[f"BBL_{self.natr_window}_{self.bb_std_dev}"].iloc[-1])))
            bb_position = float((self.market_mid_price - bb_lower) / (bb_upper - bb_lower))
            
            # Combine trend signals into a composite signal
            # Normalize RSI to -1 to 1 range (RSI is 0-100, so (RSI-50)/50 gives -1 to 1)
            rsi_signal = (weighted_rsi - 50) / 50
            
            # Bollinger signal: -1 if at bottom band, +1 if at top band
            bb_signal = (bb_position - 0.5) * 2
            
            # Combined trend signal with weights
            trend_signal = (rsi_signal * 0.4) + (macd_signal * 0.3) + (bb_signal * 0.3)
            
            # Apply trend strength factor
            trend_adjustment = trend_signal * self.trend_strength_scalar * self.trend_scalar
            
            # ---- Calculate inventory-based adjustment ----
            # Get current inventory position
            base_bal = self.connectors[self.exchange].get_balance(self.base)
            base_bal_in_quote = base_bal * self.market_mid_price
            quote_bal = self.connectors[self.exchange].get_balance(self.quote)
            total_bal_in_quote = base_bal_in_quote + quote_bal
            
            # Calculate current base percentage
            if total_bal_in_quote > 0:
                current_base_pct = float(base_bal_in_quote / total_bal_in_quote)
            else:
                current_base_pct = 0.5  # Default to target if we can't calculate
            
            # Calculate inventory imbalance
            inventory_imbalance = (self.target_base_pct - current_base_pct) / self.target_base_pct
            
            # Apply inventory adjustment to price
            inventory_adjustment = inventory_imbalance * self.inventory_range_multiplier * self.max_shift_pct
            
            # ---- Calculate final reference price ----
            # Combined adjustment from trend and inventory factors
            total_adjustment = float(trend_adjustment * self.max_shift_pct) + float(inventory_adjustment)
            
            # Cap the adjustment to prevent extreme shifts
            capped_adjustment = max(min(total_adjustment, self.max_shift_pct), -self.max_shift_pct)
            
            # Apply adjustment to mid price
            self.reference_price = self.market_mid_price * Decimal(str(1 + capped_adjustment))
            
            # ---- Apply asymmetric spread adjustments based on trend and inventory ----
            # If we need more base asset (target > current), tighten buy spread and widen sell spread
            if inventory_imbalance > 0:
                # Need more base, favor buys
                self.bid_spread_adjusted *= max(0.5, 1 - (abs(inventory_imbalance) * 0.5))
                self.ask_spread_adjusted *= min(2.0, 1 + (abs(inventory_imbalance) * 0.5))
            elif inventory_imbalance < 0:
                # Need more quote, favor sells
                self.bid_spread_adjusted *= min(2.0, 1 + (abs(inventory_imbalance) * 0.5))
                self.ask_spread_adjusted *= max(0.5, 1 - (abs(inventory_imbalance) * 0.5))
            
            # ---- Apply order size adjustments based on confidence ----
            # Calculate confidence score from multiple indicators
            confidence_score = 0.5
            
            # Add trend confidence (high when RSI is extreme)
            if primary_rsi > self.rsi_overbought or primary_rsi < self.rsi_oversold:
                confidence_score += 0.2
            else:
                confidence_score -= 0.1
                
            # Add MACD confidence
            if abs(macd_hist) > 0.5 * abs(float(primary_df[f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"].mean())):
                confidence_score += 0.15
            
            # Add volatility confidence (higher confidence in less volatile markets)
            if primary_natr < float(primary_df[f"NATR_{self.natr_window}"].mean()):
                confidence_score += 0.15
            else:
                confidence_score -= 0.1
                
            # Ensure confidence is between 0 and 1
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            # Adjust order size based on confidence
            size_multiplier = self.min_order_size_multiplier + (confidence_score * 
                                                               (self.max_order_size_multiplier - self.min_order_size_multiplier))
            
            # Store for use in create_proposal
            self.order_size_multiplier = size_multiplier
            
            # Only log if significant changes
            self.log_with_clock(
                logging.INFO,
                f"Market analysis: Vol={volatility:.6f}, Trend={trend_signal:.2f}, "
                f"Inv.Imbal={inventory_imbalance:.2f}, Conf={confidence_score:.2f}"
            )
            
        except Exception as e:
            self.log_with_clock(logging.ERROR, f"Error in market analysis: {str(e)}")
            # Fall back to default values on error
            self.bid_spread_adjusted = self.bid_spread
            self.ask_spread_adjusted = self.ask_spread
            self.reference_price = self.market_mid_price
    
    def get_primary_indicators(self):
        """Calculate technical indicators for primary timeframe"""
        candles_df = self.primary_candles.candles_df
        if candles_df.empty or len(candles_df) < max(self.primary_length, self.natr_window, self.rsi_window):
            return pd.DataFrame()
        
        # Calculate indicators
        # Normalized Average True Range for volatility
        candles_df.ta.natr(length=self.natr_window, scalar=1, append=True)
        
        # RSI for momentum
        candles_df.ta.rsi(length=self.rsi_window, append=True)
        
        # MACD for trend direction and strength
        candles_df.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=True)
        
        # Bollinger Bands for volatility and mean reversion
        candles_df.ta.bbands(length=self.natr_window, std=self.bb_std_dev, append=True)
        
        return candles_df
    
    def get_secondary_indicators(self):
        """Calculate technical indicators for secondary timeframe"""
        candles_df = self.secondary_candles.candles_df
        if candles_df.empty or len(candles_df) < max(self.secondary_length, self.natr_window, self.rsi_window):
            return pd.DataFrame()
        
        # Calculate same indicators for secondary timeframe
        candles_df.ta.natr(length=self.natr_window, scalar=1, append=True)
        candles_df.ta.rsi(length=self.rsi_window, append=True)
        
        return candles_df
    
    def get_tertiary_indicators(self):
        """Calculate technical indicators for tertiary timeframe"""
        candles_df = self.tertiary_candles.candles_df
        if candles_df.empty or len(candles_df) < max(self.tertiary_length, self.rsi_window):
            return pd.DataFrame()
        
        # Calculate RSI for longer timeframe trend confirmation
        candles_df.ta.rsi(length=self.rsi_window, append=True)
        
        return candles_df
        
    def create_proposal(self) -> List[OrderCandidate]:
        """Create order proposals with calculated parameters"""
        # Calculate prices with adjusted spreads
        buy_price = self.reference_price * Decimal(str(1 - self.bid_spread_adjusted))
        sell_price = self.reference_price * Decimal(str(1 + self.ask_spread_adjusted))
        
        # Ensure prices don't cross the order book
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)
        
        buy_price = min(buy_price, best_bid)
        sell_price = max(sell_price, best_ask)
        
        # Apply order size multiplier if defined
        adjusted_order_amount = self.order_amount
        if hasattr(self, 'order_size_multiplier'):
            adjusted_order_amount = self.order_amount * self.order_size_multiplier
        
        # Create order candidates
        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal(str(adjusted_order_amount)),
            price=buy_price
        )
        
        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=Decimal(str(adjusted_order_amount)),
            price=sell_price
        )
        
        return [buy_order, sell_order]
    
    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust order proposal according to available budget and risk parameters"""
        # First apply the exchange's budget checker
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        
        # Get current inventory status
        base_balance = self.connectors[self.exchange].get_available_balance(self.base)
        quote_balance = self.connectors[self.exchange].get_available_balance(self.quote)
        
        # Check if we're near our risk limits
        base_value_in_quote = base_balance * self.market_mid_price
        total_portfolio_value = base_value_in_quote + quote_balance
        base_percentage = float(base_value_in_quote / total_portfolio_value) if total_portfolio_value > 0 else 0
        
        adjusted_orders = []
        for order in proposal_adjusted:
            if order.order_side == TradeType.BUY:
                # Risk check for buys - don't exceed max base position
                if base_percentage >= self.max_position_pct:
                    # Skip this order if we already have too much base asset
                    self.log_with_clock(logging.INFO, f"Skipping buy order due to max position limit ({base_percentage:.2%})")
                    continue
            elif order.order_side == TradeType.SELL:
                # Risk check for sells - maintain minimum base position
                if base_percentage <= self.min_position_pct:
                    # Skip this order if we have too little base asset
                    self.log_with_clock(logging.INFO, f"Skipping sell order due to min position limit ({base_percentage:.2%})")
                    continue
            
            adjusted_orders.append(order)
            
        return adjusted_orders
    
    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place orders from the proposal"""
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)
    
    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place an individual order"""
        if order.order_side == TradeType.SELL:
            self.sell(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        elif order.order_side == TradeType.BUY:
            self.buy(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
    
    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)
    
    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fill events and track performance"""
        self.last_executed_price = event.price
        self.total_filled_orders += 1
        
        # Track PnL in base currency value
        base_value_change = 0
        if event.trade_type == TradeType.BUY:
            # Bought base asset, value increases by amount
            base_value_change = event.amount
        else:
            # Sold base asset, value decreases by amount
            base_value_change = -event.amount
            
        # Update portfolio value after trade
        self.update_portfolio_value()
        
        # Log trade details
        msg = (f"{event.trade_type.name} {round(event.amount, 6)} {event.trading_pair} at {round(event.price, 6)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
    
    def format_status(self) -> str:
        """Format current strategy status for display"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
            
        lines = []
        
        # Portfolio value and performance
        self.update_portfolio_value()
        start_val = self.start_portfolio_value
        current_val = self.current_portfolio_value
        
        if start_val > 0:
            pnl_pct = ((current_val / start_val) - 1) * 100
            lines.extend([
                "",
                f"  Portfolio Value: {current_val:.6f} {self.quote}",
                f"  PnL: {pnl_pct:.2f}%",
                f"  Total Orders Filled: {self.total_filled_orders}"
            ])
        
        # Balances
        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # Active orders
        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])
        
        # Strategy parameters
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend(["  Strategy Parameters:"])
        lines.extend([f"  Bid Spread: {float(self.bid_spread_adjusted):.6f} | Ask Spread: {float(self.ask_spread_adjusted):.6f}"])
        lines.extend([f"  Market Mid Price: {float(self.market_mid_price):.6f} | Reference Price: {float(self.reference_price):.6f}"])
        
        # Inventory metrics
        base_bal = self.connectors[self.exchange].get_balance(self.base)
        base_bal_in_quote = base_bal * self.market_mid_price
        quote_bal = self.connectors[self.exchange].get_balance(self.quote)
        total_bal_in_quote = base_bal_in_quote + quote_bal
        
        if total_bal_in_quote > 0:
            current_base_pct = float(base_bal_in_quote / total_bal_in_quote) * 100
        else:
            current_base_pct = 0
            
        target_base_pct = self.target_base_pct * 100
        
        lines.extend([f"  Base %: {current_base_pct:.2f}% (Target: {target_base_pct:.2f}%)"])
        
        # Technical analysis
        lines.extend(["\n----------------------------------------------------------------------\n"])
        lines.extend([f"  Primary Timeframe: {self.primary_interval} | Secondary: {self.secondary_interval} | Tertiary: {self.tertiary_interval}"])
        
        # Get latest indicators
        try:
            primary_df = self.get_primary_indicators()
            if not primary_df.empty:
                lines.extend(["", "  Latest Indicators:"])
                latest = primary_df.iloc[-1]
                rsi = float(latest[f"RSI_{self.rsi_window}"])
                macd = float(latest[f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"])
                macd_signal = float(latest[f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"])
                macd_hist = float(latest[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"])
                natr = float(latest[f"NATR_{self.natr_window}"])
                
                lines.extend([
                    f"  RSI({self.rsi_window}): {rsi:.2f}",
                    f"  MACD: {macd:.6f}, Signal: {macd_signal:.6f}, Hist: {macd_hist:.6f}",
                    f"  NATR({self.natr_window}): {natr:.6f}"
                ])
        except Exception as e:
            lines.extend([f"  Error displaying indicators: {str(e)}"])
            
        return "\n".join(lines) 