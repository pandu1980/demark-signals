"""
Individual Stock Pattern Analyzer
Analyzes daily chart for multiple technical patterns:
- Pullback to 21-EMA or 50-DMA
- High Tight Flag
- Earnings Gap Hold
- Relative Strength Breakout vs NASDAQ
- Ascending Triangle
- RSI Reset (40-50 Zone)
- VWAP Reclaim (Daily/Weekly)
- Consolidation After News
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')


def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_vwap(df):
    """Calculate VWAP"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap


def calculate_weekly_vwap(df):
    """Calculate Weekly VWAP"""
    df = df.copy()
    df['Week'] = df.index.to_period('W')

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = typical_price * df['Volume']

    weekly_vwap = df.groupby('Week').apply(
        lambda x: x['TP_Vol'].cumsum() / x['Volume'].cumsum()
    )

    # Flatten the result
    if isinstance(weekly_vwap.index, pd.MultiIndex):
        weekly_vwap = weekly_vwap.droplevel(0)

    return weekly_vwap


def check_pullback_to_ma(df):
    """Check for pullback to 21-EMA or 50-DMA"""
    result = {
        'pullback_21ema': False,
        'pullback_50dma': False,
        'ema_21': 0,
        'sma_50': 0,
        'distance_21ema': 0,
        'distance_50dma': 0,
        'bouncing': False,
        'description': ''
    }

    if len(df) < 50:
        return result

    df['EMA_21'] = calculate_ema(df['Close'], 21)
    df['SMA_50'] = calculate_sma(df['Close'], 50)

    current = df.iloc[-1]
    prev = df.iloc[-2]

    result['ema_21'] = round(current['EMA_21'], 2)
    result['sma_50'] = round(current['SMA_50'], 2)

    # Distance from MAs
    dist_21 = ((current['Close'] - current['EMA_21']) / current['EMA_21']) * 100
    dist_50 = ((current['Close'] - current['SMA_50']) / current['SMA_50']) * 100

    result['distance_21ema'] = round(dist_21, 2)
    result['distance_50dma'] = round(dist_50, 2)

    # Check for pullback to 21-EMA (within 2%)
    if abs(dist_21) <= 2:
        result['pullback_21ema'] = True
        # Check if bouncing (low touched EMA but closed above)
        if current['Low'] <= current['EMA_21'] <= current['Close']:
            result['bouncing'] = True
            result['description'] = 'Price pulling back to 21-EMA with bounce'
        else:
            result['description'] = 'Price at 21-EMA support zone'

    # Check for pullback to 50-DMA (within 2%)
    if abs(dist_50) <= 2:
        result['pullback_50dma'] = True
        if current['Low'] <= current['SMA_50'] <= current['Close']:
            result['bouncing'] = True
            result['description'] = 'Price pulling back to 50-DMA with bounce'
        else:
            result['description'] = 'Price at 50-DMA support zone'

    return result


def check_high_tight_flag(df):
    """
    Check for High Tight Flag pattern:
    - Stock moves up 100%+ in 4-8 weeks
    - Then consolidates 10-25% from high
    - Tight consolidation (low volatility)
    """
    result = {
        'detected': False,
        'prior_move': 0,
        'consolidation_depth': 0,
        'consolidation_days': 0,
        'tightness': 0,
        'description': ''
    }

    if len(df) < 60:
        return result

    # Look for 100%+ move in last 8 weeks (40 trading days)
    lookback = min(60, len(df) - 1)

    for start in range(lookback, 20, -5):
        period_low = df['Low'].iloc[-start:-20].min()
        period_high = df['High'].iloc[-20:].max()

        if period_low > 0:
            move_pct = ((period_high - period_low) / period_low) * 100

            if move_pct >= 80:  # 80%+ move (relaxed from 100%)
                # Check consolidation
                recent_high = df['High'].iloc[-15:].max()
                recent_low = df['Low'].iloc[-15:].min()
                current_price = df['Close'].iloc[-1]

                consolidation = ((recent_high - current_price) / recent_high) * 100
                range_pct = ((recent_high - recent_low) / recent_high) * 100

                # High tight flag: consolidation 10-25%, tight range
                if 5 <= consolidation <= 30 and range_pct <= 20:
                    result['detected'] = True
                    result['prior_move'] = round(move_pct, 1)
                    result['consolidation_depth'] = round(consolidation, 1)
                    result['consolidation_days'] = 15
                    result['tightness'] = round(100 - range_pct, 1)
                    result['description'] = f"High Tight Flag: {move_pct:.0f}% move, {consolidation:.1f}% pullback, {100-range_pct:.0f}% tight"
                    break

    return result


def check_earnings_gap_hold(df, symbol):
    """
    Check for Earnings Gap Hold:
    - Gap up on earnings (3%+)
    - Holding above gap level
    """
    result = {
        'detected': False,
        'gap_date': '',
        'gap_size': 0,
        'days_held': 0,
        'holding_above': False,
        'description': ''
    }

    if len(df) < 30:
        return result

    # Look for significant gaps in last 30 days
    for i in range(-30, -1):
        if i >= -len(df) + 1:
            prev_close = df['Close'].iloc[i-1]
            curr_open = df['Open'].iloc[i]

            gap_pct = ((curr_open - prev_close) / prev_close) * 100

            # Significant gap (3%+) - could be earnings
            if abs(gap_pct) >= 3:
                gap_level = curr_open if gap_pct > 0 else prev_close
                gap_date = df.index[i].strftime('%Y-%m-%d')
                days_since = abs(i)

                # Check if holding above gap level
                current_price = df['Close'].iloc[-1]
                prices_since = df['Close'].iloc[i:]

                if gap_pct > 0:  # Gap up
                    holding = all(prices_since >= gap_level * 0.97)  # Within 3% of gap
                    if holding and current_price >= gap_level:
                        result['detected'] = True
                        result['gap_date'] = gap_date
                        result['gap_size'] = round(gap_pct, 1)
                        result['days_held'] = days_since
                        result['holding_above'] = True
                        result['description'] = f"Gap up {gap_pct:.1f}% on {gap_date}, held for {days_since} days"
                        break

    return result


def check_relative_strength_vs_nasdaq(df, nasdaq_df):
    """
    Check for Relative Strength Breakout vs NASDAQ:
    - Stock outperforming NASDAQ
    - RS line making new highs
    """
    result = {
        'rs_breakout': False,
        'rs_new_high': False,
        'rs_trend': '',
        'outperformance_20d': 0,
        'outperformance_50d': 0,
        'description': ''
    }

    if len(df) < 50 or len(nasdaq_df) < 50:
        return result

    # Align data
    common_dates = df.index.intersection(nasdaq_df.index)
    if len(common_dates) < 50:
        return result

    stock = df.loc[common_dates]['Close']
    nasdaq = nasdaq_df.loc[common_dates]['Close']

    # Calculate Relative Strength
    rs_line = stock / nasdaq

    # Check RS trend
    rs_20 = rs_line.iloc[-20:].mean()
    rs_50 = rs_line.iloc[-50:].mean()
    current_rs = rs_line.iloc[-1]

    # RS new high (52-week)
    rs_52w_high = rs_line.iloc[-252:].max() if len(rs_line) >= 252 else rs_line.max()
    rs_new_high = current_rs >= rs_52w_high * 0.98

    # Outperformance
    stock_return_20d = ((stock.iloc[-1] - stock.iloc[-20]) / stock.iloc[-20]) * 100
    nasdaq_return_20d = ((nasdaq.iloc[-1] - nasdaq.iloc[-20]) / nasdaq.iloc[-20]) * 100
    outperf_20d = stock_return_20d - nasdaq_return_20d

    stock_return_50d = ((stock.iloc[-1] - stock.iloc[-50]) / stock.iloc[-50]) * 100
    nasdaq_return_50d = ((nasdaq.iloc[-1] - nasdaq.iloc[-50]) / nasdaq.iloc[-50]) * 100
    outperf_50d = stock_return_50d - nasdaq_return_50d

    result['outperformance_20d'] = round(outperf_20d, 2)
    result['outperformance_50d'] = round(outperf_50d, 2)
    result['rs_new_high'] = rs_new_high

    # RS breakout: RS making new highs + uptrend
    if current_rs > rs_20 > rs_50:
        result['rs_trend'] = 'UPTREND'
        if rs_new_high:
            result['rs_breakout'] = True
            result['description'] = f"RS Breakout! New high, outperforming NASDAQ by {outperf_20d:.1f}% (20d)"
        else:
            result['description'] = f"RS Uptrend, outperforming NASDAQ by {outperf_20d:.1f}% (20d)"
    elif current_rs < rs_20 < rs_50:
        result['rs_trend'] = 'DOWNTREND'
        result['description'] = f"RS Downtrend, underperforming NASDAQ by {abs(outperf_20d):.1f}% (20d)"
    else:
        result['rs_trend'] = 'NEUTRAL'
        result['description'] = f"RS Neutral vs NASDAQ ({outperf_20d:+.1f}% 20d)"

    return result


def check_ascending_triangle(df):
    """
    Check for Ascending Triangle:
    - Flat resistance (highs)
    - Rising support (higher lows)
    """
    result = {
        'detected': False,
        'resistance_level': 0,
        'support_slope': 0,
        'touches': 0,
        'breakout_imminent': False,
        'description': ''
    }

    if len(df) < 30:
        return result

    recent = df.iloc[-30:]

    # Find swing highs and lows
    highs = []
    lows = []

    for i in range(2, len(recent) - 2):
        # Swing high
        if (recent['High'].iloc[i] >= recent['High'].iloc[i-1] and
            recent['High'].iloc[i] >= recent['High'].iloc[i-2] and
            recent['High'].iloc[i] >= recent['High'].iloc[i+1] and
            recent['High'].iloc[i] >= recent['High'].iloc[i+2]):
            highs.append((i, recent['High'].iloc[i]))

        # Swing low
        if (recent['Low'].iloc[i] <= recent['Low'].iloc[i-1] and
            recent['Low'].iloc[i] <= recent['Low'].iloc[i-2] and
            recent['Low'].iloc[i] <= recent['Low'].iloc[i+1] and
            recent['Low'].iloc[i] <= recent['Low'].iloc[i+2]):
            lows.append((i, recent['Low'].iloc[i]))

    if len(highs) >= 2 and len(lows) >= 2:
        # Check for flat resistance (highs within 2% range)
        high_values = [h[1] for h in highs]
        high_range = (max(high_values) - min(high_values)) / max(high_values) * 100

        # Check for rising support (higher lows)
        low_values = [l[1] for l in lows]
        rising_lows = all(low_values[i] <= low_values[i+1] for i in range(len(low_values)-1))

        if high_range <= 3 and rising_lows:
            result['detected'] = True
            result['resistance_level'] = round(max(high_values), 2)

            # Calculate support slope
            if len(lows) >= 2:
                slope = (lows[-1][1] - lows[0][1]) / (lows[-1][0] - lows[0][0]) if lows[-1][0] != lows[0][0] else 0
                result['support_slope'] = round(slope, 2)

            result['touches'] = len(highs)

            # Breakout imminent if price near resistance
            current = recent['Close'].iloc[-1]
            if current >= result['resistance_level'] * 0.98:
                result['breakout_imminent'] = True
                result['description'] = f"Ascending Triangle - Breakout imminent! Resistance at ${result['resistance_level']:.2f}"
            else:
                result['description'] = f"Ascending Triangle forming. Resistance at ${result['resistance_level']:.2f}"

    return result


def check_rsi_reset(df):
    """
    Check for RSI Reset (40-50 Zone):
    - RSI pulling back to 40-50 after being overbought
    - Good entry for momentum stocks
    """
    result = {
        'in_reset_zone': False,
        'current_rsi': 0,
        'was_overbought': False,
        'reset_quality': '',
        'description': ''
    }

    if len(df) < 30:
        return result

    df['RSI'] = calculate_rsi(df['Close'])

    current_rsi = df['RSI'].iloc[-1]
    result['current_rsi'] = round(current_rsi, 2)

    # Check if in reset zone (40-50)
    if 38 <= current_rsi <= 52:
        result['in_reset_zone'] = True

        # Check if was recently overbought (>70 in last 20 days)
        recent_rsi = df['RSI'].iloc[-20:]
        if recent_rsi.max() >= 70:
            result['was_overbought'] = True
            result['reset_quality'] = 'HIGH'
            result['description'] = f"RSI Reset! RSI at {current_rsi:.1f} after being overbought - strong entry zone"
        elif recent_rsi.max() >= 60:
            result['reset_quality'] = 'MEDIUM'
            result['description'] = f"RSI in reset zone ({current_rsi:.1f}) - potential entry"
        else:
            result['reset_quality'] = 'LOW'
            result['description'] = f"RSI at {current_rsi:.1f} - neutral zone"
    else:
        if current_rsi > 70:
            result['description'] = f"RSI Overbought at {current_rsi:.1f}"
        elif current_rsi < 30:
            result['description'] = f"RSI Oversold at {current_rsi:.1f}"
        else:
            result['description'] = f"RSI at {current_rsi:.1f}"

    return result


def check_vwap_reclaim(df):
    """
    Check for VWAP Reclaim:
    - Price reclaiming VWAP after being below
    - Bullish signal
    """
    result = {
        'daily_reclaim': False,
        'weekly_reclaim': False,
        'above_daily_vwap': False,
        'above_weekly_vwap': False,
        'daily_vwap': 0,
        'weekly_vwap': 0,
        'description': ''
    }

    if len(df) < 10:
        return result

    # Daily VWAP (reset each day, use last 5 days avg as proxy)
    recent = df.iloc[-5:]
    daily_vwap = ((recent['High'] + recent['Low'] + recent['Close']) / 3 * recent['Volume']).sum() / recent['Volume'].sum()
    result['daily_vwap'] = round(daily_vwap, 2)

    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]

    result['above_daily_vwap'] = current_price >= daily_vwap

    # Check for reclaim (was below, now above)
    if prev_close < daily_vwap and current_price >= daily_vwap:
        result['daily_reclaim'] = True

    # Weekly VWAP
    try:
        weekly_data = df.iloc[-5:]
        weekly_vwap = ((weekly_data['High'] + weekly_data['Low'] + weekly_data['Close']) / 3 * weekly_data['Volume']).sum() / weekly_data['Volume'].sum()
        result['weekly_vwap'] = round(weekly_vwap, 2)
        result['above_weekly_vwap'] = current_price >= weekly_vwap

        # Weekly reclaim
        week_ago_price = df['Close'].iloc[-5] if len(df) >= 5 else df['Close'].iloc[0]
        if week_ago_price < weekly_vwap and current_price >= weekly_vwap:
            result['weekly_reclaim'] = True
    except:
        pass

    # Description
    if result['daily_reclaim'] and result['weekly_reclaim']:
        result['description'] = "VWAP Reclaim (Daily & Weekly) - Strong bullish signal!"
    elif result['daily_reclaim']:
        result['description'] = "Daily VWAP Reclaim - Bullish"
    elif result['weekly_reclaim']:
        result['description'] = "Weekly VWAP Reclaim - Bullish"
    elif result['above_daily_vwap'] and result['above_weekly_vwap']:
        result['description'] = "Trading above both Daily and Weekly VWAP"
    elif result['above_daily_vwap']:
        result['description'] = "Above Daily VWAP"
    else:
        result['description'] = "Below VWAP - waiting for reclaim"

    return result


def check_consolidation_after_news(df, symbol):
    """
    Check for Consolidation After News:
    - Big move (gap or range expansion)
    - Followed by tight consolidation
    - Low volatility contraction
    """
    result = {
        'detected': False,
        'news_date': '',
        'initial_move': 0,
        'consolidation_range': 0,
        'days_consolidating': 0,
        'volatility_contraction': 0,
        'description': ''
    }

    if len(df) < 20:
        return result

    # Look for big move in last 20 days
    for i in range(-20, -5):
        if i >= -len(df) + 1:
            # Check for range expansion or gap
            day_range = ((df['High'].iloc[i] - df['Low'].iloc[i]) / df['Close'].iloc[i-1]) * 100
            gap = abs((df['Open'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]) * 100

            big_move = day_range >= 5 or gap >= 3

            if big_move:
                # Check consolidation since then
                consolidation_period = df.iloc[i+1:]
                if len(consolidation_period) >= 3:
                    cons_high = consolidation_period['High'].max()
                    cons_low = consolidation_period['Low'].min()
                    cons_range = ((cons_high - cons_low) / cons_low) * 100

                    # Tight consolidation (less than 10% range)
                    if cons_range <= 12:
                        # Volatility contraction
                        avg_range_before = df.iloc[i-10:i]['High'].sub(df.iloc[i-10:i]['Low']).mean()
                        avg_range_after = consolidation_period['High'].sub(consolidation_period['Low']).mean()
                        vol_contraction = ((avg_range_before - avg_range_after) / avg_range_before) * 100 if avg_range_before > 0 else 0

                        result['detected'] = True
                        result['news_date'] = df.index[i].strftime('%Y-%m-%d')
                        result['initial_move'] = round(max(day_range, gap), 1)
                        result['consolidation_range'] = round(cons_range, 1)
                        result['days_consolidating'] = len(consolidation_period)
                        result['volatility_contraction'] = round(vol_contraction, 1)
                        result['description'] = f"Consolidating after {result['initial_move']:.1f}% move on {result['news_date']}. Range: {cons_range:.1f}%, {len(consolidation_period)} days"
                        break

    return result


def analyze_stock(symbol):
    """Main analysis function for a single stock"""
    print(f"\nFetching data for {symbol}...")

    try:
        # Fetch stock data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")

        if df.empty or len(df) < 50:
            return None, f"Insufficient data for {symbol}"

        # Fetch NASDAQ for relative strength comparison
        nasdaq = yf.Ticker("^IXIC")
        nasdaq_df = nasdaq.history(period="1y")

        # Get stock info
        try:
            info = ticker.info
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            market_cap = info.get('marketCap', 0)
        except:
            company_name = symbol
            sector = 'N/A'
            industry = 'N/A'
            market_cap = 0

        # Current price info
        current = df.iloc[-1]
        prev = df.iloc[-2]

        price = current['Close']
        change = price - prev['Close']
        change_pct = (change / prev['Close']) * 100
        volume = current['Volume']
        avg_volume = df['Volume'].tail(20).mean()

        # Run all pattern checks
        print("Analyzing patterns...")

        pullback = check_pullback_to_ma(df)
        htf = check_high_tight_flag(df)
        earnings_gap = check_earnings_gap_hold(df, symbol)
        rel_strength = check_relative_strength_vs_nasdaq(df, nasdaq_df)
        asc_triangle = check_ascending_triangle(df)
        rsi_reset = check_rsi_reset(df)
        vwap = check_vwap_reclaim(df)
        consolidation = check_consolidation_after_news(df, symbol)

        # Calculate overall score
        score = 0
        signals = []

        if pullback['pullback_21ema']:
            score += 15
            signals.append(('Pullback to 21-EMA', 'bullish', pullback['description']))
        if pullback['pullback_50dma']:
            score += 15
            signals.append(('Pullback to 50-DMA', 'bullish', pullback['description']))
        if pullback['bouncing']:
            score += 10

        if htf['detected']:
            score += 25
            signals.append(('High Tight Flag', 'bullish', htf['description']))

        if earnings_gap['detected']:
            score += 20
            signals.append(('Earnings Gap Hold', 'bullish', earnings_gap['description']))

        if rel_strength['rs_breakout']:
            score += 25
            signals.append(('RS Breakout vs NASDAQ', 'bullish', rel_strength['description']))
        elif rel_strength['rs_new_high']:
            score += 15
            signals.append(('RS New High', 'bullish', rel_strength['description']))

        if asc_triangle['detected']:
            score += 20
            signals.append(('Ascending Triangle', 'bullish', asc_triangle['description']))
            if asc_triangle['breakout_imminent']:
                score += 10

        if rsi_reset['in_reset_zone']:
            if rsi_reset['reset_quality'] == 'HIGH':
                score += 20
            elif rsi_reset['reset_quality'] == 'MEDIUM':
                score += 10
            signals.append(('RSI Reset Zone', 'bullish', rsi_reset['description']))

        if vwap['daily_reclaim'] or vwap['weekly_reclaim']:
            score += 15
            signals.append(('VWAP Reclaim', 'bullish', vwap['description']))

        if consolidation['detected']:
            score += 15
            signals.append(('Consolidation After News', 'neutral', consolidation['description']))

        results = {
            'symbol': symbol,
            'company_name': company_name,
            'sector': sector,
            'industry': industry,
            'market_cap': market_cap,
            'price': round(price, 2),
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'volume': int(volume),
            'avg_volume': int(avg_volume),
            'volume_ratio': round(volume / avg_volume, 2) if avg_volume > 0 else 0,
            'score': score,
            'signals': signals,
            'pullback': pullback,
            'high_tight_flag': htf,
            'earnings_gap': earnings_gap,
            'relative_strength': rel_strength,
            'ascending_triangle': asc_triangle,
            'rsi_reset': rsi_reset,
            'vwap': vwap,
            'consolidation': consolidation,
            'df': df  # Include dataframe for charting
        }

        return results, None

    except Exception as e:
        return None, f"Error analyzing {symbol}: {str(e)}"


def generate_html_report(results):
    """Generate interactive HTML report"""

    r = results
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format market cap
    if r['market_cap'] >= 1e12:
        market_cap_str = f"${r['market_cap']/1e12:.2f}T"
    elif r['market_cap'] >= 1e9:
        market_cap_str = f"${r['market_cap']/1e9:.2f}B"
    elif r['market_cap'] >= 1e6:
        market_cap_str = f"${r['market_cap']/1e6:.2f}M"
    else:
        market_cap_str = f"${r['market_cap']:,.0f}"

    # Generate signal cards
    signal_cards = ""
    for signal_name, signal_type, description in r['signals']:
        signal_cards += f"""
        <div class="signal-card {signal_type}">
            <div class="signal-name">{signal_name}</div>
            <div class="signal-desc">{description}</div>
        </div>
        """

    if not r['signals']:
        signal_cards = '<div class="no-signals">No significant patterns detected</div>'

    # Price change class
    price_class = 'positive' if r['change'] >= 0 else 'negative'
    change_symbol = '+' if r['change'] >= 0 else ''

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{r['symbol']} - Stock Pattern Analyzer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e4e4e4;
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .stock-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 20px;
        }}

        .stock-info {{
            flex: 1;
        }}

        .symbol {{
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .company-name {{
            font-size: 1.3rem;
            color: #888;
            margin-top: 5px;
        }}

        .stock-meta {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 0.9rem;
            color: #666;
        }}

        .price-box {{
            text-align: right;
        }}

        .current-price {{
            font-size: 2.8rem;
            font-weight: bold;
        }}

        .price-change {{
            font-size: 1.2rem;
            margin-top: 5px;
        }}

        .price-change.positive {{ color: #00ff88; }}
        .price-change.negative {{ color: #ff4757; }}

        .score-box {{
            background: linear-gradient(135deg, #00d9ff, #00ff88);
            border-radius: 15px;
            padding: 20px 30px;
            text-align: center;
        }}

        .score-value {{
            font-size: 3rem;
            font-weight: bold;
            color: #1a1a2e;
        }}

        .score-label {{
            color: #1a1a2e;
            font-size: 0.9rem;
            font-weight: 600;
        }}

        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .section h2 {{
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(255,255,255,0.1);
            color: #00d9ff;
        }}

        .signals-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }}

        .signal-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #666;
        }}

        .signal-card.bullish {{
            border-left-color: #00ff88;
            background: rgba(0, 255, 136, 0.1);
        }}

        .signal-card.bearish {{
            border-left-color: #ff4757;
            background: rgba(255, 71, 87, 0.1);
        }}

        .signal-card.neutral {{
            border-left-color: #ffa502;
            background: rgba(255, 165, 2, 0.1);
        }}

        .signal-name {{
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 8px;
        }}

        .signal-card.bullish .signal-name {{ color: #00ff88; }}
        .signal-card.bearish .signal-name {{ color: #ff4757; }}
        .signal-card.neutral .signal-name {{ color: #ffa502; }}

        .signal-desc {{
            color: #aaa;
            font-size: 0.9rem;
        }}

        .no-signals {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-style: italic;
        }}

        .pattern-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}

        .pattern-card {{
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 20px;
        }}

        .pattern-card h3 {{
            color: #00d9ff;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .pattern-card h3 .status {{
            font-size: 0.8rem;
            padding: 4px 10px;
            border-radius: 20px;
        }}

        .status.detected {{
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
        }}

        .status.not-detected {{
            background: rgba(255, 255, 255, 0.1);
            color: #666;
        }}

        .pattern-details {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}

        .detail-item {{
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 8px;
        }}

        .detail-label {{
            font-size: 0.8rem;
            color: #666;
        }}

        .detail-value {{
            font-size: 1.1rem;
            font-weight: bold;
            margin-top: 3px;
        }}

        .detail-value.positive {{ color: #00ff88; }}
        .detail-value.negative {{ color: #ff4757; }}

        .pattern-desc {{
            margin-top: 15px;
            padding: 12px;
            background: rgba(0,217,255,0.1);
            border-radius: 8px;
            font-size: 0.9rem;
            color: #aaa;
        }}

        .volume-bar {{
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            margin-top: 5px;
            overflow: hidden;
        }}

        .volume-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            border-radius: 4px;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            margin-top: 30px;
        }}

        @media (max-width: 768px) {{
            .stock-header {{
                flex-direction: column;
            }}
            .price-box {{
                text-align: left;
            }}
            .pattern-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="stock-header">
                <div class="stock-info">
                    <div class="symbol">{r['symbol']}</div>
                    <div class="company-name">{r['company_name']}</div>
                    <div class="stock-meta">
                        <span>{r['sector']}</span>
                        <span>|</span>
                        <span>{r['industry']}</span>
                        <span>|</span>
                        <span>{market_cap_str}</span>
                    </div>
                </div>
                <div class="price-box">
                    <div class="current-price">${r['price']:.2f}</div>
                    <div class="price-change {price_class}">
                        {change_symbol}{r['change']:.2f} ({change_symbol}{r['change_pct']:.2f}%)
                    </div>
                </div>
                <div class="score-box">
                    <div class="score-value">{r['score']}</div>
                    <div class="score-label">PATTERN SCORE</div>
                </div>
            </div>
        </header>

        <div class="section">
            <h2>Active Signals</h2>
            <div class="signals-grid">
                {signal_cards}
            </div>
        </div>

        <div class="section">
            <h2>Pattern Analysis</h2>
            <div class="pattern-grid">
                <!-- Pullback to MA -->
                <div class="pattern-card">
                    <h3>
                        Pullback to Moving Average
                        <span class="status {'detected' if r['pullback']['pullback_21ema'] or r['pullback']['pullback_50dma'] else 'not-detected'}">
                            {'DETECTED' if r['pullback']['pullback_21ema'] or r['pullback']['pullback_50dma'] else 'NOT DETECTED'}
                        </span>
                    </h3>
                    <div class="pattern-details">
                        <div class="detail-item">
                            <div class="detail-label">21-EMA</div>
                            <div class="detail-value">${r['pullback']['ema_21']:.2f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Distance to 21-EMA</div>
                            <div class="detail-value {'positive' if r['pullback']['distance_21ema'] >= 0 else 'negative'}">{r['pullback']['distance_21ema']:+.2f}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">50-DMA</div>
                            <div class="detail-value">${r['pullback']['sma_50']:.2f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Distance to 50-DMA</div>
                            <div class="detail-value {'positive' if r['pullback']['distance_50dma'] >= 0 else 'negative'}">{r['pullback']['distance_50dma']:+.2f}%</div>
                        </div>
                    </div>
                    <div class="pattern-desc">{r['pullback']['description'] or 'No pullback to key moving averages detected'}</div>
                </div>

                <!-- High Tight Flag -->
                <div class="pattern-card">
                    <h3>
                        High Tight Flag
                        <span class="status {'detected' if r['high_tight_flag']['detected'] else 'not-detected'}">
                            {'DETECTED' if r['high_tight_flag']['detected'] else 'NOT DETECTED'}
                        </span>
                    </h3>
                    <div class="pattern-details">
                        <div class="detail-item">
                            <div class="detail-label">Prior Move</div>
                            <div class="detail-value positive">{r['high_tight_flag']['prior_move']:.1f}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Consolidation</div>
                            <div class="detail-value">{r['high_tight_flag']['consolidation_depth']:.1f}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Days Consolidating</div>
                            <div class="detail-value">{r['high_tight_flag']['consolidation_days']}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Tightness</div>
                            <div class="detail-value">{r['high_tight_flag']['tightness']:.1f}%</div>
                        </div>
                    </div>
                    <div class="pattern-desc">{r['high_tight_flag']['description'] or 'No high tight flag pattern detected'}</div>
                </div>

                <!-- Earnings Gap Hold -->
                <div class="pattern-card">
                    <h3>
                        Earnings Gap Hold
                        <span class="status {'detected' if r['earnings_gap']['detected'] else 'not-detected'}">
                            {'DETECTED' if r['earnings_gap']['detected'] else 'NOT DETECTED'}
                        </span>
                    </h3>
                    <div class="pattern-details">
                        <div class="detail-item">
                            <div class="detail-label">Gap Date</div>
                            <div class="detail-value">{r['earnings_gap']['gap_date'] or 'N/A'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Gap Size</div>
                            <div class="detail-value positive">{r['earnings_gap']['gap_size']:.1f}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Days Held</div>
                            <div class="detail-value">{r['earnings_gap']['days_held']}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Holding Above</div>
                            <div class="detail-value">{'Yes' if r['earnings_gap']['holding_above'] else 'No'}</div>
                        </div>
                    </div>
                    <div class="pattern-desc">{r['earnings_gap']['description'] or 'No earnings gap hold pattern detected'}</div>
                </div>

                <!-- Relative Strength -->
                <div class="pattern-card">
                    <h3>
                        Relative Strength vs NASDAQ
                        <span class="status {'detected' if r['relative_strength']['rs_breakout'] else 'not-detected'}">
                            {r['relative_strength']['rs_trend']}
                        </span>
                    </h3>
                    <div class="pattern-details">
                        <div class="detail-item">
                            <div class="detail-label">20-Day Outperformance</div>
                            <div class="detail-value {'positive' if r['relative_strength']['outperformance_20d'] >= 0 else 'negative'}">{r['relative_strength']['outperformance_20d']:+.2f}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">50-Day Outperformance</div>
                            <div class="detail-value {'positive' if r['relative_strength']['outperformance_50d'] >= 0 else 'negative'}">{r['relative_strength']['outperformance_50d']:+.2f}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">RS New High</div>
                            <div class="detail-value">{'Yes' if r['relative_strength']['rs_new_high'] else 'No'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">RS Breakout</div>
                            <div class="detail-value">{'Yes' if r['relative_strength']['rs_breakout'] else 'No'}</div>
                        </div>
                    </div>
                    <div class="pattern-desc">{r['relative_strength']['description']}</div>
                </div>

                <!-- Ascending Triangle -->
                <div class="pattern-card">
                    <h3>
                        Ascending Triangle
                        <span class="status {'detected' if r['ascending_triangle']['detected'] else 'not-detected'}">
                            {'DETECTED' if r['ascending_triangle']['detected'] else 'NOT DETECTED'}
                        </span>
                    </h3>
                    <div class="pattern-details">
                        <div class="detail-item">
                            <div class="detail-label">Resistance Level</div>
                            <div class="detail-value">${r['ascending_triangle']['resistance_level']:.2f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Resistance Touches</div>
                            <div class="detail-value">{r['ascending_triangle']['touches']}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Support Slope</div>
                            <div class="detail-value">{r['ascending_triangle']['support_slope']:.2f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Breakout Imminent</div>
                            <div class="detail-value">{'Yes' if r['ascending_triangle']['breakout_imminent'] else 'No'}</div>
                        </div>
                    </div>
                    <div class="pattern-desc">{r['ascending_triangle']['description'] or 'No ascending triangle pattern detected'}</div>
                </div>

                <!-- RSI Reset -->
                <div class="pattern-card">
                    <h3>
                        RSI Reset (40-50 Zone)
                        <span class="status {'detected' if r['rsi_reset']['in_reset_zone'] else 'not-detected'}">
                            {'IN ZONE' if r['rsi_reset']['in_reset_zone'] else 'NOT IN ZONE'}
                        </span>
                    </h3>
                    <div class="pattern-details">
                        <div class="detail-item">
                            <div class="detail-label">Current RSI</div>
                            <div class="detail-value">{r['rsi_reset']['current_rsi']:.1f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Was Overbought</div>
                            <div class="detail-value">{'Yes' if r['rsi_reset']['was_overbought'] else 'No'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Reset Quality</div>
                            <div class="detail-value">{r['rsi_reset']['reset_quality'] or 'N/A'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">In Reset Zone</div>
                            <div class="detail-value">{'Yes' if r['rsi_reset']['in_reset_zone'] else 'No'}</div>
                        </div>
                    </div>
                    <div class="pattern-desc">{r['rsi_reset']['description']}</div>
                </div>

                <!-- VWAP Reclaim -->
                <div class="pattern-card">
                    <h3>
                        VWAP Reclaim
                        <span class="status {'detected' if r['vwap']['daily_reclaim'] or r['vwap']['weekly_reclaim'] else 'not-detected'}">
                            {'RECLAIMED' if r['vwap']['daily_reclaim'] or r['vwap']['weekly_reclaim'] else 'NOT RECLAIMED'}
                        </span>
                    </h3>
                    <div class="pattern-details">
                        <div class="detail-item">
                            <div class="detail-label">Daily VWAP</div>
                            <div class="detail-value">${r['vwap']['daily_vwap']:.2f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Above Daily VWAP</div>
                            <div class="detail-value">{'Yes' if r['vwap']['above_daily_vwap'] else 'No'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Weekly VWAP</div>
                            <div class="detail-value">${r['vwap']['weekly_vwap']:.2f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Above Weekly VWAP</div>
                            <div class="detail-value">{'Yes' if r['vwap']['above_weekly_vwap'] else 'No'}</div>
                        </div>
                    </div>
                    <div class="pattern-desc">{r['vwap']['description']}</div>
                </div>

                <!-- Consolidation After News -->
                <div class="pattern-card">
                    <h3>
                        Consolidation After News
                        <span class="status {'detected' if r['consolidation']['detected'] else 'not-detected'}">
                            {'DETECTED' if r['consolidation']['detected'] else 'NOT DETECTED'}
                        </span>
                    </h3>
                    <div class="pattern-details">
                        <div class="detail-item">
                            <div class="detail-label">News Date</div>
                            <div class="detail-value">{r['consolidation']['news_date'] or 'N/A'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Initial Move</div>
                            <div class="detail-value positive">{r['consolidation']['initial_move']:.1f}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Consolidation Range</div>
                            <div class="detail-value">{r['consolidation']['consolidation_range']:.1f}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Days Consolidating</div>
                            <div class="detail-value">{r['consolidation']['days_consolidating']}</div>
                        </div>
                    </div>
                    <div class="pattern-desc">{r['consolidation']['description'] or 'No consolidation after news detected'}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Volume Analysis</h2>
            <div class="pattern-details">
                <div class="detail-item">
                    <div class="detail-label">Today's Volume</div>
                    <div class="detail-value">{r['volume']:,}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">20-Day Avg Volume</div>
                    <div class="detail-value">{r['avg_volume']:,}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Volume Ratio</div>
                    <div class="detail-value {'positive' if r['volume_ratio'] >= 1.5 else ''}">{r['volume_ratio']:.2f}x</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Volume Status</div>
                    <div class="detail-value">{'Above Average' if r['volume_ratio'] > 1 else 'Below Average'}</div>
                </div>
            </div>
            <div style="margin-top: 15px;">
                <div class="detail-label">Volume Bar</div>
                <div class="volume-bar">
                    <div class="volume-fill" style="width: {min(r['volume_ratio'] * 50, 100)}%"></div>
                </div>
            </div>
        </div>

        <footer>
            <p>Stock Pattern Analyzer | Generated: {timestamp}</p>
            <p>Data from Yahoo Finance | For educational purposes only. Not financial advice.</p>
        </footer>
    </div>
</body>
</html>
"""

    return html_content


def main():
    print("=" * 60)
    print("Stock Pattern Analyzer")
    print("=" * 60)

    # Get symbol from command line or prompt
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = input("\nEnter stock symbol: ").upper().strip()

    if not symbol:
        print("No symbol provided!")
        return

    # Analyze stock
    results, error = analyze_stock(symbol)

    if error:
        print(f"\nError: {error}")
        return

    if not results:
        print(f"\nNo data available for {symbol}")
        return

    # Generate HTML report
    html_content = generate_html_report(results)
    html_file = f"stock_analysis_{symbol}.html"

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nHTML report saved: {html_file}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"ANALYSIS COMPLETE: {symbol}")
    print("=" * 60)
    print(f"\nPrice: ${results['price']:.2f} ({results['change_pct']:+.2f}%)")
    print(f"Pattern Score: {results['score']}")
    print(f"\nActive Signals ({len(results['signals'])}):")

    for signal_name, signal_type, description in results['signals']:
        print(f"  [{signal_type.upper()}] {signal_name}: {description}")

    if not results['signals']:
        print("  No significant patterns detected")

    return results, html_file


if __name__ == "__main__":
    results, html_file = main()
