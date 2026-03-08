import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# S&P 500 sample - major stocks
symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CMCSA',
    'NFLX', 'XOM', 'VZ', 'INTC', 'T', 'PFE', 'KO', 'MRK', 'PEP', 'ABT',
    'CVX', 'CSCO', 'WMT', 'CRM', 'AMD', 'COST', 'ABBV', 'ACN', 'AVGO', 'TXN',
    'NKE', 'QCOM', 'DHR', 'MDT', 'NEE', 'LIN', 'UNP', 'BMY', 'PM', 'ORCL',
    'TMO', 'LOW', 'HON', 'AMGN', 'IBM', 'SBUX', 'GS', 'BLK', 'CAT', 'BA',
    'COIN', 'PLTR', 'SQ', 'SNAP', 'UBER', 'LYFT', 'ROKU', 'ZM', 'SHOP', 'SPOT'
]

print('='*70)
print('  TRENDLINE BREAKOUT SCANNER')
print('='*70)
print(f'\nScanning {len(symbols)} stocks for trendline breakouts...\n')

def find_pivot_points(df, left=5, right=5):
    """Find swing highs and lows"""
    highs = []
    lows = []

    for i in range(left, len(df) - right):
        # Swing high
        if df['High'].iloc[i] == df['High'].iloc[i-left:i+right+1].max():
            highs.append((i, df['High'].iloc[i]))
        # Swing low
        if df['Low'].iloc[i] == df['Low'].iloc[i-left:i+right+1].min():
            lows.append((i, df['Low'].iloc[i]))

    return highs, lows

def detect_trendline_breakout(symbol):
    """Detect trendline breakouts"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='6mo')

        if len(df) < 60:
            return None

        df['Vol_SMA'] = df['Volume'].rolling(20).mean()

        highs, lows = find_pivot_points(df)

        current = df.iloc[-1]
        prev = df.iloc[-2]

        results = []

        # Check resistance breakout (need at least 2 swing highs)
        if len(highs) >= 2:
            recent_highs = highs[-3:] if len(highs) >= 3 else highs[-2:]
            resistance_level = max([h[1] for h in recent_highs])

            # Breakout above resistance with volume
            if current['Close'] > resistance_level and prev['Close'] <= resistance_level:
                vol_ratio = current['Volume'] / current['Vol_SMA'] if current['Vol_SMA'] > 0 else 1
                if vol_ratio > 1.2:
                    results.append({
                        'symbol': symbol,
                        'type': 'RESISTANCE BREAKOUT',
                        'signal': 'BULLISH',
                        'price': current['Close'],
                        'level': resistance_level,
                        'volume': f'{vol_ratio:.1f}x',
                        'change': ((current['Close']/prev['Close'])-1)*100
                    })

        # Check support breakdown (need at least 2 swing lows)
        if len(lows) >= 2:
            recent_lows = lows[-3:] if len(lows) >= 3 else lows[-2:]
            support_level = min([l[1] for l in recent_lows])

            # Breakdown below support with volume
            if current['Close'] < support_level and prev['Close'] >= support_level:
                vol_ratio = current['Volume'] / current['Vol_SMA'] if current['Vol_SMA'] > 0 else 1
                if vol_ratio > 1.2:
                    results.append({
                        'symbol': symbol,
                        'type': 'SUPPORT BREAKDOWN',
                        'signal': 'BEARISH',
                        'price': current['Close'],
                        'level': support_level,
                        'volume': f'{vol_ratio:.1f}x',
                        'change': ((current['Close']/prev['Close'])-1)*100
                    })

        # Check for potential breakout (within 2% of resistance)
        if len(highs) >= 2 and not results:
            recent_highs = highs[-3:] if len(highs) >= 3 else highs[-2:]
            resistance_level = max([h[1] for h in recent_highs])

            distance_pct = (resistance_level - current['Close']) / current['Close'] * 100
            if 0 < distance_pct < 2:
                vol_ratio = current['Volume'] / current['Vol_SMA'] if current['Vol_SMA'] > 0 else 1
                results.append({
                    'symbol': symbol,
                    'type': 'APPROACHING RESISTANCE',
                    'signal': 'WATCH',
                    'price': current['Close'],
                    'level': resistance_level,
                    'volume': f'{vol_ratio:.1f}x',
                    'change': ((current['Close']/prev['Close'])-1)*100
                })

        # Check for potential breakdown (within 2% of support)
        if len(lows) >= 2 and not results:
            recent_lows = lows[-3:] if len(lows) >= 3 else lows[-2:]
            support_level = min([l[1] for l in recent_lows])

            distance_pct = (current['Close'] - support_level) / current['Close'] * 100
            if 0 < distance_pct < 2:
                vol_ratio = current['Volume'] / current['Vol_SMA'] if current['Vol_SMA'] > 0 else 1
                results.append({
                    'symbol': symbol,
                    'type': 'APPROACHING SUPPORT',
                    'signal': 'WATCH',
                    'price': current['Close'],
                    'level': support_level,
                    'volume': f'{vol_ratio:.1f}x',
                    'change': ((current['Close']/prev['Close'])-1)*100
                })

        return results if results else None

    except Exception as e:
        return None

# Scan all symbols
breakouts = []
approaching = []

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(detect_trendline_breakout, sym): sym for sym in symbols}

    completed = 0
    for future in as_completed(futures):
        completed += 1
        if completed % 20 == 0:
            print(f'  Scanned {completed}/{len(symbols)} stocks...')

        result = future.result()
        if result:
            for r in result:
                if 'BREAKOUT' in r['type'] or 'BREAKDOWN' in r['type']:
                    breakouts.append(r)
                else:
                    approaching.append(r)

print(f'\nScan complete!')

# Display results
print('\n' + '='*70)
print('  CONFIRMED BREAKOUTS/BREAKDOWNS')
print('='*70)

if breakouts:
    # Sort by change
    breakouts.sort(key=lambda x: abs(x['change']), reverse=True)
    print(f'\n{"Symbol":<8} {"Type":<22} {"Signal":<8} {"Price":<10} {"Level":<10} {"Volume":<8} {"Change":<8}')
    print('-'*70)
    for b in breakouts:
        signal_color = 'BULLISH' if b['signal'] == 'BULLISH' else 'BEARISH'
        print(f'{b["symbol"]:<8} {b["type"]:<22} {signal_color:<8} ${b["price"]:<9.2f} ${b["level"]:<9.2f} {b["volume"]:<8} {b["change"]:+.2f}%')
else:
    print('\nNo confirmed breakouts/breakdowns found today.')

print('\n' + '='*70)
print('  APPROACHING KEY LEVELS (Watch List)')
print('='*70)

if approaching:
    # Sort by absolute change
    approaching.sort(key=lambda x: abs(x['change']), reverse=True)
    print(f'\n{"Symbol":<8} {"Type":<22} {"Signal":<8} {"Price":<10} {"Level":<10} {"Volume":<8} {"Change":<8}')
    print('-'*70)
    for a in approaching[:15]:  # Top 15
        print(f'{a["symbol"]:<8} {a["type"]:<22} {a["signal"]:<8} ${a["price"]:<9.2f} ${a["level"]:<9.2f} {a["volume"]:<8} {a["change"]:+.2f}%')
else:
    print('\nNo stocks approaching key levels.')

print('\n' + '='*70)
print(f'  Summary: {len(breakouts)} breakouts, {len(approaching)} approaching key levels')
print('='*70)
