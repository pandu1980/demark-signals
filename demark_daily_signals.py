"""
DeMark Daily Trade Signals Web App (PWA)
Features: Charts, Screening/Filtering, Portfolio Tracking, Options Trading
Install on mobile: Open in browser → Add to Home Screen
"""

from flask import Flask, render_template_string, jsonify, request, make_response
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

app = Flask(__name__)


# ============== PWA Files ==============

MANIFEST_JSON = '''{
    "name": "DeMark Trade Signals",
    "short_name": "DeMark",
    "description": "Daily trade signals based on DeMark Sequential indicators",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#0f1419",
    "theme_color": "#00d4aa",
    "orientation": "portrait-primary",
    "icons": [
        {
            "src": "/icon-192.png",
            "sizes": "192x192",
            "type": "image/png",
            "purpose": "any maskable"
        },
        {
            "src": "/icon-512.png",
            "sizes": "512x512",
            "type": "image/png",
            "purpose": "any maskable"
        }
    ]
}'''

SERVICE_WORKER_JS = '''
const CACHE_NAME = 'demark-signals-v1';
const STATIC_ASSETS = [
    '/',
    '/manifest.json',
    'https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.min.js'
];

// Install - cache static assets
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(STATIC_ASSETS);
        })
    );
    self.skipWaiting();
});

// Activate - clean old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys => {
            return Promise.all(
                keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
            );
        })
    );
    self.clients.claim();
});

// Fetch - network first, fallback to cache
self.addEventListener('fetch', event => {
    // Skip non-GET requests
    if (event.request.method !== 'GET') return;

    // For API calls, always try network first
    if (event.request.url.includes('/api/')) {
        event.respondWith(
            fetch(event.request)
                .then(response => {
                    // Clone and cache successful API responses
                    if (response.ok) {
                        const clone = response.clone();
                        caches.open(CACHE_NAME).then(cache => {
                            cache.put(event.request, clone);
                        });
                    }
                    return response;
                })
                .catch(() => {
                    // Fallback to cache if offline
                    return caches.match(event.request);
                })
        );
        return;
    }

    // For static assets, try cache first
    event.respondWith(
        caches.match(event.request).then(cached => {
            if (cached) return cached;
            return fetch(event.request).then(response => {
                if (response.ok) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => {
                        cache.put(event.request, clone);
                    });
                }
                return response;
            });
        })
    );
});
'''

# Simple SVG icons (will be converted to PNG-like data URLs)
ICON_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
    <rect width="512" height="512" rx="64" fill="#0f1419"/>
    <path d="M128 384 L256 128 L384 384" stroke="#00d4aa" stroke-width="40" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="256" cy="320" r="24" fill="#00d4aa"/>
    <text x="256" y="450" text-anchor="middle" font-family="Arial" font-size="64" font-weight="bold" fill="#00d4aa">DM</text>
</svg>'''


@app.route('/manifest.json')
def manifest():
    response = make_response(MANIFEST_JSON)
    response.headers['Content-Type'] = 'application/manifest+json'
    return response


@app.route('/sw.js')
def service_worker():
    response = make_response(SERVICE_WORKER_JS)
    response.headers['Content-Type'] = 'application/javascript'
    return response


@app.route('/icon-192.png')
@app.route('/icon-512.png')
def app_icon():
    # Return SVG as a simple icon (browsers handle this well)
    response = make_response(ICON_SVG)
    response.headers['Content-Type'] = 'image/svg+xml'
    return response

PORTFOLIO_FILE = "demark_portfolio.json"


# ============== Options Functions ==============

def get_options_chain(symbol: str, signal_type: str, price: float, strength: int) -> dict:
    """Get options chain and suggest trades based on DeMark signal"""
    try:
        stock = yf.Ticker(symbol)

        # Get available expiration dates
        expirations = stock.options
        if not expirations:
            return None

        # Filter expirations: 2-8 weeks out for short-term trades
        today = datetime.now().date()
        target_expirations = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days_to_exp = (exp_date - today).days
            if 14 <= days_to_exp <= 60:  # 2-8 weeks
                target_expirations.append(exp)

        if not target_expirations:
            # Fall back to nearest expiration
            target_expirations = expirations[:3]

        suggestions = []

        for exp in target_expirations[:3]:  # Max 3 expirations
            try:
                chain = stock.option_chain(exp)
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                days_to_exp = (exp_date - today).days

                if signal_type == "buy":
                    # Buy CALL options for bullish signals
                    calls = chain.calls
                    if calls.empty:
                        continue

                    # Find strikes near the money and slightly OTM
                    atm_strike = calls.iloc[(calls['strike'] - price).abs().argsort()[:1]]['strike'].values[0]

                    # Get OTM calls (strike > current price) within 5%
                    otm_calls = calls[(calls['strike'] >= price) & (calls['strike'] <= price * 1.05)]

                    if otm_calls.empty:
                        otm_calls = calls[calls['strike'] >= price].head(3)

                    for _, opt in otm_calls.head(2).iterrows():
                        strike = opt['strike']
                        bid = opt.get('bid', 0) or 0
                        ask = opt.get('ask', 0) or 0
                        mid_price = (bid + ask) / 2 if bid and ask else opt.get('lastPrice', 0)
                        volume = opt.get('volume', 0) or 0
                        oi = opt.get('openInterest', 0) or 0
                        iv = opt.get('impliedVolatility', 0) or 0

                        if mid_price > 0:
                            # Calculate basic metrics
                            otm_pct = ((strike - price) / price) * 100
                            breakeven = strike + mid_price
                            breakeven_pct = ((breakeven - price) / price) * 100

                            suggestions.append({
                                "type": "CALL",
                                "action": "BUY",
                                "strike": strike,
                                "expiration": exp,
                                "days_to_exp": days_to_exp,
                                "bid": round(bid, 2),
                                "ask": round(ask, 2),
                                "mid_price": round(mid_price, 2),
                                "volume": int(volume),
                                "open_interest": int(oi),
                                "iv": round(iv * 100, 1),
                                "otm_pct": round(otm_pct, 1),
                                "breakeven": round(breakeven, 2),
                                "breakeven_pct": round(breakeven_pct, 1),
                                "max_risk": round(mid_price * 100, 2),
                                "risk_reward": "Unlimited upside" if otm_pct < 3 else f"Need {breakeven_pct:.1f}% move"
                            })

                else:  # sell signal
                    # Buy PUT options for bearish signals
                    puts = chain.puts
                    if puts.empty:
                        continue

                    # Get OTM puts (strike < current price) within 5%
                    otm_puts = puts[(puts['strike'] <= price) & (puts['strike'] >= price * 0.95)]

                    if otm_puts.empty:
                        otm_puts = puts[puts['strike'] <= price].tail(3)

                    for _, opt in otm_puts.tail(2).iterrows():
                        strike = opt['strike']
                        bid = opt.get('bid', 0) or 0
                        ask = opt.get('ask', 0) or 0
                        mid_price = (bid + ask) / 2 if bid and ask else opt.get('lastPrice', 0)
                        volume = opt.get('volume', 0) or 0
                        oi = opt.get('openInterest', 0) or 0
                        iv = opt.get('impliedVolatility', 0) or 0

                        if mid_price > 0:
                            otm_pct = ((price - strike) / price) * 100
                            breakeven = strike - mid_price
                            breakeven_pct = ((price - breakeven) / price) * 100

                            suggestions.append({
                                "type": "PUT",
                                "action": "BUY",
                                "strike": strike,
                                "expiration": exp,
                                "days_to_exp": days_to_exp,
                                "bid": round(bid, 2),
                                "ask": round(ask, 2),
                                "mid_price": round(mid_price, 2),
                                "volume": int(volume),
                                "open_interest": int(oi),
                                "iv": round(iv * 100, 1),
                                "otm_pct": round(otm_pct, 1),
                                "breakeven": round(breakeven, 2),
                                "breakeven_pct": round(breakeven_pct, 1),
                                "max_risk": round(mid_price * 100, 2),
                                "risk_reward": f"Need {breakeven_pct:.1f}% drop"
                            })
            except Exception as e:
                continue

        # Sort by days to expiration and OTM %
        suggestions.sort(key=lambda x: (x['days_to_exp'], x['otm_pct']))

        # Add spread suggestions for stronger signals
        spreads = []
        if strength >= 4 and len(suggestions) >= 2:
            if signal_type == "buy":
                spreads.append({
                    "strategy": "Bull Call Spread",
                    "description": f"Buy {suggestions[0]['strike']} Call, Sell higher strike Call",
                    "risk": "Limited to net debit",
                    "reward": "Limited to spread width minus debit"
                })
            else:
                spreads.append({
                    "strategy": "Bear Put Spread",
                    "description": f"Buy {suggestions[0]['strike']} Put, Sell lower strike Put",
                    "risk": "Limited to net debit",
                    "reward": "Limited to spread width minus debit"
                })

        return {
            "symbol": symbol,
            "price": price,
            "signal_type": signal_type,
            "strength": strength,
            "suggestions": suggestions[:6],  # Top 6 suggestions
            "spreads": spreads,
            "expirations_available": len(expirations)
        }
    except Exception as e:
        return None


def get_options_for_signals(signals: list) -> list:
    """Get options suggestions for all signals"""
    options_data = []

    # Only process strong signals (strength >= 3)
    strong_signals = [s for s in signals if s.get('strength', 0) >= 3]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for sig in strong_signals[:20]:  # Limit to top 20
            future = executor.submit(
                get_options_chain,
                sig['symbol'],
                sig['signal_type'],
                sig['price'],
                sig['strength']
            )
            futures[future] = sig

        for future in as_completed(futures):
            result = future.result()
            if result and result.get('suggestions'):
                sig = futures[future]
                result['signal'] = sig['signal']
                result['trade_idea'] = sig['trade_idea']
                options_data.append(result)

    # Sort by signal strength
    options_data.sort(key=lambda x: -x['strength'])
    return options_data

# ============== DeMark Calculation Functions ==============

def calculate_td_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TD Sequential Setup counts (1-9)"""
    n = len(df)
    df = df.copy()
    df["buy_setup"] = 0
    df["sell_setup"] = 0
    df["buy_setup_complete"] = False
    df["sell_setup_complete"] = False
    df["buy_perfected"] = False
    df["sell_perfected"] = False

    buy_count = 0
    sell_count = 0

    for i in range(4, n):
        close_now = df["Close"].iloc[i]
        close_4_ago = df["Close"].iloc[i - 4]

        if close_now < close_4_ago:
            buy_count += 1
            sell_count = 0
        elif close_now > close_4_ago:
            sell_count += 1
            buy_count = 0
        else:
            buy_count = 0
            sell_count = 0

        df.iloc[i, df.columns.get_loc("buy_setup")] = min(buy_count, 9)
        df.iloc[i, df.columns.get_loc("sell_setup")] = min(sell_count, 9)

        if buy_count == 9:
            df.iloc[i, df.columns.get_loc("buy_setup_complete")] = True
            low_8 = df["Low"].iloc[i - 1]
            low_9 = df["Low"].iloc[i]
            low_6 = df["Low"].iloc[i - 3]
            low_7 = df["Low"].iloc[i - 2]
            if min(low_8, low_9) < min(low_6, low_7):
                df.iloc[i, df.columns.get_loc("buy_perfected")] = True
            buy_count = 0

        if sell_count == 9:
            df.iloc[i, df.columns.get_loc("sell_setup_complete")] = True
            high_8 = df["High"].iloc[i - 1]
            high_9 = df["High"].iloc[i]
            high_6 = df["High"].iloc[i - 3]
            high_7 = df["High"].iloc[i - 2]
            if max(high_8, high_9) > max(high_6, high_7):
                df.iloc[i, df.columns.get_loc("sell_perfected")] = True
            sell_count = 0

    return df


def calculate_td_countdown(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TD Sequential Countdown (1-13)"""
    df = df.copy()
    df["buy_countdown"] = 0
    df["sell_countdown"] = 0
    df["buy_countdown_complete"] = False
    df["sell_countdown_complete"] = False

    buy_cd_count = 0
    sell_cd_count = 0
    in_buy_countdown = False
    in_sell_countdown = False

    for i in range(2, len(df)):
        if df["buy_setup_complete"].iloc[i]:
            in_buy_countdown = True
            buy_cd_count = 0
        if df["sell_setup_complete"].iloc[i]:
            in_sell_countdown = True
            sell_cd_count = 0

        if in_buy_countdown and i >= 2:
            if df["Close"].iloc[i] <= df["Low"].iloc[i - 2]:
                buy_cd_count += 1
            df.iloc[i, df.columns.get_loc("buy_countdown")] = min(buy_cd_count, 13)
            if buy_cd_count >= 13:
                df.iloc[i, df.columns.get_loc("buy_countdown_complete")] = True
                in_buy_countdown = False
                buy_cd_count = 0

        if in_sell_countdown and i >= 2:
            if df["Close"].iloc[i] >= df["High"].iloc[i - 2]:
                sell_cd_count += 1
            df.iloc[i, df.columns.get_loc("sell_countdown")] = min(sell_cd_count, 13)
            if sell_cd_count >= 13:
                df.iloc[i, df.columns.get_loc("sell_countdown_complete")] = True
                in_sell_countdown = False
                sell_cd_count = 0

    return df


def calculate_td_combo(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TD Combo Countdown"""
    df = df.copy()
    df["buy_combo"] = 0
    df["sell_combo"] = 0
    df["buy_combo_complete"] = False
    df["sell_combo_complete"] = False

    buy_combo_count = 0
    sell_combo_count = 0

    for i in range(2, len(df)):
        if df["buy_setup"].iloc[i] == 1 and (i == 0 or df["buy_setup"].iloc[i-1] == 0):
            buy_combo_count = 0
        if df["sell_setup"].iloc[i] == 1 and (i == 0 or df["sell_setup"].iloc[i-1] == 0):
            sell_combo_count = 0

        if df["buy_setup"].iloc[i] > 0 and i >= 2:
            close_now = df["Close"].iloc[i]
            low_2_ago = df["Low"].iloc[i - 2]
            close_1_ago = df["Close"].iloc[i - 1]
            if close_now <= low_2_ago and close_now < close_1_ago:
                buy_combo_count += 1
            df.iloc[i, df.columns.get_loc("buy_combo")] = min(buy_combo_count, 13)
            if buy_combo_count >= 13:
                df.iloc[i, df.columns.get_loc("buy_combo_complete")] = True

        if df["sell_setup"].iloc[i] > 0 and i >= 2:
            close_now = df["Close"].iloc[i]
            high_2_ago = df["High"].iloc[i - 2]
            close_1_ago = df["Close"].iloc[i - 1]
            if close_now >= high_2_ago and close_now > close_1_ago:
                sell_combo_count += 1
            df.iloc[i, df.columns.get_loc("sell_combo")] = min(sell_combo_count, 13)
            if sell_combo_count >= 13:
                df.iloc[i, df.columns.get_loc("sell_combo_complete")] = True

    return df


def get_stock_data(symbol: str, days: int = 120) -> dict:
    """Get full stock data with DeMark signals for charting"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)

        if df.empty or len(df) < 20:
            return None

        df = calculate_td_setup(df)
        df = calculate_td_countdown(df)
        df = calculate_td_combo(df)

        # Prepare chart data
        chart_data = []
        for i, (idx, row) in enumerate(df.iterrows()):
            bar = {
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
                "buy_setup": int(row["buy_setup"]),
                "sell_setup": int(row["sell_setup"]),
                "buy_countdown": int(row["buy_countdown"]),
                "sell_countdown": int(row["sell_countdown"]),
                "buy_setup_complete": bool(row["buy_setup_complete"]),
                "sell_setup_complete": bool(row["sell_setup_complete"]),
                "buy_perfected": bool(row["buy_perfected"]),
                "sell_perfected": bool(row["sell_perfected"]),
            }
            chart_data.append(bar)

        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current
        price = current["Close"]
        change_pct = ((price - prev["Close"]) / prev["Close"]) * 100

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "chart_data": chart_data,
            "current": {
                "buy_setup": int(current["buy_setup"]),
                "sell_setup": int(current["sell_setup"]),
                "buy_countdown": int(current["buy_countdown"]),
                "sell_countdown": int(current["sell_countdown"]),
            }
        }
    except Exception as e:
        return None


def get_demark_signals(symbol: str, days: int = 120) -> dict:
    """Get DeMark signals for a stock"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)

        if df.empty or len(df) < 20:
            return None

        df = calculate_td_setup(df)
        df = calculate_td_countdown(df)
        df = calculate_td_combo(df)

        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current
        price = current["Close"]
        prev_price = prev["Close"]
        change_pct = ((price - prev_price) / prev_price) * 100

        # Check recent setups
        recent_buy_setup = df["buy_setup_complete"].iloc[-5:].any()
        recent_sell_setup = df["sell_setup_complete"].iloc[-5:].any()

        buy_setup_count = int(current["buy_setup"])
        sell_setup_count = int(current["sell_setup"])
        buy_combo_count = int(current["buy_combo"])
        sell_combo_count = int(current["sell_combo"])
        buy_countdown = int(current["buy_countdown"])
        sell_countdown = int(current["sell_countdown"])

        # Determine signal type and strength
        signal = "NEUTRAL"
        signal_type = "neutral"
        strength = 0
        trade_idea = ""

        if current["buy_setup_complete"]:
            if current["buy_perfected"]:
                signal = "BUY 9 PERFECTED"
                signal_type = "buy"
                strength = 5
                trade_idea = "Strong bullish reversal. Consider long entry with stop below recent low."
            else:
                signal = "BUY SETUP 9"
                signal_type = "buy"
                strength = 4
                trade_idea = "Bullish setup complete. Watch for reversal confirmation."
        elif current["sell_setup_complete"]:
            if current["sell_perfected"]:
                signal = "SELL 9 PERFECTED"
                signal_type = "sell"
                strength = 5
                trade_idea = "Strong bearish reversal. Consider short or exit longs."
            else:
                signal = "SELL SETUP 9"
                signal_type = "sell"
                strength = 4
                trade_idea = "Bearish setup complete. Watch for reversal confirmation."
        elif current["buy_countdown_complete"]:
            signal = "BUY COUNTDOWN 13"
            signal_type = "buy"
            strength = 5
            trade_idea = "Full buy countdown complete. Major reversal zone."
        elif current["sell_countdown_complete"]:
            signal = "SELL COUNTDOWN 13"
            signal_type = "sell"
            strength = 5
            trade_idea = "Full sell countdown complete. Major reversal zone."
        elif buy_setup_count >= 7:
            signal = f"BUY SETUP {buy_setup_count}"
            signal_type = "buy"
            strength = buy_setup_count - 5
            trade_idea = f"Building buy setup ({buy_setup_count}/9). Watch for completion."
        elif sell_setup_count >= 7:
            signal = f"SELL SETUP {sell_setup_count}"
            signal_type = "sell"
            strength = sell_setup_count - 5
            trade_idea = f"Building sell setup ({sell_setup_count}/9). Watch for completion."
        elif recent_buy_setup:
            signal = "RECENT BUY 9"
            signal_type = "buy"
            strength = 3
            trade_idea = "Recent buy setup. In potential reversal zone."
        elif recent_sell_setup:
            signal = "RECENT SELL 9"
            signal_type = "sell"
            strength = 3
            trade_idea = "Recent sell setup. In potential reversal zone."
        elif buy_countdown >= 10:
            signal = f"BUY CD {buy_countdown}"
            signal_type = "buy"
            strength = 2
            trade_idea = f"Buy countdown progressing ({buy_countdown}/13)."
        elif sell_countdown >= 10:
            signal = f"SELL CD {sell_countdown}"
            signal_type = "sell"
            strength = 2
            trade_idea = f"Sell countdown progressing ({sell_countdown}/13)."

        if buy_combo_count >= 10:
            strength += 1
            trade_idea += f" Combo at {buy_combo_count}/13."
        if sell_combo_count >= 10:
            strength += 1
            trade_idea += f" Combo at {sell_combo_count}/13."

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "buy_setup": buy_setup_count,
            "sell_setup": sell_setup_count,
            "buy_countdown": buy_countdown,
            "sell_countdown": sell_countdown,
            "buy_combo": buy_combo_count,
            "sell_combo": sell_combo_count,
            "buy_perfected": bool(current["buy_perfected"]),
            "sell_perfected": bool(current["sell_perfected"]),
            "signal": signal,
            "signal_type": signal_type,
            "strength": min(strength, 5),
            "trade_idea": trade_idea.strip(),
            "date": df.index[-1].strftime("%Y-%m-%d")
        }
    except Exception as e:
        return None


def scan_stocks(symbols: list, use_cache: bool = True) -> list:
    """Scan multiple stocks in parallel with caching"""
    global _scan_cache

    # Check cache
    if use_cache and _scan_cache["data"] is not None:
        cache_age = (datetime.now() - _scan_cache["timestamp"]).total_seconds()
        if cache_age < CACHE_DURATION:
            return _scan_cache["data"]

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced workers for cloud
        futures = {executor.submit(get_demark_signals, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result["signal"] != "NEUTRAL":
                    results.append(result)
            except:
                pass

    results.sort(key=lambda x: (-x["strength"], x["symbol"]))

    # Update cache
    _scan_cache["data"] = results
    _scan_cache["timestamp"] = datetime.now()

    return results


# Portfolio functions
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return {"positions": [], "watchlist": []}


def save_portfolio(data):
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# Stock Universe - Optimized for cloud (top 75 most traded)
ALL_STOCKS = [
    # Mega Cap Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Tech
    "AMD", "AVGO", "CRM", "ADBE", "ORCL", "INTC", "QCOM", "MU", "AMAT",
    # AI & Cloud
    "PLTR", "SNOW", "NET", "CRWD", "PANW", "DDOG", "SMCI",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY", "AMGN",
    # Consumer
    "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "PG", "KO", "PEP",
    # Media & Entertainment
    "DIS", "NFLX", "CMCSA",
    # Industrial
    "BA", "CAT", "GE", "HON", "UPS", "LMT", "RTX", "DE",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY",
    # EV & Clean
    "RIVN", "LCID", "ENPH", "FSLR",
    # Fintech
    "PYPL", "COIN", "SOFI", "AFRM",
    # ETFs
    "SPY", "QQQ", "IWM",
]

# Cache for scan results (5 min cache)
_scan_cache = {"data": None, "timestamp": None}
CACHE_DURATION = 300  # 5 minutes


# ============== HTML Template ==============

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>DeMark Trade Signals</title>

    <!-- PWA Meta Tags -->
    <meta name="theme-color" content="#00d4aa">
    <meta name="description" content="Daily trade signals based on DeMark Sequential indicators">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="DeMark">

    <!-- PWA Manifest -->
    <link rel="manifest" href="/manifest.json">

    <!-- Icons -->
    <link rel="icon" type="image/svg+xml" href="/icon-192.png">
    <link rel="apple-touch-icon" href="/icon-192.png">

    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1419;
            color: #e7e9ea;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
            padding: 16px 20px;
            border-bottom: 1px solid #2f3336;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(90deg, #00d4aa, #00a8e8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .tabs {
            display: flex;
            gap: 4px;
        }
        .tab {
            padding: 8px 16px;
            border: none;
            background: transparent;
            color: #71767b;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .tab:hover { color: #e7e9ea; background: #1a1f2e; }
        .tab.active { color: #00d4aa; background: rgba(0,212,170,0.1); }
        .controls {
            display: flex;
            gap: 12px;
            margin-top: 12px;
            flex-wrap: wrap;
            align-items: center;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #00d4aa, #00a8e8);
            color: #0f1419;
        }
        .btn-primary:hover { transform: translateY(-1px); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-secondary {
            background: #2f3336;
            color: #e7e9ea;
        }
        .btn-secondary:hover { background: #3a3f42; }
        .btn-secondary.active { background: #00d4aa; color: #0f1419; }
        .btn-small {
            padding: 4px 10px;
            font-size: 11px;
        }
        .filter-group { display: flex; gap: 6px; }
        .search-box {
            padding: 8px 12px;
            border: 1px solid #2f3336;
            border-radius: 6px;
            background: #1a1f2e;
            color: #e7e9ea;
            font-size: 13px;
            width: 200px;
        }
        .search-box:focus { outline: none; border-color: #00d4aa; }
        .status {
            margin-left: auto;
            color: #71767b;
            font-size: 12px;
        }
        .status.scanning { color: #00d4aa; }

        .main-content { display: flex; height: calc(100vh - 110px); }
        .panel { background: #0f1419; overflow: hidden; display: flex; flex-direction: column; }
        .panel-signals { flex: 1; border-right: 1px solid #2f3336; min-width: 400px; }
        .panel-chart { flex: 2; display: none; }
        .panel-chart.visible { display: flex; flex-direction: column; }

        .panel-header {
            padding: 12px 16px;
            border-bottom: 1px solid #2f3336;
            background: #1a1f2e;
            font-weight: 600;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .panel-content { flex: 1; overflow-y: auto; padding: 12px; }

        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin-bottom: 12px;
        }
        .stat-card {
            background: #1a1f2e;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            border: 1px solid #2f3336;
        }
        .stat-card .value { font-size: 20px; font-weight: 700; }
        .stat-card .label { font-size: 10px; color: #71767b; margin-top: 2px; }
        .stat-card.buy .value { color: #00d4aa; }
        .stat-card.sell .value { color: #f23645; }
        .stat-card.strong .value { color: #ffd700; }

        .signal-card {
            background: #1a1f2e;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #2f3336;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .signal-card:hover { border-color: #00d4aa; }
        .signal-card.selected { border-color: #00d4aa; background: rgba(0,212,170,0.05); }
        .signal-card.buy { border-left: 3px solid #00d4aa; }
        .signal-card.sell { border-left: 3px solid #f23645; }

        .signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .symbol { font-size: 16px; font-weight: 700; }
        .price-info { text-align: right; }
        .price { font-size: 14px; }
        .change { font-size: 11px; }
        .change.up { color: #00d4aa; }
        .change.down { color: #f23645; }

        .signal-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .signal-badge.buy { background: rgba(0,212,170,0.15); color: #00d4aa; }
        .signal-badge.sell { background: rgba(242,54,69,0.15); color: #f23645; }

        .trade-idea { font-size: 12px; color: #b5bac1; margin-top: 6px; line-height: 1.4; }

        .metrics { display: flex; gap: 12px; margin-top: 6px; }
        .metric { font-size: 11px; color: #71767b; }
        .metric span { color: #e7e9ea; font-weight: 600; }

        .strength-dots { display: flex; gap: 3px; }
        .strength-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: #2f3336;
        }
        .strength-dot.filled.buy { background: #00d4aa; }
        .strength-dot.filled.sell { background: #f23645; }

        #chart-container { flex: 1; min-height: 400px; }

        .chart-info {
            padding: 12px 16px;
            background: #1a1f2e;
            border-top: 1px solid #2f3336;
        }
        .chart-symbol { font-size: 20px; font-weight: 700; }
        .chart-price { font-size: 16px; margin-left: 12px; }
        .chart-signal { margin-top: 8px; }

        .portfolio-section { margin-top: 16px; }
        .portfolio-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .portfolio-title { font-size: 13px; font-weight: 600; color: #71767b; }

        .position-card {
            background: #1a1f2e;
            border-radius: 6px;
            padding: 10px;
            border: 1px solid #2f3336;
            margin-bottom: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .position-symbol { font-weight: 600; }
        .position-details { font-size: 11px; color: #71767b; }
        .position-pnl { font-weight: 600; }
        .position-pnl.profit { color: #00d4aa; }
        .position-pnl.loss { color: #f23645; }

        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #71767b;
        }
        .empty-state h3 { font-size: 16px; margin-bottom: 6px; color: #e7e9ea; }

        .loading { display: flex; align-items: center; justify-content: center; padding: 40px; }
        .spinner {
            width: 30px; height: 30px;
            border: 2px solid #2f3336;
            border-top-color: #00d4aa;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-overlay.visible { display: flex; }
        .modal {
            background: #1a1f2e;
            border-radius: 12px;
            padding: 20px;
            width: 90%;
            max-width: 400px;
            border: 1px solid #2f3336;
        }
        .modal h3 { margin-bottom: 16px; }
        .form-group { margin-bottom: 12px; }
        .form-group label { display: block; font-size: 12px; color: #71767b; margin-bottom: 4px; }
        .form-group input {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #2f3336;
            border-radius: 6px;
            background: #0f1419;
            color: #e7e9ea;
            font-size: 14px;
        }
        .form-group input:focus { outline: none; border-color: #00d4aa; }
        .modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }

        .disclaimer {
            text-align: center;
            padding: 12px;
            color: #71767b;
            font-size: 10px;
            border-top: 1px solid #2f3336;
        }

        /* Offline indicator */
        .offline-banner {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: #f23645;
            color: white;
            text-align: center;
            padding: 8px;
            font-size: 12px;
            z-index: 2000;
            display: none;
        }
        .offline-banner.visible { display: block; }

        /* Mobile responsive */
        @media (max-width: 768px) {
            .header h1 { font-size: 18px; }
            .tabs { gap: 2px; }
            .tab { padding: 6px 10px; font-size: 12px; }
            .controls { gap: 8px; }
            .btn { padding: 6px 12px; font-size: 12px; }
            .search-box { width: 120px; font-size: 12px; }
            .filter-group { display: none; }
            .main-content { flex-direction: column; height: auto; }
            .panel-signals { min-width: 100%; border-right: none; border-bottom: 1px solid #2f3336; }
            .panel-chart { min-height: 400px; }
            .signal-card { padding: 10px; }
            .symbol { font-size: 14px; }
            .trade-idea { font-size: 11px; }
            .option-row { grid-template-columns: 70px 1fr 80px; font-size: 11px; }
            .option-metrics { display: none; }
            .stat-card { padding: 8px; }
            .stat-card .value { font-size: 18px; }
        }

        /* Safe area for notched phones */
        @supports (padding-top: env(safe-area-inset-top)) {
            .header { padding-top: calc(12px + env(safe-area-inset-top)); }
            .disclaimer { padding-bottom: calc(12px + env(safe-area-inset-bottom)); }
        }

        /* Options Styles */
        .options-section { display: none; }
        .options-section.visible { display: block; }

        .options-card {
            background: #1a1f2e;
            border-radius: 10px;
            padding: 14px;
            border: 1px solid #2f3336;
            margin-bottom: 12px;
        }
        .options-card.buy { border-left: 3px solid #00d4aa; }
        .options-card.sell { border-left: 3px solid #f23645; }

        .options-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .options-symbol {
            font-size: 18px;
            font-weight: 700;
        }
        .options-signal {
            font-size: 12px;
            color: #71767b;
        }

        .options-grid {
            display: grid;
            gap: 8px;
        }
        .option-row {
            background: #0f1419;
            border-radius: 6px;
            padding: 10px 12px;
            display: grid;
            grid-template-columns: 80px 1fr 100px 80px;
            gap: 12px;
            align-items: center;
            font-size: 13px;
        }
        .option-type {
            font-weight: 700;
            padding: 4px 8px;
            border-radius: 4px;
            text-align: center;
            font-size: 11px;
        }
        .option-type.call { background: rgba(0,212,170,0.15); color: #00d4aa; }
        .option-type.put { background: rgba(242,54,69,0.15); color: #f23645; }

        .option-details {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .option-strike { font-weight: 600; }
        .option-exp { font-size: 11px; color: #71767b; }

        .option-price {
            text-align: center;
        }
        .option-mid { font-weight: 600; font-size: 14px; }
        .option-spread { font-size: 10px; color: #71767b; }

        .option-metrics {
            text-align: right;
            font-size: 11px;
            color: #71767b;
        }
        .option-iv { color: #ffd700; }

        .spread-suggestion {
            background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(0,168,232,0.1));
            border-radius: 6px;
            padding: 10px 12px;
            margin-top: 8px;
            font-size: 12px;
        }
        .spread-name { font-weight: 600; color: #00d4aa; margin-bottom: 4px; }
        .spread-desc { color: #b5bac1; }

        .options-footer {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #2f3336;
            font-size: 11px;
            color: #71767b;
        }
    </style>
</head>
<body>
    <div class="offline-banner" id="offlineBanner">You're offline. Showing cached data.</div>
    <div class="header">
        <div class="header-top">
            <h1>DeMark Daily Signals</h1>
            <div class="tabs">
                <button class="tab active" onclick="switchTab('signals')">Signals</button>
                <button class="tab" onclick="switchTab('options')">Options</button>
                <button class="tab" onclick="switchTab('portfolio')">Portfolio</button>
            </div>
        </div>
        <div class="controls">
            <button id="scanBtn" class="btn btn-primary" onclick="startScan()">Scan Market</button>
            <input type="text" class="search-box" placeholder="Search symbol..." id="searchBox" onkeyup="filterBySearch()">
            <div class="filter-group">
                <button class="btn btn-secondary btn-small active" data-filter="all" onclick="filterSignals('all', this)">All</button>
                <button class="btn btn-secondary btn-small" data-filter="buy" onclick="filterSignals('buy', this)">Buy</button>
                <button class="btn btn-secondary btn-small" data-filter="sell" onclick="filterSignals('sell', this)">Sell</button>
                <button class="btn btn-secondary btn-small" data-filter="strong" onclick="filterSignals('strong', this)">Strong</button>
            </div>
            <div id="status" class="status"></div>
        </div>
    </div>

    <div class="main-content">
        <div class="panel panel-signals">
            <div class="panel-header">
                <span>Trade Signals</span>
                <span id="signalCount">0 signals</span>
            </div>
            <div class="panel-content" id="signalsPanel">
                <div id="stats" class="stats-row"></div>
                <div id="signals"></div>

                <div class="options-section" id="optionsSection">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                        <span style="font-size:13px;color:#71767b;">Options trades based on DeMark signals (strength 3+)</span>
                        <button class="btn btn-primary btn-small" onclick="loadOptions()">Load Options</button>
                    </div>
                    <div id="optionsList"></div>
                </div>

                <div class="portfolio-section" id="portfolioSection" style="display:none;">
                    <div class="portfolio-header">
                        <span class="portfolio-title">MY POSITIONS</span>
                        <button class="btn btn-secondary btn-small" onclick="showAddPosition()">+ Add</button>
                    </div>
                    <div id="positions"></div>

                    <div class="portfolio-header" style="margin-top:16px;">
                        <span class="portfolio-title">WATCHLIST</span>
                        <button class="btn btn-secondary btn-small" onclick="showAddWatchlist()">+ Add</button>
                    </div>
                    <div id="watchlist"></div>
                </div>
            </div>
        </div>

        <div class="panel panel-chart" id="chartPanel">
            <div class="panel-header">
                <span id="chartTitle">Select a signal to view chart</span>
                <button class="btn btn-secondary btn-small" onclick="closeChart()">Close</button>
            </div>
            <div id="chart-container"></div>
            <div class="chart-info" id="chartInfo"></div>
        </div>
    </div>

    <div class="modal-overlay" id="addModal">
        <div class="modal">
            <h3 id="modalTitle">Add Position</h3>
            <div class="form-group">
                <label>Symbol</label>
                <input type="text" id="modalSymbol" placeholder="e.g., AAPL">
            </div>
            <div class="form-group" id="qtyGroup">
                <label>Quantity</label>
                <input type="number" id="modalQty" placeholder="e.g., 100">
            </div>
            <div class="form-group" id="priceGroup">
                <label>Entry Price</label>
                <input type="number" id="modalPrice" step="0.01" placeholder="e.g., 150.00">
            </div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                <button class="btn btn-primary" onclick="saveModal()">Save</button>
            </div>
        </div>
    </div>

    <div class="disclaimer">
        Educational purposes only. Not financial advice. DeMark indicators identify potential reversal zones.
    </div>

    <script>
        let allSignals = [];
        let currentFilter = 'all';
        let searchTerm = '';
        let chart = null;
        let candleSeries = null;
        let selectedSymbol = null;
        let currentTab = 'signals';
        let portfolio = { positions: [], watchlist: [] };
        let modalMode = 'position';

        async function startScan() {
            const btn = document.getElementById('scanBtn');
            const status = document.getElementById('status');
            btn.disabled = true;
            btn.textContent = 'Scanning...';
            status.textContent = 'Scanning...';
            status.className = 'status scanning';
            document.getElementById('signals').innerHTML = '<div class="loading"><div class="spinner"></div></div>';

            try {
                const response = await fetch('/api/scan');
                const data = await response.json();
                allSignals = data.signals || [];
                renderSignals();
                status.textContent = `${new Date().toLocaleTimeString()} | ${allSignals.length} signals`;
                status.className = 'status';
            } catch (error) {
                status.textContent = 'Scan failed';
                document.getElementById('signals').innerHTML = '<div class="empty-state"><h3>Scan failed</h3></div>';
            }

            btn.disabled = false;
            btn.textContent = 'Scan Market';
        }

        function filterSignals(filter, btn) {
            currentFilter = filter;
            document.querySelectorAll('.filter-group .btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderSignals();
        }

        function filterBySearch() {
            searchTerm = document.getElementById('searchBox').value.toUpperCase();
            renderSignals();
        }

        function renderSignals() {
            let filtered = allSignals;

            if (searchTerm) {
                filtered = filtered.filter(s => s.symbol.includes(searchTerm));
            }
            if (currentFilter === 'buy') {
                filtered = filtered.filter(s => s.signal_type === 'buy');
            } else if (currentFilter === 'sell') {
                filtered = filtered.filter(s => s.signal_type === 'sell');
            } else if (currentFilter === 'strong') {
                filtered = filtered.filter(s => s.strength >= 4);
            }

            const buyCount = allSignals.filter(s => s.signal_type === 'buy').length;
            const sellCount = allSignals.filter(s => s.signal_type === 'sell').length;
            const strongCount = allSignals.filter(s => s.strength >= 4).length;

            document.getElementById('stats').innerHTML = `
                <div class="stat-card buy"><div class="value">${buyCount}</div><div class="label">Buy</div></div>
                <div class="stat-card sell"><div class="value">${sellCount}</div><div class="label">Sell</div></div>
                <div class="stat-card strong"><div class="value">${strongCount}</div><div class="label">Strong</div></div>
                <div class="stat-card"><div class="value">${allSignals.length}</div><div class="label">Total</div></div>
            `;

            document.getElementById('signalCount').textContent = `${filtered.length} signals`;

            if (filtered.length === 0) {
                document.getElementById('signals').innerHTML = `
                    <div class="empty-state">
                        <h3>No signals found</h3>
                        <p>Click "Scan Market" to find signals</p>
                    </div>`;
                return;
            }

            document.getElementById('signals').innerHTML = filtered.map(s => `
                <div class="signal-card ${s.signal_type} ${selectedSymbol === s.symbol ? 'selected' : ''}" onclick="showChart('${s.symbol}')">
                    <div class="signal-header">
                        <div>
                            <span class="symbol">${s.symbol}</span>
                            <span class="signal-badge ${s.signal_type}">${s.signal}</span>
                        </div>
                        <div class="price-info">
                            <div class="price">$${s.price.toFixed(2)}</div>
                            <div class="change ${s.change_pct >= 0 ? 'up' : 'down'}">${s.change_pct >= 0 ? '+' : ''}${s.change_pct.toFixed(2)}%</div>
                        </div>
                    </div>
                    <div class="trade-idea">${s.trade_idea}</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;">
                        <div class="metrics">
                            <span class="metric">Setup: <span>${s.buy_setup || s.sell_setup}/9</span></span>
                            <span class="metric">CD: <span>${s.buy_countdown || s.sell_countdown}/13</span></span>
                        </div>
                        <div class="strength-dots">
                            ${[1,2,3,4,5].map(i => `<div class="strength-dot ${i <= s.strength ? 'filled ' + s.signal_type : ''}"></div>`).join('')}
                        </div>
                    </div>
                </div>
            `).join('');
        }

        async function showChart(symbol) {
            selectedSymbol = symbol;
            renderSignals();

            document.getElementById('chartPanel').classList.add('visible');
            document.getElementById('chartTitle').textContent = `Loading ${symbol}...`;

            try {
                const response = await fetch(`/api/chart/${symbol}`);
                const data = await response.json();

                if (data.error) {
                    document.getElementById('chartTitle').textContent = 'Error loading chart';
                    return;
                }

                document.getElementById('chartTitle').textContent = `${symbol} - DeMark Analysis`;

                if (chart) {
                    chart.remove();
                }

                const container = document.getElementById('chart-container');
                chart = LightweightCharts.createChart(container, {
                    width: container.clientWidth,
                    height: container.clientHeight,
                    layout: {
                        background: { color: '#0f1419' },
                        textColor: '#71767b',
                    },
                    grid: {
                        vertLines: { color: '#1a1f2e' },
                        horzLines: { color: '#1a1f2e' },
                    },
                    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                    rightPriceScale: { borderColor: '#2f3336' },
                    timeScale: { borderColor: '#2f3336' },
                });

                candleSeries = chart.addCandlestickSeries({
                    upColor: '#00d4aa',
                    downColor: '#f23645',
                    borderUpColor: '#00d4aa',
                    borderDownColor: '#f23645',
                    wickUpColor: '#00d4aa',
                    wickDownColor: '#f23645',
                });

                const chartData = data.chart_data.map(bar => ({
                    time: bar.date,
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                }));
                candleSeries.setData(chartData);

                // Add markers for DeMark signals
                const markers = [];
                data.chart_data.forEach(bar => {
                    if (bar.buy_setup_complete) {
                        markers.push({
                            time: bar.date,
                            position: 'belowBar',
                            color: '#00d4aa',
                            shape: 'arrowUp',
                            text: bar.buy_perfected ? 'B9 PERF' : 'B9',
                        });
                    }
                    if (bar.sell_setup_complete) {
                        markers.push({
                            time: bar.date,
                            position: 'aboveBar',
                            color: '#f23645',
                            shape: 'arrowDown',
                            text: bar.sell_perfected ? 'S9 PERF' : 'S9',
                        });
                    }
                    if (bar.buy_setup > 0 && bar.buy_setup < 9) {
                        markers.push({
                            time: bar.date,
                            position: 'belowBar',
                            color: 'rgba(0,212,170,0.5)',
                            shape: 'circle',
                            text: bar.buy_setup.toString(),
                        });
                    }
                    if (bar.sell_setup > 0 && bar.sell_setup < 9) {
                        markers.push({
                            time: bar.date,
                            position: 'aboveBar',
                            color: 'rgba(242,54,69,0.5)',
                            shape: 'circle',
                            text: bar.sell_setup.toString(),
                        });
                    }
                });
                candleSeries.setMarkers(markers);

                chart.timeScale().fitContent();

                // Update chart info
                const current = data.current;
                document.getElementById('chartInfo').innerHTML = `
                    <span class="chart-symbol">${symbol}</span>
                    <span class="chart-price">$${data.price}</span>
                    <span class="change ${data.change_pct >= 0 ? 'up' : 'down'}">${data.change_pct >= 0 ? '+' : ''}${data.change_pct}%</span>
                    <div class="chart-signal">
                        <span class="metric">Buy Setup: <span>${current.buy_setup}/9</span></span>
                        <span class="metric" style="margin-left:16px;">Sell Setup: <span>${current.sell_setup}/9</span></span>
                        <span class="metric" style="margin-left:16px;">Buy CD: <span>${current.buy_countdown}/13</span></span>
                        <span class="metric" style="margin-left:16px;">Sell CD: <span>${current.sell_countdown}/13</span></span>
                    </div>
                `;

                // Resize handler
                window.addEventListener('resize', () => {
                    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
                });

            } catch (error) {
                document.getElementById('chartTitle').textContent = 'Error loading chart';
            }
        }

        function closeChart() {
            document.getElementById('chartPanel').classList.remove('visible');
            selectedSymbol = null;
            renderSignals();
        }

        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            const tabIndex = tab === 'signals' ? 1 : tab === 'options' ? 2 : 3;
            document.querySelector(`.tab:nth-child(${tabIndex})`).classList.add('active');

            document.getElementById('stats').style.display = 'none';
            document.getElementById('signals').style.display = 'none';
            document.getElementById('optionsSection').classList.remove('visible');
            document.getElementById('portfolioSection').style.display = 'none';

            if (tab === 'signals') {
                document.getElementById('stats').style.display = 'grid';
                document.getElementById('signals').style.display = 'block';
            } else if (tab === 'options') {
                document.getElementById('optionsSection').classList.add('visible');
            } else {
                document.getElementById('portfolioSection').style.display = 'block';
                loadPortfolio();
            }
        }

        let optionsData = [];

        async function loadOptions() {
            const status = document.getElementById('status');
            status.textContent = 'Loading options...';
            status.className = 'status scanning';
            document.getElementById('optionsList').innerHTML = '<div class="loading"><div class="spinner"></div></div>';

            try {
                const response = await fetch('/api/options');
                const data = await response.json();
                optionsData = data.options || [];
                renderOptions();
                status.textContent = `${new Date().toLocaleTimeString()} | ${optionsData.length} options trades`;
                status.className = 'status';
            } catch (error) {
                status.textContent = 'Failed to load options';
                document.getElementById('optionsList').innerHTML = '<div class="empty-state"><h3>Failed to load options</h3></div>';
            }
        }

        function renderOptions() {
            if (optionsData.length === 0) {
                document.getElementById('optionsList').innerHTML = `
                    <div class="empty-state">
                        <h3>No options data</h3>
                        <p>Click "Load Options" to fetch options for active signals</p>
                    </div>`;
                return;
            }

            document.getElementById('optionsList').innerHTML = optionsData.map(opt => `
                <div class="options-card ${opt.signal_type}">
                    <div class="options-header">
                        <div>
                            <span class="options-symbol">${opt.symbol}</span>
                            <span class="signal-badge ${opt.signal_type}" style="margin-left:8px;">${opt.signal}</span>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:16px;font-weight:600;">$${opt.price}</div>
                            <div class="options-signal">Strength: ${opt.strength}/5</div>
                        </div>
                    </div>
                    <div class="trade-idea" style="margin-bottom:10px;">${opt.trade_idea}</div>
                    <div class="options-grid">
                        ${opt.suggestions.map(s => `
                            <div class="option-row">
                                <div class="option-type ${s.type.toLowerCase()}">${s.action} ${s.type}</div>
                                <div class="option-details">
                                    <span class="option-strike">$${s.strike} Strike</span>
                                    <span class="option-exp">${s.expiration} (${s.days_to_exp}d)</span>
                                </div>
                                <div class="option-price">
                                    <div class="option-mid">$${s.mid_price}</div>
                                    <div class="option-spread">${s.bid} / ${s.ask}</div>
                                </div>
                                <div class="option-metrics">
                                    <div class="option-iv">IV: ${s.iv}%</div>
                                    <div>Vol: ${s.volume}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    ${opt.spreads && opt.spreads.length > 0 ? `
                        <div class="spread-suggestion">
                            <div class="spread-name">${opt.spreads[0].strategy}</div>
                            <div class="spread-desc">${opt.spreads[0].description}</div>
                        </div>
                    ` : ''}
                    <div class="options-footer">
                        <span>Max Risk: $${opt.suggestions[0]?.max_risk || 'N/A'} per contract</span>
                        <span>${opt.suggestions[0]?.risk_reward || ''}</span>
                    </div>
                </div>
            `).join('');
        }

        async function loadOptionsForSymbol(symbol) {
            try {
                const response = await fetch(`/api/options/${symbol}`);
                const data = await response.json();
                if (!data.error) {
                    optionsData = [data];
                    switchTab('options');
                    renderOptions();
                }
            } catch (error) {
                console.error('Failed to load options for symbol');
            }
        }

        async function loadPortfolio() {
            try {
                const response = await fetch('/api/portfolio');
                portfolio = await response.json();
                renderPortfolio();
            } catch (error) {
                console.error('Failed to load portfolio');
            }
        }

        async function renderPortfolio() {
            // Render positions
            if (portfolio.positions.length === 0) {
                document.getElementById('positions').innerHTML = '<div class="empty-state"><p>No positions yet</p></div>';
            } else {
                let positionsHtml = '';
                for (const pos of portfolio.positions) {
                    const signal = allSignals.find(s => s.symbol === pos.symbol);
                    const currentPrice = signal ? signal.price : pos.entry_price;
                    const pnl = ((currentPrice - pos.entry_price) / pos.entry_price * 100).toFixed(2);
                    const pnlClass = pnl >= 0 ? 'profit' : 'loss';
                    positionsHtml += `
                        <div class="position-card" onclick="showChart('${pos.symbol}')">
                            <div>
                                <div class="position-symbol">${pos.symbol}</div>
                                <div class="position-details">${pos.qty} shares @ $${pos.entry_price}</div>
                            </div>
                            <div class="position-pnl ${pnlClass}">${pnl >= 0 ? '+' : ''}${pnl}%</div>
                        </div>
                    `;
                }
                document.getElementById('positions').innerHTML = positionsHtml;
            }

            // Render watchlist
            if (portfolio.watchlist.length === 0) {
                document.getElementById('watchlist').innerHTML = '<div class="empty-state"><p>No watchlist items</p></div>';
            } else {
                let watchlistHtml = '';
                for (const sym of portfolio.watchlist) {
                    const signal = allSignals.find(s => s.symbol === sym);
                    watchlistHtml += `
                        <div class="position-card" onclick="showChart('${sym}')">
                            <div class="position-symbol">${sym}</div>
                            ${signal ? `<span class="signal-badge ${signal.signal_type}">${signal.signal}</span>` : ''}
                        </div>
                    `;
                }
                document.getElementById('watchlist').innerHTML = watchlistHtml;
            }
        }

        function showAddPosition() {
            modalMode = 'position';
            document.getElementById('modalTitle').textContent = 'Add Position';
            document.getElementById('qtyGroup').style.display = 'block';
            document.getElementById('priceGroup').style.display = 'block';
            document.getElementById('modalSymbol').value = '';
            document.getElementById('modalQty').value = '';
            document.getElementById('modalPrice').value = '';
            document.getElementById('addModal').classList.add('visible');
        }

        function showAddWatchlist() {
            modalMode = 'watchlist';
            document.getElementById('modalTitle').textContent = 'Add to Watchlist';
            document.getElementById('qtyGroup').style.display = 'none';
            document.getElementById('priceGroup').style.display = 'none';
            document.getElementById('modalSymbol').value = '';
            document.getElementById('addModal').classList.add('visible');
        }

        function closeModal() {
            document.getElementById('addModal').classList.remove('visible');
        }

        async function saveModal() {
            const symbol = document.getElementById('modalSymbol').value.toUpperCase();
            if (!symbol) return;

            if (modalMode === 'position') {
                const qty = parseFloat(document.getElementById('modalQty').value);
                const price = parseFloat(document.getElementById('modalPrice').value);
                if (!qty || !price) return;

                portfolio.positions.push({ symbol, qty, entry_price: price });
            } else {
                if (!portfolio.watchlist.includes(symbol)) {
                    portfolio.watchlist.push(symbol);
                }
            }

            await fetch('/api/portfolio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(portfolio)
            });

            closeModal();
            renderPortfolio();
        }

        // Auto-scan on load
        startScan();

        // ============== PWA Service Worker & Install ==============
        let deferredPrompt = null;

        // Register Service Worker
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js')
                .then(reg => console.log('SW registered'))
                .catch(err => console.log('SW registration failed'));
        }

        // Offline detection
        function updateOnlineStatus() {
            const banner = document.getElementById('offlineBanner');
            if (navigator.onLine) {
                banner.classList.remove('visible');
            } else {
                banner.classList.add('visible');
            }
        }
        window.addEventListener('online', updateOnlineStatus);
        window.addEventListener('offline', updateOnlineStatus);
        updateOnlineStatus();

        // Listen for install prompt
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            showInstallBanner();
        });

        function showInstallBanner() {
            if (document.getElementById('installBanner')) return;

            const banner = document.createElement('div');
            banner.id = 'installBanner';
            banner.innerHTML = `
                <div style="position:fixed;bottom:0;left:0;right:0;background:linear-gradient(135deg,#1a1f2e,#0f1419);
                    padding:12px 16px;display:flex;justify-content:space-between;align-items:center;
                    border-top:1px solid #00d4aa;z-index:1000;">
                    <div style="display:flex;align-items:center;gap:12px;">
                        <div style="width:40px;height:40px;background:#00d4aa;border-radius:8px;display:flex;
                            align-items:center;justify-content:center;font-weight:bold;color:#0f1419;">DM</div>
                        <div>
                            <div style="font-weight:600;font-size:14px;">Install DeMark Signals</div>
                            <div style="font-size:12px;color:#71767b;">Add to home screen for quick access</div>
                        </div>
                    </div>
                    <div style="display:flex;gap:8px;">
                        <button onclick="dismissInstall()" style="padding:8px 12px;background:transparent;
                            border:1px solid #2f3336;border-radius:6px;color:#71767b;cursor:pointer;">Later</button>
                        <button onclick="installApp()" style="padding:8px 16px;background:#00d4aa;
                            border:none;border-radius:6px;color:#0f1419;font-weight:600;cursor:pointer;">Install</button>
                    </div>
                </div>
            `;
            document.body.appendChild(banner);
        }

        async function installApp() {
            if (!deferredPrompt) return;
            deferredPrompt.prompt();
            const { outcome } = await deferredPrompt.userChoice;
            deferredPrompt = null;
            dismissInstall();
        }

        function dismissInstall() {
            const banner = document.getElementById('installBanner');
            if (banner) banner.remove();
        }

        // Detect if already installed
        window.addEventListener('appinstalled', () => {
            deferredPrompt = null;
            dismissInstall();
        });

        // iOS install instructions
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
        const isStandalone = window.matchMedia('(display-mode: standalone)').matches;

        if (isIOS && !isStandalone) {
            setTimeout(() => {
                const iosBanner = document.createElement('div');
                iosBanner.id = 'iosBanner';
                iosBanner.innerHTML = `
                    <div style="position:fixed;bottom:0;left:0;right:0;background:linear-gradient(135deg,#1a1f2e,#0f1419);
                        padding:12px 16px;border-top:1px solid #00d4aa;z-index:1000;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div style="font-size:13px;">
                                <strong>Install this app:</strong> tap
                                <span style="font-size:16px;">⬆️</span> then "Add to Home Screen"
                            </div>
                            <button onclick="this.parentElement.parentElement.parentElement.remove()"
                                style="background:none;border:none;color:#71767b;font-size:18px;cursor:pointer;">✕</button>
                        </div>
                    </div>
                `;
                document.body.appendChild(iosBanner);
            }, 3000);
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/scan')
def api_scan():
    signals = scan_stocks(ALL_STOCKS)
    return jsonify({
        "signals": signals,
        "timestamp": datetime.now().isoformat(),
        "count": len(signals)
    })


@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    result = get_stock_data(symbol.upper())
    if result:
        return jsonify(result)
    return jsonify({"error": "Symbol not found"}), 404


@app.route('/api/symbol/<symbol>')
def api_symbol(symbol):
    result = get_demark_signals(symbol.upper())
    if result:
        return jsonify(result)
    return jsonify({"error": "Symbol not found"}), 404


@app.route('/api/portfolio', methods=['GET', 'POST'])
def api_portfolio():
    if request.method == 'GET':
        return jsonify(load_portfolio())
    else:
        data = request.json
        save_portfolio(data)
        return jsonify({"status": "ok"})


@app.route('/api/options')
def api_options():
    """Get options suggestions for all active signals"""
    signals = scan_stocks(ALL_STOCKS)
    options_data = get_options_for_signals(signals)
    return jsonify({
        "options": options_data,
        "timestamp": datetime.now().isoformat(),
        "count": len(options_data)
    })


@app.route('/api/options/<symbol>')
def api_options_symbol(symbol):
    """Get options for a specific symbol"""
    signal = get_demark_signals(symbol.upper())
    if not signal:
        return jsonify({"error": "Symbol not found"}), 404

    options = get_options_chain(
        signal['symbol'],
        signal['signal_type'],
        signal['price'],
        signal['strength']
    )
    if options:
        options['signal'] = signal['signal']
        options['trade_idea'] = signal['trade_idea']
        return jsonify(options)
    return jsonify({"error": "No options data available"}), 404


if __name__ == '__main__':
    import socket

    # Get port from environment (for cloud deployment) or use 5000
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    if debug_mode:
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except:
            local_ip = "localhost"

        print("\n" + "=" * 60)
        print("DeMark Daily Trade Signals (PWA)")
        print("=" * 60)
        print("\nFeatures:")
        print("  - Real-time DeMark signal scanning")
        print("  - Interactive candlestick charts with signal markers")
        print("  - OPTIONS TRADING: Call/Put suggestions based on signals")
        print("  - Portfolio tracking and watchlist")
        print("  - Installable as mobile app (PWA)")
        print("\n" + "-" * 60)
        print("ACCESS URLS:")
        print(f"  Computer:  http://localhost:{port}")
        print(f"  Mobile:    http://{local_ip}:{port}")
        print("-" * 60)
        print("\nTo install on phone:")
        print("  1. Open the Mobile URL in Safari (iOS) or Chrome (Android)")
        print("  2. Tap Share -> 'Add to Home Screen'")
        print("\nPress Ctrl+C to stop\n")

    app.run(debug=debug_mode, host='0.0.0.0', port=port)
