"""
StockSense AI  — Stock Data Intelligence Dashboard
FastAPI backend: v2.0  (upgraded with Compare, ML Prediction, Smart Insights)
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LinearRegression

# ──────────────────────────────────────────────────────────────
#  APP SETUP
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="StockSense AI",
    description="Professional stock data intelligence dashboard with ML predictions.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
#  DATABASE
# ──────────────────────────────────────────────────────────────
DB_FILE = "stock_data.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            symbol  TEXT NOT NULL,
            date    TEXT NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  INTEGER,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ──────────────────────────────────────────────────────────────
#  COMPANIES REGISTRY
# ──────────────────────────────────────────────────────────────
COMPANIES: Dict[str, str] = {
    "INFY":      "Infosys Ltd",
    "TCS":       "Tata Consultancy Services",
    "RELIANCE":  "Reliance Industries",
    "HDFCBANK":  "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
}

# ──────────────────────────────────────────────────────────────
#  DATA FETCH & CACHE HELPERS
# ──────────────────────────────────────────────────────────────

def fetch_from_yfinance(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Download historical data from Yahoo Finance (NSE suffix)."""
    try:
        ticker = yf.Ticker(symbol + ".NS")
        df = ticker.history(period=period)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df["symbol"] = symbol
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
        return df[["symbol", "date", "open", "high", "low", "close", "volume"]]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"yfinance error: {exc}")


def store_data(df: pd.DataFrame) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute(
            "INSERT OR REPLACE INTO stock_data VALUES (?,?,?,?,?,?,?)",
            (row["symbol"], str(row["date"]), row["open"],
             row["high"], row["low"], row["close"], row["volume"]),
        )
    conn.commit()
    conn.close()


def get_cached_data(symbol: str, start_date: Optional[str] = None) -> pd.DataFrame:
    conn = get_db_connection()
    query = "SELECT * FROM stock_data WHERE symbol = ?"
    params: list = [symbol]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    conn.close()
    return df


def ensure_data_available(symbol: str) -> None:
    """Fetch fresh data if cache is empty or stale (> 1 day old)."""
    df = get_cached_data(symbol)
    if df.empty:
        store_data(fetch_from_yfinance(symbol))
        return
    last_date = pd.to_datetime(df["date"].max()).date()
    if last_date < datetime.now().date() - timedelta(days=1):
        store_data(fetch_from_yfinance(symbol))

# ──────────────────────────────────────────────────────────────
#  METRICS & ANALYTICS
# ──────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, days: int = 30) -> List[Dict[str, Any]]:
    df = df.tail(days).copy()
    df["daily_return"]  = (df["close"] - df["open"]) / df["open"]
    df["moving_avg_7"]  = df["close"].rolling(window=7,  min_periods=1).mean().fillna(df["close"])
    df["moving_avg_20"] = df["close"].rolling(window=20, min_periods=1).mean().fillna(df["close"])
    result = []
    for _, row in df.iterrows():
        result.append({
            "date":         pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "open":         round(row["open"],         2),
            "high":         round(row["high"],         2),
            "low":          round(row["low"],          2),
            "close":        round(row["close"],        2),
            "volume":       int(row["volume"]),
            "daily_return": round(row["daily_return"], 6),
            "moving_avg_7": round(row["moving_avg_7"], 2),
            "moving_avg_20":round(row["moving_avg_20"],2),
        })
    return result


def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")
    daily_returns = (df["close"] - df["open"]) / df["open"]
    volatility    = daily_returns.std()
    return {
        "symbol":        df["symbol"].iloc[0],
        "current_price": round(df["close"].iloc[-1], 2),
        "high_52w":      round(df["high"].max(),      2),
        "low_52w":       round(df["low"].min(),       2),
        "avg_close":     round(df["close"].mean(),    2),
        "volatility":    round(volatility,            6),
    }


def compute_smart_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Derive human-readable insights from price data:
      • Trend       — based on last 5 closes vs simple linear slope
      • Volatility  — Low / Medium / High bucketed from std deviation
      • RSI (14)    — momentum indicator
      • Support/Resistance — rolling 20-day low/high
    """
    if len(df) < 6:
        return {"trend": "UNKNOWN", "volatility_level": "UNKNOWN", "rsi": None}

    closes = df["close"].values.astype(float)

    # ── Trend ──────────────────────────────────────────────────
    last5 = closes[-5:]
    x = np.arange(len(last5)).reshape(-1, 1)
    slope = LinearRegression().fit(x, last5).coef_[0]
    trend = "UPTREND" if slope > 0 else "DOWNTREND"

    # ── Volatility Level ───────────────────────────────────────
    daily_rets = (df["close"] - df["open"]) / df["open"]
    vol = daily_rets.std()
    if vol < 0.01:
        vol_level = "LOW"
    elif vol < 0.025:
        vol_level = "MEDIUM"
    else:
        vol_level = "HIGH"

    # ── RSI (14) ───────────────────────────────────────────────
    rsi: Optional[float] = None
    if len(closes) >= 15:
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = gains[-14:].mean()
        avg_loss = losses[-14:].mean()
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs  = avg_gain / avg_loss
            rsi = round(100 - 100 / (1 + rs), 2)

    # ── Support / Resistance ───────────────────────────────────
    window = min(20, len(df))
    support    = round(df["low"].tail(window).min(),  2)
    resistance = round(df["high"].tail(window).max(), 2)

    # ── 5-day momentum % ──────────────────────────────────────
    momentum_pct = round((closes[-1] - closes[-6]) / closes[-6] * 100, 2) if len(closes) >= 6 else None

    return {
        "trend":            trend,
        "volatility_level": vol_level,
        "volatility_value": round(float(vol), 6),
        "rsi":              rsi,
        "support":          support,
        "resistance":       resistance,
        "momentum_5d_pct":  momentum_pct,
    }

# ──────────────────────────────────────────────────────────────
#  ML PREDICTION  (Linear Regression on price index)
# ──────────────────────────────────────────────────────────────

def predict_prices(df: pd.DataFrame, horizon: int = 5) -> List[Dict[str, Any]]:
    """
    Fit a simple Linear Regression on close prices and project
    `horizon` days into the future.
    Returns a list of {date, predicted_close} dicts.
    """
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Not enough data for prediction")

    closes = df["close"].values.astype(float)
    X      = np.arange(len(closes)).reshape(-1, 1)

    model  = LinearRegression()
    model.fit(X, closes)

    last_date = pd.to_datetime(df["date"].max())
    predictions = []
    for i in range(1, horizon + 1):
        pred_idx  = len(closes) - 1 + i
        pred_val  = model.predict([[pred_idx]])[0]
        pred_date = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        predictions.append({
            "date":            pred_date,
            "predicted_close": round(float(pred_val), 2),
        })

    # Confidence: R² score on in-sample data (simple proxy)
    r2 = model.score(X, closes)

    return predictions, round(float(r2), 4)

# ──────────────────────────────────────────────────────────────
#  ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/companies", tags=["Core"])
async def list_companies():
    """Return the list of tracked companies."""
    return [{"symbol": sym, "name": name} for sym, name in COMPANIES.items()]


@app.get("/data/{symbol}", tags=["Core"])
async def get_stock_data(
    symbol: str,
    days: int = Query(30, ge=1, le=365),
):
    """OHLCV + moving averages for a symbol over the last N days."""
    symbol = symbol.upper()
    if symbol not in COMPANIES:
        raise HTTPException(status_code=404, detail="Company not found")
    ensure_data_available(symbol)
    df = get_cached_data(symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    return {"symbol": symbol, "data": compute_metrics(df, days)}


@app.get("/summary/{symbol}", tags=["Core"])
async def get_summary(symbol: str):
    """52W high/low, current price, volatility."""
    symbol = symbol.upper()
    if symbol not in COMPANIES:
        raise HTTPException(status_code=404, detail="Company not found")
    ensure_data_available(symbol)
    df = get_cached_data(symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    return compute_summary(df)


@app.get("/insights/{symbol}", tags=["Analytics"])
async def get_insights(symbol: str):
    """
    Smart insights: trend direction, volatility level,
    RSI, support/resistance, 5-day momentum.
    """
    symbol = symbol.upper()
    if symbol not in COMPANIES:
        raise HTTPException(status_code=404, detail="Company not found")
    ensure_data_available(symbol)
    df = get_cached_data(symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    return {"symbol": symbol, **compute_smart_insights(df)}


@app.get("/predict/{symbol}", tags=["ML"])
async def predict(
    symbol:  str,
    horizon: int = Query(5, ge=1, le=14, description="Days to predict (1–14)"),
):
    """
    ML-based price prediction using Linear Regression.
    Returns predicted closing prices for the next `horizon` days
    along with the model's R² confidence score.
    """
    symbol = symbol.upper()
    if symbol not in COMPANIES:
        raise HTTPException(status_code=404, detail="Company not found")
    ensure_data_available(symbol)
    df = get_cached_data(symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    predictions, r2 = predict_prices(df, horizon)
    return {
        "symbol":      symbol,
        "horizon":     horizon,
        "r2_score":    r2,
        "predictions": predictions,
    }


@app.get("/compare", tags=["Analytics"])
async def compare_stocks(
    symbol1: str = Query(..., description="First stock symbol"),
    symbol2: str = Query(..., description="Second stock symbol"),
    days:    int = Query(30, ge=5, le=365),
):
    """
    Compare two stocks over the last N days.
    Returns aligned close-price series for both.
    """
    sym1, sym2 = symbol1.upper(), symbol2.upper()
    for s in (sym1, sym2):
        if s not in COMPANIES:
            raise HTTPException(status_code=404, detail=f"Company not found: {s}")
        ensure_data_available(s)

    df1 = get_cached_data(sym1).tail(days)
    df2 = get_cached_data(sym2).tail(days)

    if df1.empty or df2.empty:
        raise HTTPException(status_code=404, detail="Data missing for one or both symbols")

    # Align on common trading dates
    dates1 = set(pd.to_datetime(df1["date"]).dt.date)
    dates2 = set(pd.to_datetime(df2["date"]).dt.date)
    common = sorted(dates1 & dates2)
    if not common:
        raise HTTPException(status_code=404, detail="No overlapping trading dates")

    df1["date_py"] = pd.to_datetime(df1["date"]).dt.date
    df2["date_py"] = pd.to_datetime(df2["date"]).dt.date

    d1_map = dict(zip(df1["date_py"], df1["close"]))
    d2_map = dict(zip(df2["date_py"], df2["close"]))

    return {
        "symbol1":      sym1,
        "symbol2":      sym2,
        "name1":        COMPANIES[sym1],
        "name2":        COMPANIES[sym2],
        "dates":        [d.strftime("%Y-%m-%d") for d in common],
        sym1:           [round(d1_map[d], 2) for d in common],
        sym2:           [round(d2_map[d], 2) for d in common],
    }


@app.get("/top-stocks", tags=["Core"])
async def get_top_gainers_losers():
    """Daily gainers and losers across all tracked companies."""
    results = []
    for sym in COMPANIES:
        try:
            ensure_data_available(sym)
            df = get_cached_data(sym)
            if df.empty or len(df) < 2:
                continue
            latest = df.iloc[-1]
            prev   = df.iloc[-2]
            daily_return = (latest["close"] - prev["close"]) / prev["close"]
            results.append({
                "symbol":        sym,
                "name":          COMPANIES[sym],
                "daily_return":  round(float(daily_return), 6),
                "current_price": round(float(latest["close"]), 2),
            })
        except Exception:
            continue

    if not results:
        raise HTTPException(status_code=404, detail="No data available")

    sorted_r = sorted(results, key=lambda x: x["daily_return"], reverse=True)
    return {
        "gainers": sorted_r[:3],
        "losers":  sorted_r[-3:][::-1],
    }