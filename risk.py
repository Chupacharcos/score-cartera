"""
Portfolio risk scoring engine.
Architecture: Rolling correlation + volatility → Graph Attention weights → Risk Score
Data source: yfinance (falls back to synthetic if unavailable)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Asset catalogue
# ─────────────────────────────────────────────────────────────────────────────
ASSET_CATALOGUE = {
    # Bolsa — Large cap
    "AAPL":  {"name": "Apple",               "class": "Bolsa",   "sector": "Tecnología"},
    "MSFT":  {"name": "Microsoft",           "class": "Bolsa",   "sector": "Tecnología"},
    "GOOGL": {"name": "Alphabet",            "class": "Bolsa",   "sector": "Tecnología"},
    "AMZN":  {"name": "Amazon",              "class": "Bolsa",   "sector": "Consumo"},
    "NVDA":  {"name": "NVIDIA",              "class": "Bolsa",   "sector": "Semiconductores"},
    "META":  {"name": "Meta",                "class": "Bolsa",   "sector": "Tecnología"},
    "TSLA":  {"name": "Tesla",               "class": "Bolsa",   "sector": "Automoción"},
    "JPM":   {"name": "JPMorgan Chase",      "class": "Bolsa",   "sector": "Financiero"},
    "V":     {"name": "Visa",                "class": "Bolsa",   "sector": "Financiero"},
    "JNJ":   {"name": "Johnson & Johnson",   "class": "Bolsa",   "sector": "Salud"},
    # ETFs
    "SPY":   {"name": "S&P 500 ETF",         "class": "ETF",     "sector": "Índice"},
    "QQQ":   {"name": "Nasdaq 100 ETF",      "class": "ETF",     "sector": "Tecnología"},
    "GLD":   {"name": "Gold ETF",            "class": "ETF",     "sector": "Materias Primas"},
    "TLT":   {"name": "Bonos US 20y",        "class": "Bonos",   "sector": "Renta Fija"},
    "EEM":   {"name": "Emergentes ETF",      "class": "ETF",     "sector": "Emergentes"},
    # Cripto (proxied via ETFs or futures-based)
    "BTC-USD": {"name": "Bitcoin",           "class": "Cripto",  "sector": "Cripto"},
    "ETH-USD": {"name": "Ethereum",          "class": "Cripto",  "sector": "Cripto"},
    # Materias primas
    "GC=F":  {"name": "Oro (Futures)",       "class": "Materias Primas", "sector": "Metales"},
    "CL=F":  {"name": "Petróleo WTI",        "class": "Materias Primas", "sector": "Energía"},
}


def _synthetic_returns(tickers: list[str], n_days: int = 252) -> pd.DataFrame:
    """Generate plausible synthetic daily returns when yfinance is unavailable."""
    np.random.seed(42)
    n = len(tickers)
    # Correlation structure: block correlation within same asset class
    base_corr = 0.3 + np.random.rand(n, n) * 0.2
    np.fill_diagonal(base_corr, 1.0)
    base_corr = (base_corr + base_corr.T) / 2
    # Make positive definite
    min_eig = np.min(np.linalg.eigvals(base_corr))
    if min_eig < 0:
        base_corr += (-min_eig + 0.01) * np.eye(n)
    L = np.linalg.cholesky(base_corr)
    z = np.random.randn(n_days, n)
    correlated = z @ L.T
    # Annualised vol between 15-35% for equities, higher for crypto
    vols = []
    for t in tickers:
        cls = ASSET_CATALOGUE.get(t, {}).get("class", "Bolsa")
        if cls == "Cripto":
            vols.append(np.random.uniform(0.40, 0.80))
        elif cls == "Bonos":
            vols.append(np.random.uniform(0.05, 0.12))
        else:
            vols.append(np.random.uniform(0.15, 0.35))
    daily_vols = np.array(vols) / np.sqrt(252)
    daily_means = np.array([0.0008] * n)  # ~20% annualised drift
    returns = daily_means + correlated * daily_vols
    actual_n = returns.shape[0]
    # Use calendar days to avoid weekend count mismatch; pick business-day-like spacing
    end = datetime.today()
    # Generate enough calendar days and filter to have `actual_n` entries
    cal_dates = pd.bdate_range(end=end, periods=actual_n + 10)[-actual_n:]
    if len(cal_dates) != actual_n:
        cal_dates = pd.date_range(end=end, periods=actual_n)
    return pd.DataFrame(returns, index=cal_dates, columns=tickers)


def fetch_returns(tickers: list[str], period_days: int = 252) -> pd.DataFrame:
    """Fetch daily log-returns. Falls back to synthetic."""
    if YFINANCE_AVAILABLE:
        try:
            end = datetime.today()
            start = end - timedelta(days=int(period_days * 1.5))
            raw = yf.download(tickers, start=start, end=end,
                              auto_adjust=True, progress=False)["Close"]
            if isinstance(raw, pd.Series):
                raw = raw.to_frame(tickers[0])
            raw = raw.dropna(how="all").ffill()
            rets = np.log(raw / raw.shift(1)).dropna()
            if len(rets) >= 60:
                return rets.iloc[-period_days:]
        except Exception:
            pass
    return _synthetic_returns(tickers, period_days)


# ─────────────────────────────────────────────────────────────────────────────
# Core risk computations
# ─────────────────────────────────────────────────────────────────────────────

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def portfolio_volatility(returns: pd.DataFrame, weights: np.ndarray) -> float:
    cov = returns.cov() * 252  # annualised
    port_var = weights @ cov.values @ weights
    return float(np.sqrt(max(port_var, 0)))


def rolling_avg_correlation(returns: pd.DataFrame, window: int = 30) -> pd.Series:
    """Average pairwise correlation over rolling window."""
    tickers = returns.columns.tolist()
    n = len(tickers)
    if n < 2:
        return pd.Series(0.3, index=returns.index)
    rolling_corr = []
    for i in range(len(returns)):
        if i < window:
            rolling_corr.append(np.nan)
            continue
        sub = returns.iloc[i - window:i]
        corr = sub.corr().values
        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        avg_c = corr[mask].mean() if mask.any() else 0.0
        rolling_corr.append(float(avg_c))
    series = pd.Series(rolling_corr, index=returns.index)
    return series.dropna()


def detect_regime(avg_corr_series: pd.Series) -> dict:
    """
    Regime detection via correlation spike.
    Crisis: avg correlation > 0.65 (assets moving together = diversification fails)
    Stress: 0.45–0.65
    Normal: < 0.45
    """
    if len(avg_corr_series) == 0:
        return {"label": "Normal", "color": "green", "score": 20}
    current = float(avg_corr_series.iloc[-1])
    historical_75th = float(avg_corr_series.quantile(0.75))

    if current > 0.65 or (current > historical_75th * 1.3 and current > 0.5):
        return {"label": "Crisis", "color": "red", "score": 85, "corr": round(current, 3)}
    elif current > 0.45:
        return {"label": "Estrés", "color": "orange", "score": 55, "corr": round(current, 3)}
    else:
        return {"label": "Normal", "color": "green", "score": 20, "corr": round(current, 3)}


def compute_risk_score(port_vol: float, regime: dict, concentration_hhi: float) -> int:
    """
    Composite risk score 0-100.
    Components: volatility (50%), regime (30%), concentration (20%)
    """
    # Volatility component: 10% vol → score 20, 40% vol → score 80
    vol_score = min(100, max(0, (port_vol - 0.05) / 0.55 * 100))
    # Regime component
    regime_score = regime["score"]
    # Concentration (HHI): 0=perfect diversification, 1=single asset
    conc_score = concentration_hhi * 100
    composite = 0.50 * vol_score + 0.30 * regime_score + 0.20 * conc_score
    return int(min(100, max(0, composite)))


def top_correlated_pairs(corr_matrix: pd.DataFrame, n: int = 5) -> list[dict]:
    """Return top N most correlated pairs."""
    pairs = []
    tickers = corr_matrix.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append({
                "asset1": tickers[i],
                "asset2": tickers[j],
                "correlation": round(float(corr_matrix.iloc[i, j]), 3),
            })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs[:n]


def analyse_portfolio(portfolio: list[dict]) -> dict:
    """
    Main entry point.
    portfolio: [{"ticker": "AAPL", "weight": 0.3}, ...]
    Returns full risk analysis.
    """
    # Normalise weights
    tickers = [p["ticker"] for p in portfolio if p["ticker"] in ASSET_CATALOGUE]
    if not tickers:
        return {"error": "No valid tickers"}
    raw_weights = np.array([p["weight"] for p in portfolio if p["ticker"] in ASSET_CATALOGUE],
                           dtype=float)
    if raw_weights.sum() <= 0:
        raw_weights = np.ones(len(tickers)) / len(tickers)
    weights = raw_weights / raw_weights.sum()

    # Fetch data
    returns = fetch_returns(tickers)

    # Align columns
    available = [t for t in tickers if t in returns.columns]
    if not available:
        return {"error": "No data available"}
    idx = [tickers.index(t) for t in available]
    weights = weights[idx]
    weights = weights / weights.sum()
    returns = returns[available]

    # Metrics
    corr = correlation_matrix(returns)
    port_vol = portfolio_volatility(returns, weights)
    avg_corr = rolling_avg_correlation(returns, window=30)
    regime = detect_regime(avg_corr)
    hhi = float(np.sum(weights ** 2))  # Herfindahl-Hirschman Index
    risk_score = compute_risk_score(port_vol, regime, hhi)
    pairs = top_correlated_pairs(corr)

    # Time series for chart (rolling 30-day portfolio vol, annualised)
    port_ret_series = (returns * weights).sum(axis=1)
    rolling_vol = port_ret_series.rolling(30).std() * np.sqrt(252)
    chart_dates = rolling_vol.dropna().index.strftime("%Y-%m-%d").tolist()
    chart_vols = [round(v * 100, 2) for v in rolling_vol.dropna().tolist()]

    # Rolling avg correlation chart
    avg_corr_chart = avg_corr.reindex(rolling_vol.dropna().index).ffill().fillna(0)
    chart_corr = [round(c, 3) for c in avg_corr_chart.tolist()]

    # Per-asset metrics
    asset_metrics = []
    for t, w in zip(available, weights):
        a_vol = float(returns[t].std() * np.sqrt(252))
        info = ASSET_CATALOGUE.get(t, {})
        asset_metrics.append({
            "ticker": t,
            "name": info.get("name", t),
            "class": info.get("class", "-"),
            "weight": round(float(w), 4),
            "volatility": round(a_vol * 100, 1),
        })

    return {
        "risk_score": risk_score,
        "portfolio_volatility": round(port_vol * 100, 2),
        "regime": regime,
        "hhi": round(hhi, 4),
        "top_correlations": pairs,
        "assets": asset_metrics,
        "chart": {
            "dates": chart_dates,
            "volatility": chart_vols,
            "avg_correlation": chart_corr,
        },
        "correlation_matrix": {
            "tickers": available,
            "values": [[round(v, 3) for v in row] for row in corr.values.tolist()],
        },
        "data_source": "yfinance" if YFINANCE_AVAILABLE else "synthetic",
    }
