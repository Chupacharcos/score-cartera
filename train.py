"""
Entrenamiento del modelo de scoring de riesgo de cartera.
Arquitectura: HMM (Hidden Markov Model) 3 estados (Bull/Neutral/Bear)
              entrenado sobre retornos reales de yfinance.

Ejecutar offline:
  cd /var/www/score-cartera
  source /var/www/chatbot/venv/bin/activate
  python3 train.py

Genera en artifacts/:
  hmm_model.joblib     — GaussianHMM (3 estados de mercado)
  scaler.joblib        — StandardScaler para features HMM
  regime_map.json      — mapeo estado → {Bull, Neutral, Bear}
  backtest.joblib      — métricas de backtest sobre datos históricos
  metadata.json        — métricas, fecha, descripción
"""

import json
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
from hmmlearn.hmm import GaussianHMM

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# Índices de referencia para entrenar el HMM (mercados globales + España)
TRAIN_TICKERS = ["SPY", "^IBEX", "EWP", "^STOXX50E", "TLT", "GLD"]
LOOKBACK_DAYS  = 5 * 252   # 5 años de datos diarios
N_STATES       = 3          # Bull / Neutral / Bear


def download_market_data() -> tuple[np.ndarray, np.ndarray]:
    """Descarga retornos históricos de mercado vía yfinance."""
    try:
        import yfinance as yf
        print("  Descargando datos reales de mercado (yfinance)...")
        all_returns = []
        for ticker in TRAIN_TICKERS:
            try:
                hist = yf.Ticker(ticker).history(period="5y", interval="1d", auto_adjust=True)
                closes = hist["Close"].dropna().values.astype(float)
                if len(closes) > 100:
                    ret = np.diff(np.log(closes))
                    all_returns.append(ret[-LOOKBACK_DAYS:])
                    print(f"    {ticker}: {len(ret)} días reales")
            except Exception as e:
                print(f"    {ticker}: error — {e}")
                continue

        if len(all_returns) < 2:
            raise RuntimeError("Insuficientes tickers descargados")

        # Usar el índice de referencia principal (SPY) como base
        base_ret = all_returns[0]
        n = min(len(r) for r in all_returns)
        all_returns = [r[-n:] for r in all_returns]

        # Promedio ponderado (SPY 40%, IBEX 20%, STOXX 20%, resto 20%)
        weights = [0.40, 0.20, 0.15, 0.15, 0.05, 0.05]
        weights = weights[:len(all_returns)]
        weights = np.array(weights) / sum(weights)
        combined = np.average(all_returns, axis=0, weights=weights)

        print(f"  Total: {n} observaciones diarias reales")
        return combined, np.array([True] * n)

    except Exception as e:
        print(f"  yfinance no disponible ({e}) — usando GARCH sintético calibrado")
        return _garch_synthetic(LOOKBACK_DAYS)


def _garch_synthetic(n: int) -> tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1) sintético calibrado con parámetros históricos del IBEX35."""
    rng = np.random.default_rng(42)
    omega, alpha, beta = 0.000002, 0.08, 0.91
    sigma2 = np.zeros(n)
    ret    = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * ret[t-1]**2 + beta * sigma2[t-1]
        ret[t]    = rng.normal(0, np.sqrt(sigma2[t]))
    return ret, np.array([False] * n)


def build_hmm_features(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Features para el HMM:
      - Retorno diario (log)
      - Volatilidad rolling 20 días
      - Momentum 5 días
      - Momentum 20 días
    """
    n = len(returns)
    vol20 = np.array([
        returns[max(0, t-window):t+1].std() if t >= 5 else returns[:t+1].std()
        for t in range(n)
    ])
    mom5  = np.array([
        returns[max(0, t-4):t+1].sum() if t >= 4 else returns[:t+1].sum()
        for t in range(n)
    ])
    mom20 = np.array([
        returns[max(0, t-19):t+1].sum() if t >= 19 else returns[:t+1].sum()
        for t in range(n)
    ])
    return np.column_stack([returns, vol20, mom5, mom20]).astype(np.float32)


def map_states_to_regimes(
    states: np.ndarray, returns: np.ndarray
) -> dict[int, str]:
    """
    Asigna etiquetas Bull/Neutral/Bear a los estados HMM
    según el retorno medio de cada estado.
    """
    state_returns = {}
    for s in range(N_STATES):
        mask = states == s
        if mask.sum() > 0:
            state_returns[s] = float(returns[mask].mean())
    sorted_states = sorted(state_returns, key=lambda s: state_returns[s])
    names = ["Bear", "Neutral", "Bull"]
    return {s: names[i] for i, s in enumerate(sorted_states)}


def compute_backtest_metrics(
    returns: np.ndarray, states: np.ndarray, regime_map: dict
) -> dict:
    """
    Métricas de backtest:
      - Sharpe ratio (anualizado)
      - Sortino ratio
      - VaR 95%
      - CVaR 95% (Expected Shortfall)
      - Max Drawdown
      - F1 / MCC: precisión del HMM en identificar Bear (clase positiva)
    """
    ann = np.sqrt(252)

    # ── Estrategia HMM: largo solo en Bull/Neutral, fuera en Bear ─────────────
    strategy_mask = np.array([regime_map.get(s, "Neutral") != "Bear" for s in states])
    strat_returns = returns * strategy_mask.astype(float)

    # Buy-and-hold
    bah_ret = returns

    def sharpe(r, rf=0.0):
        excess = r - rf / 252
        return ann * excess.mean() / (excess.std() + 1e-9)

    def sortino(r, rf=0.0):
        excess  = r - rf / 252
        downside = excess[excess < 0].std() + 1e-9
        return ann * excess.mean() / downside

    def max_drawdown(r):
        cum = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return float(dd.min())

    def var_cvar(r, level=0.05):
        var  = float(np.percentile(r, level * 100))
        cvar = float(r[r <= var].mean()) if (r <= var).sum() > 0 else var
        return var, cvar

    var95, cvar95 = var_cvar(returns)
    strat_var, strat_cvar = var_cvar(strat_returns)

    # ── Evaluación clasificadora: ¿detecta el Bear correctamente? ────────────
    # Generar etiqueta "Bear real" = retornos en el peor cuartil
    p25 = np.percentile(returns, 25)
    y_true_bear = (returns <= p25).astype(int)
    y_pred_bear = (np.array([regime_map.get(s, "Neutral") for s in states]) == "Bear").astype(int)

    f1_bear  = float(f1_score(y_true_bear, y_pred_bear, zero_division=0))
    mcc_bear = float(matthews_corrcoef(y_true_bear, y_pred_bear))

    # F1 multiclase sobre los 3 regímenes (Bull/Neutral/Bear)
    # Etiqueta "real" basada en retorno: terciles
    p33, p66 = np.percentile(returns, 33), np.percentile(returns, 66)
    y_true_mc = np.where(returns <= p33, 0, np.where(returns <= p66, 1, 2))  # 0=Bear,1=Neutral,2=Bull
    y_pred_mc = np.array([
        0 if regime_map.get(s, "Neutral") == "Bear"
        else 1 if regime_map.get(s, "Neutral") == "Neutral"
        else 2
        for s in states
    ])
    f1_mc  = float(f1_score(y_true_mc, y_pred_mc, average="macro", zero_division=0))
    mcc_mc = float(matthews_corrcoef(y_true_mc, y_pred_mc))

    return {
        "buy_and_hold": {
            "sharpe":       round(sharpe(bah_ret), 4),
            "sortino":      round(sortino(bah_ret), 4),
            "var_95":       round(var95, 5),
            "cvar_95":      round(cvar95, 5),
            "max_drawdown": round(max_drawdown(bah_ret), 4),
            "return_anual": round(float(bah_ret.mean() * 252), 4),
        },
        "hmm_strategy": {
            "sharpe":       round(sharpe(strat_returns), 4),
            "sortino":      round(sortino(strat_returns), 4),
            "var_95":       round(strat_var, 5),
            "cvar_95":      round(strat_cvar, 5),
            "max_drawdown": round(max_drawdown(strat_returns), 4),
            "return_anual": round(float(strat_returns.mean() * 252), 4),
        },
        "clasificacion_bear": {
            "f1_score":     round(f1_bear, 4),
            "mcc":          round(mcc_bear, 4),
            "descripcion":  "Bear = retornos en cuartil inferior (peor 25%)",
        },
        "clasificacion_multiclase": {
            "f1_macro":     round(f1_mc, 4),
            "mcc":          round(mcc_mc, 4),
            "descripcion":  "3 clases Bull/Neutral/Bear vs terciles de retorno real",
        },
    }


def train():
    print("\n" + "=" * 60)
    print("HMM — Detección de Regímenes de Mercado (Bull/Neutral/Bear)")
    print("=" * 60)

    # 1. Datos
    returns, real_flags = download_market_data()
    n_real = int(real_flags.sum())

    # 2. Features HMM
    X_hmm = build_hmm_features(returns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hmm)

    # 3. Entrenar HMM
    print(f"\n  Entrenando GaussianHMM ({N_STATES} estados) sobre {len(returns)} días...")
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        tol=1e-4,
        random_state=42,
    )
    model.fit(X_scaled)

    states = model.predict(X_scaled)
    regime_map = map_states_to_regimes(states, returns)
    print(f"  Mapeo de estados: {regime_map}")

    # Distribución de regímenes
    for s in range(N_STATES):
        mask = states == s
        name = regime_map[s]
        ret_mean = returns[mask].mean() * 252
        vol_mean = returns[mask].std() * np.sqrt(252)
        print(f"    Estado {s} ({name}): {mask.sum()} días "
              f"| retorno anual: {ret_mean*100:.1f}% "
              f"| vol anual: {vol_mean*100:.1f}%")

    # 4. Backtest y métricas
    print("\n  Calculando métricas de backtest y clasificación...")
    backtest = compute_backtest_metrics(returns, states, regime_map)

    bah = backtest["buy_and_hold"]
    hmm = backtest["hmm_strategy"]
    clf_bear = backtest["clasificacion_bear"]
    clf_mc   = backtest["clasificacion_multiclase"]

    print(f"\n  ── Métricas de clasificación (regímenes) ──")
    print(f"    F1 Bear detection : {clf_bear['f1_score']:.4f}")
    print(f"    MCC Bear          : {clf_bear['mcc']:.4f}")
    print(f"    F1 multiclase     : {clf_mc['f1_macro']:.4f}")
    print(f"    MCC multiclase    : {clf_mc['mcc']:.4f}")
    print(f"\n  ── Backtest (Sharpe / CVaR / Max Drawdown) ──")
    print(f"    Buy & Hold  — Sharpe: {bah['sharpe']:.3f} | "
          f"CVaR: {bah['cvar_95']*100:.2f}% | MaxDD: {bah['max_drawdown']*100:.1f}%")
    print(f"    HMM Strat.  — Sharpe: {hmm['sharpe']:.3f} | "
          f"CVaR: {hmm['cvar_95']*100:.2f}% | MaxDD: {hmm['max_drawdown']*100:.1f}%")

    # 5. Guardar artifacts
    print("\n  Guardando artifacts...")
    joblib.dump(model,  ARTIFACTS / "hmm_model.joblib")
    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
    joblib.dump(backtest, ARTIFACTS / "backtest.joblib")
    (ARTIFACTS / "regime_map.json").write_text(
        json.dumps({str(k): v for k, v in regime_map.items()}, indent=2)
    )

    metadata = {
        "fecha_entrenamiento":  datetime.now().isoformat(),
        "modelo":               "GaussianHMM (3 estados: Bull/Neutral/Bear)",
        "arquitectura":         "HMM con features: retorno, vol_20d, mom_5d, mom_20d",
        "n_estados":            N_STATES,
        "n_dias_entrenamiento": int(len(returns)),
        "n_real":               n_real,
        "tickers_train":        TRAIN_TICKERS,
        "dataset":              "yfinance real (SPY, IBEX35, EWP, STOXX50E, TLT, GLD) 5 años",
        "regime_map":           {str(k): v for k, v in regime_map.items()},
        "clasificacion_bear":   clf_bear,
        "clasificacion_multiclase": clf_mc,
        "backtest":             backtest,
        "sharpe_mejora":        round(hmm["sharpe"] - bah["sharpe"], 4),
        "cvar_reduccion_pct":   round((bah["cvar_95"] - hmm["cvar_95"]) / abs(bah["cvar_95"]) * 100, 1),
        "maxdd_reduccion_pct":  round((bah["max_drawdown"] - hmm["max_drawdown"]) / abs(bah["max_drawdown"]) * 100, 1),
    }
    (ARTIFACTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    print(f"\n  Artifacts guardados en {ARTIFACTS}")
    print(f"\n  RESULTADOS FINALES:")
    print(f"    F1 Bear = {clf_bear['f1_score']:.4f}  |  MCC = {clf_bear['mcc']:.4f}")
    print(f"    Sharpe HMM = {hmm['sharpe']:.3f}  vs  B&H = {bah['sharpe']:.3f}")
    print(f"    CVaR reducido un {metadata['cvar_reduccion_pct']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    train()
