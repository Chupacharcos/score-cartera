# Score de Riesgo de Cartera en Tiempo Real

Sistema de análisis y scoring de riesgo para carteras de inversión. Combina un **GaussianHMM entrenado** (Hidden Markov Model, 3 regímenes: Bull/Neutral/Bear) con análisis de correlaciones dinámicas y métricas de riesgo en tiempo real (VaR, CVaR, Sharpe, Sortino, MaxDrawdown).

Demo en producción: [adrianmoreno-dev.com/demo/score-riesgo-cartera](https://adrianmoreno-dev.com/demo/score-riesgo-cartera)

---

## Resultados del Modelo (Backtest 5 años)

| Métrica | HMM Strategy | Buy & Hold |
|---------|-------------|------------|
| **Sharpe ratio** | **2.12** | 1.11 |
| **Max Drawdown** | **-7.5%** | -21.9% |
| **CVaR 95%** | **-1.5%/día** | -2.2%/día |
| CVaR reducción | **-32%** | — |
| F1 Bear detection | 0.33 | — |

> El HMM es un modelo **no supervisado** — descubre regímenes estadísticos latentes sin etiquetas. Su calidad se mide por Sharpe y reducción de riesgo de cola, no por F1 vs. una etiqueta arbitraria.

---

## Arquitectura

```
Datos reales yfinance (SPY, IBEX35, EWP, STOXX50E, TLT, GLD — 5 años)
        │
        ▼
┌──────────────────────────────────┐
│  Features HMM (4)                │
│  · retorno diario log            │
│  · volatilidad rolling 20d       │
│  · momentum 5d                   │
│  · momentum 20d                  │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  GaussianHMM (3 estados)         │
│  covariance_type="full"          │
│  n_iter=200                      │
└──────────────┬───────────────────┘
               │
               ▼
    Mapeo: estado → {Bull, Neutral, Bear}
    (por retorno medio de cada estado)
               │
               ▼
┌──────────────────────────────────┐
│  Risk Score (0-100)              │
│  50% volatilidad cartera         │
│  30% régimen HMM                 │
│  20% concentración HHI           │
└──────────────────────────────────┘
```

### Regímenes detectados (datos reales 2020-2025)

| Estado | Régimen | Retorno anual | Volatilidad anual |
|--------|---------|--------------|-------------------|
| 0 | Bear | -29.4% | Alta |
| 1 | Neutral | +12.4% | Media |
| 2 | Bull | +35.4% | Baja |

---

## Métricas de riesgo en tiempo real

Para cada cartera analizada se calculan sobre datos reales de Yahoo Finance:

- **VaR 95%**: pérdida diaria que se supera solo en el 5% de los peores días
- **CVaR 95% (Expected Shortfall)**: pérdida media en ese peor 5%
- **Sharpe ratio**: rentabilidad por unidad de riesgo (anualizado, √252)
- **Sortino ratio**: igual que Sharpe pero penalizando solo la volatilidad bajista
- **Max Drawdown**: mayor caída desde máximos históricos

---

## Activos disponibles (19)

| Clase | Tickers |
|-------|---------|
| Bolsa Large-cap | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ |
| ETFs | SPY, QQQ, GLD, TLT, EEM |
| Cripto | BTC-USD, ETH-USD |
| Materias primas | GC=F (Oro), CL=F (Petróleo WTI) |

---

## Estructura del proyecto

```
score-cartera/
├── train.py          # Entrenamiento GaussianHMM + backtest (ejecutar offline)
├── risk.py           # Motor de análisis: HMM, VaR/CVaR/Sharpe, correlaciones
├── router.py         # Endpoints FastAPI (/ml/cartera/*)
├── api.py            # App FastAPI standalone (puerto 8096)
└── artifacts/        # Modelo entrenado (excluido de git)
    ├── hmm_model.joblib
    ├── scaler.joblib
    ├── regime_map.json
    ├── backtest.joblib
    └── metadata.json
```

---

## Endpoints REST

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/ml/cartera/analyse` | Análisis completo de la cartera |
| `GET` | `/ml/cartera/assets` | Lista de activos disponibles |
| `GET` | `/ml/cartera/health` | Estado del servicio |

### `POST /ml/cartera/analyse`

**Body:**
```json
{
  "portfolio": [
    {"ticker": "AAPL", "weight": 0.4},
    {"ticker": "SPY",  "weight": 0.3},
    {"ticker": "TLT",  "weight": 0.3}
  ]
}
```

**Respuesta (extracto):**
```json
{
  "risk_score": 38,
  "portfolio_volatility": 14.2,
  "regime": {"label": "Neutral", "color": "orange", "score": 45, "hmm": true, "bear_pct_20d": 15.0},
  "risk_metrics": {
    "sharpe": 1.42, "sortino": 2.01,
    "var_95": -1.12, "cvar_95": -1.58,
    "max_drawdown": -12.3
  },
  "hhi": 0.34,
  "top_correlations": [...],
  "assets": [...],
  "chart": {"dates": [...], "volatility": [...], "avg_correlation": [...]},
  "data_source": "yfinance"
}
```

---

## Entrenamiento

```bash
cd /var/www/score-cartera
source /var/www/chatbot/venv/bin/activate
python3 train.py
```

Descarga 5 años de datos reales de yfinance (SPY, IBEX35, EWP, STOXX50E, TLT, GLD), entrena el HMM y ejecuta el backtest completo (~2 min). Si yfinance no está disponible, usa GARCH(1,1) sintético calibrado.

## Arranque del servicio

```bash
uvicorn api:app --host 127.0.0.1 --port 8096 --reload
```

---

## Stack técnico

- **Python 3.12** · **hmmlearn** (GaussianHMM) · **NumPy / Pandas**
- **yfinance** — datos de mercado reales
- **scikit-learn** — StandardScaler, métricas
- **joblib** — serialización del modelo
- **FastAPI / Uvicorn** — API REST

---

*Parte del portafolio de proyectos IA/ML — [adrianmoreno-dev.com](https://adrianmoreno-dev.com)*
