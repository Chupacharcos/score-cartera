from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from risk import analyse_portfolio, ASSET_CATALOGUE

router = APIRouter()


class AssetWeight(BaseModel):
    ticker: str
    weight: float


class PortfolioRequest(BaseModel):
    portfolio: list[AssetWeight]


@router.post("/analyse")
def analyse(req: PortfolioRequest):
    portfolio = [{"ticker": a.ticker, "weight": a.weight} for a in req.portfolio]
    if not portfolio:
        raise HTTPException(status_code=400, detail="Portfolio vacío")
    result = analyse_portfolio(portfolio)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/assets")
def get_assets():
    grouped: dict[str, list] = {}
    for ticker, info in ASSET_CATALOGUE.items():
        cls = info["class"]
        if cls not in grouped:
            grouped[cls] = []
        grouped[cls].append({
            "ticker": ticker,
            "name": info["name"],
            "sector": info.get("sector", "-"),
        })
    return {"groups": grouped, "total": len(ASSET_CATALOGUE)}


@router.get("/health")
def health():
    return {"status": "ok"}
