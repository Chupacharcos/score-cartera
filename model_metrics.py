from fastapi import APIRouter
from risk import ASSET_CATALOGUE

router = APIRouter()

@router.get("/metrics")
def get_metrics():
    total_assets = len(ASSET_CATALOGUE)
    classes = list({info["class"] for info in ASSET_CATALOGUE.values()})
    return {"total_assets": total_assets, "classes": classes}