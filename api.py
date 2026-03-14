from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router

app = FastAPI(
    title="Score de Riesgo de Cartera",
    description="Real-time portfolio risk scoring with TFT-lite + Graph Attention correlations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://adrianmoreno-dev.com", "http://127.0.0.1", "http://localhost"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/cartera")


@app.get("/health")
def health():
    return {"status": "ok", "service": "score-cartera", "version": "1.0.0"}
