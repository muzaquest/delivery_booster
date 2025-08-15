from fastapi import FastAPI

app = FastAPI(title="Restaurant Sales Analytics API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Healthcheck endpoint returning service status."""
    return {"status": "ok"}