from fastapi import FastAPI
from app.api.routes import extract

app = FastAPI(
    title="Violence Detection API",
    version="0.1.0",
    description="API for extracting and detecting violence-related content"
)

app.include_router(extract.router)

@app.get("/")
async def home():
    return {"message": "Backend is running successfully!"}
