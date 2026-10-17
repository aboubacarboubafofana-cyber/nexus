from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

app = FastAPI(
    title="NEXUS API",
    description="Backend de l'assistant IA multi-agents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"name": "NEXUS API", "status": "ok", "version": "0.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/chat.html")
def serve_chat():
    return FileResponse(Path(__file__).parent / "chat.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    return {"response": f"Tu as dit : {req.message}"}


@app.on_event("startup")
def startup():
    logger.info("NEXUS API démarrée")
