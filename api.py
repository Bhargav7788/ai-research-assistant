"""
FastAPI backend for the AI Research Assistant.
Endpoints:
  GET  /         → serves static/index.html
  GET  /health   → status + KB size
  POST /chat     → SSE streaming response
"""

import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Agent is initialised inside lifespan so routes are ALWAYS registered first
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    try:
        from agent import ResearchAgent
        _agent = ResearchAgent()
        print("✓ Agent initialised successfully")
    except Exception as exc:
        print(f"✗ Agent init failed: {exc}")
        _agent = None
    yield
    # shutdown — nothing to clean up


app = FastAPI(title="AI Research Assistant", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if _agent is None:
        return {"status": "starting", "model": "gemini-2.5-flash", "kb_chunks": 0}
    return {
        "status": "ok",
        "model": "gemini-2.5-flash",
        "kb_chunks": _agent.rag.total_chunks(),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent is still starting up — please retry in a few seconds")

    async def event_stream():
        try:
            async for event in _agent.stream(req.message):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# Serve the single-page frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Dev entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,          # lifespan + reload don't mix well
    )
