
import json
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# Paths and settings
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:4b"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant and English teacher. "
    "Explain things clearly, use examples, and gently correct "
    "the user's English when appropriate."
)

app = FastAPI(title="My Local AI")

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
async def chat(body: ChatRequest):
    # The browser sends the conversation history.
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *body.messages,
        ],
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "num_predict": 512
        },
    }

    async def generate():
        try:
            timeout = httpx.Timeout(300.0, connect=10.0)

            async with httpx.AsyncClient(
                timeout=timeout
            ) as client:
                async with client.stream(
                    "POST",
                    OLLAMA_URL,
                    json=payload,
                ) as response:

                    if response.is_error:
                        details = (
                            await response.aread()
                        ).decode("utf-8", errors="replace")

                        yield json.dumps({
                            "error": (
                                f"Ollama HTTP {response.status_code}: "
                                f"{details}"
                            )
                        }) + "\n"
                        return

                    # Forward Ollama's streamed JSON lines.
                    async for line in response.aiter_lines():
                        if line:
                            yield line + "\n"

        except httpx.TimeoutException:
            yield json.dumps({
                "error": "Ollama took too long to respond."
            }) + "\n"

        except httpx.RequestError as error:
            yield json.dumps({
                "error": (
                    "Could not connect to Ollama. "
                    "Check that Ollama is running. "
                    f"Details: {error}"
                )
            }) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )
