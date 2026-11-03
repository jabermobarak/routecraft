"""Stateless routing API. No text persistence or external model calls."""

import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .engine import Engine

STATIC = Path(__file__).parent / "static"


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    texts: list[str] = Field(min_length=1, max_length=16)

    @field_validator("texts")
    @classmethod
    def valid_texts(cls, values):
        if any(not 2 <= len(t.strip()) <= 1000 for t in values):
            raise ValueError("Each text must contain 2..1000 characters")
        return [t.strip() for t in values]


def create_app(directory=None, api_token=None):
    directory = Path(directory or os.getenv("RC_ARTIFACTS", "artifacts"))
    token = os.getenv("RC_API_TOKEN", "") if api_token is None else api_token

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = None
        try:
            app.state.engine = Engine(directory)
        except (OSError, ValueError, KeyError, RuntimeError):
            pass
        yield

    app = FastAPI(title="RouteCraft", version="1.0.0", lifespan=lifespan)

    def authorize(authorization: Annotated[str | None, Header()] = None):
        if token and not secrets.compare_digest(authorization or "", "Bearer " + token):
            raise HTTPException(401, "Valid bearer token required")

    def ready_engine():
        if app.state.engine is None:
            raise HTTPException(
                503, "Model bundle unavailable or invalid. Run python -m routecraft.cli verify"
            )
        return app.state.engine

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready(engine: Annotated[Engine, Depends(ready_engine)]):
        return {"status": "ready", "model_version": engine.version}

    @app.get("/api/config", dependencies=[Depends(authorize)])
    def config(engine: Annotated[Engine, Depends(ready_engine)]):
        return {
            "model_version": engine.version,
            "labels": engine.report["labels"],
            "max_batch": 16,
            "examples": [
                "Please find me a flight from Moscow to Beirut.",
                "Can you check where my order is?",
                "Switch the assistant language to English.",
                "I would like a table for two tomorrow evening.",
                "Explain the theory of quantum gravity.",
            ],
            "language": "English",
            "max_tokens": 48,
            "data_source": "CLINC150 ten-intent subset",
        }

    @app.get("/api/report", dependencies=[Depends(authorize)])
    def report(engine: Annotated[Engine, Depends(ready_engine)]):
        return engine.report

    @app.post("/api/route", dependencies=[Depends(authorize)])
    def route(body: Batch, engine: Annotated[Engine, Depends(ready_engine)]):
        return {"request_id": str(uuid.uuid4()), **engine.route(body.texts)}

    @app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(authorize)])
    def metrics(engine: Annotated[Engine, Depends(ready_engine)]):
        return (
            f"# TYPE routecraft_requests_total counter\nroutecraft_requests_total {engine.requests}\n"
            f"# TYPE routecraft_texts_total counter\nroutecraft_texts_total {engine.texts_scored}\n"
            f"# TYPE routecraft_seconds_sum counter\nroutecraft_seconds_sum {engine.total_seconds}\n"
        )

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
