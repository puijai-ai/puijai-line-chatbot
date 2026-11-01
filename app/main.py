"""FastAPI: Risk API + LINE Webhook + Interactive Demo Dashboard สำหรับแชทบอทปุยใจ (คนที่ B).

รัน:        uvicorn app.main:app --reload
Dashboard:  http://127.0.0.1:8000/
API Docs:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# รองรับการโหลดแพ็กเกจ ML จาก D: อัตโนมัติ (กรณีไดรฟ์ C: พื้นที่จำกัด)
for _extra_pkg in ("D:/miniconda3/Lib/site-packages", "D:/puijai_env/site-packages"):
    if Path(_extra_pkg).exists() and str(Path(_extra_pkg)) not in sys.path:
        sys.path.insert(0, str(Path(_extra_pkg)))

from puijai_risk.classifiers import RiskClassifier, decide_level, get_classifier
from puijai_risk.config import Settings, get_settings
from puijai_risk.preprocessing import tokenize
from puijai_risk.safety.policy import SafetyAction, decide_action
from puijai_risk.schemas import RiskLevel, RiskResult
from puijai_risk.windowing import Window, build_window

from .line_handler import reply_message, verify_signature
from .messages import build_reply_messages
from .session_store import InMemorySessionStore, hash_user_id

logger = logging.getLogger("puijai.app")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000, description="ข้อความล่าสุดของผู้ใช้")
    user_id: str | None = Field(default=None, description="ถ้าส่งมา ระบบจะจำข้อความก่อนหน้าให้เอง")
    history: list[str] | None = Field(
        default=None, description="ข้อความก่อนหน้า (เก่า→ใหม่) กรณีระบบปุยใจเดิมเก็บประวัติเอง"
    )
    classifier: str | None = Field(default=None, description="เลือกโมเดลเฉพาะกิจ: keyword | tfidf | wangchanberta")
    n_turns: int | None = Field(default=None, ge=1, le=10, description="จำนวนข้อความใน Sliding Window")
    critical_threshold: float | None = Field(default=None, ge=0.05, le=0.95)


class ClearSessionRequest(BaseModel):
    user_id: str


def notify_staff(user_hash: str, result: RiskResult) -> None:
    """แจ้งเตือนผู้ดูแลเมื่อพบความเสี่ยงระดับวิกฤต (ไม่บันทึกข้อความดิบของผู้ใช้ตามหลัก PDPA)."""
    logger.warning(
        "CRITICAL risk user=%s p_critical=%.2f",
        user_hash[:12],
        result.probabilities.get(RiskLevel.CRITICAL, 0.0),
    )


def _resolve_model_path(name: str, default_path: str) -> str | None:
    if name == "keyword":
        return None
    if name == "tfidf":
        for p in (default_path, "models/tfidf.joblib"):
            if Path(p).is_file():
                return p
        return "models/tfidf.joblib"
    if name == "wangchanberta":
        for p in (default_path, "models/wcb-n3", "models/wangchanberta-risk"):
            if (Path(p) / "config.json").is_file():
                return p
        return default_path
    return default_path


def _read_csv_if_exists(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def create_app(settings: Settings | None = None, classifier: RiskClassifier | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or get_settings()
        app.state.settings = s
        model_path = _resolve_model_path(s.risk_model, s.model_path)
        default_clf = classifier or get_classifier(
            s.risk_model,
            model_path,
            critical_threshold=s.critical_threshold,
            warning_threshold=s.warning_threshold,
        )
        app.state.classifier = default_clf
        app.state.classifier_cache = {default_clf.name: default_clf}
        app.state.sessions = InMemorySessionStore(s.session_max_turns, s.session_ttl_seconds)
        logger.info("loaded classifier=%s", app.state.classifier.name)
        yield

    app = FastAPI(title="Puijai Risk API", version="1.0.0", lifespan=lifespan)

    def get_or_load_classifier(name: str | None) -> RiskClassifier:
        s: Settings = app.state.settings
        if not name or name == app.state.classifier.name:
            return app.state.classifier
        cache: dict[str, RiskClassifier] = app.state.classifier_cache
        if name not in cache:
            mpath = _resolve_model_path(name, s.model_path)
            cache[name] = get_classifier(
                name,
                mpath,
                critical_threshold=s.critical_threshold,
                warning_threshold=s.warning_threshold,
            )
        return cache[name]

    async def assess(
        turns: list[str],
        clf_name: str | None = None,
        n_turns: int | None = None,
        crit_thresh: float | None = None,
    ) -> tuple[RiskResult, SafetyAction, Window, float, str]:
        s: Settings = app.state.settings
        clf = await run_in_threadpool(get_or_load_classifier, clf_name)
        eff_n = n_turns or s.n_turns
        window = build_window(turns, n_turns=eff_n, max_tokens=s.max_tokens)
        t0 = time.perf_counter()
        result: RiskResult = (await run_in_threadpool(clf.predict_windows, [window]))[0]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if crit_thresh is not None and crit_thresh != clf.critical_threshold:
            new_lvl = decide_level(result.probabilities, crit_thresh, clf.warning_threshold)
            evidence = clf.explain(window.text) if new_lvl != RiskLevel.NORMAL else []
            result = RiskResult(
                level=new_lvl,
                probabilities=result.probabilities,
                n_tokens=result.n_tokens,
                n_turns=result.n_turns,
                evidence=evidence,
            )
        return result, decide_action(result), window, latency_ms, clf.name

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        index_file = STATIC_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return HTMLResponse("<h1>Puijai Risk API</h1><p><a href='/docs'>Open API Docs</a></p>")

    @app.get("/health")
    async def health():
        return {"status": "ok", "classifier": app.state.classifier.name}

    @app.get("/models")
    async def list_models():
        return {
            "active": app.state.classifier.name,
            "available": {
                "keyword": True,
                "tfidf": Path("models/tfidf.joblib").is_file(),
                "wangchanberta": (Path("models/wcb-n3") / "config.json").is_file()
                or (Path("models/wangchanberta-risk") / "config.json").is_file(),
            },
        }

    @app.get("/experiments")
    async def get_experiments():
        res_dir = Path("results")
        return {
            "summary": _read_csv_if_exists(res_dir / "model_comparison_summary.csv"),
            "exp1": {
                m: _read_csv_if_exists(res_dir / f"length_experiment_{m}_exp1.csv")
                for m in ("keyword", "tfidf", "wangchanberta")
            },
            "exp2": {
                m: _read_csv_if_exists(res_dir / f"length_experiment_{m}_exp2.csv")
                for m in ("keyword", "tfidf", "wangchanberta")
            },
        }

    @app.post("/session/clear")
    async def clear_session(req: ClearSessionRequest):
        s: Settings = app.state.settings
        app.state.sessions.clear(hash_user_id(req.user_id, s.user_id_salt))
        return {"status": "cleared"}

    @app.post("/predict")
    async def predict(req: PredictRequest):
        s: Settings = app.state.settings
        if req.history is not None:
            turns = [*req.history, req.text]
        elif req.user_id:
            turns = app.state.sessions.append(hash_user_id(req.user_id, s.user_id_salt), req.text)
        else:
            turns = [req.text]
        result, action, window, latency_ms, used_clf = await assess(
            turns, req.classifier, req.n_turns, req.critical_threshold
        )
        return {
            "classifier": used_clf,
            "latency_ms": latency_ms,
            "window_text": window.text,
            "tokens": tokenize(window.text),
            "risk": result.to_dict(),
            "action": action.to_dict(),
        }

    @app.post("/callback")
    async def callback(request: Request, x_line_signature: str = Header(default="")):
        s: Settings = app.state.settings
        if not s.line_channel_secret:
            raise HTTPException(status_code=503, detail="LINE_CHANNEL_SECRET is not configured")
        body = await request.body()
        if not verify_signature(body, x_line_signature, s.line_channel_secret):
            raise HTTPException(status_code=400, detail="invalid signature")

        payload = json.loads(body or b"{}")
        for event in payload.get("events", []):
            message = event.get("message") or {}
            if event.get("type") != "message" or message.get("type") != "text":
                continue
            user_hash = hash_user_id(event.get("source", {}).get("userId", "anonymous"), s.user_id_salt)
            turns = app.state.sessions.append(user_hash, message.get("text", ""))
            result, action, _, _, _ = await assess(turns)
            await reply_message(event["replyToken"], build_reply_messages(action), s.line_channel_access_token)
            if action.notify_staff:
                notify_staff(user_hash, result)
        return {"status": "ok"}

    return app


app = create_app()
