import asyncio
import hashlib
import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import store, harness
from .detector import validate, DetectionError, inventory
from .settings import ROOT, MAX_BYTES
from .conversation import route, infer_language

@asynccontextmanager
async def lifespan(app):
    store.init()
    yield
    tasks = list(harness.JOBS.values())
    for task in tasks: task.cancel()
    if tasks: await asyncio.gather(*tasks, return_exceptions=True)

app = FastAPI(title="HybridSense API", version="0.2.0", lifespan=lifespan)

class Submission(BaseModel):
    code: str = Field(max_length=MAX_BYTES)
    language: str
    filename: str = Field(default="untitled", max_length=120)
    use_agent: bool = True


class SourceAttachment(BaseModel):
    filename: str = Field(max_length=120)
    language: str | None = None
    code: str = Field(max_length=MAX_BYTES)


class ChatSubmission(BaseModel):
    session_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    message: str = Field(default="", max_length=150000)
    sources: list[SourceAttachment] = Field(default_factory=list, max_length=10)
    use_agent: bool = True


@app.post("/api/chat", status_code=202)
async def chat(data: ChatSubmission):
    if len(harness.JOBS) >= 4: raise HTTPException(429, "任务队列已满，请稍后发送。")
    previous = store.conversation_runs(data.session_id)
    if any(r["id"] in harness.JOBS for r in previous): raise HTTPException(409, "当前对话仍在处理上一条消息。")
    sources = []
    for source in data.sources:
        language = source.language or infer_language(source.code, source.filename)
        if not language: raise HTTPException(422, f"无法识别 {source.filename} 的语言，请指定受支持的代码语言。")
        sources.append({**source.model_dump(), "language": language})
    try: intent, sources = route(data.message, sources)
    except (ValueError, DetectionError) as exc: raise HTTPException(422, str(exc))
    run = {"id": uuid.uuid4().hex, "kind": "conversation", "session_id": data.session_id,
        "intent": intent, "message": data.message, "sources": sources, "use_agent": data.use_agent,
        "status": "queued", "created_at": store.now(), "attempt": 1,
        "filename": sources[0]["filename"] if sources else "通用对话",
        "source_sha256": hashlib.sha256(json_source({"message": data.message, "sources": sources}).encode()).hexdigest()}
    store.save(run)
    store.event(run["id"], "route", "默认代码审计：生成整体与片段证据报告" if intent == "detect" else "按用户意图进行通用交流")
    harness.launch(run["id"])
    return {"id": run["id"], "intent": intent, "status": "queued"}


def json_source(sources):
    import json
    return json.dumps(sources, ensure_ascii=False, sort_keys=True)


@app.get("/api/chat/{session_id}")
def chat_history(session_id: str):
    return store.conversation_runs(session_id)

def require_run(run_id):
    run = store.get(run_id)
    if not run: raise HTTPException(404, "任务不存在。")
    return run

@app.get("/api/health")
def health():
    return {"status": "ok", "agent_configured": bool(os.getenv("DEEPSEEK_API_KEY")), "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), **inventory()}

@app.get("/api/runs")
def runs(): return store.listing()

@app.post("/api/runs", status_code=202)
async def submit(data: Submission):
    try: validate(data.code, data.language)
    except DetectionError as exc: raise HTTPException(422, str(exc))
    if len(harness.JOBS) >= 4: raise HTTPException(429, "任务队列已满，请等待当前任务完成。")
    run = {"id": uuid.uuid4().hex, "status": "queued", "created_at": store.now(), "attempt": 1,
        "filename": data.filename.replace("\\", "/").rsplit("/", 1)[-1], "language": data.language,
        "code": data.code, "use_agent": data.use_agent, "source_sha256": hashlib.sha256(data.code.encode()).hexdigest()}
    store.save(run)
    store.event(run["id"], "queued", "提交已接收，等待模型资源")
    harness.launch(run["id"])
    return {"id": run["id"], "status": "queued"}

@app.post("/api/upload", status_code=202)
async def upload(file: UploadFile = File(...), language: str = Form(...), use_agent: bool = Form(True)):
    raw = await file.read(MAX_BYTES + 1)
    await file.close()
    if len(raw) > MAX_BYTES: raise HTTPException(413, "文件上限为 128 KiB。")
    try: code = raw.decode("utf-8-sig")
    except UnicodeDecodeError: raise HTTPException(422, "请上传 UTF-8 编码的源代码文件。")
    return await submit(Submission(code=code, language=language, filename=(file.filename or "uploaded")[:120], use_agent=use_agent))

@app.get("/api/runs/{run_id}")
def get_run(run_id: str): return require_run(run_id)

@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str, after: int = 0):
    require_run(run_id)
    return store.events(run_id, after)

@app.post("/api/runs/{run_id}/cancel")
async def cancel(run_id: str):
    require_run(run_id)
    task = harness.JOBS.get(run_id)
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return require_run(run_id)

@app.post("/api/runs/{run_id}/resume", status_code=202)
async def resume(run_id: str):
    run = require_run(run_id)
    if run_id in harness.JOBS: raise HTTPException(409, "任务正在运行。")
    if len(harness.JOBS) >= 4: raise HTTPException(429, "任务队列已满。")
    run.update(status="queued", attempt=run["attempt"] + 1, error=None)
    store.save(run)
    harness.launch(run_id)
    return {"id": run_id, "status": "queued"}

@app.get("/api/runs/{run_id}/report")
def report(run_id: str, format: str = "json"):
    run = require_run(run_id)
    if format == "markdown":
        general = run.get("kind") == "conversation" and run.get("intent") == "chat"
        title = "对话记录" if general else "审计报告"
        text = f"# HybridSense {title}\n\n文件 / 会话：{run['filename']}\n\n任务：{run_id}\n\n{run.get('report', '内容尚未生成')}\n\n输入 SHA256：{run['source_sha256']}\n"
        if run.get("result"):
            text += "\n## 方法与研究依据\n\n分数未经独立校准，片段定位为探索性复核。完整计算定义见项目 RESEARCH.md。\n\n- [概率校准研究：Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)\n- [混写定位基准：HybridCodeAuthorship 2026](https://arxiv.org/abs/2606.12620)\n- [长代码窗口处理：CodeDet-NITS 2026](https://aclanthology.org/2026.semeval-1.397/)\n"
        return Response(text, media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="hybridsense-{run_id[:8]}.md"'})
    return {k: v for k, v in run.items() if k not in {"code", "sources", "message"}} | {"events": store.events(run_id)}

dist = ROOT / "frontend" / "dist"
if dist.exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    @app.get("/{path:path}")
    def frontend(path: str):
        if path.startswith("api/"): raise HTTPException(404)
        return FileResponse(dist / "index.html")
