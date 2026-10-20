import json
import sqlite3
from datetime import datetime, timezone
from .settings import RUNTIME

DB = RUNTIME / "runs.sqlite3"

def now(): return datetime.now(timezone.utc).isoformat()

def connect():
    db = sqlite3.connect(DB, timeout=15)
    db.row_factory = sqlite3.Row
    return db

def init():
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, data TEXT NOT NULL)")
    # A restarted process must not leave orphan tasks apparently running.
    for run in listing(full=True):
        if run["status"] in {"queued", "running"}:
            run.update(status="interrupted", error="服务曾重启，可从已保存的检测证据继续。")
            save(run)

def save(run):
    run["updated_at"] = now()
    with connect() as db:
        db.execute("INSERT INTO runs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data", (run["id"], json.dumps(run, ensure_ascii=False)))

def get(run_id):
    with connect() as db: row = db.execute("SELECT data FROM runs WHERE id=?", (run_id,)).fetchone()
    return json.loads(row[0]) if row else None

def listing(full=False):
    with connect() as db: rows = db.execute("SELECT data FROM runs ORDER BY rowid DESC LIMIT 100").fetchall()
    runs = [json.loads(r[0]) for r in rows]
    if full: return runs
    return [{k: v for k, v in r.items() if k not in {"code", "sources", "file_evidence", "result", "report", "message"}} | {"ai_probability": (r.get("result") or {}).get("ai_probability")} for r in runs if r.get("kind") != "conversation"]


def conversation_runs(session_id, limit=40):
    with connect() as db:
        rows = db.execute("SELECT data FROM runs WHERE json_extract(data, '$.session_id')=? ORDER BY rowid DESC LIMIT ?", (session_id, limit)).fetchall()
    return [json.loads(r[0]) for r in reversed(rows)]

def event(run_id, kind, message, **detail):
    data = {"time": now(), "kind": kind, "message": message, **detail}
    with connect() as db:
        cursor = db.execute("INSERT INTO events(run_id,data) VALUES (?,?)", (run_id, json.dumps(data, ensure_ascii=False)))
        return {"seq": cursor.lastrowid, **data}

def events(run_id, after=0):
    with connect() as db: rows = db.execute("SELECT seq,data FROM events WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after)).fetchall()
    return [{"seq": r[0], **json.loads(r[1])} for r in rows]
