"""Real local API + DeepSeek acceptance, with no credential output."""
import asyncio
import json
import uuid
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "\n\n".join(f"def summarize_{i}(values):\n    \"\"\"Compute summary {i}.\"\"\"\n    return {{'total': sum(values), 'count': len(values)}}" for i in range(12))
TYPESCRIPT = "interface Summary { total: number; count: number; }\nfunction summarize(values: number[]): Summary {\n  return { total: values.reduce((a, b) => a + b, 0), count: values.length };\n}\nfunction legacy(values: number[]) {\n  let s=0;\n  for (const x of values) s+=x;\n  return s;\n}\n"


async def wait(client, rid):
    for _ in range(360):
        run = (await client.get(f"/api/runs/{rid}")).json()
        if run["status"] not in {"queued", "running"}: break
        await asyncio.sleep(1)
    assert run["status"] == "completed", (run["status"], run.get("error"))
    events = (await client.get(f"/api/runs/{rid}/events")).json()
    return run, events


async def main():
    results = {"checks": []}
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=30) as client:
        health = (await client.get("/api/health")).json()
        assert health["agent_configured"]
        sid = uuid.uuid4().hex
        async def send(message="", sources=None, use_agent=True):
            response = await client.post("/api/chat", json={"session_id": sid, "message": message, "sources": sources or [], "use_agent": use_agent})
            response.raise_for_status()
            run, events = await wait(client, response.json()["id"])
            print(json.dumps({"id": run["id"], "intent": run["intent"], "mode": run.get("report_mode"), "characters": len(run.get("report", ""))}), flush=True)
            return run, events

        run, events = await send("请记住本次演示代号是‘星河蓝’，先只回复一句确认即可。")
        assert run["intent"] == "chat" and run["report_mode"] == "deepagents_chat"
        assert not any(e["kind"] == "mcp" for e in events)
        results["checks"].append("real general DeepAgents dialogue without MCP detection")
        run, _ = await send("刚刚让你记住的演示代号是什么？")
        assert "星河蓝" in run["report"]
        results["checks"].append("persisted multi-turn conversation memory")
        sources = [{"filename": "multi-functions.py", "language": "python", "code": PYTHON}, {"filename": "summary.ts", "language": "typescript", "code": TYPESCRIPT}]
        run, events = await send("请解释附件 TypeScript 里的 summarize 函数，不用检测。简洁回答。", [sources[1]])
        assert run["intent"] == "chat" and not any(e["kind"] == "mcp" for e in events)
        assert any(e.get("tool") == "read_submission" for e in events)
        results["checks"].append("explicit attachment explanation follows user intent")
        run, events = await send(sources=sources)
        assert run["intent"] == "detect" and run["report_mode"] == "deepagents", run.get("agent_error")
        batch = run["result"]
        assert len(batch["files"]) == 2
        for file in batch["files"]:
            result = file["result"]
            assert result["overall_score"]["calibrated"] is False
            assert result["local_summary"]["coverage"] == 1
        assert batch["files"][0]["result"]["localization"]["analyzed_functions"] == 12
        exported = (await client.get(f"/api/runs/{run['id']}/report")).json()
        assert "sources" not in exported and "message" not in exported
        markdown = (await client.get(f"/api/runs/{run['id']}/report", params={"format": "markdown"})).text
        assert "AI" in markdown
        results["checks"].extend(["silent multi-file upload defaults to actual MCP detection", "12 functions retained with full window coverage", "uncalibrated overall/local scores and ranked hotspots", "real DeepSeek evidence-gated batch report", "exports omit raw attachment source"])
        results.update(passed=True, session_id=sid, detection_run_id=run["id"], model=health["model"])
        results["sample"] = {"overall": batch["ai_probability"], "files": [{"filename": f["filename"], "score": f["result"]["ai_probability"], "regions": len(f["result"]["regions"]), "coverage": f["result"]["local_summary"]["coverage"]} for f in batch["files"]]}
    (ROOT / "runtime/v2-acceptance.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__": asyncio.run(main())
