"""Validate cancellation/recovery and actual API report export against localhost."""
import asyncio
import json
from pathlib import Path
import httpx

async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=20) as client:
        invalid = await client.post("/api/runs", json={"code":"", "language":"python"})
        assert invalid.status_code == 422
        invalid = await client.post("/api/upload", data={"language":"python"}, files={"file":("too_big.py", b"x" * 131073)})
        assert invalid.status_code == 413
        invalid = await client.post("/api/upload", data={"language":"python"}, files={"file":("bad.py", b"\xff\xfe\x00")})
        assert invalid.status_code == 422
        response = await client.post("/api/runs", json={"code":"def square(x):\n    return x * x\n", "language":"python", "filename":"recovery-check.py", "use_agent":False})
        response.raise_for_status()
        run_id = response.json()["id"]
        await asyncio.sleep(.5)
        cancelled = await client.post(f"/api/runs/{run_id}/cancel")
        assert cancelled.json()["status"] == "cancelled", cancelled.text
        resumed = await client.post(f"/api/runs/{run_id}/resume")
        resumed.raise_for_status()
        for _ in range(90):
            await asyncio.sleep(1)
            run = (await client.get(f"/api/runs/{run_id}")).json()
            if run["status"] not in {"queued", "running"}: break
        assert run["status"] == "completed", run
        assert run["attempt"] == 2
        assert run["result"]["encoder"]["revision"] != "unrecorded"
        exported = await client.get(f"/api/runs/{run_id}/report")
        assert "code" not in exported.json()
        markdown = await client.get(f"/api/runs/{run_id}/report?format=markdown")
        assert markdown.status_code == 200 and "attachment" in markdown.headers["content-disposition"]
        resumed = await client.post(f"/api/runs/{run_id}/resume")
        for _ in range(30):
            await asyncio.sleep(.5)
            run = (await client.get(f"/api/runs/{run_id}")).json()
            if run["status"] not in {"queued", "running"}: break
        assert run["status"] == "completed"
        events = (await client.get(f"/api/runs/{run_id}/events")).json()
        assert any(e["kind"]=="checkpoint" and "恢复" in e["message"] for e in events)
        output={"passed":True,"run_id":run_id,"checks":["empty source rejected","oversize upload rejected","invalid UTF-8 rejected","cancel","resume","encoder revision recorded","JSON export excludes source","Markdown attachment","reuse saved evidence"]}
        (Path(__file__).resolve().parents[1]/"runtime/api-validation.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
        print("API validation passed")

if __name__ == "__main__": asyncio.run(main())
