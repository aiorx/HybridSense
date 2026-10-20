import json
from unittest.mock import patch
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from backend import harness, store

class EvidenceModel(BaseChatModel):
    step: int = 0
    seen: list = []
    @property
    def _llm_type(self): return "evidence-test-model"
    def bind_tools(self, tools, **kwargs):
        self.seen.extend(getattr(t, "name", "") for t in tools)
        return self
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.step += 1
        if self.step == 1:
            msg = AIMessage(content="", tool_calls=[{"name":"write_todos", "args":{"todos":[{"content":"读取 MCP 证据", "status":"in_progress"}]}, "id":"plan-1"}])
        elif self.step == 2:
            msg = AIMessage(content="", tool_calls=[{"name":"inspect_detection", "args":{}, "id":"evidence-1"}])
        elif self.step == 3:
            msg = AIMessage(content="", tool_calls=[{"name":"write_todos", "args":{"todos":[{"content":"读取 MCP 证据", "status":"completed"}]}, "id":"plan-2"}])
        else: msg = AIMessage(content="结论：模型分数为 50%，证据不确定。请人工复核。")
        return ChatResult(generations=[ChatGeneration(message=msg)])

@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "runs.sqlite3")
    monkeypatch.setattr(harness, "RUNTIME", tmp_path)
    store.init()
    return tmp_path

@pytest.mark.asyncio
async def test_deepagents_plan_tool_and_checkpoint(isolated_store, monkeypatch):
    model = EvidenceModel()
    run = {"id":"test-evidence-run", "code":"print(1)", "result":{"ai_probability":.5,"regions":[],"warnings":["uncertain"]}}
    async def fake_load(session): return []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-placeholder")
    with patch("langchain_openai.ChatOpenAI", return_value=model), patch("langchain_mcp_adapters.tools.load_mcp_tools", side_effect=fake_load):
        report = await harness.interpret(run, None)
    assert "证据不确定" in report
    assert "inspect_detection" in model.seen
    assert "execute" not in model.seen
    assert (isolated_store / "agent-checkpoints.sqlite3").exists()
    assert any(e["kind"] == "tool" for e in store.events(run["id"]))

@pytest.mark.asyncio
async def test_actual_mcp_protocol():
    async with harness.mcp_session() as session:
        tools = await session.list_tools()
        assert {"list_detectors", "analyze_code"} <= {t.name for t in tools.tools}
        result = harness.unwrap(await session.call_tool("list_detectors", {}))
        assert len(result["languages"]) == 6
        error = await session.call_tool("analyze_code", {"code":"", "language":"python"})
        assert error.isError
        real = harness.unwrap(await session.call_tool("analyze_code", {"code":"def add(a, b):\n    return a + b\n", "language":"python", "localize":False}))
        assert real["feature_count"] == 400
        assert 0 <= real["ai_probability"] <= 1

def test_restart_marks_orphan_task_interrupted(isolated_store):
    store.save({"id":"orphan", "status":"running"})
    store.init()
    assert store.get("orphan")["status"] == "interrupted"
