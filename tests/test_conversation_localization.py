import json
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from backend import store, harness
from backend.app import app
from backend.conversation import route, aggregate
from backend.detector import semantic_encoder
from backend.localization import make_regions, summarize_regions

SOURCE = {"filename": "sample.py", "language": "python", "code": "def f(x):\n    return x + 1\n"}


@pytest.mark.parametrize("message,sources,expected", [
    ("你好，可以帮我规划答辩吗", [], "chat"),
    ("", [SOURCE], "detect"),
    ("请解释并改进附件代码", [SOURCE], "chat"),
    ("不用检测，帮我修改代码", [SOURCE], "chat"),
    ("请检测这些代码的 AI 痕迹", [SOURCE], "detect"),
    ("```python\ndef f():\n    return 1\n```", [], "detect"),
    ("解释这个函数\n```python\ndef f():\n    return 1\n```", [], "chat"),
    ("def f():\n    return 1\n", [], "detect"),
    ("print(1)", [], "detect"),
    ("你能介绍一下 AI 代码检测的局限吗", [], "chat"),
])
def test_intent_routing(message, sources, expected):
    intent, files = route(message, sources)
    assert intent == expected
    if expected == "detect": assert files


def test_multiple_fenced_snippets_keep_languages():
    intent, files = route("```python\nprint(1)\n```\n```typescript\nconst n: number = 1;\n```", [])
    assert intent == "detect"
    assert [f["language"] for f in files] == ["python", "typescript"]


def test_overlapping_windows_are_not_counted_twice():
    summary = summarize_regions("abcdef", [
        {"kind": "window", "start_offset": 0, "end_offset": 6, "ai_probability": .2},
        {"kind": "window", "start_offset": 2, "end_offset": 6, "ai_probability": .8},
        {"kind": "function", "start_offset": 0, "end_offset": 6, "ai_probability": 1},
    ])
    assert summary["score"] == pytest.approx(.4)
    assert summary["coverage"] == 1
    assert summary["thresholds"]["validated"] is False


def test_many_functions_and_tail_are_scanned():
    tokenizer, _ = semantic_encoder()
    code = "\n\n".join(f"def part_{i}(x):\n    return x + {i}" for i in range(22)) + "\n# 尾部代码\nprint(part_21(1))\n"
    regions, info = make_regions(code, "python", tokenizer)
    functions = [r for r in regions if r["kind"] == "function"]
    assert len(functions) == 22  # Regression: the old implementation stopped after 8.
    assert info["parser_status"] == "ok"
    for r in regions: r["ai_probability"] = .5
    summary = summarize_regions(code, regions)
    assert summary["coverage"] == 1
    assert summary["line_scores"][-1]["score"] == pytest.approx(.5)
    assert max(r["end_offset"] for r in regions if r["kind"] == "window") == len(code)


def test_unicode_offsets_preserve_exact_source_after_emoji():
    tokenizer, _ = semantic_encoder()
    code = "# 🙂 中文\n\ndef 开始(值):\n    return 值 + 1\n"
    regions, _ = make_regions(code, "python", tokenizer)
    function = next(r for r in regions if r["kind"] == "function")
    assert function["offset_unit"] == "unicode_codepoints"
    assert code[function["start_offset"]:function["end_offset"]] == "def 开始(值):\n    return 值 + 1"


@pytest.mark.parametrize("language,code", [
    ("c", "int first(int x) { return x + 1; }\nint second(int x) { return x * 2; }"),
    ("cpp", "int first(int x) { return x + 1; }\nint second(int x) { return x * 2; }"),
    ("java", "class Sample { int first(int x) { return x + 1; } int second(int x) { return x * 2; } }"),
    ("javascript", "function first(x) { return x + 1; }\nfunction second(x) { return x * 2; }"),
    ("typescript", "function first(x: number) { return x + 1; }\nfunction second(x: number) { return x * 2; }"),
])
def test_ast_function_boundaries_for_other_languages(language, code):
    tokenizer, _ = semantic_encoder()
    regions, info = make_regions(code, language, tokenizer)
    assert info["parser_status"] == "ok"
    assert {r["name"] for r in regions if r["kind"] == "function"} == {"first", "second"}


def test_window_budget_reports_incomplete_coverage(monkeypatch):
    from backend import localization
    monkeypatch.setattr(localization, "MAX_WINDOWS", 2)
    tokenizer, _ = semantic_encoder()
    code = "x = 1\n" * 1000
    regions, info = make_regions(code, "python", tokenizer)
    for r in regions: r["ai_probability"] = .5
    assert info["budget_limited"]
    assert info["analyzed_windows"] == 2
    assert 0 < summarize_regions(code, regions)["coverage"] < 1


def test_batch_weights_and_hotspot_filename():
    files = [{"filename": "one.py", "language": "python", "code": "a ", "result": {"ai_probability": .8, "regions": [{"start": 1, "end": 1, "ai_probability": .9}]}},
             {"filename": "two.ts", "language": "typescript", "code": "bbb", "result": {"ai_probability": .2, "regions": []}}]
    result = aggregate(files)
    assert result["ai_probability"] == pytest.approx(.35)
    assert result["overall_score"]["calibrated"] is False
    assert result["hotspots"][0]["filename"] == "one.py"
    assert all("code" not in f for f in result["files"])


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "runs.sqlite3")
    monkeypatch.setattr(harness, "RUNTIME", tmp_path)
    store.init()
    return tmp_path


def test_chat_api_routes_and_reports_omit_sources(isolated):
    sid = "a" * 32
    with patch.object(harness, "launch") as launch, TestClient(app) as client:
        uploaded = client.post("/api/chat", json={"session_id": sid, "sources": [SOURCE]})
        assert uploaded.status_code == 202
        assert uploaded.json()["intent"] == "detect"
        rid = uploaded.json()["id"]
        exported = client.get(f"/api/runs/{rid}/report").json()
        assert "sources" not in exported and "message" not in exported
        assert client.get(f"/api/chat/{sid}").json()[0]["sources"][0]["code"] == SOURCE["code"]
        chat = client.post("/api/chat", json={"session_id": sid, "message": "请帮我规划答辩"})
        assert chat.json()["intent"] == "chat"
        assert launch.call_count == 2
        assert client.post("/api/chat", json={"session_id": sid}).status_code == 422
        assert client.get("/api/runs").json() == []


class GeneralModel(BaseChatModel):
    seen: list = []
    @property
    def _llm_type(self): return "general-dialogue-test"
    def bind_tools(self, tools, **kwargs):
        self.seen.extend(getattr(t, "name", "") for t in tools)
        return self
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="可以，我们先确定答辩的中心论点。"))])


@pytest.mark.asyncio
async def test_general_chat_uses_deepagents_without_detection_gate(isolated, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-placeholder")
    model = GeneralModel()
    run = {"id": "general", "kind": "conversation", "intent": "chat", "session_id": "b" * 32, "message": "帮我规划答辩", "sources": []}
    with patch("langchain_openai.ChatOpenAI", return_value=model):
        answer = await harness.interpret(run, None)
    assert "中心论点" in answer
    assert "read_submission" in model.seen
    assert "inspect_detection" not in model.seen
    assert "execute" not in model.seen
