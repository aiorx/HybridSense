"""Intent routing and transparent multi-file score summaries."""
import re
from pathlib import PurePath
from .detector import validate

EXTENSIONS = {"py": "python", "c": "c", "h": "c", "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "java": "java", "js": "javascript", "jsx": "javascript", "mjs": "javascript", "ts": "typescript", "tsx": "typescript"}
ALIASES = {**EXTENSIONS, "python": "python", "c++": "cpp", "javascript": "javascript", "typescript": "typescript"}
DETECTION = re.compile(r"检测|判别|鉴别|AI.{0,8}(?:痕迹|生成|倾向|可疑|占比|标识)|(?:是否|是不是).{0,6}AI|(?:detect|audit|authorship|AI.generated)", re.I)


def infer_language(code, filename="", hint=""):
    if hint.lower() in ALIASES: return ALIASES[hint.lower()]
    ext = PurePath(filename).suffix.lstrip(".").lower()
    if ext in EXTENSIONS: return EXTENSIONS[ext]
    if re.search(r"^\s*(?:def |from \w+ import|import \w+\s*$|async def )", code, re.M): return "python"
    if re.search(r"\binterface\s|:\s*(?:string|number|boolean)\b", code): return "typescript"
    if re.search(r"\b(?:public class|System\.out|public static)\b", code): return "java"
    if "#include" in code: return "cpp" if "std::" in code or "<iostream>" in code else "c"
    if re.search(r"\b(?:function|const|let|var)\s", code): return "javascript"
    return None


def route(message, sources):
    sources = [dict(s) for s in sources]
    message = message.strip()
    fenced = list(re.finditer(r"```([^\n`]*)\n([\s\S]*?)```", message))
    residual = re.sub(r"```[^\n`]*\n[\s\S]*?```", "", message).strip()
    no_detection = bool(re.search(r"(?:不要|不用|无需|不必|别).{0,3}(?:检测|审计|判别)|do not detect|don't detect", residual, re.I))
    intent = "detect" if (sources and not message) or (DETECTION.search(residual) and not no_detection) else "chat"
    if fenced and (not residual or intent == "detect"):
        for index, match in enumerate(fenced):
            lang = infer_language(match[2], hint=match[1].strip())
            if not lang: raise ValueError("无法识别代码块语言，请在 ``` 后标注 python、c、cpp、java、javascript 或 typescript。")
            sources.append({"filename": f"snippet-{index + 1}.{next(k for k, v in EXTENSIONS.items() if v == lang)}", "language": lang, "code": match[2]})
        intent = "detect"
    if not sources and not fenced:
        lang = infer_language(message)
        first = next((x.strip() for x in message.splitlines() if x.strip()), "")
        # Only unmistakable code-only submissions trigger the default audit.
        if not lang and re.match(r"^(?:print\(|[A-Za-z_]\w*\s*=)", first): lang = "python"
        if lang and re.match(r"(?:def |async def |from |import |#include|public class|function |const |let |interface |print\(|[A-Za-z_]\w*\s*=)", first):
            sources = [{"filename": "pasted-code", "language": lang, "code": message}]
            intent = "detect"
    if intent == "detect" and not sources: intent = "chat"
    if not sources and not message: raise ValueError("请输入消息或添加代码文件。")
    if len(sources) > 10: raise ValueError("每次最多提交 10 个文件或代码块。")
    if sum(len(s["code"].encode("utf-8")) for s in sources) > 512 * 1024: raise ValueError("每次代码总大小最多 512 KiB。")
    for source in sources:
        validate(source["code"], source["language"])
        source["filename"] = source["filename"].replace("\\", "/").rsplit("/", 1)[-1][:120]
    return intent, sources


def aggregate(files):
    weights = [sum(not c.isspace() for c in f["code"]) for f in files]
    total = max(1, sum(weights))
    score = sum(f["result"]["ai_probability"] * w for f, w in zip(files, weights)) / total
    hotspots = []
    for index, file in enumerate(files):
        for ri, r in enumerate(file["result"]["regions"]):
            hotspots.append({"file_index": index, "region_index": ri, "filename": file["filename"], **r})
    return {"is_batch": True, "ai_probability": score,
        "overall_score": {"value": score, "label": "多文件汇总倾向分数", "method": "nonwhitespace_character_weighted_file_scores", "calibrated": False},
        "files": [{k: v for k, v in file.items() if k != "code"} for file in files],
        "hotspots": sorted(hotspots, key=lambda r: r["ai_probability"], reverse=True)[:30],
        "warnings": ["多文件分数是按非空白字符数加权的文件级模型分数；跨语言模型尚未统一校准，只用于汇总参考。", "高分片段用于安排人工复核，不是已确认的 AI 作者标识。"],
        "method_version": "hybridsense-batch-1"}


def evidence_view(result):
    if result.get("is_batch"):
        return {**result, "files": [{**f, "result": evidence_view(f["result"])} for f in result["files"]], "hotspots": result["hotspots"][:10]}
    return {k: v for k, v in result.items() if k not in {"encoder", "regions", "local_summary"}} | {
        "region_count": len(result.get("regions", [])),
        "regions": sorted([dict(r, index=i) for i, r in enumerate(result.get("regions", []))], key=lambda r: r["ai_probability"], reverse=True)[:20],
        "local_summary": {k: v for k, v in result.get("local_summary", {}).items() if k != "line_scores"}}
