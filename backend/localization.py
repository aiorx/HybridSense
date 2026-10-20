"""Exploratory localization, with explicit coverage and overlap correction.

These are engineering policies, not calibrated authorship estimators. See RESEARCH.md.
"""
import ast
import bisect
import importlib
import numpy as np

METHOD_VERSION = "hybridsense-localization-2"
HIGH_THRESHOLD = .80
REVIEW_THRESHOLD = .65
MAX_WINDOWS = 128
MAX_FUNCTIONS = 96


def review_label(score):
    return "high_signal" if score >= HIGH_THRESHOLD else "review" if score >= REVIEW_THRESHOLD else "lower_signal"


def make_regions(code, language, tokenizer):
    starts = [0] + [i + 1 for i, char in enumerate(code) if char == "\n"]
    def region(a, b, kind, name):
        text = code[a:b]
        token_count = len(tokenizer.encode(text, add_special_tokens=False, verbose=False))
        return {"start": bisect.bisect_right(starts, a), "end": bisect.bisect_right(starts, max(a, b - 1)),
            "start_offset": a, "end_offset": b, "offset_unit": "unicode_codepoints", "kind": kind, "name": name, "token_count": token_count,
            "nonblank_lines": sum(bool(line.strip()) for line in text.splitlines()),
            "context_flags": (["short_context"] if sum(bool(x.strip()) for x in text.splitlines()) < 12 else []) + (["semantic_truncation"] if token_count > 510 else [])}

    encoded = tokenizer(code, add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False)
    offsets = encoded["offset_mapping"]
    windows = []
    for index in range(0, len(offsets), 384):
        stop = min(index + 480, len(offsets))
        a = 0 if index == 0 else offsets[index][0]
        b = len(code) if stop == len(offsets) else offsets[stop][0]
        if b > a: windows.append(region(a, b, "window", f"上下文窗口 {len(windows) + 1}"))
        if stop == len(offsets): break
    total_windows = len(windows)
    # Uniform sampling preserves both head and tail when the resource budget is hit.
    if total_windows > MAX_WINDOWS:
        windows = [windows[i] for i in np.linspace(0, total_windows - 1, MAX_WINDOWS, dtype=int)]
    functions, parser_status = [], "ok"
    try:
        if language == "python":
            lines = code.splitlines(keepends=True)
            parsed = ast.parse(code)
            for node in ast.walk(parsed):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
                a = starts[node.lineno - 1] + len(lines[node.lineno - 1].encode("utf-8")[:node.col_offset].decode("utf-8"))
                b = starts[node.end_lineno - 1] + len(lines[node.end_lineno - 1].encode("utf-8")[:node.end_col_offset].decode("utf-8"))
                functions.append(region(a, b, "function", node.name))
            functions.sort(key=lambda r: r["start_offset"])
        else:
            functions, parser_status = tree_functions(code, language, region)
    except (ImportError, ValueError, RecursionError, SyntaxError):
        parser_status = "window_fallback"
    total_functions = len(functions)
    functions = functions[:MAX_FUNCTIONS]
    return windows + functions, {"version": METHOD_VERSION, "parser_status": parser_status,
        "total_windows": total_windows, "analyzed_windows": len(windows), "total_functions": total_functions,
        "analyzed_functions": len(functions), "budget_limited": total_windows > MAX_WINDOWS or total_functions > MAX_FUNCTIONS,
        "window_tokens": 480, "stride_tokens": 384, "max_windows": MAX_WINDOWS, "max_functions": MAX_FUNCTIONS}


def tree_functions(code, language, region):
    from tree_sitter import Language, Parser
    module = importlib.import_module("tree_sitter_" + language)
    grammar = module.language_typescript() if language == "typescript" else module.language()
    tree = Parser(Language(grammar)).parse(code.encode("utf-8"))
    parser_status = "partial_syntax" if tree.root_node.has_error else "ok"
    raw = code.encode("utf-8")
    stack, functions = [tree.root_node], []
    types = {"function_definition", "function_declaration", "method_declaration", "method_definition", "generator_function_declaration", "arrow_function"}
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.children))
        if node.type not in types: continue
        name_node = node.child_by_field_name("name")
        if name_node is None and language in {"c", "cpp"}:
            declarator = node.child_by_field_name("declarator")
            while declarator is not None:
                if declarator.type in {"identifier", "field_identifier", "qualified_identifier", "operator_name"}:
                    name_node = declarator
                    break
                declarator = declarator.child_by_field_name("declarator")
        name = raw[name_node.start_byte:name_node.end_byte].decode("utf-8") if name_node else f"匿名函数 · L{node.start_point.row + 1}"
        a = len(raw[:node.start_byte].decode("utf-8"))
        b = len(raw[:node.end_byte].decode("utf-8"))
        functions.append(region(a, b, "function", name))
    return functions, parser_status


def summarize_regions(code, regions):
    sums = np.zeros(len(code), dtype=float)
    counts = np.zeros(len(code), dtype=np.int16)
    for r in regions:
        if r["kind"] != "window": continue
        a, b = r["start_offset"], r["end_offset"]
        sums[a:b] += r["ai_probability"]
        counts[a:b] += 1
    nonspace = np.array([not c.isspace() for c in code], dtype=bool)
    covered = (counts > 0) & nonspace
    values = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    line_scores, offset = [], 0
    for index, line in enumerate(code.splitlines(keepends=True)):
        mask = covered[offset:offset + len(line)]
        local = values[offset:offset + len(line)][mask]
        line_scores.append({"line": index + 1, "score": float(local.mean()) if len(local) else None})
        offset += len(line)
    peak = max((r["ai_probability"] for r in regions), default=None)
    return {"score": float(values[covered].mean()) if covered.any() else None,
        "method": "mean_over_windows_per_character_then_mean_over_covered_nonwhitespace_characters",
        "coverage": float(covered.sum() / max(1, nonspace.sum())), "coverage_unit": "nonwhitespace_characters",
        "high_signal_regions": sum(r["ai_probability"] >= HIGH_THRESHOLD for r in regions),
        "peak_score": peak, "line_scores": line_scores, "calibrated": False,
        "thresholds": {"high_signal": HIGH_THRESHOLD, "review": REVIEW_THRESHOLD, "validated": False},
        "interpretation": "局部汇总与高信号标识用于人工复核；不是 AI 代码比例、行级作者标签或置信区间。"}
