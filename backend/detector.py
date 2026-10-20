"""Adapters over the original extractors. Never execute submitted source code."""
import csv
import json
import hashlib
import importlib.util
import re
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .settings import LANGUAGES, MODEL_ROOT, EMBEDDING_PATH, MAX_BYTES
from .localization import make_regions, summarize_regions, review_label

class DetectionError(ValueError):
    pass

def validate(code: str, language: str):
    if language not in LANGUAGES:
        raise DetectionError("请选择六种受支持的编程语言之一。")
    if not code.strip() or "\x00" in code:
        raise DetectionError("请输入非空的文本代码，不能包含 NUL 字符。")
    if len(code.encode("utf-8")) > MAX_BYTES:
        raise DetectionError("单文件上限为 128 KiB。")
    if len(code.splitlines()) > 3000:
        raise DetectionError("单文件最多 3000 行，请按模块拆分。")

@lru_cache(maxsize=12)
def load_source(language: str, filename: str):
    folder = MODEL_ROOT / LANGUAGES[language][0]
    name = f"hybridsense_original_{language}_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, folder / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def artifacts(language):
    folder, tag, _, _ = LANGUAGES[language]
    base = MODEL_ROOT / folder / tag
    models = sorted(base.rglob("*.joblib"))
    schemas = sorted(base.rglob("feature_schema.json"))
    tables = sorted(base.rglob("features*.csv"))
    if not models or not (schemas or tables):
        raise DetectionError(f"{folder} 缺少已训练模型或特征表。")
    if schemas:
        names = json.loads(schemas[0].read_text(encoding="utf-8"))["features"]
    else:
        with tables[0].open(encoding="utf-8-sig", newline="") as f:
            names = [n for n in next(csv.reader(f)) if n not in {"label", "file_name", "filename"}]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise DetectionError("训练特征顺序表格式无效。")
    return models[0], names

@lru_cache(maxsize=6)
def load_model(language):
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    path, names = artifacts(language)
    # A pickle version mismatch is a deployment error, not a reason to guess.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(path)
    if model.n_features_in_ != len(names):
        raise DetectionError("模型维度与训练特征表不匹配，已拒绝预测。")
    # Bound CPU consumption across all saved ensemble components.
    params = {k: 2 for k in model.get_params() if k == "n_jobs" or k.endswith("__n_jobs")}
    if params:
        model.set_params(**params)
    for estimator in getattr(model, "estimators_", []):
        if "n_jobs" in estimator.get_params(): estimator.set_params(n_jobs=2)
    return model, names, path

@lru_cache(maxsize=1)
def semantic_encoder():
    if not (EMBEDDING_PATH / "config.json").exists():
        raise DetectionError("缺少 CodeBERTa 语义权重。请先运行 scripts/prepare_models.py；系统不会用全零语义特征伪造判别。")
    # On Windows the bundled OpenMP/BLAS DLLs must be initialized before torch.
    # detect() naturally loads sklearn first; standalone localization does not.
    import sklearn
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(str(EMBEDDING_PATH), local_files_only=True)
    model = AutoModel.from_pretrained(str(EMBEDDING_PATH), local_files_only=True)
    model.eval()
    if model.config.hidden_size != 768:
        raise DetectionError("CodeBERTa 的隐藏维度与训练要求的 768 维不一致。")
    return tokenizer, model

def embeddings(codes):
    import torch
    tokenizer, model = semantic_encoder()
    result = []
    for offset in range(0, len(codes), 8):
        inputs = tokenizer(codes[offset:offset + 8], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.inference_mode():
            values = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        result.extend({f"codeberta_dim_{i}": float(v) for i, v in enumerate(row)} for row in values)
    return result

def manual_features(code, language):
    file = "feature_extractor_plus.py" if language == "cpp" else "feature_extractor.py"
    module = load_source(language, file)
    original = load_source(language, "config.py")
    config = SimpleNamespace(**{k: getattr(original, k) for k in dir(original) if k.isupper()})
    # Checked-in configs may describe a later ablation, while data_all's CSV
    # contains features from the complete experiment. Extract every manual
    # group and select only the columns retained by that saved model.
    config.FEATURES = {k: True if isinstance(v, bool) else v for k, v in config.FEATURES.items()}
    config.FEATURES["codeberta"] = False
    wrapper = LANGUAGES[language][2]
    if wrapper:
        if language == "javascript" and not module.ESPRIMA_AVAILABLE:
            raise DetectionError("缺少 esprima，无法复原 JavaScript 语法特征。")
        if language == "java" and not module.JAVALANG_AVAILABLE:
            raise DetectionError("缺少 javalang，无法复原 Java 语法特征。")
        ex = getattr(module, wrapper)(config)
        values = ex.extract_batch([code])[0]
        return dict(zip(ex.get_feature_names(), map(float, values)))
    result = {}
    for cls in ["LexicalFeatureExtractor", "CppSpecificFeatureExtractor" if language == "cpp" else "CSpecificFeatureExtractor", "HumanErrorFeatureExtractor"]:
        ex = getattr(module, cls)(config)
        result.update(zip(ex.get_feature_names(), map(float, ex.extract(code))))
    ex = module.SyntacticFeatureExtractor(config)
    if not ex.use_tree_sitter:
        raise DetectionError("Tree-sitter 解析器不可用，无法复原 C/C++ 语法特征。")
    result.update(zip(ex.get_structural_feature_names(), map(float, ex.extract_structural_features([code])[0])))
    for n in [2, 3, 4]:
        values = ex.extract_ngram_features([code], n)[0]
        result.update(zip(ex.get_feature_names(n), map(float, values)))
    return result

def feature_group(name):
    if name.startswith("codeberta_"): return "semantic"
    if name.startswith(("ast_", "nttf_", "ntad_", "ntmd_")) or "ast" in name: return "syntactic"
    if any(k in name for k in ["human", "micro_", "magic", "informal", "space", "commented", "trailing", "indent", "redundant"]): return "human"
    if name.startswith(("py_", "ts_", "cpp_", "c_", "java_", "js_")) or any(k in name for k in ["pointer", "stl_", "arrow", "type_", "template", "builtin"]): return "language"
    return "lexical"

def detect(code: str, language: str, localize: bool = True):
    validate(code, language)
    model, names, path = load_model(language)
    tokenizer, _ = semantic_encoder()
    chunks, localization = make_regions(code, language, tokenizer) if localize else ([], {"enabled": False})
    sources = [code] + [code[s["start_offset"]:s["end_offset"]] for s in chunks]
    semantic = embeddings(sources) if any(n.startswith("codeberta_") for n in names) else [{} for _ in sources]
    rows, maps = [], []
    for source, emb in zip(sources, semantic):
        values = {**manual_features(source, language), **emb}
        missing = [n for n in names if n not in values and not n.startswith("ast_n")]
        if missing:
            raise DetectionError(f"原始提取器与模型特征不兼容：{', '.join(missing[:8])}")
        row = [values.get(n, 0.0) for n in names]
        if not np.isfinite(row).all():
            raise DetectionError("特征含非有限数值，已拒绝预测。")
        rows.append(row)
        maps.append(values)
    X = np.asarray(rows)
    classes = list(model.classes_)
    if set(classes) != {0, 1}: raise DetectionError("模型标签与训练数据的 0=Human、1=AI 不一致。")
    probs = model.predict_proba(X)[:, classes.index(1)]
    p = float(probs[0])
    voters = []
    region_votes = []
    for name, voter in getattr(model, "named_estimators_", {}).items():
        vp = voter.predict_proba(X)[:, list(voter.classes_).index(1)]
        voters.append({"name": name, "model": type(voter).__name__, "ai_probability": float(vp[0])})
        region_votes.append(vp[1:])
    regions = []
    for index, (chunk, prob) in enumerate(zip(chunks, probs[1:])):
        scores = [float(v[index]) for v in region_votes] or [float(prob)]
        regions.append({**chunk, "ai_probability": float(prob), "ai_score": float(prob), "review_label": review_label(prob),
            "voter_range": [min(scores), max(scores)], "method": "exploratory_fragment_prediction"})
    local_summary = summarize_regions(code, regions)
    mixed = len(regions) > 1 and max(probs[1:]) >= .65 and min(probs[1:]) <= .35
    verdict = "mixed_signals" if mixed else "ai_leaning" if p >= .65 else "human_leaning" if p <= .35 else "uncertain"
    counts = Counter(feature_group(n) for n in names)
    top = []
    importance = None
    if hasattr(model, "feature_importances_"): importance = model.feature_importances_
    elif getattr(model, "named_estimators_", None):
        importance = next((e.feature_importances_ for e in model.named_estimators_.values() if hasattr(e, "feature_importances_")), None)
    if importance is not None:
        for i in np.argsort(importance)[-10:][::-1]:
            top.append({"name": names[i], "value": rows[0][i], "global_importance": float(importance[i]), "group": feature_group(names[i])})
    warnings_list = ["AI 生成倾向分数为未校准模型输出，不是真实后验概率或 AI 代码占比。", "片段分数与局部汇总属于探索性定位；高信号阈值 0.80、复核阈值 0.65 为可复核的工程设置，尚未经独立验证集确定。", "片段分布可能偏离文件级训练分布；行色来自覆盖窗口均值，不是逐行作者标签；模型间范围不是置信区间。", "五维标签是按特征名整理的展示分组，并非独立的五个分类器。"]
    tokenizer, _ = semantic_encoder()
    if len(tokenizer.encode(code, truncation=False)) > 512:
        warnings_list.append("整体分数沿用原文件级分类器，其 CodeBERTa 向量最多 512 token；另以 480 token 重叠窗口扫描全文，覆盖情况见局部汇总。")
    if localization.get("budget_limited"): warnings_list.append("片段数量超过执行预算，窗口均匀抽样、函数仅保留前 96 个；请以实际覆盖率解读，建议拆分提交。")
    if localization.get("parser_status") != "ok" and localize: warnings_list.append("语法存在不完整结构或解析器回退；仍保留上下文窗口扫描，函数边界可能不完整。")
    if len(code.splitlines()) < 12: warnings_list.append("代码较短，建议提交完整函数或文件后复核。")
    if language == "typescript": warnings_list.append("原始 TypeScript 提取器可选 ts_morph 未启用；选定模型保留的特征不依赖该解析器。")
    manifest_path = EMBEDDING_PATH / "hybridsense-manifest.json"
    encoder_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"repo":"huggingface/CodeBERTa-small-v1", "revision":"unrecorded"}
    return {"language": language, "ai_probability": p, "ai_score": p,
        "overall_score": {"value": p, "label": "整体 AI 生成倾向分数", "method": "original_file_classifier", "calibrated": False},
        "local_summary": local_summary, "localization": localization,
        "hotspots": sorted(range(len(regions)), key=lambda i: regions[i]["ai_probability"], reverse=True),
        "verdict": verdict, "mixed_signals": bool(mixed),
        "feature_count": len(names), "groups": [{"id": g, "count": counts[g], "available": counts[g] > 0} for g in ["lexical", "syntactic", "human", "language", "semantic"]],
        "voters": voters or [{"name": "classifier", "model": type(model).__name__, "ai_probability": p}],
        "regions": regions, "top_features": top, "model": str(path.relative_to(MODEL_ROOT)),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "schema_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "encoder":encoder_manifest, "extractor_sha256":hashlib.sha256((MODEL_ROOT / LANGUAGES[language][0] / ("feature_extractor_plus.py" if language == "cpp" else "feature_extractor.py")).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(code.encode()).hexdigest(), "line_count": len(code.splitlines()), "warnings": warnings_list}

def inventory():
    results = []
    for key, (folder, tag, _, ext) in LANGUAGES.items():
        try:
            path, names = artifacts(key)
            results.append({"id": key, "name": folder, "extension": ext, "features": len(names), "model": str(path.relative_to(MODEL_ROOT)), "artifact_present": True})
        except Exception as exc:
            results.append({"id": key, "name": folder, "artifact_present": False, "error": str(exc)})
    return {"languages": results, "semantic_weights_present": (EMBEDDING_PATH / "config.json").exists()}
