import json
import pytest
import numpy as np
from backend.detector import detect, manual_features, load_model, validate, DetectionError

SAMPLES = {
    "python": "def sum_positive(values):\n    return sum(v for v in values if v > 0)\n\ndef legacy_sum(values):\n    s=0\n    for v in values:\n        if v>0:s+=v\n    return s\n",
    "c": "#include <stdio.h>\nint sum_positive(const int *v,int n) { int s=0; for(int i=0;i<n;i++) { if(v[i]>0) s+=v[i]; } return s; }\n",
    "cpp": "#include <vector>\nint sum_positive(const std::vector<int>& values) { int s=0; for(auto v:values) { if(v>0) s+=v; } return s; }\n",
    "java": "public class Example { public static int sumPositive(int[] values) { int s = 0; for(int v : values) { if(v > 0) s += v; } return s; } }\n",
    "javascript": "function sumPositive(values) { var sum = 0; for (var i=0; i<values.length; i++) { if(values[i]>0) sum+=values[i]; } return sum; }\n",
    "typescript": "function sumPositive(values: number[]): number { return values.filter(v => v > 0).reduce((a,b) => a+b, 0); }\n"
}

@pytest.mark.parametrize("language", SAMPLES)
def test_real_model_and_original_features(language):
    result = detect(SAMPLES[language], language)
    assert 0 <= result["ai_probability"] <= 1
    assert result["feature_count"] in {300, 400}
    assert len(result["model_sha256"]) == 64
    assert sum(g["count"] for g in result["groups"]) == result["feature_count"]
    assert all(np.isfinite(f["value"]) for f in result["top_features"])
    assert result["voters"]
    json.dumps(result, allow_nan=False)

def test_java_ablation_config_does_not_drop_trained_features():
    values = manual_features(SAMPLES["java"], "java")
    assert values["micro_missing_space_kw_ratio"] > 0

@pytest.mark.parametrize("code,lang", [("", "python"), ("print(1)\x00", "python"), ("x"*131073,"c"), ("x","rust")], ids=["empty", "binary", "oversize", "unsupported"])
def test_reject_invalid_source(code, lang):
    with pytest.raises(DetectionError): validate(code, lang)

def test_no_execution_of_source(tmp_path):
    target = tmp_path / "executed.txt"
    code = f"from pathlib import Path\nPath({str(target)!r}).write_text('bad')\n"
    result = detect(code, "python", localize=False)
    assert not target.exists()
    assert result["source_sha256"]
