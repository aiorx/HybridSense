import hashlib
import json
import pytest
from backend import detector
from backend.settings import ROOT, LANGUAGES


def test_published_asset_manifest_and_schemas():
    base = ROOT / "models"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["languages"]) == set(LANGUAGES)
    for relative, expected in manifest["files"].items():
        path = base / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    for data in manifest["languages"].values():
        schema = json.loads((base / data["schema"]).read_text(encoding="utf-8"))
        assert len(schema["features"]) == data["feature_count"]
        assert hashlib.sha256("\n".join(schema["features"]).encode()).hexdigest() == schema["schema_sha256"]
    assert not list(base.rglob("*.csv"))


def asset_directory(tmp_path, monkeypatch):
    folder, tag, _, _ = LANGUAGES["python"]
    base = tmp_path / folder / tag
    base.mkdir(parents=True)
    (base / "model.joblib").write_bytes(b"not-loaded-in-this-schema-test")
    monkeypatch.setattr(detector, "MODEL_ROOT", tmp_path)
    return base


def test_json_schema_avoids_training_csv_dependency(tmp_path, monkeypatch):
    base = asset_directory(tmp_path, monkeypatch)
    (base / "feature_schema.json").write_text(json.dumps({"features": ["b", "a"]}), encoding="utf-8")
    _, names = detector.artifacts("python")
    assert names == ["b", "a"]


def test_original_csv_header_remains_supported(tmp_path, monkeypatch):
    base = asset_directory(tmp_path, monkeypatch)
    (base / "features.csv").write_text("filename,z,a,label\nprivate.py,1,2,0\n", encoding="utf-8")
    _, names = detector.artifacts("python")
    assert names == ["z", "a"]


def test_reject_duplicate_feature_columns(tmp_path, monkeypatch):
    base = asset_directory(tmp_path, monkeypatch)
    (base / "feature_schema.json").write_text(json.dumps({"features": ["a", "a"]}), encoding="utf-8")
    with pytest.raises(detector.DetectionError): detector.artifacts("python")
