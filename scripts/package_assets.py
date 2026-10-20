"""Export inference assets only; never copy training rows or change source experiments."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.settings import LANGUAGES, ROOT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(source, target):
    source, target = source.resolve(), target.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and destination must be separate directories.")
    if target.exists():
        raise FileExistsError("Destination already exists; choose a new directory to preserve existing assets.")
    # Validate all inputs before creating the destination.
    plans = []
    for language, (folder, experiment, _, _) in LANGUAGES.items():
        base = source / folder / experiment
        models = sorted(base.rglob("*.joblib"))
        tables = sorted(base.rglob("features*.csv"))
        extractor = "feature_extractor_plus.py" if language == "cpp" else "feature_extractor.py"
        files = [source / folder / "config.py", source / folder / extractor]
        if not models or not tables or not all(p.is_file() for p in files):
            raise FileNotFoundError(f"Missing original inference assets for {language}.")
        with tables[0].open(encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f))
        names = [n for n in header if n not in {"label", "file_name", "filename"}]
        if not names or len(set(names)) != len(names): raise ValueError(f"Invalid schema: {language}")
        plans.append((language, folder, experiment, models[0], files, tables[0], names))
    manifest = {"format_version": 1, "purpose": "inference-only assets; no training data rows", "languages": {}}
    for language, folder, experiment, model, files, table, names in plans:
        for original in [*files, model]:
            destination = target / original.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
        schema = target / folder / experiment / "feature_schema.json"
        schema.write_text(json.dumps({"features": names, "label_mapping": {"0": "human", "1": "AI"},
            "source_header": table.relative_to(source).as_posix(),
            "schema_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["languages"][language] = {"model": model.relative_to(source).as_posix(), "model_sha256": sha256(model),
            "feature_count": len(names), "schema": schema.relative_to(target).as_posix()}
    manifest["files"] = {p.relative_to(target).as_posix(): sha256(p) for p in sorted(target.rglob("*")) if p.is_file()}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"languages": len(plans), "files": len(manifest["files"]) + 1,
        "total_bytes": sum(p.stat().st_size for p in target.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "emb-_cbt")
    parser.add_argument("--target", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    package(args.source, args.target)
