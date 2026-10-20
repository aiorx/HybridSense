import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values

ROOT = Path(__file__).resolve().parents[1]
if os.getenv("HYBRIDSENSE_DETECTOR") == "1":
    for key, value in dotenv_values(ROOT / ".env").items():
        if key.startswith("HYBRIDSENSE_") and value is not None:
            os.environ.setdefault(key, value)
else:
    load_dotenv(ROOT / ".env")

def project_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

MODEL_ROOT = project_path(os.getenv("HYBRIDSENSE_MODEL_ROOT", "models" if (ROOT / "models").is_dir() else "emb-_cbt"))
EMBEDDING_PATH = project_path(os.getenv("HYBRIDSENSE_EMBEDDING_PATH", "runtime/models/codeberta"))
RUNTIME = ROOT / "runtime"
RUNTIME.mkdir(exist_ok=True)
TASK_TIMEOUT = int(os.getenv("HYBRIDSENSE_TASK_TIMEOUT", "240"))
AGENT_TIMEOUT = int(os.getenv("HYBRIDSENSE_AGENT_TIMEOUT", "120"))
MAX_BYTES = 128 * 1024
LANGUAGES = {
    "python": ("Python", "data_all", "PythonBatchExtractorWrapper", ".py"),
    "c": ("C", "data_all1", None, ".c"),
    "cpp": ("C++", "data_all", None, ".cpp"),
    "java": ("Java", "data_all", "JavaBatchExtractorWrapper", ".java"),
    "javascript": ("Javascript", "data_12345", "JSBatchExtractorWrapper", ".js"),
    "typescript": ("Typescript", "data_all", "TSBatchExtractorWrapper", ".ts"),
}
