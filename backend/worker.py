"""One bounded inference job. Source arrives as JSON data over stdin."""
import json
import sys
from .detector import detect

if __name__ == "__main__":
    try:
        data = json.loads(sys.stdin.buffer.read())
        answer = detect(**data)
    except Exception as exc:
        answer = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.buffer.write(json.dumps(answer, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    sys.stdout.buffer.flush()
