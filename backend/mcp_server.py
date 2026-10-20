"""Run: python -m backend.mcp_server [--http]. stdio emits no application logs."""
import asyncio
import logging
import json
import os
import sys
from mcp.server.fastmcp import FastMCP
from .detector import inventory, validate
from .settings import TASK_TIMEOUT

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
mcp = FastMCP("HybridSense", host="127.0.0.1", port=8001)
_capacity = asyncio.Semaphore(1)

@mcp.tool()
def list_detectors() -> dict:
    """List six trusted local detectors and semantic weight availability."""
    return inventory()

@mcp.tool()
async def analyze_code(code: str, language: str, localize: bool = True) -> dict:
    """Predict AI signals using trained models; probabilities are not authorship proof. Language: python/c/cpp/java/javascript/typescript. Never executes source."""
    validate(code, language)
    async with _capacity:
        # A fresh worker also avoids Windows/Conda multiprocessing spawn-pipe
        # deadlocks and bounds a pathological parser's entire process lifetime.
        process = await asyncio.create_subprocess_exec(sys.executable, "-m", "backend.worker",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **({"creationflags": 0x08000000} if os.name == "nt" else {}))
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(json.dumps({"code":code,"language":language,"localize":localize}).encode()), timeout=TASK_TIMEOUT)
            if process.returncode:
                raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1800:])
            answer = json.loads(stdout)
            if "error" in answer: raise RuntimeError(answer["error"])
            return answer
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                if os.name == "nt":
                    killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(process.pid), "/T", "/F", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, creationflags=0x08000000)
                    await killer.wait()
                else: process.kill()
                await process.wait()
            raise

if __name__ == "__main__":
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
