"""End-to-end check: mcp-slicer <-> 3D Slicer Web Server REST API.

Boots Slicer with analysis/mcp_slicer_check/slicer_webserver_boot.py, then talks Model Context
Protocol to the installed ``mcp-slicer`` server over stdio (newline-delimited JSON-RPC) and calls
every tool it exposes: execute_python_code, list_nodes, capture_screenshot.

Usage (host Python, no dependencies):

    python analysis/mcp_slicer_check/verify_mcp_slicer.py [--slicer PATH] [--server PATH]

Exit code 0 when every check passes.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_SLICER = "D:/Program Files/3D Slicer 5.12.3/Slicer.exe"
DEFAULT_SERVER = r"C:\Users\{user}\.local\bin\mcp-slicer.exe"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


# --------------------------------------------------------------------------- MCP stdio client
class McpClient:
    """Minimal MCP client: newline-delimited JSON-RPC 2.0 over a child process."""

    def __init__(self, command: list[str], env: dict | None = None) -> None:
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self.lines: queue.Queue = queue.Queue()
        self.stderr: list[str] = []
        self._next_id = 0
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self.server_info: dict = {}
        self.protocol_version = ""

    def _pump_stdout(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if line:
                self.lines.put(line)

    def _pump_stderr(self) -> None:
        for line in self.proc.stderr:  # type: ignore[union-attr]
            self.stderr.append(line.rstrip())
            if len(self.stderr) > 200:
                del self.stderr[:-200]

    def _write(self, payload: dict) -> None:
        self.proc.stdin.write(json.dumps(payload) + "\n")  # type: ignore[union-attr]
        self.proc.stdin.flush()  # type: ignore[union-attr]

    def notify(self, method: str, params: dict | None = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method: str, params: dict | None = None, timeout: float = 90.0) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"mcp-slicer exited ({self.proc.returncode}): " + " | ".join(self.stderr[-5:])
                    )
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") == request_id:
                return message
        raise TimeoutError(
            f"no reply to {method} within {timeout}s; stderr: " + " | ".join(self.stderr[-5:])
        )

    def initialize(self) -> dict:
        reply = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "cbct-mcp-probe", "version": "1.0"},
            },
            # First launch of a freshly installed executable can be slow (antivirus scan).
            timeout=120,
        )
        result = reply.get("result") or {}
        self.server_info = result.get("serverInfo") or {}
        self.protocol_version = result.get("protocolVersion", "")
        self.notify("notifications/initialized")
        return result

    def call(self, tool: str, arguments: dict | None = None, timeout: float = 120.0) -> dict:
        reply = self.request("tools/call", {"name": tool, "arguments": arguments or {}}, timeout=timeout)
        if "error" in reply:
            raise RuntimeError(f"{tool} failed: {reply['error']}")
        return reply.get("result") or {}

    def close(self) -> None:
        try:
            self.proc.stdin.close()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def tool_text(result: dict) -> str:
    for item in result.get("content") or []:
        if item.get("type") == "text":
            return str(item.get("text", ""))
    return ""


def tool_structured(result: dict) -> dict:
    """Tools returning ``dict`` come back as structuredContent (or as a JSON text block)."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        # FastMCP wraps non-dict results; a plain dict is returned as-is.
        return structured.get("result", structured) if "result" in structured else structured
    text = tool_text(result)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {"result": payload}


def tool_images(result: dict) -> list[dict]:
    return [item for item in (result.get("content") or []) if item.get("type") == "image"]


# ------------------------------------------------------------------------------- Slicer boot
def wait_for_ready(ready_file: Path, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_file.exists():
            try:
                return json.loads(ready_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        time.sleep(0.3)
    raise TimeoutError(f"Slicer did not report ready within {timeout}s")


def http_delete(url: str) -> None:
    request = urllib.request.Request(url, method="DELETE")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        opener.open(request, timeout=10).read()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slicer", default=os.environ.get("SLICER_EXE", DEFAULT_SLICER))
    parser.add_argument("--server", default=os.environ.get("MCP_SLICER_EXE", DEFAULT_SERVER.format(user=os.environ.get("USERNAME", ""))))
    parser.add_argument("--port", type=int, default=2016)
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()

    server_exe = Path(args.server)
    if not server_exe.exists():
        found = shutil.which("mcp-slicer")
        if not found:
            print(f"mcp-slicer not found ({server_exe}); install with: uv tool install mcp-slicer")
            return 2
        server_exe = Path(found)
    if not Path(args.slicer).exists():
        print(f"Slicer not found at {args.slicer}")
        return 2

    probe_dir = Path(tempfile.mkdtemp(prefix="mcp_slicer_probe_"))
    ready_file = probe_dir / "ready.json"
    env = dict(os.environ)
    env["MCP_PROBE_DIR"] = str(probe_dir)
    env["MCP_PROBE_PORT"] = str(args.port)

    module_dir = REPO / "CBCTPlanner"
    slicer_args = [args.slicer, "--no-splash"]
    if (module_dir / "CBCTPlanner.py").exists():
        slicer_args += ["--additional-module-paths", str(module_dir)]
    slicer_args += ["--python-script", str(HERE / "slicer_webserver_boot.py")]

    print(f"booting {args.slicer} ...", flush=True)
    slicer_proc = subprocess.Popen(
        slicer_args,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    client: McpClient | None = None
    try:
        info = wait_for_ready(ready_file, timeout=240)
    except TimeoutError as exc:
        print(f"FAIL: {exc}")
        slicer_proc.terminate()
        return 2

    port = int(info["port"])
    check(
        "Slicer REST API started on loopback",
        0 < port < 65536,
        f"Slicer {info.get('version')} rev {str(info.get('revision'))[:10]} on 127.0.0.1:{port}",
    )

    try:
        client = McpClient([str(server_exe)])
        handshake = client.initialize()
        check(
            "MCP handshake over stdio",
            bool(handshake.get("protocolVersion")) and bool(client.server_info.get("name")),
            f"server {client.server_info.get('name')} {client.server_info.get('version')} "
            f"(protocol {client.protocol_version})",
        )

        listed = client.request("tools/list", timeout=30).get("result") or {}
        tools = {tool["name"]: tool for tool in listed.get("tools", [])}
        check(
            "tools/list exposes the three Slicer tools",
            {"list_nodes", "execute_python_code", "capture_screenshot"} <= set(tools),
            ", ".join(sorted(tools)),
        )

        # --- execute_python_code: does the hardcoded endpoint + raw-body exec work on this Slicer?
        result = client.call(
            "execute_python_code",
            {
                "code": "import sys\n"
                "__execResult = {'app': slicer.app.applicationVersion, "
                "'py': sys.version.split()[0], 'nodes': slicer.mrmlScene.GetNumberOfNodes(), "
                "'main': [slicer.util.mainWindow().width, slicer.util.mainWindow().height]}"
            },
        )
        payload = tool_structured(result)
        check(
            "execute_python_code runs inside Slicer",
            payload.get("success") is True
            and isinstance(payload.get("message"), dict)
            and payload["message"].get("app"),
            f"{payload.get('message')}",
        )

        # --- error propagation
        result = client.call("execute_python_code", {"code": "__execResult = 1 / 0"})
        payload = tool_structured(result)
        message = str(payload.get("message", ""))
        check(
            "python errors come back as failure, not a crash",
            payload.get("success") is False and ("ZeroDivision" in message or "division" in message),
            message.splitlines()[0][:160] if message else "(empty)",
        )

        # --- scene mutation + list_nodes
        client.call(
            "execute_python_code",
            {
                "code": (
                    "n = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'MCPProbeMarkups')\n"
                    "n.AddControlPoint(slicer.util.vtkVector3d(0, 0, 0), 'A')\n"
                    "n.AddControlPoint(slicer.util.vtkVector3d(0, 0, 10), 'B')\n"
                    "__execResult = n.GetID()"
                )
            },
        )
        names = tool_structured(client.call("list_nodes", {"filter_type": "names"}))
        ids = tool_structured(
            client.call("list_nodes", {"filter_type": "ids", "class_name": "vtkMRMLMarkupsFiducialNode"})
        )
        props = tool_structured(
            client.call("list_nodes", {"filter_type": "properties", "name": "MCPProbeMarkups"})
        )
        node_names = names.get("nodes") if isinstance(names.get("nodes"), list) else []
        node_ids = ids.get("nodes") if isinstance(ids.get("nodes"), list) else []
        prop_nodes = props.get("nodes") if isinstance(props.get("nodes"), dict) else {}
        prop_names = [
            str(v.get("Name") or v.get("name") or "") for v in prop_nodes.values() if isinstance(v, dict)
        ]
        check(
            "list_nodes names/ids/properties reflect the live scene",
            "MCPProbeMarkups" in node_names
            and len(node_ids) >= 1
            and "MCPProbeMarkups" in prop_names,
            f"{len(node_names)} names, {len(node_ids)} fiducial ids, "
            f"{len(prop_nodes)} property entries ({', '.join(prop_names)[:60]})",
        )

        # --- can an agent drive THIS project's module through the MCP?
        module_reply = tool_structured(
            client.call(
                "execute_python_code",
                {
                    "code": "import slicer\n"
                    "names = list(slicer.util.moduleNames())\n"
                    "__execResult = {'cbctplanner': 'CBCTPlanner' in names, 'count': len(names), "
                    "'near': [n for n in names if 'cbct' in n.lower()]}"
                },
            )
        ).get("message")
        module_info = module_reply if isinstance(module_reply, dict) else {}
        check(
            "CBCTPlanner is visible to the agent as a Slicer module",
            module_info.get("cbctplanner") is True,
            f"{module_info.get('count')} modules registered, matches {module_info.get('near')}",
        )

        # --- view sizes: a slice grab of a not-yet-laid-out view is the known failure mode
        size_code = (
            "import slicer\n"
            "mw = slicer.util.mainWindow()\n"
            "lm = slicer.app.layoutManager()\n"
            "w = lm.sliceWidget('Red')\n"
            "__execResult = {'main': [mw.width, mw.height], 'visible': bool(mw.visible), "
            "'layout': int(lm.layout), 'slice': [w.width, w.height] if w else None, "
            "'sliceView': [w.sliceView().width, w.sliceView().height] if w else None}"
        )
        before = tool_structured(client.call("execute_python_code", {"code": size_code})).get("message")
        client.call(
            "execute_python_code",
            {
                "code": "import slicer\n"
                "slicer.util.mainWindow().showMaximized()\n"
                "slicer.app.processEvents()\n"
                "__execResult = 1"
            },
        )
        after = tool_structured(client.call("execute_python_code", {"code": size_code})).get("message")
        after = after if isinstance(after, dict) else {}
        slice_size = after.get("slice") or [0, 0]
        check(
            "main window and slice views are laid out",
            bool(after.get("visible")) and min(slice_size) >= 40,
            f"window {after.get('main')}, red slice {slice_size}, slice view {after.get('sliceView')}",
        )

        # --- screenshots
        shot_dims: dict[str, tuple[int, int]] = {}
        for view_type, kwargs in (
            ("application", {}),
            ("slice", {"view_name": "red"}),
            ("3d", {"camera_axis": "A"}),
        ):
            result = client.call("capture_screenshot", {"view_type": view_type, **kwargs}, timeout=120)
            images = tool_images(result)
            ok = bool(images)
            size = ""
            dimensions = ""
            if ok:
                try:
                    raw = base64.b64decode(images[0]["data"])
                    width = height = 0
                    index = 2
                    while index < len(raw) - 9:
                        if raw[index] != 0xFF:
                            index += 1
                            continue
                        marker = raw[index + 1]
                        length = int.from_bytes(raw[index + 2 : index + 4], "big")
                        if marker in (0xC0, 0xC1, 0xC2):
                            height = int.from_bytes(raw[index + 5 : index + 7], "big")
                            width = int.from_bytes(raw[index + 7 : index + 9], "big")
                            break
                        index += 2 + length
                    size = f"{len(raw) // 1024} KiB {images[0].get('mimeType', '')}"
                    dimensions = f"{width}x{height}"
                    shot_dims[view_type] = (width, height)
                    ok = raw[:2] == b"\xff\xd8" and min(width, height) >= 60
                    # The slice grab must be the red slice widget, not some other view.
                    if view_type == "slice":
                        ok = ok and abs(width - int(slice_size[0])) <= 4
                except Exception as exc:  # pragma: no cover - diagnostics only
                    ok = False
                    dimensions = f"decode error: {exc}"
            check(
                f"capture_screenshot({view_type}{', ' + ','.join(f'{k}={v}' for k, v in kwargs.items()) if kwargs else ''}) returns a usable JPEG",
                ok,
                f"{dimensions} {size} | {tool_text(result)[:80]}",
            )

        if args.keep_open:
            print("leaving Slicer running; press Enter to stop", flush=True)
            input()
    finally:
        if client is not None:
            client.close()
        http_delete(f"http://127.0.0.1:{port}/slicer/system")
        (probe_dir / "quit").write_text("1", encoding="utf-8")
        try:
            slicer_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            slicer_proc.terminate()
        shutil.rmtree(probe_dir, ignore_errors=True)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n=== SUMMARY ===")
    print(json.dumps({"passed": len(RESULTS) - len(failed), "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
