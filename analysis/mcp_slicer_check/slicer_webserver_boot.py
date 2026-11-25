"""Start 3D Slicer with its REST API ready for the MCP server (mcp-slicer).

Two ways to use it:

1. As a launcher for interactive work:

       "D:/Program Files/3D Slicer 5.12.3/Slicer.exe" --no-splash --python-script \\
           analysis/mcp_slicer_check/slicer_webserver_boot.py

   The Slicer Web Server REST API comes up on 127.0.0.1:2016 with the Slicer API and its ``exec``
   interface enabled (the GUI module leaves ``exec`` off by default), so ``mcp-slicer`` can drive
   the scene, list nodes and grab screenshots immediately.

2. As the Slicer side of verify_mcp_slicer.py, which sets MCP_PROBE_DIR: the chosen port is written
   to ``ready.json`` and Slicer exits when a file named ``quit`` appears there.

Binding is loopback-only (not 0.0.0.0), so Windows does not raise a firewall prompt. ``exec`` is
remote code execution: keep the server on loopback and stop it (Web Server module > Stop server)
when you are done.
"""

from __future__ import annotations

import json
import os

import qt
import slicer

PROBE_DIR = os.environ.get("MCP_PROBE_DIR")  # only set when driven by verify_mcp_slicer.py
READY_FILE = os.path.join(PROBE_DIR, "ready.json") if PROBE_DIR else None
QUIT_FILE = os.path.join(PROBE_DIR, "quit") if PROBE_DIR else None


def _start_rest_api(port: int = 2016) -> int:
    import WebServer

    logic = WebServer.WebServerLogic(
        port=port,
        enableSlicer=True,
        enableExec=True,
        enableDICOM=False,
        enableStaticPages=False,
    )
    logic.stop()
    port = WebServer.SlicerHTTPServer.findFreePort(port)
    logic.server = WebServer.SlicerHTTPServer(
        requestHandlers=logic.requestHandlers,
        docroot=logic.docroot,
        server_address=("127.0.0.1", port),  # loopback only
        logMessage=logic.logMessage,
        certfile=None,
        keyfile=None,
        enableCORS=False,
    )
    logic.server.start()
    logic.port = port
    logic.serverStarted = True
    # Keep the logic object referenced so the server thread is not collected.
    slicer._mcpProbeWebServer = logic
    return port


def main() -> None:
    port = _start_rest_api(int(os.environ.get("MCP_PROBE_PORT", "2016")))
    print(f"[boot] Slicer {slicer.app.applicationVersion} REST API on 127.0.0.1:{port}", flush=True)

    # A maximised window gives slice/3D grabs a sane geometry; a restored 965x400 window yields
    # 300x70 slice captures, which is not useful for an agent looking at the case.
    main_window = slicer.util.mainWindow()
    if main_window is not None:
        main_window.showMaximized()
    slicer.app.processEvents()

    if not PROBE_DIR:
        return
    os.makedirs(PROBE_DIR, exist_ok=True)
    try:
        os.remove(QUIT_FILE)
    except OSError:
        pass
    with open(READY_FILE, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "port": port,
                "version": slicer.app.applicationVersion,
                "revision": slicer.app.revision,
            },
            handle,
        )

    # Give the probe something to look at, and quit when it asks.
    slicer.mrmlScene.Clear()
    volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "MCPProbeVolume")
    volume.CreateDefaultDisplayNodes()

    quit_timer = qt.QTimer()
    quit_timer.setInterval(400)

    def _check_quit() -> None:
        if os.path.exists(QUIT_FILE):
            quit_timer.stop()
            print("[boot] quit requested, exiting", flush=True)
            qt.QTimer.singleShot(200, slicer.util.exit)

    quit_timer.timeout.connect(_check_quit)
    quit_timer.start()


main()
