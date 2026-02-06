"""construct view — local web UI for browsing scenario run outputs."""

from __future__ import annotations

import json
import sys
import webbrowser
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

_VIEWER_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>construct viewer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif;
         background: #0f1117; color: #e1e4e8; display: flex; height: 100vh; overflow: hidden; }

  /* Sidebar */
  .sidebar { width: 260px; min-width: 260px; background: #161b22; border-right: 1px solid #30363d;
             display: flex; flex-direction: column; }
  .sidebar-header { padding: 16px; font-size: 14px; font-weight: 600; color: #8b949e;
                    text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid #30363d; }
  .run-list { flex: 1; overflow-y: auto; padding: 8px; }
  .run-item { padding: 10px 12px; border-radius: 6px; cursor: pointer; margin-bottom: 4px;
              font-size: 13px; display: flex; justify-content: space-between; align-items: center; }
  .run-item:hover { background: #1c2129; }
  .run-item.active { background: #1f6feb33; border: 1px solid #1f6feb; }
  .run-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: 600; margin-left: 8px; flex-shrink: 0; }
  .badge.pass { background: #23863633; color: #3fb950; }
  .badge.fail { background: #da363633; color: #f85149; }

  /* Main */
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  /* Meta bar */
  .meta-bar { padding: 14px 20px; background: #161b22; border-bottom: 1px solid #30363d;
              display: flex; gap: 24px; flex-wrap: wrap; align-items: center; font-size: 13px; }
  .meta-bar .label { color: #8b949e; margin-right: 4px; }
  .meta-bar .value { color: #e1e4e8; }

  /* Frame display */
  .frame-area { flex: 1; display: flex; align-items: center; justify-content: center;
                padding: 20px; overflow: hidden; background: #0d1117; }
  .frame-area img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; }
  .placeholder { color: #484f58; font-size: 16px; }

  /* Step nav */
  .step-nav { display: flex; align-items: center; justify-content: center; gap: 16px;
              padding: 12px; background: #161b22; border-top: 1px solid #30363d; border-bottom: 1px solid #30363d; }
  .step-nav button { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 14px;
                     border-radius: 6px; cursor: pointer; font-size: 14px; }
  .step-nav button:hover:not(:disabled) { background: #30363d; }
  .step-nav button:disabled { opacity: .4; cursor: default; }
  .step-counter { font-size: 14px; color: #8b949e; min-width: 100px; text-align: center; }

  /* Step detail */
  .step-detail { padding: 16px 20px; background: #161b22; overflow-y: auto; max-height: 200px; font-size: 13px; }
  .step-detail .row { margin-bottom: 8px; display: flex; }
  .step-detail .row .label { color: #8b949e; min-width: 100px; flex-shrink: 0; }
  .step-detail .row .value { color: #c9d1d9; word-break: break-word; }
  .step-detail .reasoning { color: #8b949e; font-style: italic; line-height: 1.5; }

  /* Empty state */
  .empty-state { flex: 1; display: flex; align-items: center; justify-content: center;
                 flex-direction: column; gap: 8px; color: #484f58; }
</style>
</head>
<body>

<div class="sidebar">
  <div class="sidebar-header">Runs</div>
  <div class="run-list" id="runList"></div>
</div>

<div class="main" id="mainArea">
  <div class="empty-state" id="emptyState">
    <div style="font-size:32px">&#x1f916;</div>
    <div>Select a run from the sidebar</div>
  </div>
</div>

<script>
(function() {
  const runList = document.getElementById('runList');
  const mainArea = document.getElementById('mainArea');
  let currentRun = null;
  let currentStep = 0;
  let runData = null;

  async function loadRuns() {
    const resp = await fetch('/api/runs');
    const runs = await resp.json();
    runList.innerHTML = '';
    if (runs.length === 0) {
      runList.innerHTML = '<div style="padding:12px;color:#484f58;font-size:13px">No runs found</div>';
      return;
    }
    runs.forEach(function(r) {
      const el = document.createElement('div');
      el.className = 'run-item';
      el.dataset.name = r.name;
      el.innerHTML = '<span class="run-name">' + esc(r.name) + '</span>'
        + '<span class="badge ' + (r.passed ? 'pass' : 'fail') + '">'
        + (r.passed ? 'PASS' : 'FAIL') + '</span>';
      el.addEventListener('click', function() { selectRun(r.name); });
      runList.appendChild(el);
    });
  }

  async function selectRun(name) {
    currentRun = name;
    currentStep = 0;

    document.querySelectorAll('.run-item').forEach(function(el) {
      el.classList.toggle('active', el.dataset.name === name);
    });

    const resp = await fetch('/api/runs/' + encodeURIComponent(name));
    if (!resp.ok) return;
    runData = await resp.json();

    renderRun();
  }

  function renderRun() {
    if (!runData) return;
    const d = runData;
    const step = d.steps[currentStep] || null;
    const totalSteps = d.steps.length;
    const frameCount = d.frame_count || 0;

    let html = '';

    // Meta bar
    html += '<div class="meta-bar">';
    html += '<div><span class="label">Scenario:</span><span class="value">' + esc(d.scenario.name) + '</span></div>';
    html += '<div><span class="label">Prompt:</span><span class="value">' + esc(d.scenario.prompt) + '</span></div>';
    html += '<div><span class="label">Result:</span><span class="value">'
      + '<span class="badge ' + (d.passed ? 'pass' : 'fail') + '">' + (d.passed ? 'PASS' : 'FAIL') + '</span>'
      + '</span></div>';
    html += '<div><span class="label">Termination:</span><span class="value">' + esc(d.termination_reason) + '</span></div>';
    if (d.total_latency_ms) {
      html += '<div><span class="label">Total latency:</span><span class="value">' + (d.total_latency_ms / 1000).toFixed(1) + 's</span></div>';
    }
    if (d.total_cost_usd) {
      html += '<div><span class="label">Total cost:</span><span class="value">$' + d.total_cost_usd.toFixed(4) + '</span></div>';
    }
    html += '</div>';

    // Frame
    html += '<div class="frame-area">';
    if (currentStep < frameCount) {
      const fname = 'step_' + String(currentStep).padStart(4, '0') + '.png';
      html += '<img src="/frames/' + encodeURIComponent(currentRun) + '/' + fname + '" alt="Frame ' + currentStep + '">';
    } else {
      html += '<div class="placeholder">No frame available for this step</div>';
    }
    html += '</div>';

    // Step nav
    html += '<div class="step-nav">';
    html += '<button id="prevBtn"' + (currentStep <= 0 ? ' disabled' : '') + '>&larr;</button>';
    html += '<span class="step-counter">Step ' + (currentStep + 1) + ' / ' + totalSteps + '</span>';
    html += '<button id="nextBtn"' + (currentStep >= totalSteps - 1 ? ' disabled' : '') + '>&rarr;</button>';
    html += '</div>';

    // Step detail
    html += '<div class="step-detail">';
    if (step) {
      html += '<div class="row"><span class="label">Action:</span><span class="value">' + esc(step.action.name) + '</span></div>';
      if (step.action.parameters && Object.keys(step.action.parameters).length > 0) {
        html += '<div class="row"><span class="label">Parameters:</span><span class="value">' + esc(JSON.stringify(step.action.parameters)) + '</span></div>';
      }
      if (step.reasoning) {
        html += '<div class="row"><span class="label">Reasoning:</span><span class="value reasoning">' + esc(step.reasoning) + '</span></div>';
      }
      if (step.latency_ms) {
        html += '<div class="row"><span class="label">Latency:</span><span class="value">' + Math.round(step.latency_ms) + 'ms</span></div>';
      }
      if (step.cost_usd) {
        html += '<div class="row"><span class="label">Cost:</span><span class="value">$' + step.cost_usd.toFixed(4) + '</span></div>';
      }
    }
    html += '</div>';

    mainArea.innerHTML = html;

    // Bind nav buttons
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    if (prevBtn) prevBtn.addEventListener('click', function() { if (currentStep > 0) { currentStep--; renderRun(); } });
    if (nextBtn) nextBtn.addEventListener('click', function() { if (currentStep < totalSteps - 1) { currentStep++; renderRun(); } });
  }

  function esc(s) {
    if (s == null) return '';
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

  document.addEventListener('keydown', function(e) {
    if (!runData) return;
    if (e.key === 'ArrowLeft' && currentStep > 0) { currentStep--; renderRun(); }
    if (e.key === 'ArrowRight' && currentStep < runData.steps.length - 1) { currentStep++; renderRun(); }
  });

  loadRuns();
})();
</script>
</body>
</html>
"""


class ViewerHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving the viewer UI and run data APIs."""

    outputs_dir: Path = Path("outputs")

    def log_message(self, format, *args):  # noqa: A002
        # Silence default stderr logging
        pass

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]  # strip query params

        if path == "/":
            self._serve_html()
        elif path == "/api/runs":
            self._serve_runs_list()
        elif path.startswith("/api/runs/"):
            name = path[len("/api/runs/"):]
            self._serve_run_detail(name)
        elif path.startswith("/frames/"):
            self._serve_frame(path[len("/frames/"):])
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_html(self):
        data = _VIEWER_HTML.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_json(self, obj, status=HTTPStatus.OK):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_runs_list(self):
        outputs = self.outputs_dir
        if not outputs.is_dir():
            self._serve_json([])
            return

        runs = []
        for d in sorted(outputs.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            result_file = d / "result.json"
            if not result_file.exists():
                continue
            try:
                result = json.loads(result_file.read_text())
                runs.append({
                    "name": d.name,
                    "passed": result.get("passed", False),
                    "steps": len(result.get("steps", [])),
                    "timestamp": d.name.rsplit("_", 1)[-1] if "_" in d.name else "",
                })
            except (json.JSONDecodeError, OSError):
                continue

        self._serve_json(runs)

    def _serve_run_detail(self, name: str):
        run_dir = self.outputs_dir / name
        if not run_dir.is_relative_to(self.outputs_dir) or not run_dir.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        result_file = run_dir / "result.json"
        if not result_file.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            result = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError):
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        frames_dir = run_dir / "frames"
        if frames_dir.is_dir():
            step_dirs = sorted(
                d for d in frames_dir.iterdir() if d.is_dir() and d.name.startswith("step_")
            )
            if step_dirs:
                # Nested format: frames/step_XXXX/frame_XXXX.png
                result["frame_format"] = "nested"
                result["frames_per_step"] = [
                    len(list(sd.glob("*.png"))) for sd in step_dirs
                ]
                result["frame_count"] = len(step_dirs)
            else:
                # Flat (legacy) format: frames/step_XXXX.png
                flat_count = len(list(frames_dir.glob("*.png")))
                result["frame_format"] = "flat"
                result["frames_per_step"] = [1] * flat_count
                result["frame_count"] = flat_count
        else:
            result["frame_format"] = "flat"
            result["frames_per_step"] = []
            result["frame_count"] = 0
        self._serve_json(result)

    def _serve_frame(self, rel_path: str):
        # rel_path is "<run_name>/<filename>" → maps to outputs/<run_name>/frames/<filename>
        parts = rel_path.split("/", 1)
        if len(parts) != 2:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        run_name, filename = parts
        base = self.outputs_dir.resolve()
        target = (self.outputs_dir / run_name / "frames" / filename).resolve()
        if not target.is_relative_to(base):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def view_command(args) -> int:
    """Run the viewer web server (called from CLI dispatch)."""
    outputs_dir = Path(args.outputs_dir).resolve()
    if not outputs_dir.is_dir():
        print(f"Outputs directory not found: {outputs_dir}", file=sys.stderr)
        return 1

    ViewerHandler.outputs_dir = outputs_dir

    server = HTTPServer(("127.0.0.1", args.port), ViewerHandler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Serving construct viewer at {url}")
    print(f"Outputs directory: {outputs_dir}")
    print("Press Ctrl+C to stop")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
        server.server_close()

    return 0
