"""Tests for the construct viewer web UI."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pytest

from construct.frame_utils import frame_to_png_bytes
from construct.viewer import ViewerHandler, view_command


def _make_result_json(name="test_run", passed=True, steps=1, termination="done"):
    return json.dumps({
        "scenario": {
            "name": name,
            "prompt": "test prompt",
            "success_criteria": "do the thing",
            "max_steps": 20,
            "timeout_s": 120.0,
            "tags": ["test"],
        },
        "termination_reason": termination,
        "passed": passed,
        "steps": [
            {
                "step_index": i,
                "action": {"name": "act", "parameters": {"k": "v"}},
                "reasoning": "because",
                "latency_ms": 100.0,
                "cost_usd": 0.001,
            }
            for i in range(steps)
        ],
        "eval_scores": {},
        "total_latency_ms": 100.0 * steps,
        "total_cost_usd": 0.001 * steps,
        "error": None,
    })


def _make_run_dir(
    outputs_dir: Path,
    name: str,
    *,
    passed=True,
    steps=2,
    frames=2,
    nested_frames=False,
    frames_per_step=3,
):
    """Create a mock run directory with result.json and optional frames.

    If *nested_frames* is True, creates the new nested format with
    ``step_XXXX/frame_XXXX.png`` subdirectories.  Otherwise creates
    the legacy flat format with ``step_XXXX.png`` files directly in
    the frames directory.
    """
    run_dir = outputs_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(_make_result_json(name=name, passed=passed, steps=steps))

    if frames > 0:
        frames_dir = run_dir / "frames"
        frames_dir.mkdir()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        png_bytes = frame_to_png_bytes(frame)
        if nested_frames:
            for i in range(frames):
                step_dir = frames_dir / f"step_{i:04d}"
                step_dir.mkdir()
                for j in range(frames_per_step):
                    (step_dir / f"frame_{j:04d}.png").write_bytes(png_bytes)
        else:
            for i in range(frames):
                (frames_dir / f"step_{i:04d}.png").write_bytes(png_bytes)

    return run_dir


@pytest.fixture()
def viewer_server(tmp_path):
    """Start a ViewerHandler server in a background thread, yield (base_url, outputs_dir)."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    # Create a fresh handler class to avoid polluting other tests
    class Handler(ViewerHandler):
        pass

    Handler.outputs_dir = outputs_dir

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}", outputs_dir

    server.shutdown()


def _get(url: str) -> tuple[int, bytes, str]:
    """Fetch a URL, return (status, body, content_type)."""
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class TestRunsListAPI:
    def test_returns_all_runs_sorted_newest_first(self, viewer_server):
        base_url, outputs_dir = viewer_server
        _make_run_dir(outputs_dir, "alpha_20260101T000000Z", passed=True)
        _make_run_dir(outputs_dir, "beta_20260102T000000Z", passed=False)

        status, body, ct = _get(f"{base_url}/api/runs")
        assert status == 200
        assert "application/json" in ct
        runs = json.loads(body)
        assert len(runs) == 2
        # sorted reverse by name → beta before alpha
        assert runs[0]["name"] == "beta_20260102T000000Z"
        assert runs[1]["name"] == "alpha_20260101T000000Z"

    def test_empty_outputs_returns_empty_list(self, viewer_server):
        base_url, _ = viewer_server
        status, body, _ = _get(f"{base_url}/api/runs")
        assert status == 200
        assert json.loads(body) == []

    def test_skips_dirs_without_result_json(self, viewer_server):
        base_url, outputs_dir = viewer_server
        (outputs_dir / "no_result").mkdir()
        _make_run_dir(outputs_dir, "has_result_20260101T000000Z")

        status, body, _ = _get(f"{base_url}/api/runs")
        runs = json.loads(body)
        assert len(runs) == 1
        assert runs[0]["name"] == "has_result_20260101T000000Z"


class TestRunDetailAPI:
    def test_returns_result_with_frame_count_flat(self, viewer_server):
        base_url, outputs_dir = viewer_server
        _make_run_dir(outputs_dir, "detail_run", steps=3, frames=2)

        status, body, ct = _get(f"{base_url}/api/runs/detail_run")
        assert status == 200
        data = json.loads(body)
        assert data["scenario"]["name"] == "detail_run"
        assert len(data["steps"]) == 3
        assert data["frame_count"] == 2
        assert data["frame_format"] == "flat"
        assert data["frames_per_step"] == [1, 1]

    def test_returns_result_with_nested_frames(self, viewer_server):
        base_url, outputs_dir = viewer_server
        _make_run_dir(outputs_dir, "nested_run", steps=2, frames=2, nested_frames=True, frames_per_step=4)

        status, body, _ = _get(f"{base_url}/api/runs/nested_run")
        assert status == 200
        data = json.loads(body)
        assert data["frame_format"] == "nested"
        assert data["frame_count"] == 2
        assert data["frames_per_step"] == [4, 4]

    def test_nonexistent_run_returns_404(self, viewer_server):
        base_url, _ = viewer_server
        status, _, _ = _get(f"{base_url}/api/runs/nonexistent")
        assert status == 404

    def test_run_without_frames_dir(self, viewer_server):
        base_url, outputs_dir = viewer_server
        _make_run_dir(outputs_dir, "no_frames_run", frames=0)

        status, body, _ = _get(f"{base_url}/api/runs/no_frames_run")
        assert status == 200
        data = json.loads(body)
        assert data["frame_count"] == 0
        assert data["frame_format"] == "flat"
        assert data["frames_per_step"] == []


class TestFrameServing:
    def test_serves_correct_png_bytes_flat(self, viewer_server):
        base_url, outputs_dir = viewer_server
        run_dir = _make_run_dir(outputs_dir, "frame_run", frames=1)
        expected = (run_dir / "frames" / "step_0000.png").read_bytes()

        status, body, ct = _get(f"{base_url}/frames/frame_run/step_0000.png")
        assert status == 200
        assert "image/png" in ct
        assert body == expected

    def test_serves_nested_frame(self, viewer_server):
        base_url, outputs_dir = viewer_server
        run_dir = _make_run_dir(outputs_dir, "nested_frame_run", frames=1, nested_frames=True, frames_per_step=2)
        expected = (run_dir / "frames" / "step_0000" / "frame_0000.png").read_bytes()

        status, body, ct = _get(f"{base_url}/frames/nested_frame_run/step_0000/frame_0000.png")
        assert status == 200
        assert "image/png" in ct
        assert body == expected

    def test_nonexistent_frame_returns_404(self, viewer_server):
        base_url, outputs_dir = viewer_server
        _make_run_dir(outputs_dir, "frame_run2", frames=1)

        status, _, _ = _get(f"{base_url}/frames/frame_run2/step_9999.png")
        assert status == 404

    def test_path_traversal_rejected(self, viewer_server):
        base_url, _ = viewer_server
        status, _, _ = _get(f"{base_url}/frames/../../etc/passwd")
        assert status in (403, 404)


class TestHTMLServing:
    def test_root_serves_html(self, viewer_server):
        base_url, _ = viewer_server
        status, body, ct = _get(f"{base_url}/")
        assert status == 200
        assert "text/html" in ct
        assert b"construct viewer" in body


# ---------------------------------------------------------------------------
# view_command tests
# ---------------------------------------------------------------------------


class TestViewCommand:
    def test_missing_outputs_dir_returns_1(self, tmp_path, capsys):
        class Args:
            outputs_dir = str(tmp_path / "nonexistent")
            port = 0
            no_open = True

        code = view_command(Args())
        assert code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()
