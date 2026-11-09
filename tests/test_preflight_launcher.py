import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "preflight_launcher",
    Path(__file__).resolve().parents[1] / "scripts/run_after_preflight.py",
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_gate_and_runner_share_interpreter_and_runner_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["placeholder"])
    gate = tmp_path / "gate.py"
    runner = tmp_path / "runner.py"
    out = tmp_path / "result.txt"
    gate.write_text("import h5py,sys\nassert h5py.__version__\n")
    runner.write_text(
        "import h5py,sys\nfrom pathlib import Path\nPath(sys.argv[2]).write_text(sys.executable)\n"
    )
    launcher.execute(gate, runner, ["--output", str(out)])
    assert out.read_text() == sys.executable
