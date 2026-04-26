import os
import sys

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "training"))

from config_paths import expand_config_paths, expand_path, ft_checkpoint_paths


def test_expand_path_env(monkeypatch):
    monkeypatch.setenv("SCRATCH", "/tmp/scratch_test")
    assert expand_path("$SCRATCH/msa/x.h5") == "/tmp/scratch_test/msa/x.h5"


def test_expand_path_home(monkeypatch):
    monkeypatch.setenv("HOME", "/home/user")
    # os.path.expanduser typically uses HOME env var on Unix
    assert expand_path("~/data/file.txt") == "/home/user/data/file.txt"


def test_expand_path_non_string():
    assert expand_path(None) is None
    assert expand_path(123) == 123
    assert expand_path({"path": "/a/b"}) == {"path": "/a/b"}


def test_expand_path_no_expansion():
    assert expand_path("/absolute/path/to/file") == "/absolute/path/to/file"
    assert expand_path("relative/path/to/file") == "relative/path/to/file"


def test_expand_config_paths_presaved_null():
    cfg = {"training": {"presaved": None}}
    expand_config_paths(cfg)
    assert cfg["training"]["presaved"] is None


def test_expand_config_paths_presaved_empty_string():
    cfg = {"training": {"presaved": "  "}}
    expand_config_paths(cfg)
    assert cfg["training"]["presaved"] is None


def test_expand_config_paths_all_fields(monkeypatch):
    monkeypatch.setenv("SCRATCH", "/scratch")
    monkeypatch.setenv("HOME", "/home/user")

    cfg = {
        "data": {
            "datafile": "$SCRATCH/data.h5",
            "ft_datafile": "~/ft_data.h5",
            "other": "not_expanded",
        },
        "model": {
            "saved_weights": "$SCRATCH/weights.pth",
            "hidden_dim": 512,
        },
        "saving": {
            "model_str": "$SCRATCH/model.pth",
            "log_file": "~/train.log",
        },
        "training": {
            "presaved": "$SCRATCH/presaved",
        },
        "finetuning": {
            "ensemble_path": "$SCRATCH/ensemble",
        },
    }

    expand_config_paths(cfg)

    assert cfg["data"]["datafile"] == "/scratch/data.h5"
    assert cfg["data"]["ft_datafile"] == "/home/user/ft_data.h5"
    assert cfg["data"]["other"] == "not_expanded"
    assert cfg["model"]["saved_weights"] == "/scratch/weights.pth"
    assert cfg["model"]["hidden_dim"] == 512
    assert cfg["saving"]["model_str"] == "/scratch/model.pth"
    assert cfg["saving"]["log_file"] == "/home/user/train.log"
    assert cfg["training"]["presaved"] == "/scratch/presaved"
    assert cfg["finetuning"]["ensemble_path"] == "/scratch/ensemble"


def test_ft_checkpoint_paths_ensemble_suffix():
    cfg = {
        "finetuning": {"ensemble": True},
        "saving": {"model_str": "/out/m.pth", "log_file": "/out/train.log"},
    }
    ms, lg = ft_checkpoint_paths(cfg, 42)
    assert ms == "/out/m_seed42.pth"
    assert lg == "/out/train_seed42.log"


def test_ft_checkpoint_paths_no_ensemble():
    cfg = {
        "finetuning": {"ensemble": False},
        "saving": {"model_str": "/out/m.pth", "log_file": "/out/train.log"},
    }
    assert ft_checkpoint_paths(cfg, 99) == ("/out/m.pth", "/out/train.log")
