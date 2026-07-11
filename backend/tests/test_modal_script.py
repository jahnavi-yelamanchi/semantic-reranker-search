import importlib.util
import sys
import types
from pathlib import Path


def test_modal_training_script_defines_entrypoint():
    modal_stub = types.SimpleNamespace(
        App=lambda *_args, **_kwargs: types.SimpleNamespace(
            function=lambda *_args, **_kwargs: lambda fn: fn,
            local_entrypoint=lambda *_args, **_kwargs: lambda fn: fn,
        ),
        Image=types.SimpleNamespace(
            debian_slim=lambda **_kwargs: types.SimpleNamespace(
                pip_install=lambda *_args, **_kwargs: types.SimpleNamespace()
            )
        ),
        Volume=types.SimpleNamespace(
            from_name=lambda *_args, **_kwargs: types.SimpleNamespace(commit=lambda: None)
        ),
    )
    sys.modules["modal"] = modal_stub

    script_path = Path(__file__).resolve().parents[2] / "modal" / "train.py"
    spec = importlib.util.spec_from_file_location("modal_train", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    assert hasattr(module, "train_and_export")
    assert hasattr(module, "build_synthetic_pairs")
