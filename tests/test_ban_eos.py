"""CPU-only regression checks for ExLlamaV3 EOS stop construction."""

from pathlib import Path
import importlib.util


MODULE = Path(__file__).resolve().parents[1] / "tabbyAPI/backends/exllamav3/stop_conditions.py"
spec = importlib.util.spec_from_file_location("_test_stop_conditions", MODULE)
stop_conditions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop_conditions)


def test_ban_eos_preserves_user_stop_only():
    user = ["END"]
    result = stop_conditions.build_stop_conditions(user, [151645], [151645, 151643], True)
    assert result == ["END"]
    assert user == ["END"]


def test_normal_generation_keeps_model_and_backend_eos():
    result = stop_conditions.build_stop_conditions(["END"], [151645], [151645, 151643], False)
    assert set(result) == {"END", 151645, 151643}
