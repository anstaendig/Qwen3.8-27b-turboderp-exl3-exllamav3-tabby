"""Exercise the public patch scripts from a clean, offline upstream checkout."""
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "a3fed43efae3b18ec6181618d6a4a66153a9d14e"


def test_clean_apply_idempotence_and_working_source_match(tmp_path):
    shutil.copytree(ROOT / "patches", tmp_path / "patches")
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    tabby = tmp_path / "tabbyAPI"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT / "tabbyAPI"), str(tabby)], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", PIN], cwd=tabby, check=True)
    def apply():
        for name in ["metrics", "ban-eos", "thinking"]:
            subprocess.run([str(tmp_path / f"scripts/apply-{name}")], check=True, capture_output=True)
    def snapshot():
        return {str(p.relative_to(tabby)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tabby.rglob("*") if p.is_file() and ".git" not in p.relative_to(tabby).parts}
    apply()
    before = snapshot()
    apply()
    assert snapshot() == before
    changes = subprocess.check_output(["git", "diff", "--name-only"], cwd=tabby, text=True).splitlines()
    changes += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=tabby, text=True).splitlines()
    for name in changes:
        assert (tabby / name).read_bytes() == (ROOT / "tabbyAPI" / name).read_bytes(), name
    for name in ["thinking", "ban-eos", "metrics"]:
        subprocess.run(["git", "apply", "--reverse", str(tmp_path / f"patches/tabbyapi-{name}.patch")], cwd=tabby, check=True)
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=tabby) == b""
