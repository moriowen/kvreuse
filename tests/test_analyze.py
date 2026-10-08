import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(path, scores, budget=1.0, ms=100.0):
    rows = [{"sample_id": f"conv-{i % 3}", "qa_index": i, "category": 1 + i % 5, "score": s,
             "budget": budget, "prefill_ms": ms, "n_tokens": 2000} for i, s in enumerate(scores)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_analyze_table(tmp_path):
    ref, run = tmp_path / "full.jsonl", tmp_path / "reposition.jsonl"
    _write(ref, [1.0, 0.5, 0.0, 1.0, 1.0, 0.8, 0.2, 1.0, 0.0, 0.6])
    _write(run, [0.5, 0.5, 0.0, 1.0, 0.0, 0.4, 0.2, 1.0, 0.0, 0.3], budget=0.0, ms=25.0)
    out = subprocess.run([sys.executable, str(ROOT / "scripts/analyze.py"), str(ref), str(run), "--boot", "200"],
                         capture_output=True, text=True, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("| reposition"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[1] == "10"
    assert cells[2] == "3"
    assert cells[4].startswith("-")  # reuse lost F1 here
    assert cells[16] == "4.00x"  # 100 ms / 25 ms
