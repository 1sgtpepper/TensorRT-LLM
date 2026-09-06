# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare real tree-worker output while changing only per-node grammar masks."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    source = Path(__file__).with_name("guided_tree_worker.py").read_bytes()
    patch = Path(__file__).with_name("guided_tree_worker_control.patch").resolve()
    evidence = Path("evidence").resolve()
    evidence.mkdir(exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix="guided-tree-control-") as directory:
        folder = Path(directory)
        fixture = folder / "guided_tree_worker.py"
        for phase, expected_exit in (("baseline", 1), ("control", 0), ("reversion", 1)):
            fixture.write_bytes(source)
            if phase == "control":
                subprocess.run(
                    ["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
                    cwd=folder,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                assert fixture.read_bytes() == source
            output = evidence / f"tree-{phase}.json"
            result = subprocess.run(
                [sys.executable, str(fixture), "--mode", "tree", "--output", str(output)],
                capture_output=True,
                text=True,
                timeout=180,
            )
            (evidence / f"tree-{phase}.log").write_text(result.stdout + result.stderr)
            assert output.exists(), f"{phase}: no worker record"
            report = json.loads(output.read_text())
            assert report["phase"] == "completed", f"{phase}: setup or generation did not complete"
            assert result.returncode == expected_exit, (phase, result.returncode, report)
            if expected_exit:
                assert "Real worker output violates the finite language" in result.stderr
            results[phase] = {
                "exit": result.returncode,
                "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "report": report,
            }
            print(
                json.dumps({"phase": phase, "exit": result.returncode, "actual": report["actual"]})
            )
    assert results["baseline"] == results["reversion"], "Reversion did not restore the same result"
    baseline = results["baseline"]["report"]["trace"]
    control = results["control"]["report"]["trace"]
    assert len(baseline) == len(control)
    changed = 0
    for before, after in zip(baseline, control, strict=True):
        for key in (
            "contexts",
            "generations",
            "committed_before",
            "tree_valid",
            "topology",
            "drafts",
        ):
            assert before.get(key) == after.get(key), key
        if after.get("corrected_rows"):
            assert before["committed_before"] == [1, 1]
            assert before["requests"][0]["previous_accepted_drafts"] == 0
            assert before["advanced"] == [1]
            assert before["accepted"] != after["accepted"]
            changed += 1
        else:
            assert before == after
    assert changed == 1
    (evidence / "tree-control-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("One real tree differs only in per-node masks; correction and exact reversion confirmed.")


if __name__ == "__main__":
    main()
