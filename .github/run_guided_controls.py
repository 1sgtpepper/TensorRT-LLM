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
"""Compare actual component failures with isolated corrections and reversion."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    package = Path(importlib.metadata.distribution("tensorrt-llm").locate_file("tensorrt_llm"))
    original = package / "_torch/pyexecutor/guided_decoder.py"
    baseline = original.read_bytes()
    assert hashlib.sha256(baseline).hexdigest() == (
        "fb154d90e873c5337c61294f3e35a5c47f8e150dc0439deb43eeafc22d73439c"
    )
    records = []
    with tempfile.TemporaryDirectory(
        prefix="guided-controls-", dir=os.environ["RUNNER_TEMP"]
    ) as tmp:
        temporary = Path(tmp)
        overlay = temporary / "overlay"
        owner = overlay / "tensorrt_llm/_torch/pyexecutor/guided_decoder.py"
        for relative, copied_child in (
            ("", "_torch"),
            ("_torch", "pyexecutor"),
            ("_torch/pyexecutor", "guided_decoder.py"),
        ):
            destination = overlay / "tensorrt_llm" / relative
            destination.mkdir(parents=True, exist_ok=True)
            for source in (package / relative).iterdir():
                if source.name not in (copied_child, "__pycache__"):
                    (destination / source.name).symlink_to(
                        source, target_is_directory=source.is_dir()
                    )
        owner.write_bytes(baseline)
        (overlay / "triton_kernels").symlink_to(
            package.parent / "triton_kernels", target_is_directory=True
        )
        probes = temporary / "probes"
        probes.mkdir()
        files = ["test_guided_recompute.py", "test_guided_tree.py"]
        for name in files:
            shutil.copy2(root / ".github" / name, probes / name)
        environment = dict(os.environ, PYTHONPATH=str(overlay), PYTHONDONTWRITEBYTECODE="1")

        def run(
            label: str, selected: list[str], count: int, expected_failures: int
        ) -> dict[str, str]:
            source_hash = hashlib.sha256(owner.read_bytes()).hexdigest()
            xml_path = evidence / f"{label}.xml"
            command = [
                sys.executable,
                "-c",
                "import hashlib,sys; from pathlib import Path; "
                "import tensorrt_llm.bindings as native; "
                "import tensorrt_llm._torch.pyexecutor.guided_decoder as guide; "
                "assert hashlib.sha256(Path(guide.__file__).read_bytes()).hexdigest()==sys.argv[1]; "
                "assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()=="
                "'da99747464b85dc7b0fc9d4ae76f69fb2d31de0c235657a0374d6168ed874e13'; "
                "print('Imported source:',guide.__file__,sys.argv[1]); "
                "import pytest; raise SystemExit(pytest.main(sys.argv[2:]))",
                source_hash,
                "-s",
                "-q",
                "-rA",
                "--tb=short",
                f"--junitxml={xml_path}",
                *selected,
            ]
            with (evidence / f"{label}.log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=probes,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                )
            record = {"label": label, "exit_code": result.returncode, "source_sha256": source_hash}
            records.append(record)
            (evidence / "control-results.json").write_text(json.dumps(records, indent=2) + "\n")
            assert result.returncode == (1 if expected_failures else 0), record
            assert xml_path.exists(), f"{label} failed before pytest; inspect {label}.log"
            suite = ET.parse(xml_path).getroot().find("testsuite")
            assert int(suite.attrib["tests"]) == count
            assert int(suite.attrib["errors"]) == int(suite.attrib["skipped"]) == 0
            failures = {
                case.attrib["classname"] + "." + case.attrib["name"]: case.find("failure")
                .attrib["message"]
                .splitlines()[0]
                for case in suite.findall("testcase")
                if case.find("failure") is not None
            }
            assert len(failures) == expected_failures, failures
            if expected_failures:
                continuation = [
                    m for n, m in failures.items() if n.startswith("test_guided_recompute.")
                ]
                tree = [m for n, m in failures.items() if n.startswith("test_guided_tree.")]
                assert len(continuation) == 10 and len(tree) == 1
                assert all(
                    "transition=" in m and ("output='aa'" in m or "output='aba'" in m)
                    for m in continuation
                )
                assert "Emitted prefix 'aadx'" in tree[0]
            record.update(tests=count, failures=failures, errors=0, skipped=0)
            (evidence / "control-results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "label": label,
                        "tests": count,
                        "failures": len(failures),
                        "exit_code": result.returncode,
                    }
                ),
                flush=True,
            )
            return failures

        first = run("baseline", files, 18, 11)
        subprocess.run(
            ["git", "apply", str(root / ".github/guided_recompute_control.patch")],
            cwd=overlay,
            check=True,
        )
        shutil.copy2(owner, evidence / "guided_decoder-control.py")
        run("recompute-control", files[:1], 16, 0)
        owner.write_bytes(baseline)
        subprocess.run(
            ["git", "apply", str(root / ".github/guided_tree_control.patch")],
            cwd=probes,
            check=True,
        )
        shutil.copy2(probes / files[1], evidence / "test_guided_tree-control.py")
        run("tree-control", files[1:], 2, 0)
        shutil.copy2(root / ".github" / files[1], probes / files[1])
        restored = run("baseline-reversion", files, 18, 11)
        assert first == restored, "Reversion must restore the same behavioral discrepancies"
    assert original.read_bytes() == baseline, "The installed package must remain unchanged"


if __name__ == "__main__":
    main()
