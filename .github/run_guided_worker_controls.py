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

"""Compare real worker output under baseline, committed-history restoration and reversion."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
        prefix="guided-worker-controls-", dir=os.environ["RUNNER_TEMP"]
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
        (overlay / "triton_kernels").symlink_to(
            package.parent / "triton_kernels", target_is_directory=True
        )
        owner.write_bytes(baseline)
        probe = temporary / "guided_worker.py"
        shutil.copy2(root / ".github/guided_worker.py", probe)
        environment = dict(os.environ, PYTHONPATH=str(overlay), PYTHONDONTWRITEBYTECODE="1")
        for label, expected_exit in (("baseline", 1), ("control", 0), ("reversion", 1)):
            if label == "control":
                subprocess.run(
                    ["git", "apply", str(root / ".github/guided_recompute_control.patch")],
                    cwd=overlay,
                    check=True,
                )
                shutil.copy2(owner, evidence / "worker-guided-decoder-control.py")
            elif label == "reversion":
                owner.write_bytes(baseline)
            source_hash = hashlib.sha256(owner.read_bytes()).hexdigest()
            output = evidence / f"worker-{label}.json"
            with (evidence / f"worker-{label}.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import hashlib,runpy,sys; from pathlib import Path; "
                        "import tensorrt_llm.bindings as native; "
                        "import tensorrt_llm._torch.pyexecutor.guided_decoder as guide; "
                        "assert hashlib.sha256(Path(guide.__file__).read_bytes()).hexdigest()==sys.argv[1]; "
                        "assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()=="
                        "'da99747464b85dc7b0fc9d4ae76f69fb2d31de0c235657a0374d6168ed874e13'; "
                        "print('Imported source:',guide.__file__,sys.argv[1],flush=True); "
                        "sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')",
                        source_hash,
                        str(probe),
                        "--kv-tokens",
                        "128",
                        "--output",
                        str(output),
                    ],
                    cwd=temporary,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
            assert output.exists(), f"{label} failed before worker setup; inspect its log"
            report = json.loads(output.read_text())
            record = {
                "label": label,
                "exit_code": result.returncode,
                "source_sha256": source_hash,
                **report,
            }
            records.append(record)
            (evidence / "worker-control-results.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            assert result.returncode == expected_exit, record
            assert report["phase"] == "completed", record
            assert len(report["pauses"]) == 1 and len(report["pauses"][0]["committed"]) == 64, (
                record
            )
            word = "a" * 31 + "b" * 32 + "c" * 32
            assert report["expected"] == word and len(report["actual"]) == 2
            if label == "control":
                assert report["actual"] == [word, word], record
            else:
                assert report["actual"] == [word, word[:64] + "a" * 31], record
            print(
                json.dumps(
                    {
                        "phase": label,
                        "exit_code": result.returncode,
                        "committed_at_pause": 64,
                        "correct_outputs": sum(value == word for value in report["actual"]),
                    }
                ),
                flush=True,
            )
        assert records[0]["tokens"] == records[2]["tokens"]
        assert records[0]["pauses"] == records[2]["pauses"]
    assert original.read_bytes() == baseline


if __name__ == "__main__":
    main()
