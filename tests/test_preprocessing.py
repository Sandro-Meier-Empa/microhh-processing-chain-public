from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import microhh_processing_chain.preprocessing as pre


def test_create_run_4x_script_uses_requested_mode(tmp_path: Path) -> None:
    cases = [
        SimpleNamespace(name="case", scratch_path=tmp_path / f"case_{index}")
        for index in range(4)
    ]
    recorded: dict[str, Any] = {}

    class FakeAlpsSetup:
        def create_sbatch_script(
            self,
            commands: list[str],
            job_name: str,
            file_path: Path,
            scratch_path: Path,
        ) -> Path:
            recorded.update(
                {
                    "commands": commands,
                    "job_name": job_name,
                    "file_path": file_path,
                    "scratch_path": scratch_path,
                }
            )
            file_path.write_text("\n".join(commands))
            return file_path

    job_file = pre.create_run_4x_script(
        cases=cases,
        alps_setup=FakeAlpsSetup(),
        scratch_path=tmp_path,
        mode=pre.RunMode.RERUN,
    )

    commands = "\n".join(recorded["commands"])

    assert job_file == tmp_path / "rerun_microhh_4x.sbatch"
    assert recorded["job_name"] == "rerun_microhh_4x"
    assert "# Launch 'run' phase" in commands
    assert "# Launch 'init' phase" not in commands


def test_start_run_4x_simulation_submits_script_and_sets_case_job_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [SimpleNamespace(job_id=None) for _ in range(4)]
    alps_setup = SimpleNamespace(ssh_connect="ela", partition_key="normal")
    job_file = tmp_path / "run_microhh_4x.sbatch"
    calls: list[dict[str, Any]] = []

    def fake_create_run_4x_script(**kwargs: object) -> Path:
        calls.append(kwargs)
        return job_file

    class FakeSlurmJob:
        def __init__(self, path: Path, ssh_connect: str) -> None:
            self.path = path
            self.ssh_connect = ssh_connect

        def start(self, wait: bool) -> int:
            calls.append(
                {
                    "path": self.path,
                    "ssh_connect": self.ssh_connect,
                    "wait": wait,
                }
            )
            return 1234

    monkeypatch.setattr(pre, "create_run_4x_script", fake_create_run_4x_script)
    monkeypatch.setattr(pre.cscs, "SlurmJob", FakeSlurmJob)

    job_id = pre.start_run_4x_simulation(
        cases=cases,
        alps_setup=alps_setup,
        scratch_path=tmp_path,
        mode=pre.RunMode.RERUN,
        overwrite=True,
        wait=True,
        jobreport_path="/path/to/jobreport",
    )

    assert job_id == 1234
    assert [case.job_id for case in cases] == [1234, 1234, 1234, 1234]
    assert calls[0] == {
        "cases": cases,
        "alps_setup": alps_setup,
        "scratch_path": str(tmp_path),
        "mode": pre.RunMode.RERUN,
        "overwrite": True,
        "jobreport_path": "/path/to/jobreport",
    }
    assert calls[1] == {
        "path": job_file,
        "ssh_connect": "ela",
        "wait": True,
    }


def test_start_run_4x_simulation_skips_debug_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [SimpleNamespace(job_id=9999) for _ in range(4)]
    alps_setup = SimpleNamespace(partition_key="debug")

    def fake_jobs_in_queue() -> pd.DataFrame:
        return pd.DataFrame({"PARTITION": ["debug"]})

    def fail_create_run_4x_script(**kwargs: object) -> Path:
        pytest.fail("script should not be created")

    monkeypatch.setattr(pre.cscs, "jobs_in_queue", fake_jobs_in_queue)
    monkeypatch.setattr(pre, "create_run_4x_script", fail_create_run_4x_script)

    job_id = pre.start_run_4x_simulation(
        cases=cases,
        alps_setup=alps_setup,
        scratch_path=tmp_path,
    )

    assert job_id is None
    assert [case.job_id for case in cases] == [None, None, None, None]
