import os
import subprocess
from pathlib import Path

import microhh_processing_chain.cscs as cscs


def _write_sbatch(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(["#!/usr/bin/env bash", *lines, "echo done"]) + "\n")
    return path


def test_slurm_job_output_path_replaces_job_id(tmp_path: Path) -> None:
    job_file = _write_sbatch(
        tmp_path / "run_microhh.sbatch",
        ["#SBATCH --output=run_microhh_%j.out"],
    )

    job = cscs.SlurmJob(job_file, id=12345)

    assert job.output_path == tmp_path / "run_microhh_12345.out"
    assert job.id == 12345


def test_slurm_job_output_path_uses_chdir_for_relative_output(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs"
    scratch_dir = tmp_path / "scratch"
    job_dir.mkdir()
    scratch_dir.mkdir()
    job_file = _write_sbatch(
        job_dir / "post.sbatch",
        [
            "#SBATCH --chdir=../scratch",
            "#SBATCH -o logs/post_%j.out",
        ],
    )

    job = cscs.SlurmJob(job_file, id=5678)

    assert job.output_path == scratch_dir / "logs/post_5678.out"


def test_slurm_job_output_path_without_job_id_selects_newest_match(
    tmp_path: Path,
) -> None:
    job_file = _write_sbatch(
        tmp_path / "run_microhh.sbatch",
        ["#SBATCH --output=run_microhh_%j.out"],
    )
    older = tmp_path / "run_microhh_111.out"
    newer = tmp_path / "run_microhh_222.out"
    older.write_text("older")
    newer.write_text("newer")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    job = cscs.SlurmJob(job_file)

    assert job.output_path == newer


def test_slurm_job_id_is_read_from_output_file(tmp_path: Path) -> None:
    job_file = _write_sbatch(
        tmp_path / "run_microhh.sbatch",
        ["#SBATCH --output=run_microhh_%j.out"],
    )
    (tmp_path / "run_microhh_31415.out").write_text("output")

    job = cscs.SlurmJob(job_file)

    assert job.id == 31415


def test_slurm_job_start_stores_job_id(tmp_path: Path, monkeypatch) -> None:
    job_file = _write_sbatch(
        tmp_path / "run_microhh.sbatch",
        ["#SBATCH --output=run_microhh_%j.out"],
    )
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="Submitted batch job 4567\n",
            stderr="",
        )

    monkeypatch.setattr(cscs.subprocess, "run", fake_run)

    job = cscs.SlurmJob(job_file)
    job_id = job.start(wait=False)

    assert job_id == 4567
    assert job.id == 4567
    assert calls[0][0][0] == ["sbatch", str(job_file)]
    assert calls[0][1]["cwd"] == tmp_path


def test_slurm_job_output_opens_resolved_file(tmp_path: Path, monkeypatch) -> None:
    job_file = _write_sbatch(
        tmp_path / "run_microhh.sbatch",
        ["#SBATCH --output=run_microhh_%j.out"],
    )
    out_file = tmp_path / "run_microhh_7890.out"
    out_file.write_text("output")
    opened = []

    monkeypatch.setattr(cscs.utils, "open_in_vscode", lambda path: opened.append(path))

    job = cscs.SlurmJob(job_file, id=7890)

    assert job.output_path == out_file

    job.view_output()

    assert opened == [out_file]


def test_slurm_job_job_id_alias_sets_id(tmp_path: Path) -> None:
    job_file = _write_sbatch(
        tmp_path / "run_microhh.sbatch",
        ["#SBATCH --output=run_microhh_%j.out"],
    )

    job = cscs.SlurmJob(job_file, id=2468)
    job.id = 1357

    assert job.id == 1357
    assert job.output_path == tmp_path / "run_microhh_1357.out"
