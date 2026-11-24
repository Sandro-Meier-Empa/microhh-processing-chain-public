"""Tests for the pre-processing helpers."""

import logging
from pathlib import Path
from typing import Optional

import cscs
import microhh_py.microhh_tools as mht
import numpy as np
import pytest

import microhh_processing_chain.initialisation as init
from microhh_processing_chain import preprocessing as pre

_LOGGER_NAME = "microhh_processing_chain.preprocessing"


def _make_case(tmp_path: Path, wallclocklimit: Optional[str] = None) -> init.Case:
    """Create a case directory holding a minimal INI file."""
    case_directory = tmp_path / "test.wallclock"
    case_directory.mkdir()

    lines = ["[master]", "npx=1", "npy=1"]
    if wallclocklimit is not None:
        lines.append(f"wallclocklimit={wallclocklimit}")
    lines += ["", "[time]", "starttime=0", "endtime=3600", ""]
    (case_directory / "test.ini").write_text("\n".join(lines))

    return init.read_case(case_directory)


def _make_setup(run_time: str) -> cscs.AlpsSetup:
    """Create an Alps setup with the given Slurm run time."""
    return cscs.AlpsSetup(
        cluster="eiger",
        processor="cpu",
        account="test",
        run_time=run_time,
        partition="normal",
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the warnings logged by the pre-processing module."""
    return [
        record
        for record in caplog.records
        if record.name == _LOGGER_NAME and record.levelno >= logging.WARNING
    ]


def test_missing_wallclock_limit_is_set_from_slurm_run_time(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    case = _make_case(tmp_path)
    alps_setup = _make_setup("03:20:00")

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        pre._ensure_wallclock_limit(case, alps_setup)
        # The written value must compare equal to the Slurm run time when the
        # case is set up again.
        pre._ensure_wallclock_limit(case, alps_setup)

    expected = 12000.0 / 3600.0
    assert case.config["master"]["wallclocklimit"] == expected
    on_disk = mht.Read_namelist(case.ini_file_path)
    assert on_disk["master"]["wallclocklimit"] == expected
    assert on_disk["master"]["npx"] == 1
    assert on_disk["time"]["endtime"] == 3600
    assert not _warnings(caplog)


def test_shorter_slurm_run_time_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    case = _make_case(tmp_path, wallclocklimit="12")
    ini_before = case.ini_file_path.read_text()

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        pre._ensure_wallclock_limit(case, _make_setup("06:00:00"))

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "shorter than 'wallclocklimit'" in warnings[0].getMessage()
    assert case.ini_file_path.read_text() == ini_before


def test_longer_slurm_run_time_keeps_existing_limit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    case = _make_case(tmp_path, wallclocklimit="0.25")
    ini_before = case.ini_file_path.read_text()

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        pre._ensure_wallclock_limit(case, _make_setup("06:00:00"))

    assert not _warnings(caplog)
    assert case.config["master"]["wallclocklimit"] == 0.25
    assert case.ini_file_path.read_text() == ini_before


def test_minimum_run_time_sets_wallclock_limit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    case = _make_case(tmp_path)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        pre._ensure_wallclock_limit(case, _make_setup("02:00:00"))

    assert case.config["master"]["wallclocklimit"] == 2.0
    assert not _warnings(caplog)


@pytest.mark.parametrize("run_time", ["00:10:00", "00:15:00", "01:59:59"])
def test_short_run_time_is_not_set(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, run_time: str
) -> None:
    case = _make_case(tmp_path)
    ini_before = case.ini_file_path.read_text()

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        pre._ensure_wallclock_limit(case, _make_setup(run_time))

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "Not setting 'wallclocklimit'" in warnings[0].getMessage()
    assert case.config.get("master", "wallclocklimit", None) is None
    assert case.ini_file_path.read_text() == ini_before


def test_non_numeric_wallclock_limit_raises(tmp_path: Path) -> None:
    case = _make_case(tmp_path, wallclocklimit="soon")

    with pytest.raises(ValueError, match="wallclocklimit"):
        pre._ensure_wallclock_limit(case, _make_setup("06:00:00"))


def _make_gpu_setup() -> cscs.AlpsSetup:
    """Create a single-GPU Alps setup."""
    return cscs.AlpsSetup(
        cluster="santis",
        processor="gpu",
        account="test",
        run_time="10:00:00",
        partition="normal",
        gpus_per_node=1,
    )


@pytest.mark.parametrize("jobreport_path", [None, "/opt/jobreport"])
def test_single_run_command_changes_into_case_directory(
    tmp_path: Path, jobreport_path: Optional[str]
) -> None:
    case = _make_case(tmp_path)
    case.scratch_path = tmp_path / "case_dir"

    commands = pre._get_single_run_command(
        ["init", "run"], case=case, jobreport_path=jobreport_path
    )

    prefix = f"srun --chdir={case.scratch_path} "
    assert len(commands) == 2
    assert all(command.startswith(prefix) for command in commands)
    assert commands[0] == f"{prefix}{case.scratch_path}/microhh init test"
    if jobreport_path is None:
        assert commands[1] == f"{prefix}{case.scratch_path}/microhh run test"
    else:
        assert commands[1] == (
            f"{prefix}{jobreport_path} -- {case.scratch_path}/microhh run test"
        )


def test_run_4x_script_with_one_case_runs_in_case_directory(tmp_path: Path) -> None:
    # Regression: with a single case, the sbatch script runs from the shared
    # scratch directory, so MicroHH could not find the case INI file.
    case = _make_case(tmp_path)
    case.scratch_path = tmp_path / "case_dir"
    case.scratch_path.mkdir()

    slurm_job = pre.create_run_4x_script(
        [case], _make_gpu_setup(), scratch_path=tmp_path, mode="restart"
    )

    script = Path(slurm_job.path).read_text().splitlines()
    assert f"#SBATCH --chdir={tmp_path}" in script
    srun_lines = [line for line in script if line.startswith("srun")]
    assert srun_lines == [
        f"srun --chdir={case.scratch_path} {case.scratch_path}/microhh run test"
    ]


_ITOT, _JTOT, _KTOT = 4, 3, 2


def _make_restart_case(
    tmp_path: Path, itemsize: int = 4, iotimeprec: Optional[int] = None
) -> init.Case:
    """Create a case with two scalars and a restart field u at 60 s."""
    case_directory = tmp_path / "test.restart"
    case_directory.mkdir()

    lines = [
        "[grid]",
        f"itot={_ITOT}",
        f"jtot={_JTOT}",
        f"ktot={_KTOT}",
        "",
        "[fields]",
        "slist=a,b",
        "",
        "[time]",
        "starttime=60",
        "endtime=120",
    ]
    if iotimeprec is not None:
        lines.append(f"iotimeprec={iotimeprec}")
    (case_directory / "test.ini").write_text("\n".join([*lines, ""]))

    case = init.read_case(case_directory)
    timestr = pre._restart_time_str(case, 60)
    (case_directory / f"u.{timestr}").write_bytes(
        b"\x01" * (_ITOT * _JTOT * _KTOT * itemsize)
    )
    return case


@pytest.mark.parametrize("itemsize", [4, 8])
def test_zero_restart_fields_are_written_for_all_scalars(
    tmp_path: Path, itemsize: int
) -> None:
    case = _make_restart_case(tmp_path, itemsize=itemsize)

    written = pre.write_zero_restart_fields(case, 60)

    names = ["a.0000060", "a_gradbot.0000060", "b.0000060", "b_gradbot.0000060"]
    assert written == [case.directory / name for name in names]
    for path in written:
        count = _ITOT * _JTOT * (1 if "_gradbot" in path.name else _KTOT)
        assert path.stat().st_size == count * itemsize
        data = np.fromfile(path, dtype=np.float32 if itemsize == 4 else np.float64)
        assert np.all(data == 0)


def test_existing_scalar_is_skipped_by_default(tmp_path: Path) -> None:
    case = _make_restart_case(tmp_path)
    # The restart of scalar a is a symbolic link into another case
    other = tmp_path / "a.0000060"
    other.write_bytes(b"\x02" * (_ITOT * _JTOT * _KTOT * 4))
    (case.directory / "a.0000060").symlink_to(other)

    written = pre.write_zero_restart_fields(case, 60)

    assert [path.name for path in written] == ["b.0000060", "b_gradbot.0000060"]
    assert other.read_bytes() == b"\x02" * (_ITOT * _JTOT * _KTOT * 4)


def test_requested_existing_scalar_raises(tmp_path: Path) -> None:
    case = _make_restart_case(tmp_path)
    (case.directory / "a_gradbot.0000060").write_bytes(b"\x02")

    with pytest.raises(FileExistsError, match="a_gradbot"):
        pre.write_zero_restart_fields(case, 60, scalars=["a", "b"])

    # Nothing is written when one target exists
    assert not (case.directory / "b.0000060").exists()
    assert (case.directory / "a_gradbot.0000060").read_bytes() == b"\x02"


def test_unknown_scalar_raises(tmp_path: Path) -> None:
    case = _make_restart_case(tmp_path)

    with pytest.raises(ValueError, match="not in \\[fields\\] slist"):
        pre.write_zero_restart_fields(case, 60, scalars=["c"])


def test_missing_reference_field_raises(tmp_path: Path) -> None:
    case = _make_restart_case(tmp_path)

    with pytest.raises(FileNotFoundError, match=r"u\.0000120"):
        pre.write_zero_restart_fields(case, 120)


def test_reference_field_of_wrong_size_raises(tmp_path: Path) -> None:
    case = _make_restart_case(tmp_path)
    (case.directory / "u.0000060").write_bytes(b"\x01" * (_ITOT * _JTOT * _KTOT * 2))

    with pytest.raises(ValueError, match="neither single nor double precision"):
        pre.write_zero_restart_fields(case, 60)


def test_zero_restart_fields_use_iotimeprec(tmp_path: Path) -> None:
    # iotimeprec = -1: the file names count tenths of a second
    case = _make_restart_case(tmp_path, iotimeprec=-1)

    written = pre.write_zero_restart_fields(case, 60, scalars=["a"])

    assert [path.name for path in written] == ["a.0000600", "a_gradbot.0000600"]
