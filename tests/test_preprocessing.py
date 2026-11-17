"""Tests for the pre-processing helpers."""

import logging
from pathlib import Path
from typing import Optional

import cscs
import microhh_py.microhh_tools as mht
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
