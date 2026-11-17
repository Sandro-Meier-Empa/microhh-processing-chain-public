"""Tests for the TUV photolysis wrapper in :mod:`case_input`.

Most tests run a fake ``tuv`` executable that mimics the TUV 5.4 interface:
it reads ``INPUTS/microh`` from its working directory and writes
``../microh.txt`` in the TUV output format. ``FAKE_TUV_MODE`` makes it fail
in specific ways. Set ``TUV_PATH`` to a compiled TUV 5.4 root directory to
also run the test against the real TUV.
"""

import logging
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microhh_processing_chain import case_input as ci

LON, LAT = 14.458, 51.837

FAKE_TUV = f"""#!{sys.executable}
import math, os, re, sys
from pathlib import Path

mode = os.environ.get("FAKE_TUV_MODE", "ok")
sys.stdin.read()
print("fake TUV: calculating....")

text = Path("INPUTS/microh").read_text()
values = dict(re.findall(r"(\\w+) =\\s+(\\S+)", text))
tstart, tstop, nt = float(values["tstart"]), float(values["tstop"]), int(values["nt"])

if mode == "exit":
    print("Fortran runtime error: Cannot open file 'DATAJ1/x'", file=sys.stderr)
    sys.exit(2)
if mode == "nooutput":
    sys.exit(0)

lines = ["Photolysis rate coefficients, s-1"]
lines += [f"   {{i}} = reaction {{i}}" for i in range(1, 9)]
lines += ["values at z =      0.500 km", "Columns: time, sza, photo-reactions"]
lines += ["time, hrs.  sza, deg." + "".join(f"{{i:11d}}" for i in range(1, 9))]
times = [tstart + i * (tstop - tstart) / (nt - 1) for i in range(nt)]
if mode == "short":
    times = times[:-1]
for t in times:
    # Rates encode the hour of day, so tests can check the time axis.
    rates = [f"{{(k + 1) * 1e-3 * (1 + t):10.3E}}" for k in range(8)]
    if mode == "nan" and t == times[-1]:
        rates[2] = "       NaN"
    lines.append(f"{{t:10.4f}} {{50 + t:10.3f}} " + " ".join(rates))
lines.append("-" * 60)
Path("../microh.txt").write_text("\\n".join(lines) + "\\n")
"""


@pytest.fixture
def fake_tuv(tmp_path: Path) -> Path:
    """Create a fake TUV root directory with an executable ``tuv``."""
    root = tmp_path / "TUV" / "V5.4"
    (root / "INPUTS").mkdir(parents=True)
    for name in ("DATAE1", "DATAJ1", "DATAS1"):
        (root / name).mkdir()
    (root / "helpin").write_text("")
    executable = root / "tuv"
    executable.write_text(FAKE_TUV)
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return root


def expected_rate(start: datetime, n_times: int, k: int = 0) -> np.ndarray:
    """Rate written by the fake TUV for reaction ``k``, every 15 min from ``start``."""
    # At midnight between two runs the next day's 0 h row is kept, not the
    # previous day's 24 h row (identical in real TUV output).
    hours = (start.hour + start.minute / 60 + 0.25 * np.arange(n_times)) % 24
    return (k + 1) * 1e-3 * (1 + hours)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2021-08-12 08:00", "2021-08-12 18:00", [("2021-08-12", 8.0, 18.0)]),
        (
            "2021-08-12 08:30",
            "2021-08-13 06:15",
            [("2021-08-12", 8.5, 24.0), ("2021-08-13", 0.0, 6.25)],
        ),
        ("2021-08-12 12:00", "2021-08-13 00:00", [("2021-08-12", 12.0, 24.0)]),
    ],
)
def test_tuv_day_windows(start: str, end: str, expected: list) -> None:
    windows = ci._tuv_day_windows(
        datetime.fromisoformat(start), datetime.fromisoformat(end)
    )
    assert [(str(d.date()), t0, t1) for d, t0, t1 in windows] == expected


def test_multi_day_rates(fake_tuv: Path) -> None:
    start, end = datetime(2021, 8, 12, 12), datetime(2021, 8, 13, 6)
    df = ci.calc_tuv_photolysis(fake_tuv, start, end, LON, LAT)

    expected_index = pd.date_range(start, end, freq="15min")
    assert df.index.equals(expected_index)
    assert list(df.columns) == ["sza", *ci.TUV_RATE_COLUMNS]
    assert np.isfinite(df.to_numpy()).all()
    assert df["jo31d"].to_numpy() == pytest.approx(expected_rate(start, len(df), 0))
    assert df["jch3o2h"].to_numpy() == pytest.approx(expected_rate(start, len(df), 7))


def test_start_not_on_full_hour(fake_tuv: Path) -> None:
    start, end = datetime(2021, 8, 12, 8, 30), datetime(2021, 8, 12, 10, 45)
    df = ci.calc_tuv_photolysis(fake_tuv, start, end, LON, LAT)

    assert df.index[0] == pd.Timestamp(start)
    assert df.index[-1] == pd.Timestamp(end)
    assert df["jo31d"].to_numpy() == pytest.approx(expected_rate(start, len(df)))


def test_installation_is_not_modified(fake_tuv: Path) -> None:
    ci.calc_tuv_photolysis(
        fake_tuv, datetime(2021, 8, 12, 8), datetime(2021, 8, 12, 9), LON, LAT
    )
    assert list((fake_tuv / "INPUTS").iterdir()) == []
    assert not (fake_tuv.parent / "microh.txt").exists()


def test_stale_output_is_not_used(
    fake_tuv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A previous run's output next to the installation must never be read.
    (fake_tuv.parent / "microh.txt").write_text("stale")
    monkeypatch.setenv("FAKE_TUV_MODE", "nooutput")
    with pytest.raises(ci.TuvError, match="wrote no output"):
        ci.calc_tuv_photolysis(
            fake_tuv, datetime(2021, 8, 12, 8), datetime(2021, 8, 12, 9), LON, LAT
        )


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("exit", r"exited with code 2(.|\n)*Fortran runtime error"),
        ("nooutput", "wrote no output"),
        ("nan", "missing or non-finite at 1 of 5 times"),
        ("short", "missing or non-finite at 1 of 5 times"),
    ],
)
def test_tuv_failure_raises(
    fake_tuv: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    monkeypatch.setenv("FAKE_TUV_MODE", mode)
    with pytest.raises(ci.TuvError, match=message):
        ci.calc_tuv_photolysis(
            fake_tuv, datetime(2021, 8, 12, 8), datetime(2021, 8, 12, 9), LON, LAT
        )


@pytest.mark.parametrize("mode", ["exit", "nooutput", "nan", "short"])
def test_tuv_failure_without_raise_fills_zero(
    fake_tuv: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: str,
) -> None:
    monkeypatch.setenv("FAKE_TUV_MODE", mode)
    with caplog.at_level(logging.WARNING, logger=ci.logger.name):
        df = ci.calc_tuv_photolysis(
            fake_tuv,
            datetime(2021, 8, 12, 8),
            datetime(2021, 8, 12, 9),
            LON,
            LAT,
            raise_on_error=False,
        )
    assert len(df) == 5
    assert np.isfinite(df.to_numpy()).all()
    assert (df.iloc[-1][list(ci.TUV_RATE_COLUMNS)] == 0).any()
    assert "setting them to zero" in caplog.text


def test_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="TUV executable not found"):
        ci.calc_tuv_photolysis(
            tmp_path, datetime(2021, 8, 12, 8), datetime(2021, 8, 12, 9), LON, LAT
        )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (datetime(2021, 8, 12, 8, 10), datetime(2021, 8, 12, 9), "start_date"),
        (datetime(2021, 8, 12, 8), datetime(2021, 8, 12, 9, 5), "end_date"),
        (datetime(2021, 8, 12, 9), datetime(2021, 8, 12, 8), "must be after"),
        (datetime(2021, 8, 12, 9), datetime(2021, 8, 12, 9), "must be after"),
    ],
)
def test_invalid_period(
    fake_tuv: Path, start: datetime, end: datetime, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ci.calc_tuv_photolysis(fake_tuv, start, end, LON, LAT)


@pytest.mark.parametrize("suppress", [True, False])
def test_suppress_stdout(
    fake_tuv: Path, capfd: pytest.CaptureFixture, suppress: bool
) -> None:
    ci.calc_tuv_photolysis(
        fake_tuv,
        datetime(2021, 8, 12, 8),
        datetime(2021, 8, 12, 9),
        LON,
        LAT,
        suppress_stdout=suppress,
    )
    assert ("fake TUV: calculating" in capfd.readouterr().out) is not suppress


@pytest.mark.parametrize("float_type", [np.float32, np.float64])
def test_calc_photolysis_rates_method(fake_tuv: Path, float_type: type) -> None:
    # Bypass __init__, which needs a full case, domain and grid.
    icbc = ci.InitialBoundaryConditions.__new__(ci.InitialBoundaryConditions)
    icbc._float_type = float_type
    icbc._df_tuv = None
    icbc.tuv_path = fake_tuv
    icbc.start_date = datetime(2021, 8, 12, 8)
    icbc.end_date = datetime(2021, 8, 12, 10)
    icbc.central_lon, icbc.central_lat = LON, LAT

    df = icbc.calc_photolysis_rates()

    assert set(df.dtypes) == {np.dtype(float_type)}
    assert icbc._df_tuv is df
    tdep = icbc.get_tdep_chem()
    assert tdep["time_chem"][0] == 0
    assert tdep["time_chem"][-1] == 7200


@pytest.mark.skipif(
    "TUV_PATH" not in os.environ, reason="set TUV_PATH to run the real TUV"
)
def test_real_tuv() -> None:
    """Run the real TUV for a summer day around Jänschwalde (~1 min)."""
    df = ci.calc_tuv_photolysis(
        os.environ["TUV_PATH"],
        datetime(2021, 8, 12, 0),
        datetime(2021, 8, 13, 0),
        LON,
        LAT,
    )
    assert len(df) == 97
    assert np.isfinite(df.to_numpy()).all()
    # Sun near its highest point at ~11 UTC (13:00 CEST) at 14.5 deg east.
    assert np.argmin(df["sza"].to_numpy()) // 4 == 11  # 15 min steps from 00 UTC
    assert df["sza"].min() == pytest.approx(37.0, abs=0.5)
    assert 5e-3 < df["jno2"].max() < 1.5e-2
    assert df.loc["2021-08-12 00:00", "jno2"] < 1e-6
