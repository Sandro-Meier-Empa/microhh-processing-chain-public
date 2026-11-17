"""Tests for the general utility helpers."""

from pathlib import Path

import microhh_py.microhh_tools as mht
import numpy as np
import pytest

from microhh_processing_chain import utils


def test_small_strength_is_not_rounded_to_zero() -> None:
    strengths = [100 / 3600, 1.0, 2.5e-5]

    result = utils.to_comma_separated_list(strengths, nsignificant=3)

    assert result == "0.0278,1,2.5e-05"


def test_large_value_keeps_significant_digits() -> None:
    assert utils.to_comma_separated_list([1234.567], nsignificant=6) == "1234.57"


def test_default_hides_float64_artifacts_and_keeps_small_values() -> None:
    assert utils.to_comma_separated_list([209.20000000000002]) == "209.2"
    assert utils.to_comma_separated_list(np.array([1.23e-6])) == "1.23e-06"


def test_default_hides_float32_artifacts() -> None:
    values = np.array([0.1, 100 / 3600], dtype=np.float32)

    assert utils.to_comma_separated_list(values) == "0.1,0.0277778"


def test_non_float_items_and_repetition() -> None:
    assert utils.to_comma_separated_list(["co2", "ch4"], n=2) == "co2,ch4,co2,ch4"
    assert utils.to_comma_separated_list([1, np.int64(2)], nsignificant=2) == "1,2"


def test_nsignificant_smaller_than_one_raises() -> None:
    with pytest.raises(ValueError, match="nsignificant"):
        utils.to_comma_separated_list([1.0], nsignificant=0)


def test_strength_survives_ini_round_trip(tmp_path: Path) -> None:
    strengths = np.array([100 / 3600, 2.5e-5, 1500.0])
    ini_file = tmp_path / "test.ini"

    ini_file.write_text("[source]\nstrength=0\n")

    namelist = mht.Read_namelist(ini_file)
    namelist.set_value(
        group="source",
        variable="strength",
        value=utils.to_comma_separated_list(strengths, nsignificant=4),
    )
    namelist.save(ini_file, allow_overwrite=True)

    read_back = np.atleast_1d(mht.Read_namelist(ini_file)["source"]["strength"])

    np.testing.assert_allclose(read_back, strengths, rtol=1e-3)
    assert np.all(read_back > 0)
