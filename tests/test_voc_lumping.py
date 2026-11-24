"""Tests for the lumping of VOCs into the propene species of MicroHH."""

from pathlib import Path

import ls2d
import numpy as np
import pytest
import xarray as xr
from microhhpy.constants import xm_cams

from microhh_processing_chain import case_input as ci
from microhh_processing_chain import initialisation as init
from microhh_processing_chain import misc, spatial


def test_propene_is_the_reference() -> None:
    weights = misc.propene_equivalent_weights(misc.K_OH_CAMS_GLOB)

    assert weights["propene"] == 1.0
    assert weights["ethene"] == pytest.approx(8.52e-12 / 2.44e-11)
    # Slow reacting compounds hardly count, fast reacting ones count more.
    assert weights["ethane"] < 0.02
    assert weights["isoprene"] > 4.0


def test_emission_classes_use_the_rate_constants(tmp_path: Path) -> None:
    files = misc.CamsGlobInventoryPaths(tmp_path, 2021)

    assert files.lumping_vocs == misc.propene_equivalent_weights(misc.K_OH_CAMS_GLOB)
    # A custom lumping is still taken as given.
    custom = misc.CamsGlobInventoryPaths(tmp_path, 2021, lumping_vocs=["propene"])
    assert custom.lumping_vocs == {"propene": 1.0}


def test_background_and_emissions_share_the_rate_constants() -> None:
    """A compound must count the same in the emissions and in the background."""
    same_compound = {
        "c2h4": "ethene",
        "c3h6": "propene",
        "c3h8": "propane",
        "c5h8": "isoprene",
        "c10h16": "monoterpenes",
        "ole": "other-alkenes-alkynes",
    }
    for eac4, glob in same_compound.items():
        assert misc.K_OH_CAMS_EAC4[eac4] == misc.K_OH_CAMS_GLOB[glob]
    # A paraffinic carbon atom reacts about as fast as one of hexane.
    per_carbon = misc.K_OH_CAMS_GLOB["hexanes"] / 6
    assert misc.K_OH_CAMS_EAC4["par"] == pytest.approx(per_carbon, rel=0.05)
    # Every species has a molar mass to convert the CAMS mass mixing ratio.
    assert set(misc.K_OH_CAMS_EAC4) <= set(xm_cams)


def test_background_vocs_are_lumped_by_oh_reactivity(tmp_path: Path) -> None:
    case = init.Case("case", "test")
    case.directory = tmp_path
    domain = spatial.Domain(
        xsize=2000, ysize=2000, itot=2, jtot=2, lon=14.45, lat=51.6, anchor="center"
    )
    vgrid = ls2d.grid.Grid_linear_stretched(kmax=2, dz0=25, alpha=0.01)
    icbc = ci.InitialBoundaryConditions(case, domain, vgrid)

    lumped = {name for group in icbc.lumping_species.values() for name in group}
    species = (set(icbc.chemical_species) - {"co2"}) | lumped
    # 1 ppb of propene and 10 ppb of every other species, as mass mixing
    # ratios like in CAMS.
    ppb = {name: 1.0 if name == "c3h6" else 10.0 for name in species}
    icbc.cams_eac4_les_input = xr.Dataset(
        {
            name: (
                ("time", "z"),
                np.full((1, 2), ppb[name] * 1e-9 * xm_cams[name] / xm_cams["air"]),
            )
            for name in species
        }
    )

    icbc._import_cams_species()
    background = icbc._species_cams
    assert background is not None

    weights = misc.propene_equivalent_weights(misc.K_OH_CAMS_EAC4)
    expected = sum(weights[name] * ppb[name] for name in weights) * 1e-9
    np.testing.assert_allclose(background["c3h6"], expected)
    # Paraffins and alcohols react slowly: together 30 ppb of them count as
    # less than 2.1 ppb of propene.
    slow = sum(weights[name] for name in ("par", "ch3oh", "c2h5oh")) * 10.0
    assert slow < 2.1
    # Species that are not lumped keep their mixing ratio, and the sums of
    # peroxides and peroxy radicals are not weighted.
    np.testing.assert_allclose(background["no2"], 10e-9)
    np.testing.assert_allclose(background["rooh"], 20e-9)
    np.testing.assert_allclose(background["ro2"], 30e-9)
