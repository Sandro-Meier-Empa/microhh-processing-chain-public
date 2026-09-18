"""Generate MicroHH initial, boundary, surface, and forcing inputs."""

import logging
import os
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import ls2d
import microhhpy  # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd
import xarray as xr

from microhh_processing_chain import initialisation as init
from microhh_processing_chain import spatial

logger = logging.getLogger(__name__)

FloatType = Union[type[np.float32], type[np.float64]]
NUDGEFAC_MIN = 1e-6


def nudge_factor(
    z: Union[float, np.ndarray, list[float], tuple[float, ...]],
    zi: float,
    tau_bl: float,
    tau_ft: float,
    z_ramp: Optional[float] = None,
    ramp_width: Optional[float] = None,
) -> np.ndarray:
    """Calculate a height-dependent MicroHH nudging factor in s-1.

    Args:
        z: Heights at which to evaluate the nudging factor.
        zi: Boundary-layer height.
        tau_bl: Relaxation timescale inside the boundary layer.
        tau_ft: Relaxation timescale in the free troposphere.
        z_ramp: Center of the transition. Defaults to ``1.2 * zi``.
        ramp_width: Half-width of the tanh ramp. Defaults to ``0.2 * zi``.

    Returns:
        Nudging factor at each supplied height.

    Notes:
        Pass ``tau_bl=np.inf`` to leave the boundary layer essentially free.
    """
    z = np.asarray(z, dtype=np.float64)
    z_ramp = 1.2 * zi if z_ramp is None else z_ramp
    ramp_width = 0.2 * zi if ramp_width is None else ramp_width

    # The transition fraction is zero in the boundary layer and one aloft.
    transition = 0.5 * (1.0 + np.tanh((z - z_ramp) / ramp_width))
    factor = (1.0 - transition) / tau_bl + transition / tau_ft

    assert factor.max() >= NUDGEFAC_MIN, "max(nudgefac) below MicroHH's 1e-6 floor"
    return factor


class InitialBoundaryConditions:
    """Prepare MicroHH inputs from ERA5 and CAMS EAC4 data.

    Calculates the fields necessary for generating the input netCDF which is generated
    from the io module. The binary fields for the soil can also be generated and copy
    the lookup tables for the radiation and chemical reactions.

    Args:
        case: Case to Simulate
        domain: Domain of the Case
        vgrid: vertical Grid from LS2D
        float_type: Precision type from numpy: np.float64 or np.float32
    """

    def __init__(
        self,
        case: init.Case,
        domain: spatial.Domain,
        vgrid: ls2d.grid._Grid,
        float_type: FloatType = np.float32,
    ) -> None:
        """Initialize input generation for a case, domain, and vertical grid."""
        self.case = case
        self._name = case.name
        self._working_dir = case.directory
        self.domain = domain
        self.vgrid = vgrid
        self._start_date: Optional[datetime] = None
        self._end_date: Optional[datetime] = None
        self._chemical_species: list[str] = [
            "co",
            "no",
            "no3",
            "no2",
            "o3",
            "hcho",
            "c3h6",
            "co2",
            "hno3",
            "h2o2",
            "hcho",
            "ho2",
            "oh",
            "n2o5",
        ]
        self._float_type = float_type
        self.tau_nudge = 10800  # seconds

        self._era5: Optional[ls2d.Read_era5] = None
        self._era5_les_input: Optional[xr.Dataset] = None
        self._era5_les_input_mean: Optional[xr.Dataset] = None
        self._cams_eac4: Optional[ls2d.Read_cams] = None
        self._cams_eac4_les_input: Optional[xr.Dataset] = None
        self._species_cams: Optional[dict[str, np.ndarray]] = None

        self._lcc_path: Union[Path, str]
        self._tuv_path: Union[Path, str]
        self._microhh_path: Union[Path, str]
        self._gpt_path: Union[Path, str]

        self._lon = self.domain.lon
        self._lat = self.domain.lat

        lonmin, lonmax, latmin, latmax = self.domain.bbox
        self.central_lon = lonmin + (lonmax - lonmin) / 2
        self.central_lat = latmin + (latmax - latmin) / 2
        self._df_tuv: Optional[pd.DataFrame] = None

        # Default values for generating initial and boundary conditions
        # Settings:
        # ------------------
        # Root fraction coefficients (see IFS documentation):
        self.a_r = 10.739
        self.b_r = 2.608

        # TODO: this method prints output that does not represent the central_lat
        # and central_lon values.
        self.rfmip = microhhpy.chem.get_rfmip_species(
            self.central_lat, self.central_lon, exp=0
        )

        self.co2 = 400.0e-6
        self.ch4 = 1650.0e-9
        self.n2o = 306.0e-9
        self.n2 = 0.7808
        self.o2 = 0.2095
        self.co = 2.49477023097387e-08
        self.no2 = 1.95490375341816e-12

        self.lumping_species = {
            "c3h6": [
                "par",
                "c2h4",
                "ole",
                "c5h8",
                "ch3oh",
                "c2h5oh",
                "c3h8",
                "c3h6",
                "c10h16",
            ],
            "rooh": ["rooh", "ch3ooh"],
            "ro2": ["ch3o2", "c2o3", "aco2"],
        }
        return None

    @property
    def chemical_species(self) -> list[str]:
        """Return the chemical species included in the generated inputs."""
        return self._chemical_species

    @chemical_species.setter
    def chemical_species(self, value: list[str]) -> None:
        """Set the chemical species. Overwrites the default.

        Args:
            value: List of str with the chemical species.
        """
        self._chemical_species = value

    @property
    def lon(self) -> np.ndarray:
        """Return the domain longitudes."""
        return self._lon

    @property
    def lat(self) -> np.ndarray:
        """Return the domain latitudes."""
        return self._lat

    @property
    def lcc_path(self) -> Path:
        """Return the path to the land-cover classification data."""
        return Path(self._lcc_path)

    @lcc_path.setter
    def lcc_path(self, value: Union[Path, str]) -> None:
        """Set the land-cover classification data path.

        Args:
            value: Path to the land use data.
        """
        self._lcc_path = Path(value)

    @property
    def tuv_path(self) -> Path:
        """Return the path to the TUV installation."""
        return Path(self._tuv_path)

    @tuv_path.setter
    def tuv_path(self, value: Union[Path, str]) -> None:
        """Set the TUV installation path.

        Args:
            value: Path to TUV data
        """
        self._tuv_path = Path(value)

    @property
    def microhh_path(self) -> Path:
        """Return the path to the MicroHH installation."""
        return Path(self._microhh_path)

    @microhh_path.setter
    def microhh_path(self, value: Union[Path, str]) -> None:
        """Set the MicroHH installation path.

        Args:
            value: Path to microhh source code.
        """
        self._microhh_path = Path(value)

    @property
    def gpt_path(self) -> Path:
        """Return the path containing the RRTMGP coefficient files."""
        return Path(self._gpt_path)

    @gpt_path.setter
    def gpt_path(self, value: Union[Path, str]) -> None:
        """Set the RRTMGP coefficient path.

        Args:
            value: Path
        """
        self._gpt_path = Path(value)

    @property
    def start_date(self) -> datetime:
        """Return the start date for the generated input data."""
        if self._start_date is None:
            raise RuntimeError("Start date has not been set yet.")
        return self._start_date

    @start_date.setter
    def start_date(self, value: datetime) -> None:
        """Set the input-data start date.

        Args:
            value: Start date of the input data.
        """
        self._start_date = value

    @property
    def end_date(self) -> datetime:
        """Return the end date for the generated input data."""
        if self._end_date is None:
            raise RuntimeError("End date has not been set yet.")
        return self._end_date

    @end_date.setter
    def end_date(self, value: datetime) -> None:
        """Set the input-data end date.

        Args:
            value: Enddate for the input data
        """
        self._end_date = value

    @property
    def float_type(self) -> FloatType:
        """Return the floating-point type used for generated data."""
        return self._float_type

    @property
    def era5(self) -> ls2d.Read_era5:
        """Return the configured ERA5 input reader, if available."""
        if self._era5 is None:
            raise RuntimeError("ERA5 input has not been set yet.")
        return self._era5

    @era5.setter
    def era5(self, value: ls2d.Read_era5) -> None:
        """Set the ERA5 input reader and calculates the LES input.

        Args:
            value: era5 data from ls2d.Read_era
        """
        self.era5_les_input = self._get_les_input(value)
        self._era5 = value

    @property
    def cams_eac4(self) -> ls2d.Read_cams:
        """Return the configured CAMS EAC4 input reader, if available."""
        if self._cams_eac4 is None:
            raise RuntimeError("CAMS EAC4 input has not been set yet.")
        return self._cams_eac4

    @cams_eac4.setter
    def cams_eac4(self, value: ls2d.Read_cams) -> None:
        """Set the CAMS EAC4 input reader.

        Args:
            value: cams data form ls2d.Read_cams
        """
        if "go3" in value.ds_ml.variables:
            value.ds_ml = value.ds_ml.rename({"go3": "o3"})
        self.cams_eac4_les_input = self._get_les_input(value)
        self._cams_eac4 = value

    @property
    def era5_les_input(self) -> xr.Dataset:
        """Return ERA5 data interpolated to the LES grid."""
        if self._era5_les_input is None:
            raise RuntimeError("ERA5 data has not been interpolated to the LES grid.")
        return self._era5_les_input

    @era5_les_input.setter
    def era5_les_input(self, value: xr.Dataset) -> None:
        """Set ERA5 data on the LES grid.

        Args:
            value: les input
        """
        # Remove top ERA5 level, to ensure that pressure stays
        # above the minimum reference pressure in RRTMGP
        les_input = value.sel(lay=slice(0, 135), lev=slice(0, 136))
        self._era5_les_input = les_input
        self._era5_les_input_mean = les_input.mean(dim="time")

    @property
    def era5_les_input_mean(self) -> xr.Dataset:
        """Return the time-mean ERA5 data on the LES grid."""
        if self._era5_les_input_mean is None:
            raise RuntimeError("Mean ERA5 LES input has not been calculated.")
        return self._era5_les_input_mean

    @era5_les_input_mean.setter
    def era5_les_input_mean(self, value: xr.Dataset) -> None:
        """Set the time-mean ERA5 data on the LES grid.

        Args:
            value: les input
        """
        self._era5_les_input_mean = value

    @property
    def cams_eac4_les_input(self) -> xr.Dataset:
        """Return CAMS EAC4 data interpolated to the LES grid."""
        if self._cams_eac4_les_input is None:
            raise RuntimeError(
                "CAMS EAC4 data has not been interpolated to the LES grid."
            )
        return self._cams_eac4_les_input

    @cams_eac4_les_input.setter
    def cams_eac4_les_input(self, value: xr.Dataset) -> None:
        """Set CAMS EAC4 data on the LES grid.

        Args:
            value: les input
        """
        self._cams_eac4_les_input = value

    def calc_photolysis_rates(self) -> pd.DataFrame:
        """Calculate photolysis rates using TUV wrapper."""
        name = "microh"  # Must be exactly 6 characters!

        # Default input file. Only start/end date and lat/lon location are updated.
        input_file = microhhpy.utils.get_data_file("microhh_tuv.base")

        tuv_df = microhhpy.chem.calc_tuv_photolysis(
            input_file,
            self.tuv_path,
            name,
            self.start_date,
            self.end_date,
            self.central_lon,
            self.central_lat,
            suppress_stdout=True,
        )

        for name, value in tuv_df.items():
            value = np.asarray(value, dtype=self.float_type)
            if np.any(~np.isfinite(value)):
                logging.warning(
                    "Chemistry timedep %s contains NaNs/Infs; replacing with zero.",
                    name,
                )
            tuv_df[name] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        self._df_tuv = tuv_df
        return tuv_df

    def get_init_profiles(self) -> dict:
        """Build the initial-profile mapping from the configured input data."""
        self._check_les_input_era5()
        self._check_les_input_cams_eac4()
        self._import_cams_species()
        species_cams = self._species_cams
        if species_cams is None:
            raise RuntimeError("CAMS chemical species could not be initialized.")

        nudge_fac = np.ones(self.vgrid.kmax) / self.tau_nudge

        # Mean radiation profiles on LES grid:
        qt_les = self.era5_les_input["qt"].mean(axis=0)
        # Constants
        rd = 287.04
        rv = 461.5
        ep = rd / rv

        # Conversion moisture from mass to volume mixing ratio
        h2o_les = qt_les / (ep - ep * qt_les)

        init_profiles = {
            "z": self.vgrid.z,
            "thl": self.era5_les_input["thl"][0, :],
            "qt": self.era5_les_input["qt"][0, :],
            "u": self.era5_les_input["u"][0, :],
            "v": self.era5_les_input["v"][0, :],
            "nudgefac": nudge_fac,
            "co2_bg": species_cams["co2"][0, :],
            "co2": self.co2,
            "ch4": self.ch4,
            "n2o": self.n2o,
            "n2": self.n2,
            "o2": self.o2,
            "h2o": h2o_les,
            "no3": species_cams["no3"][0, :],
            "co": species_cams["co"][0, :],
            "no": species_cams["no"][0, :],
            "no2": species_cams["no2"][0, :],
            "hno3": species_cams["hno3"][0, :],
            "h2o2": species_cams["h2o2"][0, :],
            "o3": species_cams["o3"][0, :],
            "hcho": species_cams["hcho"][0, :],
            "ho2": species_cams["ho2"][0, :],
            "oh": species_cams["oh"][0, :],
            "n2o5": species_cams["n2o5"][0, :],
            "rooh": species_cams["rooh"][0, :],
            "c3h6": species_cams["c3h6"][0, :],
            "ro2": species_cams["ro2"][0, :],
        }
        return init_profiles

    def get_timedep_surface(self) -> dict:
        """Build the time-dependent surface-input mapping."""
        self._check_les_input_era5()
        timedep_surface = {
            "time_surface": self.era5_les_input["time_sec"],
            "p_sbot": self.era5_les_input["ps"],
            "thl_sbot": self.era5_les_input["wth"],
            "qt_sbot": self.era5_les_input["wq"],
        }
        return timedep_surface

    def get_timedep_ls(self) -> dict:
        """Build the time-dependent large-scale forcing mapping."""
        self._import_cams_species()
        self._check_les_input_era5()
        species_cams = self._species_cams
        if species_cams is None:
            raise RuntimeError("CAMS chemical species could not be initialized.")
        timedep_ls: dict[str, Union[xr.DataArray, np.ndarray]] = {
            "time_ls": self.era5_les_input["time_sec"],
            "u_geo": self.era5_les_input["ug"],
            "v_geo": self.era5_les_input["vg"],
            "w_ls": self.era5_les_input["wls"],
            "thl_ls": self.era5_les_input["dtthl_advec"],
            "qt_ls": self.era5_les_input["dtqt_advec"],
            "u_ls": self.era5_les_input["dtu_advec"],
            "v_ls": self.era5_les_input["dtv_advec"],
            "thl_nudge": self.era5_les_input["thl"],
            "qt_nudge": self.era5_les_input["qt"],
            "u_nudge": self.era5_les_input["u"],
            "v_nudge": self.era5_les_input["v"],
        }

        for name, conc in species_cams.items():
            if name == "co2":
                name = "co2_bg"
                timedep_ls["co2_t_nudge"] = np.zeros_like(conc)
                timedep_ls["co2_t_inflow"] = np.zeros_like(conc)
            timedep_ls[f"{name}_nudge"] = conc
            timedep_ls[f"{name}_inflow"] = conc

        return timedep_ls

    def get_radiation(self) -> dict:
        """Build the radiation-input mapping."""
        self._check_les_input_era5()
        self._check_era5_les_input_mean()
        radiation = {
            "z_lay": self.era5_les_input_mean["z_lay"],
            "z_lev": self.era5_les_input_mean["z_lev"],
            "p_lay": self.era5_les_input_mean["p_lay"],
            "p_lev": self.era5_les_input_mean["p_lev"],
            "t_lay": self.era5_les_input_mean["t_lay"],
            "t_lev": self.era5_les_input_mean["t_lev"],
            "o3": self.era5_les_input_mean["o3_lay"] * 1e-6,
            "h2o": self.era5_les_input_mean["h2o_lay"],
            "ch4": self.ch4,
            "n2o": self.n2o,
            "o2": self.o2,
            "n2": self.n2,
            "co2": self.co2,
            "co": self.co,
            "no2": self.no2,
        }
        return radiation

    def get_tdep_chem(self) -> dict:
        """Build the time-dependent chemistry-input mapping."""
        if self._df_tuv is None:
            self.calc_photolysis_rates()
        tuv_df = self._df_tuv
        if tuv_df is None:
            raise RuntimeError("Photolysis rates could not be calculated.")
        time_chem = np.array(
            (tuv_df.index - tuv_df.index[0]).total_seconds(),
            dtype=self.float_type,
        )
        emi_isop = np.zeros_like(time_chem)
        emi_no = np.zeros_like(time_chem)
        tdep_chem = {
            "time_chem": time_chem,
            "jo31d": tuv_df["jo31d"],
            "jh2o2": tuv_df["jh2o2"],
            "jno2": tuv_df["jno2"],
            "jno3": tuv_df["jno3"],
            "jn2o5": tuv_df["jn2o5"],
            "jch2or": tuv_df["jch2or"],
            "jch2om": tuv_df["jch2om"],
            "jch3o2h": tuv_df["jch3o2h"],
            "emi_isop": emi_isop,
            "emi_no": emi_no,
        }
        return tdep_chem

    def get_soil(self) -> dict:
        """Build the soil-input mapping from ERA5 data."""
        self._check_les_input_era5()
        # NOTE: not used with heterogeneous surface, but still required by MicroHH.
        soil_index = int(
            self.era5_les_input.type_soil - 1
        )  # -1 = Fortran -> C indexing
        soil = {
            "z": self.era5_les_input.zs[::-1],
            "theta_soil": self.era5_les_input.theta_soil[0, ::-1],
            "t_soil": self.era5_les_input.t_soil[0, ::-1],
            "index_soil": np.ones(4) * soil_index,
            "root_frac": self.era5_les_input.root_frac_low_veg[::-1],
        }
        return soil

    def _get_les_input(self, ds: Union[ls2d.Read_era5, ls2d.Read_cams]) -> xr.Dataset:
        """Interpolate input data to the LES grid.

        Args:
            ds: dataset from eather cams eac4 or era4
        """
        return ds.get_les_input(self.vgrid.z)

    def _check_era5_les_input_mean(self) -> None:
        if self._era5_les_input_mean is None:
            self.era5_les_input_mean = self.era5_les_input.mean(dim="time")

    def _check_les_input_era5(self) -> None:
        if self._era5_les_input is None:
            self.era5_les_input = self._get_les_input(self.era5)

    def _check_les_input_cams_eac4(self) -> None:
        if self._cams_eac4_les_input is None:
            self.cams_eac4_les_input = self._get_les_input(self.cams_eac4)

    def _import_cams_species(self) -> None:
        if self._species_cams is not None:
            return

        self._check_les_input_cams_eac4()

        species_cams = {}
        for specie in self.chemical_species:
            if specie == "co2":
                logging.warning(
                    "CO2 BG not available, setting to default value %s.", self.co2
                )
                species_cams["co2"] = np.full_like(
                    self.cams_eac4_les_input["no2"], self.co2
                )
            elif specie not in self.lumping_species:
                species_cams[specie] = (
                    self.cams_eac4_les_input[specie].values
                    * microhhpy.constants.xm_cams["air"]
                    / microhhpy.constants.xm_cams[specie]
                )

        for output_specie, sub_species in self.lumping_species.items():
            species_cams[output_specie] = np.zeros_like(self.cams_eac4_les_input["no2"])
            for sub_specie in sub_species:
                species_cams[output_specie] += (
                    self.cams_eac4_les_input[sub_specie].values
                    * microhhpy.constants.xm_cams["air"]
                    / microhhpy.constants.xm_cams[sub_specie]
                )

        self._species_cams = species_cams
        return

    def create_surface_input(self) -> None:
        """Create land-surface (vegetation) and sea (SST) input."""
        logger.info("Creating spatial (land-) surface input")

        # Default soil depths IFS.
        z_soil = np.array([-0.035, -0.175, -0.64, -1.945])[::-1]

        # Land-surface / vegetation properties from global LCC dataset
        # (100 m resolution).
        microhhpy.land.create_land_surface_input(
            self.domain.lon,
            self.domain.lat,
            z_soil,
            land_use_source="lcc_100m",
            land_use_tiff=self.lcc_path,
            save_binaries=True,
            output_dir=str(self._working_dir),
            save_netcdf=True,
            netcdf_file="lsm_input.nc",
        )

        land_surface_2d_fields = [
            "alb_dif",
            "alb_dir",
            "c_veg",
            "cs_veg",
            "gD",
            "index_veg",
            "lai",
            "lambda_stable",
            "lambda_unstable",
            "rs_soil_min",
            "rs_veg_min",
            "water_mask",
            "z0h",
            "z0m",
        ]

        for field in land_surface_2d_fields:
            ensure_binary_precision(
                f"{self._working_dir}/{field}.0000000",
                (self.domain.jtot, self.domain.itot),
                self.float_type,
            )

        ensure_binary_precision(
            f"{self._working_dir}/root_frac.0000000",
            (len(z_soil), self.domain.jtot, self.domain.itot),
            self.float_type,
        )

        # TODO: Init soil from HiHydroSoil. For now spatially homogeneous.
        soil = microhhpy.land.Land_surface_input(
            self.domain.itot,
            self.domain.jtot,
            4,
            exclude_veg=True,
            debug=True,
            float_type=self.float_type,
        )

        soil.theta_soil[:, :, :] = self.era5_les_input_mean.theta_soil.values[
            ::-1, None, None
        ]
        soil.t_soil[:, :, :] = self.era5_les_input_mean.t_soil.values[::-1, None, None]
        soil.index_soil[:, :, :] = (
            int(self.era5_les_input_mean.type_soil) - 1
        )  # FORTRAN -> C
        soil.to_binaries(path=str(self._working_dir), allow_overwrite=True)

        assert self.era5.sst is not None

        # Create SSTs from ERA5.
        sst_les = microhhpy.real.create_sst_from_regular_latlon(
            self.era5.sst[0],
            self.era5.lons,
            self.era5.lats,
            self.domain.lon,
            self.domain.lat,
            float_type=self.float_type,
        )

        if np.any(np.isnan(sst_les)):
            logger.warning("SSTs contain NaNs! Setting to 290K...")
            sst_les[:, :] = 290.0

        # Water temperatures over land are unknown, and extrapolated SSTs are
        # not very accurate.
        # TODO: get inland water mask from Corine/LCC and let the user define
        # water/lake temperatures.
        sst_les[sst_les < 280] = 280
        sst_les.tofile(f"{self._working_dir}/t_bot_water.0000000")

    def copy_lookup_tables(self) -> None:
        """Copy required land-surface and radiation lookup tables."""
        logger.info("Copying lookup tables")

        rrtmgp_path = f"{self.microhh_path}/rte-rrtmgp-cpp/"
        rrtmgp_data_path = f"{self.microhh_path}/rte-rrtmgp-cpp/rrtmgp-data"

        to_copy = [
            (f"{self.gpt_path}/rrtmgp-gas-lw-g056-cf2.nc", "coefficients_lw.nc"),
            (f"{self.gpt_path}/rrtmgp-gas-sw-g049-cf2.nc", "coefficients_sw.nc"),
            (f"{rrtmgp_data_path}/rrtmgp-clouds-lw.nc", "cloud_coefficients_lw.nc"),
            (f"{rrtmgp_data_path}/rrtmgp-clouds-sw.nc", "cloud_coefficients_sw.nc"),
            (f"{rrtmgp_path}/data/aerosol_optics.nc", "aerosol_optics.nc"),
            (
                f"{self.microhh_path}/misc/van_genuchten_parameters.nc",
                "van_genuchten_parameters.nc",
            ),
        ]

        for f in to_copy:
            target = f"{self._working_dir}/{f[1]}"
            if not Path(target).exists():
                shutil.copy(f[0], target)


def init_icbc(
    case: init.Case,
    domain: spatial.Domain,
    vgrid: ls2d.grid._Grid,
    start_date: datetime,
    end_date: datetime,
    lcc_path: Union[Path, str],
    tuv_path: Union[Path, str],
    microhh_src_path: Union[Path, str],
    gpt_path: Union[Path, str],
    era5: Optional[ls2d.Read_era5] = None,
    cams_eac4: Optional[ls2d.Read_cams] = None,
    float_type: FloatType = np.float32,
) -> InitialBoundaryConditions:
    """Initialize the initial and boundary condition generator.

    Args:
        case: Case to simulate.
        domain: Domain of the case.
        vgrid: Vertical grid from LS2D.
        start_date: Start date of the simulation.
        end_date: End date of the simulation.
        lcc_path: Path to the land-cover classification data.
        tuv_path: Path to the TUV photolysis lookup data.
        microhh_src_path: Path to the MicroHH source directory.
        gpt_path: Path to the RRTMGP coefficient data.
        era5: Optional preloaded ERA5 data.
        cams_eac4: Optional preloaded CAMS EAC4 data.
        float_type: NumPy precision type, either ``np.float32`` or ``np.float64``.

    Returns:
        Configured initial and boundary condition generator.
    """
    icbc = InitialBoundaryConditions(
        case=case,
        domain=domain,
        vgrid=vgrid,
        float_type=float_type,
    )
    icbc.start_date = start_date
    icbc.end_date = end_date

    icbc.lcc_path = lcc_path
    icbc.tuv_path = tuv_path
    icbc.microhh_path = microhh_src_path
    icbc.gpt_path = gpt_path

    if era5 is None:
        print("ERA 5 to be initialized.")
    else:
        icbc.era5 = era5

    if cams_eac4 is None:
        print("CAMS data to be initialized.")
    else:
        icbc.cams_eac4 = cams_eac4

    return icbc


def ensure_binary_precision(
    path: Union[Path, str],
    shape: tuple[int, ...],
    dtype: FloatType,
) -> None:
    """Cast a raw MicroHH binary file to the requested dtype.

    Args:
        path: Path to the binary file.
        shape: Shape of the array stored in the file.
        dtype: Either single- or double-precision floating-point dtype.

    Raises:
        ValueError: If the requested dtype is unsupported or the file size does
            not match the supplied shape in single or double precision.
    """
    target_dtype = np.dtype(dtype)
    supported_dtypes = {np.dtype(np.float32), np.dtype(np.float64)}
    if target_dtype not in supported_dtypes:
        raise ValueError(
            "Binary precision must be either np.float32 or np.float64, "
            f"not {target_dtype}."
        )

    file_path = Path(path)
    count = int(np.prod(shape))
    size = file_path.stat().st_size

    if size == count * target_dtype.itemsize:
        return

    if size == count * np.dtype(np.float32).itemsize:
        source_dtype = np.float32
    elif size == count * np.dtype(np.float64).itemsize:
        source_dtype = np.float64
    else:
        raise ValueError(
            f"Unexpected size for {file_path}: {size} bytes for shape {shape}"
        )

    data = np.fromfile(file_path, dtype=source_dtype)
    converted = data.astype(target_dtype)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            converted.tofile(temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        temp_path.chmod(stat.S_IMODE(file_path.stat().st_mode))
        temp_path.replace(file_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    logger.info(
        "Converted %s from %s to %s",
        file_path,
        np.dtype(source_dtype),
        target_dtype,
    )
