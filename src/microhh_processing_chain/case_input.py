"""Generate MicroHH initial, boundary, surface, and forcing inputs."""

# The public types for this migrated legacy module are not yet established,
# and argument descriptions are intentionally left blank for later completion.
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D417

import logging
import shutil
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional, Union

import ls2d
import numpy as np

import microhhpy  # pyright: ignore[reportMissingImports]
from microhh_processing_chain import initialisation as init
from microhh_processing_chain import spatial

logger = logging.getLogger(__name__)


class InitialBoundaryConditions:
    """Prepare MicroHH inputs from ERA5 and CAMS EAC4 data.

    TODO: Document the required state and lifecycle of the external ERA5, CAMS,
    TUV, and MicroHH inputs.

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
        vgrid: ls2d.grid,
        float_type=np.float32,
    ):
        """Initialize input generation for a case, domain, and vertical grid."""
        self.case = case
        self._name = case.name
        self._working_dir = case.directory
        self.domain = domain
        self.vgrid = vgrid
        self._start_date: Optional[datetime] = None
        self._end_date: Optional[datetime] = None
        self._chemical_species: Union[str, list[str]] = [
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
        self._era5_les_input = None
        self._era5_les_input_mean = None
        self._cams_eac4: Optional[ls2d.Read_cams] = None
        self._cams_eac4_les_input = None
        self._species_cams = None

        self._lcc_path: Union[Path, str]
        self._tuv_path: Union[Path, str]
        self._microhh_path: Union[Path, str]
        self._gpt_path: Union[Path, str]

        self._lon = self.domain.lon
        self._lat = self.domain.lat

        lonmin, lonmax, latmin, latmax = self.domain.bbox
        self.central_lon = lonmin + (lonmax - lonmin) / 2
        self.central_lat = latmin + (latmax - latmin) / 2
        self._mhhpy_domain = None
        self._df_tuv = None

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
    def chemical_species(self):
        """Return the chemical species included in the generated inputs."""
        return self._chemical_species

    @chemical_species.setter
    def chemical_species(self, value: list[str]):
        """Set the chemical species. Overwrites the default.

        Args:
            value: List of str with the chemical species.
        """
        self._chemical_species = value

    @property
    def lon(self) -> float:
        """Return the domain longitudes."""
        return self._lon

    @property
    def lat(self) -> float:
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
    def start_date(self):
        """Return the start date for the generated input data."""
        return self._start_date

    @start_date.setter
    def start_date(self, value):
        """Set the input-data start date.

        Args:
            value: Start date of the input data.
        """
        self._start_date = value

    @property
    def end_date(self):
        """Return the end date for the generated input data."""
        return self._end_date

    @end_date.setter
    def end_date(self, value: datetime):
        """Set the input-data end date.

        Args:
            value: Enddate for the input data
        """
        self._end_date = value

    @property
    def float_type(self):
        """Return the floating-point type used for generated data."""
        return self._float_type

    @property
    def era5(self):
        """Return the configured ERA5 input reader, if available."""
        if self._era5 is None:
            print("WARNING: ERA5 needs to be defined.")
        return self._era5

    @era5.setter
    def era5(self, value: ls2d.Read_era5):
        """Set the ERA5 input reader and calculates the LES input.

        Args:
            value: era5 data from ls2d.Read_era
        """
        self.era5_les_input = self._get_les_input(value)
        self._era5 = value

    @property
    def cams_eac4(self):
        """Return the configured CAMS EAC4 input reader, if available."""
        if self._cams_eac4 is None:
            print("WARNING: CAMS EAC4 data needs to be defined")
        return self._cams_eac4

    @cams_eac4.setter
    def cams_eac4(self, value):
        """Set the CAMS EAC4 input reader.

        Args:
            value: cams data form ls2d.Read_cams
        """
        if "go3" in value.ds_ml.variables:
            value.ds_ml = value.ds_ml.rename({"go3": "o3"})
        self.cams_eac4_les_input = self._get_les_input(value)
        self._cams_eac4 = value

    @property
    def era5_les_input(self):
        """Return ERA5 data interpolated to the LES grid."""
        return self._era5_les_input

    @era5_les_input.setter
    def era5_les_input(self, value):
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
    def era5_les_input_mean(self):
        """Return the time-mean ERA5 data on the LES grid."""
        return self._era5_les_input_mean

    @era5_les_input_mean.setter
    def era5_les_input_mean(self, value):
        """Set the time-mean ERA5 data on the LES grid.

        Args:
            value: les input
        """
        self._era5_les_input_mean = value

    @property
    def cams_eac4_les_input(self):
        """Return CAMS EAC4 data interpolated to the LES grid."""
        return self._cams_eac4_les_input

    @cams_eac4_les_input.setter
    def cams_eac4_les_input(self, value):
        """Set CAMS EAC4 data on the LES grid.

        Args:
            value: les input
        """
        self._cams_eac4_les_input = value

    @wraps(microhhpy.chem.calc_tuv_photolysis)
    def calc_photolysis_rates(self):
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
        self._df_tuv = tuv_df
        return tuv_df

    def get_init_profiles(self):
        """Build the initial-profile mapping from the configured input data."""
        self._check_les_input_era5()
        self._check_les_input_cams_eac4()
        self._import_cams_species()

        nudge_fac = np.ones(self.vgrid.kmax) / self.tau_nudge

        # Mean radiation profiles on LES grid:
        qt_les = self.era5_les_input["qt"].mean(axis=0)
        o3_les = self.era5_les_input["o3"].mean(axis=0)

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
            "co2_bg": self._species_cams["co2"][0, :],
            "co2": self.co2,
            "ch4": self.ch4,
            "n2o": self.n2o,
            "n2": self.n2,
            "o2": self.o2,
            "o3": o3_les * 1e-6,
            "h2o": h2o_les,
            "no3": self._species_cams["no3"][0, :],
            "co": self._species_cams["co"][0, :],
            "no": self._species_cams["no"][0, :],
            "no2": self._species_cams["no2"][0, :],
            "hno3": self._species_cams["hno3"][0, :],
            "h2o2": self._species_cams["h2o2"][0, :],
            "o3": self._species_cams["o3"][0, :],  # noqa: F601
            "hcho": self._species_cams["hcho"][0, :],
            "ho2": self._species_cams["ho2"][0, :],
            "oh": self._species_cams["oh"][0, :],
            "n2o5": self._species_cams["n2o5"][0, :],
            "rooh": self._species_cams["rooh"][0, :],
            "c3h6": self._species_cams["c3h6"][0, :],
            "ro2": self._species_cams["ro2"][0, :],
        }
        return init_profiles

    def get_timedep_surface(self):
        """Build the time-dependent surface-input mapping."""
        self._check_les_input_era5()
        timedep_surface = {
            "time_surface": self.era5_les_input["time_sec"],
            "p_sbot": self.era5_les_input["ps"],
            "thl_sbot": self.era5_les_input["wth"],
            "qt_sbot": self.era5_les_input["wq"],
        }
        return timedep_surface

    def get_timedep_ls(self):
        """Build the time-dependent large-scale forcing mapping."""
        self._import_cams_species()
        self._check_les_input_era5()
        timedep_ls = {
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

        for name, conc in self._species_cams.items():
            if name == "co2":
                name = "co2_bg"
                timedep_ls["co2_t_nudge"] = np.zeros_like(conc)
                timedep_ls["co2_t_inflow"] = np.zeros_like(conc)
            timedep_ls[f"{name}_nudge"] = conc
            timedep_ls[f"{name}_inflow"] = conc

        return timedep_ls

    def get_radiation(self):
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

    def get_tdep_chem(self):
        """Build the time-dependent chemistry-input mapping."""
        if self._df_tuv is None:
            self.calc_photolysis_rates()
        time_chem = np.array(
            (self._df_tuv.index - self._df_tuv.index[0]).total_seconds(),
            dtype=self.float_type,
        )
        emi_isop = np.zeros_like(time_chem)
        emi_no = np.zeros_like(time_chem)
        tdep_chem = {
            "time_chem": time_chem,
            "jo31d": self._df_tuv.jo31d,
            "jh2o2": self._df_tuv.jh2o2,
            "jno2": self._df_tuv.jno2,
            "jno3": self._df_tuv.jno3,
            "jn2o5": self._df_tuv.jn2o5,
            "jch2or": self._df_tuv.jch2or,
            "jch2om": self._df_tuv.jch2om,
            "jch3o2h": self._df_tuv.jch3o2h,
            "emi_isop": emi_isop,
            "emi_no": emi_no,
        }
        return tdep_chem

    def get_soil(self):
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

    def _get_les_input(self, ds: Optional[Union[ls2d.Read_era5, ls2d.Read_cams]]):
        """Interpolate input data to the LES grid.

        Args:
            ds: dataset from eather cams eac4 or era4
        """
        return ds.get_les_input(self.vgrid.z)

    def _check_era5_les_input_mean(self):
        if self.era5_les_input_mean is None:
            self.check_era5_les_input_mean = self.era5_les_input.mean(dim="time")

    def _check_les_input_era5(self):
        if self.era5_les_input is None:
            self.era5_les_input = self._get_les_input(self.era5)

    def _check_les_input_cams_eac4(self):
        if self.cams_eac4_les_input is None:
            self.cams_eac4_les_input = self._get_les_input(self.cams_eac4)

    def _import_cams_species(self):
        if self._species_cams is not None:
            return

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

    def create_surface_input(self):
        """Create land-surface (vegetation) and sea (SST) input."""
        logger.info("Creating spatial (land-) surface input")

        # Default soil depths IFS.
        z_soil = np.array([-0.035, -0.175, -0.64, -1.945])[::-1]

        # Land-surface / vegetation properties from global LCC dataset
        # (100 m resolution).
        lu_lcc = microhhpy.land.create_land_surface_input(  # noqa: F841
            self.domain.lon,
            self.domain.lat,
            z_soil,
            land_use_source="lcc_100m",
            land_use_tiff=self.lcc_path,
            save_binaries=True,
            output_dir=self._working_dir,
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
        soil.to_binaries(path=self._working_dir, allow_overwrite=True)

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

    def copy_lookup_tables(self):
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


###############################
# End of class
###############################


def ensure_binary_precision(path, shape, dtype):
    """Cast a raw MicroHH binary file to the requested dtype.

    Args:
        path: path to file
        shape: shape of the array
        dtype: datatype ether double or single precision
    """
    dtype = np.dtype(dtype)
    path = str(path)
    count = int(np.prod(shape))
    size = Path(path).stat().st_size

    if size == count * dtype.itemsize:
        return

    if size == count * np.dtype(np.float32).itemsize:
        source_dtype = np.float32
    elif size == count * np.dtype(np.float64).itemsize:
        source_dtype = np.float64
    else:
        raise ValueError(f"Unexpected size for {path}: {size} bytes for shape {shape}")

    data = np.fromfile(path, dtype=source_dtype)
    data.astype(dtype).tofile(path)
    logger.info("Converted %s from %s to %s", path, np.dtype(source_dtype), dtype)


def create_surface_input(era5, era5_mean, domain, lcc_path, float_type):
    """Create land-surface (vegetation) and sea (SST) input.

    Args:
        era5:
        era5_mean:
        domain:
        lcc_path:
        float_type:
    """
    logger.info("Creating spatial (land-) surface input")

    # Default soil depths IFS.
    z_soil = np.array([-0.035, -0.175, -0.64, -1.945])[::-1]

    # Land-surface / vegetation properties from global LCC dataset
    # (100 m resolution).
    lu_lcc = microhhpy.land.create_land_surface_input(  # noqa: F841
        domain.proj.lon,
        domain.proj.lat,
        z_soil,
        land_use_source="lcc_100m",
        land_use_tiff=lcc_path,
        save_binaries=True,
        output_dir=domain.work_dir,
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
            f"{domain.work_dir}/{field}.0000000", (domain.jtot, domain.itot), float_type
        )

    ensure_binary_precision(
        f"{domain.work_dir}/root_frac.0000000",
        (len(z_soil), domain.jtot, domain.itot),
        float_type,
    )

    # TODO: Init soil from HiHydroSoil. For now spatially homogeneous.
    soil = microhhpy.land.Land_surface_input(
        domain.itot, domain.jtot, 4, exclude_veg=True, debug=True, float_type=float_type
    )

    soil.theta_soil[:, :, :] = era5_mean.theta_soil.values[::-1, None, None]
    soil.t_soil[:, :, :] = era5_mean.t_soil.values[::-1, None, None]
    soil.index_soil[:, :, :] = int(era5_mean.type_soil) - 1  # FORTRAN -> C
    soil.to_binaries(path=domain.work_dir, allow_overwrite=True)

    # Create SSTs from ERA5.
    sst_les = microhhpy.real.create_sst_from_regular_latlon(
        era5.sst[0],
        era5.lons,
        era5.lats,
        domain.proj.lon,
        domain.proj.lat,
        float_type=float_type,
    )

    if np.any(np.isnan(sst_les)):
        logger.warning("SSTs contain NaNs! Setting to 290K...")
        sst_les[:, :] = 290.0

    # Water temperatures over land are unknown, and extrapolated SSTs are not
    # very accurate.
    # TODO: get inland water mask from Corine/LCC and let the user define
    # water/lake temperatures.
    sst_les[sst_les < 280] = 280
    sst_les.tofile(f"{domain.work_dir}/t_bot_water.0000000")


@wraps(microhhpy.chem.calc_tuv_photolysis)
def calc_photolysis_rates(tuv_path, domain):
    """Calculate photolysis rates using TUV wrapper.

    Args:
        tuv_path:
        domain:
    """
    name = "microh"  # Must be exactly 6 characters!

    # Default input file. Only start/end date and lat/lon location are updated.
    input_file = microhhpy.utils.get_data_file("microhh_tuv.base")

    tuv_df = microhhpy.chem.calc_tuv_photolysis(
        input_file,
        tuv_path,
        name,
        domain.start_date,
        domain.end_date,
        domain.proj.central_lon,
        domain.proj.central_lat,
        suppress_stdout=True,
    )

    return tuv_df


def copy_lookup_tables(microhh_path, gpt_path, domain):
    """Copy required land-surface and radiation lookup tables.

    Args:
        microhh_path:
        gpt_path:
        domain:
    """
    logger.info("Copying lookup tables")

    rrtmgp_path = f"{microhh_path}/rte-rrtmgp-cpp/"
    rrtmgp_data_path = f"{microhh_path}/rte-rrtmgp-cpp/rrtmgp-data"

    to_copy = [
        (f"{gpt_path}/rrtmgp-gas-lw-g056-cf2.nc", "coefficients_lw.nc"),
        (f"{gpt_path}/rrtmgp-gas-sw-g049-cf2.nc", "coefficients_sw.nc"),
        (f"{rrtmgp_data_path}/rrtmgp-clouds-lw.nc", "cloud_coefficients_lw.nc"),
        (f"{rrtmgp_data_path}/rrtmgp-clouds-sw.nc", "cloud_coefficients_sw.nc"),
        (f"{rrtmgp_path}/data/aerosol_optics.nc", "aerosol_optics.nc"),
        (
            f"{microhh_path}/misc/van_genuchten_parameters.nc",
            "van_genuchten_parameters.nc",
        ),
    ]

    for f in to_copy:
        target = f"{domain.work_dir}/{f[1]}"
        if not Path(target).exists():
            shutil.copy(f[0], target)
