import argparse
import logging
from enum import StrEnum

import cartopy.crs as ccrs
import ddeq
import ddeq.gui
import numpy as np
import pandas as pd
import xarray as xr
from ddeq.gui.params import CSFPlotParams, PlumeDetectionParams
from tqdm.auto import tqdm

import microhh_processing_chain.initialisation as init
import microhh_processing_chain.postprocessing as post
from microhh_processing_chain import io

logging.basicConfig(level=logging.INFO)


# =============================================================================
# Argument Parsing and other Helper Functions
# =============================================================================
class RunPhase(StrEnum):
    """Enumeration for different simulation phases."""

    SPINUP = "spinup"
    RELEASE = "release"
    FULL = "full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process MicroHH simulation output and generate 2D/3D datasets."
    )

    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to the MicroHH case directory...",
    )

    return parser.parse_args()


def get_case_times(
    case_df: pd.DataFrame, run_phase: RunPhase, savetime_seconds: int
) -> dict:
    """Calculates start/end times and durations based on the run phase.

    Args:
        case_df (pd.DataFrame): DataFrame containing release timestamps.
        run_phase (RunPhase): Enum or value defining the simulation phase.
        savetime_seconds (int): The interval used for rounding durations.
    """
    # 1. Calculate the absolute baseline boundaries
    case_start_time_full = pd.to_datetime(
        case_df["Release Start (UTC)"].iloc[0]
    ).tz_localize(None) - pd.Timedelta(hours=1, minutes=15)

    case_end_time_full = pd.to_datetime(
        case_df["Release End (UTC)"].iloc[0]
    ).tz_localize(None) + pd.Timedelta(minutes=5)

    one_hour = pd.Timedelta(hours=1)

    # 2. Determine offsets based on the phase
    if run_phase == RunPhase.SPINUP:
        start_offset = pd.Timedelta(0)
        end_offset = one_hour
    elif run_phase == RunPhase.RELEASE:
        start_offset = one_hour
        end_offset = case_end_time_full - case_start_time_full
    elif run_phase == RunPhase.FULL:
        start_offset = pd.Timedelta(0)
        end_offset = case_end_time_full - case_start_time_full
    else:
        raise ValueError(
            f"Unknown run phase: {run_phase}. Use SPINUP, RELEASE, or FULL."
        )

    # 3. Calculate phase-specific windows
    case_start_time = case_start_time_full + start_offset
    # case_end_time = case_start_time_full + end_offset

    case_start_sec = int(start_offset.total_seconds())
    case_end_sec = int(end_offset.total_seconds())

    # 4. Handle rounding to the next savetime
    duration = case_end_time_full - case_start_time
    rounded_duration = duration.ceil(pd.Timedelta(seconds=savetime_seconds))
    final_end_time = case_start_time + rounded_duration

    return {
        "start_time": case_start_time,
        "end_time": final_end_time,
        "start_sec": case_start_sec,
        "end_sec": case_end_sec,
        "duration_rounded": rounded_duration,
    }


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def main() -> None:
    # -------------------------------------------------------------------------
    # 1) Initialization
    # -------------------------------------------------------------------------
    args = parse_args()
    logging.info("Processing case: %s", args.directory)

    case = init.read_case(args.directory)
    stats = io.open_statistics(case)
    case_df = pd.read_csv(case.directory / f"{case.name}_case_info.csv")
    sources = xr.open_dataset(case.directory / f"{case.name}_sources.nc")

    sampletime = case.config.cross.sampletime
    case_times = get_case_times(case_df, RunPhase.FULL, sampletime)
    noise_lvl = 280  # ppmv m, based on coarsened data with 1m resolution
    crs = ccrs.epsg(32630)  # UTM zone 30N

    # create folder for images
    (case.directory / "images").mkdir(exist_ok=True)

    post_processing = post.PostProcessing(case)

    # -------------------------------------------------------------------------
    # 3) GUI Parameters
    # -------------------------------------------------------------------------
    gui_cmap = "Reds"
    gui_color = "black"
    gui_headless = True
    gui_detection_method = "Threshold"
    gui_thr = 0.0003
    gui_filter_size = 1
    gui_center_line = True
    gui_node_nr = 2
    gui_plume_area = True
    gui_fix_plume_width = True
    gui_emissions = True
    gui_emissions_method = "csf"
    gui_background = "linear"
    gui_lmax = 50
    gui_delta = 3
    gui_plume_width = 75

    plume_detection_params = PlumeDetectionParams(
        var_sys=0, filter_type="gaussian", lmax=gui_lmax
    )
    csf_plot_params = CSFPlotParams(
        variable_map="{gas}_mass",
        variable_std="{gas}_mass_precision",
        variable_across="{gas}_mass",
        vmins=0,
        cmap="Reds",
        line_color="black",
    )

    # -------------------------------------------------------------------------
    # 4) Generate Synthetic Wind Field
    # -------------------------------------------------------------------------
    stats = post_processing.convert_microhh_timestep_to_date(
        stats, start_date=case_times["start_time"]
    )

    plume_height = case_df["Z (m)"].iloc[0] + case_df["plume_rise (m)"].iloc[0]
    winds = stats.sel(z=plume_height, method="nearest")[["u", "v"]]

    winds["direction"] = ddeq.wind.calculate_wind_direction(winds.u, winds.v)
    winds["speed"] = np.sqrt(winds.u**2 + winds.v**2)
    winds["speed_precision"] = (
        0.1 * winds["speed"]
    )  # Assume 10% uncertainty in wind speed

    winds["direction"].attrs["units"] = "degrees"
    winds["speed"].attrs["units"] = "m s-1"
    winds["speed_precision"].attrs["units"] = "m s-1"
    winds.attrs.update({"METHOD": "Calculated from MicroHH simulation at plume height"})
    winds = winds.assign_coords(source=["source1"])

    # -------------------------------------------------------------------------
    # 5) Read Input and Pre-process
    # -------------------------------------------------------------------------
    dd_ch4 = io.open_microhh_nc(case, "ch4_path.xy")

    # rename variable to CH4
    dd_ch4 = dd_ch4.rename({"ch4_path": "CH4"})
    dd_ch4.CH4.attrs["units"] = "kg m-2"

    dd_coord = post_processing.convert_microhh_grid_to_coords(
        dd_ch4, sources.isel(source=0), microhh_crs="epsg:32630"
    )
    dd_coord = post_processing.convert_microhh_timestep_to_date(
        dd_coord,
        case_times["start_time"],
    )
    data_coord = ddeq.misc.add_corners(dd_coord)
    data_coord = data_coord.rename({"CH4": "CH4_mass"})

    data_coord = ddeq.plume_coords.compute_pixel_areas(data_coord, crs)

    data_coord_noisy = post.add_gaussian_noise(
        data_coord, var_name="CH4_mass", gas="CH4", std=noise_lvl, noise_units="ppmv m"
    )

    data_coord_noisy_slice = data_coord_noisy.sel(
        time=slice(
            pd.Timestamp(case_df["Release Start (UTC)"].iloc[0]).tz_convert(None),
            pd.Timestamp(case_df["Release End (UTC)"].iloc[0]).tz_convert(None),
        )
    )

    domain = ddeq.misc.Domain(
        name="Pau",
        startlat=float(data_coord.lat.min()),
        stoplat=float(data_coord.lat.max()),
        startlon=float(data_coord.lon.min()),
        stoplon=float(data_coord.lon.max()),
    )

    # -------------------------------------------------------------------------
    # 6) Estimate Emissions
    # -------------------------------------------------------------------------
    data_file_path = case.directory / "data_csf.nc"
    results_file_path = case.directory / "results_csf.nc"
    if data_file_path.exists() and results_file_path.exists():
        logging.info(
            "Data and results files already exists for case %s. Skipping processing.",
            args.directory,
        )
        return

    # -------------------------------------------------------------------------
    # 7) Time-Step Loop and CSF Statistics
    # -------------------------------------------------------------------------
    data_arr = []
    results_arr = []
    for t in tqdm(data_coord_noisy_slice.time, desc="Processing timesteps"):
        data_coord_subset_t = data_coord_noisy_slice.sel(time=t)
        _tolerance = pd.Timedelta(seconds=sampletime)
        winds_subset_t = winds.sel(
            time=t, method="nearest", tolerance=_tolerance.total_seconds()
        )

        try:
            ui = ddeq.gui.UserInterface(
                gas="CH4",
                variable="CH4_mass",
                variable_std="CH4_mass_precision",
                data=data_coord_subset_t,
                sources=sources.sel(source=["source1"]),
                winds=winds_subset_t,
                crs=crs,
                domain=domain,
                cmap=gui_cmap,
                color=gui_color,
                headless=gui_headless,
                plume_detection_params=plume_detection_params,
                csf_plot_params=csf_plot_params,
            )

            ui.thr_value_slider.value = gui_thr  # pyright: ignore
            ui.filter_size_slider.value = gui_filter_size  # pyright: ignore
            ui.detection_dropdown.value = gui_detection_method  # pyright: ignore
            ui.node_dropdown.value = gui_node_nr  # pyright: ignore
            ui.plume_area_button.value = gui_plume_area  # pyright: ignore
            ui.fix_plume_width_checkbox.value = gui_fix_plume_width  # pyright: ignore
            ui.plume_width_slider.value = gui_plume_width  # pyright: ignore
            ui.center_line_button.value = gui_center_line  # pyright: ignore
            ui.emissions_method_dropdown.value = gui_emissions_method  # pyright: ignore
            ui.background_dropdown.value = gui_background  # pyright: ignore
            ui.lmax_slider.value = gui_lmax  # pyright: ignore
            ui.delta_slider.value = gui_delta  # pyright: ignore
            ui.emissions_button.value = gui_emissions  # pyright: ignore

            if ui.results is None:
                logging.warning(
                    "No results returned by the GUI for time %s. "
                    "Skipping this timestep.",
                    t.values,
                )
                continue

            # save results fig:
            if ui.results_fig is not None:
                t_str = pd.Timestamp(t.values).strftime("%Y%m%d_%H%M%S")
                fig_path = case.directory / "images" / f"results_fig_{t_str}.png"
                ui.results_fig.savefig(fig_path, dpi=300)

            results_stats = ddeq.csf.compute_fit_statistics(
                ui.results,
                "CH4",
                variable="CH4_mass",
                variable_std="CH4_mass_precision",
            )

            data_arr.append(ui.data)
            results_arr.append(results_stats)

        except Exception:
            logging.exception("Error processing time %s", t.values)
            continue

    if not results_arr:
        logging.warning(
            "No valid timesteps were processed. No results file was written."
        )
        return

    # -------------------------------------------------------------------------
    # 8) Merge and Save Results
    # -------------------------------------------------------------------------
    data_arr_merged = xr.concat(data_arr, dim="time", join="outer")
    if not data_file_path.exists():
        data_arr_merged.to_netcdf(data_file_path)

    results_arr_merged = xr.concat(results_arr, dim="time", join="outer")
    if not results_file_path.exists():
        results_arr_merged.to_netcdf(results_file_path)

    logging.info(
        "Processing complete for case %s. Results saved to %s",
        args.directory,
        results_file_path,
    )


if __name__ == "__main__":
    main()
