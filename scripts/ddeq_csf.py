import argparse
import logging

import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

import ddeq
import microhh_processing_chain.initialisation as init
import microhh_processing_chain.postprocessing as post
from microhh_processing_chain import io

logging.basicConfig(level=logging.INFO)

WIND_METHOD = "Assume 1 m/s and 90 degrees wind direction"


# =============================================================================
# Argument Parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process MicroHH simulation output and generate 2D/3D datasets."
    )

    parser.add_argument(
        "--file_directory",
        type=str,
        required=True,
        help="Path to the MicroHH case directory...",
    )

    return parser.parse_args()


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1) Initialization
    # -------------------------------------------------------------------------
    args = parse_args()
    logging.info(f"Processing case: {args.directory}")

    case = init.read_case(args.directory)
    post_processing = post.PostProcessing(case)

    # -------------------------------------------------------------------------
    # 2) Variables and Configuration
    # -------------------------------------------------------------------------
    raise NotImplementedError(
        "This script is a template and requires further implementation."
    )
    noise_lvl = 450  # ppmv
    lat_list = None  # TODO: fill in source latitudes
    lon_list = None  # TODO: fill in source longitudes
    crs = ccrs.epsg(32630)
    start_timestamp = pd.Timestamp("2024-09-16 09:00:00")
    release_start_timestamp = pd.Timestamp("2024-09-16 10:00:00")
    release_end_timestamp = pd.Timestamp("2024-09-16 10:45:00")

    # -------------------------------------------------------------------------
    # 3) GUI Parameters
    # -------------------------------------------------------------------------
    gui_cmap = "Reds"
    gui_color = "black"
    gui_headless = True
    gui_detection_method = "Threshold"
    gui_thr = 0.0005
    gui_filter_size = 1
    gui_center_line = True
    gui_node = 2
    gui_plume_area = True
    gui_fix_plume_width = True
    gui_emissions = True
    gui_emissions_method = "csf"
    gui_background = "linear"
    gui_lmax = 20
    gui_delta = 3
    gui_plume_width = 50

    # -------------------------------------------------------------------------
    # 4) Generate Synthetic Wind Field
    # -------------------------------------------------------------------------
    coords = {"source": ["source1"]}

    # Create dataset with only the required variables
    winds = xr.Dataset(
        data_vars={
            "speed": (["source"], [1.0], {"units": "m s-1"}),
            "speed_precision": (["source"], [0.1], {"units": "m s-1"}),
            "direction": (["source"], [90.0], {"units": "degrees"}),
        },
        coords=coords,
    )
    winds.attrs.update({"wind method": WIND_METHOD})

    # -------------------------------------------------------------------------
    # 5) Read Input and Pre-process
    # -------------------------------------------------------------------------
    post_processing = post.PostProcessing(case)
    dd_ch4_path = io.open_microhh_nc(case, "ch4_path.xy")

    # rename variable to CH4
    dd_ch4_path = dd_ch4_path.rename({"ch4_path": "CH4"})
    dd_ch4_path.CH4.attrs["units"] = "kg m-2"
    sources = post_processing.create_sources_dataset(
        lat=lat_list,
        lon=lon_list,
    )
    dd_coord = post_processing.convert_microhh_grid_to_coords(
        dd_ch4_path, sources.isel(source=0), microhh_crs="epsg:32630"
    )
    dd_coord = post_processing.convert_microhh_timestep_to_date(
        dd_coord, start_timestamp
    )
    data_coord = ddeq.misc.add_corners(dd_coord)
    data_coord = data_coord.rename({"CH4": "CH4_mass"})

    data_coord = post.add_gaussian_noise(
        data_coord,
        var_name="CH4_mass",
        gas="CH4",
        std=noise_lvl,
        noise_units="ppmv m",
    )

    data_coord = ddeq.plume_coords.compute_pixel_areas(data_coord, crs)

    # -------------------------------------------------------------------------
    # 6) Estimate Emissions
    # -------------------------------------------------------------------------
    data_coord_subset = data_coord.sel(
        time=slice(release_start_timestamp, release_end_timestamp)
    )

    domain = ddeq.misc.Domain(
        name="Pau",
        startlat=float(data_coord_subset.lat.min()),
        stoplat=float(data_coord_subset.lat.max()),
        startlon=float(data_coord_subset.lon.min()),
        stoplon=float(data_coord_subset.lon.max()),
    )

    results_file_path = case.directory / "results_csf.nc"
    if results_file_path.exists():
        logging.info(
            "Results file already exists for case %s. Skipping processing.",
            args.directory,
        )
        return

    # -------------------------------------------------------------------------
    # 7) Time-Step Loop and CSF Statistics
    # -------------------------------------------------------------------------
    results_arr = []
    for t in tqdm(data_coord_subset.time, desc="Processing timesteps"):
        data_coord_subset_t = data_coord_subset.sel(time=t)

        try:
            ui = ddeq.gui.UserInterface(
                gas="CH4",
                variable="CH4_mass",
                variable_std="CH4_mass_precision",
                data=data_coord_subset_t,
                sources=sources.sel(source=["source1"]),
                winds=winds,
                crs=crs,
                domain=domain,
                cmap=gui_cmap,
                color=gui_color,
                headless=gui_headless,
            )

            ui.detection_dropdown.value = gui_detection_method
            ui.thr_value_slider.value = gui_thr
            ui.filter_size_slider.value = gui_filter_size
            ui.center_line_button.value = gui_center_line
            ui.node_dropdown.value = gui_node
            ui.plume_area_button.value = gui_plume_area
            ui.fix_plume_width_checkbox.value = gui_fix_plume_width
            ui.emissions_button.value = gui_emissions
            ui.emissions_method_dropdown.value = gui_emissions_method
            ui.background_dropdown.value = gui_background
            ui.lmax_slider.value = gui_lmax
            ui.delta_slider.value = gui_delta
            ui.plume_width_slider.value = gui_plume_width

            results_stats = ddeq.csf.compute_fit_statistics(
                ui.results,
                "CH4",
                variable="CH4_mass",
                variable_std="CH4_mass_precision",
            )

            results_arr.append(results_stats)
        except Exception as e:
            logging.exception("Error processing time %s: %s", t.values, e)
            continue

    if not results_arr:
        logging.warning(
            "No valid timesteps were processed. No results file was written."
        )
        return

    # -------------------------------------------------------------------------
    # 8) Merge and Save Results
    # -------------------------------------------------------------------------
    results_arr_merged = xr.concat(results_arr, dim="time", join="outer")
    results_arr_merged.attrs.update({"wind method": WIND_METHOD})
    results_arr_merged.to_netcdf(results_file_path)


if __name__ == "__main__":
    main()
