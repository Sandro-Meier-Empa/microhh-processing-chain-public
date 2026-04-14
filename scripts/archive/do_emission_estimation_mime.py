#!/usr/bin/env python3

import sys
import argparse
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

import microhh_processing.initialisation as init
import microhh_processing.processing as ch4_proc
import microhh_processing.wind as wind_proc
import microhh_processing.ime_v2 as ime


sys.path.append("/users/smeier/ddeq/src/")
import ddeq


def mean_ime(
    data: xr.Dataset,
    winds: xr.Dataset,
    sources: xr.Dataset,
    curves,
    gas: str,
    variable: str = "{gas}_minus_estimated_background_mass",
    L_min: float = 0,
    L_max: float = None,
    A_std: float = 0,
    decay_time: float = np.nan,
    min_pixel_number: int = 10,
    allowed_upstream_detected: int = 5,
    return_uncertainty_contributions: bool = False,
):

    pixel_size = np.sqrt(np.mean(data.pixel_area))

    if L_max is None:
        L_max = (
            np.max(np.absolute(data.xp.values[data.detected_plume])) + 0.5 * pixel_size
        )

    all_results = []
    for n, L in enumerate(np.arange(0.5 * pixel_size, L_max, pixel_size)):
        results = ddeq.ime.estimate_emissions(
            data=data,
            winds=winds,
            sources=sources,
            curves=curves,
            gas=gas,
            variable=variable,
            L_min=L_min,
            L_max=L,
            A_std=A_std,
            decay_time=decay_time,
            min_pixel_number=min_pixel_number,
            allowed_upstream_detected=allowed_upstream_detected,
            return_uncertainty_contributions=return_uncertainty_contributions,
        )
        results = results.assign_coords(time=data.time)
        results = results.assign_coords(plume_length_index=n)

        all_results.append(results)
    data_combined = xr.concat(all_results, dim="plume_length_index")

    return data_combined


def process_case(case_name, case_suffix, **kwargs):
    case = init.Case(
        case_name=case_name,
        suffix=case_suffix,
        scratch_path="/scratch/snx3000",
        store_path="/capstor/store/cscs/empa/em05/smeier/microhh_output",
        location="scratch",
    )

    ############################################################################
    CH4_emissions_all = []
    CH4_emissions_all_bg = []

    lat_zh = 47.3786
    lon_zh = 8.5400
    save = True

    noise = 300
    noise_units = "ppmv m"
    var = "CH4"
    filter_size = 1  # is std of gaussian
    q = 0.999
    sigma = 10
    crs = ccrs.epsg(32632)
    radius = 200

    # get time step where turbulence starts
    stats = init.read_statistics(case)

    # continue if simulation failed:
    if stats.time.size == 0:
        print("Simulation failed. Skipping...")
        return

    start_time_step, start_time_value = ch4_proc.get_2D_tke_threshold_time(
        stats,
        tke_threshold=True,
        threshold_value=0.3,
    )

    degradation_levels = [1]
    noise_levels = [noise]
    wind_integration = ["height"]

    wind_processing = wind_proc.WindProcessing(case)
    post_processing = ch4_post.PostProcessing(case)

    sources = post_processing.create_sources_dataset(lat_zh, lon_zh)

    data = post_processing.convert_to_columns("CH4", "kg m-2")
    data.CH4.attrs["units"] = "kg m-2"
    data = post_processing.convert_microhh_timestep_to_date(
        data, start_date="2020-01-01 08:00:00"
    )
    data = post_processing.convert_microhh_grid_to_coords(data, "source1", sources)

    b = 0.0001

    domain = ddeq.misc.Domain(
        name="Zurich",
        startlat=float(data.lat.min() - b),
        stoplat=float(data.lat.max() + b),
        startlon=float(data.lon.min() - b),
        stoplon=float(data.lon.max() + b),
    )

    for w, vert_int in enumerate(wind_integration):
        wind = wind_processing.create_wind_dataset(
            "source1", sources, radius=radius, method=vert_int, height=10, save=False
        )
        wind = post_processing.convert_microhh_timestep_to_date(
            wind, start_date="2020-01-01 08:00:00"
        )
        stats = post_processing.convert_microhh_timestep_to_date(
            stats, start_date="2020-01-01 08:00:00"
        )

        for t, sim_time in enumerate(data.time.values[start_time_step:]):
            data_subset = data.sel(time=sim_time)
            wind_subset = wind.sel(time=sim_time)

            for d, degradation in enumerate(degradation_levels):
                sources = post_processing.create_sources_dataset(
                    lat_zh, lon_zh, source_diameter=degradation * 5
                )

                dd1 = post_processing.degrade_resolution(
                    data_subset, degradation, degradation
                )

                for n, noise_lvl in enumerate(noise_levels):
                    dd2 = post_processing.add_gaussian_noise(
                        dd1,
                        var=var,
                        std=noise_lvl
                        / np.sqrt(
                            degradation**2
                        ),  # due to binning, noise decreases with sqrt(x_dim*y_dim)
                        noise_units=noise_units,
                    )

                    # calculate emissions
                    try:
                        dd3, curves, results = ime.estimate_emissions(
                            dd2,
                            wind_subset,
                            sources,
                            filter_size,
                            q,
                            sigma=sigma,
                            background_dilation=3,
                            crs=crs,
                        )
                    except Exception as err:
                        print(f"ERROR: {err}")
                        continue

                    # add effective wind speed
                    results = ime.compute_effective_wind_speed(
                        results, true_emissions=sources.strength.values / 60**2
                    )

                    #################### without bg subtraction ####################
                    dd_all = mean_ime(
                        dd3,
                        wind_subset,
                        sources,
                        curves,
                        gas="CH4",
                        variable="{gas}_mass",
                        min_pixel_number=1,
                        allowed_upstream_detected=15,
                    )
                    CH4_emissions_all.append(dd_all)

                    #################### with bg subtraction ####################
                    dd_all_bg = mean_ime(
                        dd3,
                        wind_subset,
                        sources,
                        curves,
                        gas="CH4",
                        variable="{gas}_mass_minus_estimated_background_mass",
                        min_pixel_number=1,
                        allowed_upstream_detected=15,
                    )
                    CH4_emissions_all_bg.append(dd_all_bg)

    dd_time_combined = xr.concat(CH4_emissions_all, dim="time")
    dd_time_combined.to_netcdf(path=case.scratch_path / "ime_results_mass.nc")
    dd_time_combined_bg = xr.concat(CH4_emissions_all_bg, dim="time")
    dd_time_combined_bg.to_netcdf(
        path=case.scratch_path / "ime_results_mass_minus_bg.nc"
    )

    ############################################################################


def main():
    parser = argparse.ArgumentParser(description="Process some cases.")
    parser.add_argument(
        "--case-name", required=True, help="Name of the case to process"
    )
    parser.add_argument(
        "--case-suffix", required=True, help="Suffix of the case to process"
    )

    # Allow for additional arguments
    parser.add_argument(
        "--additional-args",
        nargs="*",
        help="Additional arguments in the form of key=value",
    )

    args = parser.parse_args()

    # Parse additional arguments into a dictionary
    additional_args = {}
    if args.additional_args:
        for arg in args.additional_args:
            key, value = arg.split("=")
            additional_args[key] = value

    process_case(args.case_name, args.case_suffix, **additional_args)


if __name__ == "__main__":
    main()
