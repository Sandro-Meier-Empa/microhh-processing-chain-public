#!/usr/bin/env python3

import os
import sys
import argparse
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

sys.path.append(os.path.join("/users", os.environ["USER"], "processing_chain", "src"))
import processing as proc
import initialisation as init
import wind as wind_proc
import ime_v2 as ime

sys.path.append(f"/users/smeier/ddeq/src/")
import ddeq


def process_case(case_name, case_suffix, **kwargs):
    case = init.Case(
        case_name=case_name,
        suffix=case_suffix,
        scratch_path="/scratch/snx3000",
        store_path="/store/empa/em05/smeier/microhh_output",
        location="scratch",
    )

    ############################################################################
    surface_roughness = kwargs["surface_roughness"]
    sensible_heat_flux = kwargs["sensible_heat_flux"]

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

    start_time_step, start_time_value = proc.get_2D_tke_threshold_time(
        stats,
        tke_threshold=True,
        threshold_value=0.3,
    )

    degradation_levels = [1]
    noise_levels = [noise]
    wind_integration = ["height"]

    wind_processing = wind_proc.WindProcessing(case)
    post_processing = proc.PostProcessing(case)

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

        for t, sim_time in enumerate(data.time.values[:start_time_step]):
            data_subset = data.sel(time=sim_time)
            wind_subset = wind.sel(time=sim_time)
            stats_subset = stats.sel(time=sim_time)

            for d, degradation in enumerate(degradation_levels):
                sources = post_processing.create_sources_dataset(
                    lat_zh, lon_zh, source_radius=degradation * 5
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

                    # add sens heat flux, monin obukhov length and surface roughness

                    results["sensible_heat_flux"] = (
                        results.dims,
                        np.array([sensible_heat_flux]),
                        {"units": "W m-2"},
                    )
                    results["monin_obukhov_length"] = (
                        results.dims,
                        np.array([stats_subset.obuk.values]),
                        stats_subset.obuk.attrs,
                    )
                    results["surface_roughness"] = (
                        results.dims,
                        np.array([surface_roughness]),
                        {"units": "m"},
                    )

                    # assign noise lvl and degradation to dataset
                    results = results.assign_coords(
                        {
                            "time": np.array([sim_time], dtype=np.datetime64),
                            "wind_integration": vert_int,
                            "noise_level": noise_lvl,
                            "degradation": degradation,
                        }
                    )

                    # combine data with different noise levles
                    if n == 0:
                        all_results = results
                    else:
                        all_results = xr.concat(
                            [all_results, results], dim="noise_level"
                        )

                # combine data with different degradations
                if d == 0:
                    all_results2 = all_results
                else:
                    all_results2 = xr.concat(
                        [all_results2, all_results], dim="degradation"
                    )

            # combine data of different time steps
            if t == 0:
                all_results3 = all_results2
            else:
                all_results3 = xr.concat([all_results3, all_results2], dim="time")

        # combine data of different wind integrations
        if w == 0:
            all_results_final = all_results3
        else:
            all_results_final = xr.concat(
                [all_results_final, all_results3], dim="wind_integration"
            )

        all_results_final["degradation"].attrs["units"] = "-"
        all_results_final["degradation"].attrs[
            "description"
        ] = "Factor of resolution degradation in x and y direction"
        all_results_final["noise_level"].attrs["units"] = "ppmv m"
        all_results_final["noise_level"].attrs["description"] = "Gaussian noise"

    if save:
        all_results_final.to_netcdf(
            path=f"{case.scratch_path}/estimated_emissions_stable.nc"
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
