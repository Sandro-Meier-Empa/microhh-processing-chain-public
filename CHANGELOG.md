# Changelog

## 2026-11-16

### Fixed

- `export.export_plume_column()` records the emission point MicroHH actually
  uses as `source_x/y/z`. With `[source] swtimedep_location = True` the
  position comes from `/timedep source_{x,y,z}0_{n}` of the initial-condition
  file; a source that moves before the last exported snapshot raises
  `ValueError` instead of exporting the `.ini` position.
- `[source]` keys holding one value are broadcast to every source, as in
  MicroHH, instead of raising `IndexError` for `source_index > 0`.

### Changed

- Line sources (non-zero `line_x/y/z`) are rejected by the export, since a
  `PlumeColumn` file records one emission point.

## 2026-10-14

### Breaking changes

- Removed the unused `ml` module (`flip_data()`, `flip_variable()`,
  `scale_emissions()`); `flip_variable()` dropped the mirrored coordinates.
  Plume augmentation now lives in `spectra.plume.augment`.

### Added

- Added `export.export_plume_column()`, which writes the vertical columns of
  one tracer and one source as a `spectra.PlumeColumn` NetCDF file
  (`{case_id}__{tracer}.plumecolumn.nc`): UTC `time`, `simulation_time`,
  `column` in kg m-2 from `{tracer}_path.xy.nc` or the integrated 3-D field,
  `emission_rate` in kg s-1, the horizontal-mean wind at a height, and the
  source, grid and provenance attributes.

### Changed

- Declared `astropy`, which `misc` imports, as a dependency.
- `sources.get_source_strength()` accepts a float `time_s`.

## 2026-10-08

### Breaking changes

- Moved the optical-flow plume velocimetry to the stand-alone
  [`plumeflow`](https://gitlab.empa.ch/abt503/users/mesa/plumeflow) package.
  `wind.compute_optical_flow_displacement()`, `wind.compute_grid_resolution()`,
  `wind.calculate_optical_flow_effective_wind_speed()` and
  `visualisation.plot_optical_flow_wind_snapshots()` are now
  `plumeflow.compute_displacement()`, `plumeflow.compute_grid_resolution()`,
  `plumeflow.compute_effective_wind_speed()` and
  `plumeflow.plot_displacement()`. The `optical-flow` extra was removed.
- The example notebook `effective_wind_speed_from_2d_ch4.ipynb` moved from
  `microhh-notebooks` to `plumeflow/notebooks/`.

### Changed

- Corrected the license classifier in `pyproject.toml` to LGPL-3, matching the
  `LICENSE` file.

## 2026-09-22

### Breaking changes

- Renamed the `Case` attributes `name_spec`, `file_directory`, `ini_path`, and
  `ini_file` to `full_name`, `directory`, `ini_file_path`, and `config`,
  respectively.
- Replaced `PreProcessing.prepare_simulation()` and
  `PreProcessing.prepare_restart_simulation()` with the explicit
  `validate()`, `copy_required_files()`, and `create_run_script()` workflow.
- Changed `io.open_initial_cond()` and `io.open_statistics()` to return
  `xarray.DataTree` objects instead of flattened `xarray.Dataset` objects.
- Removed `microhh_processing_chain.cscs`. Cluster and Slurm utilities now come
  from the companion `cscs` package, while MicroHH progress helpers are
  available from `microhh_processing_chain.monitoring`.
- Moved `PostProcessing.convert_microhh_grid_to_coords()` and
  `PostProcessing.convert_microhh_timestep_to_date()` to module-level
  functions, removed `PostProcessing.run_postprocessing()`, and renamed
  `PostProcessing.calculate_3D_tke()` to `calculate_3d_tke()`.
- Renamed `WindProcessing.create_2D_wind()` to `create_2d_wind()` and removed
  the placeholder `EffectiveWind` class.

### Added

- Added the `case_input` module with `InitialBoundaryConditions`, `init_icbc()`,
  `nudge_factor()`, and binary-precision handling for MicroHH input creation.
- Added the `thermodynamics` module with potential, absolute, and virtual
  temperature conversions and the `PotentialTemperatureProfile` helper.
- Added optical-flow wind estimation and spatial and temporal turbulence
  spectrum functions to `wind`.
- Added simulation progress monitoring, MicroHH and GPU job-log parsers,
  restart-statistics merging, parallel 2D/3D postprocessing script generation,
  and case file summaries.
