# Changelog

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
