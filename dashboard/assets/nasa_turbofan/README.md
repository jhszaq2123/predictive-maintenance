# NASA Turbofan Assets

This directory is reserved for future SCADA/HMI visuals for the `NASA Turbofan` mode.

## Expected formats

- `SVG` for illustrative turbofan diagrams
- `PNG` for overlays or fallback visual states

## Naming convention

- `nasa_turbofan_overview.svg`
- `nasa_turbofan_component_<name>.svg`
- `nasa_turbofan_overlay_<name>.png`

## SVG component identifiers

Future SVG files should use stable element identifiers, for example:

- `fan`
- `low_pressure_compressor`
- `high_pressure_compressor`
- `combustor`
- `high_pressure_turbine`
- `low_pressure_turbine`
- `nozzle`

The NASA component view remains illustrative because the source sensors are anonymous.
