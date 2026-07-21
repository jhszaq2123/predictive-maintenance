# Shared Dashboard Assets

This directory is reserved for shared presentation assets reused across multiple dashboard modes.

## Expected formats

- `SVG` for reusable diagrams and frames
- `PNG` for static backgrounds or fallback previews

## Naming convention

- `shared_<purpose>.svg`
- `shared_<purpose>.png`

## Current compatibility note

The existing `dashboard/assets/machine_diagram.svg` file remains at its original location in Sprint SCADA 0 to avoid breaking current imports.
Future shared machine assets can be migrated here once all references are updated safely.
