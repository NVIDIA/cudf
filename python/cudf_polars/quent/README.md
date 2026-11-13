# cudf-polars-quent

This directory contains the [model](./model.yaml) defining cudf-polar's
telemetry and Rust crates to capture and analyze that telemetry.

- [cudf-polars-quent-bridge](./bridge/README.md): Build the `cudf_polars_quent` instrumentation library, including Python bindings for cudf-polars.
- [cudf-polars-quent-analyzer](./analyzer/README.md): Rust crate for analyzing cudf-polars' custom events. Used by the UI to analyzer and transform the raw events.
- [cudf-polars-quent-store](./store/README.md): Shared generated event types, used by the analyzer and browser crates.
- [cudf-polars-quent-browser](./browser/README.md): Implementation of the interface for compiling the cudf-polars analyzer to WASM and loading `.quent-browser` bundles.
