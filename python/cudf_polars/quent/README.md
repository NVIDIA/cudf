# cudf-polars-quent

This directory contains the [model](./model.yaml) defining cudf-polars'
telemetry and the Rust crates that capture and analyze that telemetry.

- [cudf-polars-quent-bridge](./bridge/README.md): Builds the `cudf_polars_quent` instrumentation library, including Python bindings for cudf-polars.
- [cudf-polars-quent-analyzer](./analyzer/README.md): Rust crate that turns cudf-polars events into the views used by the Quent UI.
- [cudf-polars-quent-store](./store/README.md): Shared generated event types, used by the analyzer and browser crates.
- [cudf-polars-quent-browser](./browser/README.md): Compiles the cudf-polars analyzer to WASM and loads `.quent-browser` bundles.

## Pinning Quent

All Quent dependencies inherit the same full commit SHA from the workspace
manifest's `[workspace.dependencies]`. Cargo keys git sources on the exact
`rev` spelling, so a branch, tag, or abbreviated SHA builds a second copy of
those crates. The generated model, the analyzer, and the `quent-open` viewer
traits then fail to unify.

See the bridge's [Updating Quent](./bridge/README.md#updating-quent) section for
the update procedure.
