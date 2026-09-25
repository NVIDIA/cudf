# cudf-polars Quent bindings

This package generates the `cudf_polars._quent` extension, an *optional*
dependency for cudf-polars.

It uses [Quent] to define a schema for cudf-polars execution and telemetry
model. A Python instrumentation library is generated from this model definition
(via generated Rust code).

The main cudf-polars package imports `cudf_polars._quent` extension only when
telemetry collection is enabled.

## Distributed filesystem workaround

The generated bindings currently export NDJSON directly from every driver and
worker process. For multi-node execution, `QuentContext.output_root` (or
`CUDF_POLARS__EXECUTOR__QUENT_OUTPUT_ROOT`) must be the same writable
shared-filesystem path on every node. Each process writes a distinct context
UUID directory, which rank 0 packages after all sessions have closed.

This is a temporary workaround until Quent provides supported Python bindings
for its Collector. Node-local output paths do not produce a complete
multi-node archive.

## Local Development

Install [maturin] into your cudf-polars development environment and build the
extension:

```sh
python -m maturin develop
```

## Updating Quent

Quent is pinned by full Git commit SHA in both `bridge/Cargo.toml`. To update
Quent:

1. Replace every Quent dependency's `rev` in both manifests with the same full
   commit SHA. Do not use a branch, tag, abbreviated SHA, or different revision
   spelling: Cargo must resolve one package identity for the generated model,
   analyzer, and `quent-open` viewer traits.
2. Resolve each crate once without `--locked` so Cargo replaces the old Quent
   Git source and records the new source SHA and transitive dependency graph:

   ```sh
   # From python/cudf_polars/quent/bridge
   cargo update
   ```

   Review the then commit the changes (including the lockfiles).
3. Activate the cudf development environment and rebuild from
   `python/cudf_polars/quent/bridge`:

   ```sh
   python -m maturin develop
   ```

   The build script generates the Rust bridge and a PEP 561 stub under Cargo's
   `OUT_DIR`; it does not overwrite the tracked
   `../../cudf_polars/_quent.pyi`. When the generated API changes, refresh that
   tracked stub from the most recent debug build:

   ```sh
   generated_stub="$(
     ls -t target/debug/build/cudf-polars-quent-*/out/_quent/__init__.pyi |
       head -n 1
   )"
   cp "${generated_stub}" ../../cudf_polars/_quent.pyi
   ```

   The generated output already includes the cudf-specific transformations
   made in `build.rs`. Review and commit the stub diff, then run mypy so stale
   handle names, event methods, and argument types are caught.
4. Run the checks at `ci/run_cudf_polars_quent_tests`.
5. Commit and push the analyzer changes to the Git remote recorded in the
   bridge's build provenance. Rebuild the bridge after committing, then
   regenerate traces. `quent-open` checks out the analyzer package at the exact
   remote and commit embedded in `model.qmi`; a local-only commit or a trace
   generated from the previous commit cannot be loaded elsewhere.

New trace archives embed the updated Quent Git provenance in `model.qmi`.
Existing archives retain the revision with which they were generated.


[maturin]: https://www.maturin.rs/installation.html
[Quent]: https://github.com/rapidsai/quent
