# cudf-polars Quent bindings

This package builds the top-level `cudf_polars_quent` extension. The
distribution is named `cudf-polars-quent` and is an *optional* dependency for
cudf-polars.

It uses [Quent] to define a schema for cudf-polars execution and telemetry
model. A Python instrumentation library is generated from this model definition
(via generated Rust code).

The main cudf-polars package imports `cudf_polars_quent` only when telemetry
collection is enabled. Keeping the extension at the top level lets the optional
distribution own its complete import namespace and allows the generated
bindings to be imported without first initializing `cudf_polars`.

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
python -c "import cudf_polars_quent"
```

## Updating Quent

Quent is pinned by full Git commit SHA in the `Cargo.toml` for both
the `bridge` and `analyzer`. To update Quent:

1. Replace every Quent dependency's `rev` in both manifests with the same full
   commit SHA. Do not use a branch, tag, abbreviated SHA, or different revision
   spelling: Cargo must resolve one package identity for the generated model,
   analyzer, and `quent-open` viewer traits.
2. Refresh the Cargo lockfile.

   ```sh
   # From python/cudf_polars/quent/bridge
   cargo update
   ```

   Review the then commit the changes (including the lockfiles).
3. Activate the cudf development environment and rebuild from the repository
   root:

   ```sh
   ./build.sh cudf_polars_quent
   ```

   The build script generates the Rust instrumentation library from the
   `model.yaml` and Python bindings to the Rust instrumentation library. The
   build also outputs the checked-in
   `python/cudf_polars/quent/bridge/cudf_polars_quent.pyi` type stub. Maturin
   packages that stub alongside the extension so editors and static type
   checkers can use the generated API from the installed distribution.


4. Run the checks at `ci/run_cudf_polars_quent_tests.sh`.
5. Commit and push the analyzer changes to the Git remote recorded in the
   bridge's build provenance. Rebuild the bridge after committing, then
   regenerate traces. `quent-open` checks out the analyzer package at the exact
   remote and commit embedded in `model.qmi`; a local-only commit or a trace
   generated from the previous commit cannot be loaded elsewhere.

New trace archives embed the updated Quent Git provenance in `model.qmi`.
Existing archives retain the revision with which they were generated.


[maturin]: https://www.maturin.rs/installation.html
[Quent]: https://github.com/rapidsai/quent
