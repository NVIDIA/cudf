# cudf-polars Quent analyzer

This crate is the entrypoint for `quent-open`. [./build.rs](./build.rs)
generates typed event importers from the same [Model Schema](../model.yaml) that
cudf-polars uses for generating and exporting events.

`cudf-polars-quent-analyzer` is typically built on demand by `quent-open` using
the versions embedded in the archive's `model.qmi`. Check the local contents with:

```sh
cargo check
cargo test
```

Quent upgrades must update every `rev` in this crate and in
[`../bridge/Cargo.toml`](../bridge/Cargo.toml) to the same full commit SHA, then
refresh and commit both Cargo lockfiles. See the bridge's [Updating
Quent](../bridge/README.md#updating-quent) section for the complete update and
verification checklist.

## Project structure

- `build.rs` generates Rust event types from `../model.yaml`; `lib.rs` exposes
  those types and the analyzer entrypoint.
- `model.rs` builds the query-engine model (the struct `CudfPolarsModel` and the
   traits implements).
- `actor.rs`, `evaluate.rs`, and `resource.rs` reconstruct execution spans and
   resource usage.
- `analyzer/` converts imported events into Quent UI query bundles, entity
  lists, and resource timelines:
  - `construction.rs` assembles the model, execution spans, and declared
    resources from the event stream.
  - `query_bundle.rs` builds the query, plan, operator, and resource views used
    by the UI.
  - `entities.rs` implements filtered and paginated entity lists.
  - `timeline.rs` builds individual and bulk resource timelines.
  - `mod.rs` connects these operations through the Quent analyzer interface.
- `viewer.rs` discovers contexts and imports their event streams; `tests.rs`
  exercises the analyzer end to end.
