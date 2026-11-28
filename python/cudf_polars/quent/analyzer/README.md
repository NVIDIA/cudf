# cudf-polars Quent analyzer

This crate is the entrypoint for `quent-open`. Event types come from
[`cudf-polars-quent-store`](../store), which generates them from the same
[model schema](../model.yaml) that cudf-polars uses for exporting events.

`cudf-polars-quent-analyzer` is typically built on demand by `quent-open` using
the versions embedded in the archive's `model.qmi`. Check the local contents with:

```sh
cargo check
cargo test
```

Quent upgrades must update every Quent dependency's `rev` in the workspace
[`Cargo.toml`](../Cargo.toml) to the same full commit SHA, then refresh and
commit the workspace `Cargo.lock`. See the bridge's
[Updating Quent](../bridge/README.md#updating-quent) section for the complete
update and verification checklist.

## Project structure

- `lib.rs` re-exports the store's generated event types and the analyzer
  entrypoint. The `filesystem` feature adds `viewer.rs`, which `quent-open`
  uses to import events from disk. The WASM build leaves that feature off.
- `model.rs` builds the query-engine model (the struct `CudfPolarsModel` and the
   traits implements).
- `actor.rs`, `evaluate.rs`, `memory_sample.rs`, and `resource.rs` reconstruct
   execution spans and resource usage. `memory_sample.rs` turns the `sampling`
   transitions of each `MemoryResourceSampler` FSM into `SampledMemoryResource`
   occupancy usages.
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
