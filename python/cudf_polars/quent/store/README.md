# cudf-polars-quent Store

Shared event types generated from the [model](../model.yaml). The analyzer and
browser crates depend on this crate.

`build.rs` also publishes `SCHEMA_DIGEST`, a SHA-256 digest of `model.yaml` in
the `sha256:<hex>` form used by Quent's simulator WASM crate. Bundle writers
and the browser analyzer both read that constant.
