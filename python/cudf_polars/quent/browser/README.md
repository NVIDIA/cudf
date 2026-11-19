# cudf-polars Quent browser analyzer

This crate exposes the cudf-polars analyzer through WASM. The optional
`generate` feature adds native filesystem support for creating browser bundles;
the [`bridge`](../bridge) crate enables this feature so Python can write
`.quent-browser` files.

Benchmark runs using `--collect-traces` write both artifacts automatically.
The ZIP remains the archival `quent-open` input. The `.quent-browser` file is
the relocatable static-viewer payload and contains exactly one engine with all
of its controller and worker contexts. Python callers generate that payload
with `cudf_polars.quent._export.write_quent_browser_bundle`.

Build the analyzer:

```bash
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen \
  target/wasm32-unknown-unknown/release/cudf_polars_quent_browser.wasm \
  --target web \
  --out-dir generated
```

The `wasm-bindgen` CLI must match the `wasm-bindgen` crate version selected by
`Cargo.lock`.

`cargo metadata --no-deps` prints `target_directory` when `CARGO_TARGET_DIR`
is set and that path differs from `target/`.

The generated JavaScript module exports Quent's standardized `BrowserServer`
interface, so no cudf-polars-specific Web Worker is required. Host the generated
JavaScript and WASM files on the same origin as the Quent UI, then select them
with the static viewer URL using a querystring like:

```text
?module=/analyzers/cudf-polars/cudf_polars_quent_browser.js&data=https://traces.example/run.quent-browser#/
```
