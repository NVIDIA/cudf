# pylibcudf regex-search example

This example demonstrates a GPU regex-search pipeline built directly with
pylibcudf:

- read a plain-text file as one GPU string per line;
- compile and run a libcudf regular expression through pylibcudf;
- report end-to-end and resident-data scan throughput; and
- optionally compare POSIX I/O with GPUDirect Storage in isolated processes.

Run it in a current RAPIDS GPU environment:

```bash
python python/pylibcudf/examples/regex_search.py big.log ERROR
```

When a trusted reference count is available, enforce correctness as part of
the benchmark:

```bash
python python/pylibcudf/examples/regex_search.py big.log ERROR \
  --expected-matches 14757803
```

Use `--gds-mode both` to request a GDS comparison. This deliberately starts two
child processes. KvikIO reads its compatibility mode during process
initialization, so changing the environment variable after pylibcudf has loaded
would not provide a trustworthy comparison.
The parent also verifies that both modes report identical byte, line, and match
counts before accepting the measurements.

The GDS-on path requires the KvikIO Python package and queries cuFile's
`is_gds_available` and `allow_compat_mode` runtime properties. It requires
native GDS to be available and compatibility fallback to be disabled, so an
unsupported target file fails instead of producing a misleading GDS result. A
successful GDS-on result therefore requires a host and target filesystem with
native GDS configured; installing the user-space cuFile library alone is not
sufficient.

Each result records the installed pylibcudf version. Preserve that field, the
GPU model, driver version, input-generation command, and filesystem/storage
configuration when publishing results so another contributor can reproduce the
comparison.

The timings use the best synchronized result from the requested repeats. The
initial correctness pass also warms the file cache and resident-data scan, so
record that behavior when comparing the output with another tool.

This is a focused performance example, not a replacement for grep or ripgrep.
