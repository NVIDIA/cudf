# Prepared pack examples

This example demonstrates the non-chunked prepared-pack use cases requested by shuffle and spill
callers:

- uncompressed output written directly into mapped pinned-host memory;
- asynchronous compressed spill using reserved capacity;
- compact compressed output suitable for shuffle framing;
- restoring compressed host data into an owning GPU table;
- zero-copy reconstruction of an uncompressed device-resident payload; and
- reuse of one plan with a second destination for the same unchanged input batch.

The examples are job-local. They do not define a durable storage format or cross-version wire
compatibility.

From `cpp/examples`, build all examples with:

```bash
./build.sh
```

Or configure only this example with:

```bash
cmake -S pack -B pack/build -Dcudf_ROOT=../build
cmake --build pack/build
```
