# cuDF Java Deployment Modes Example

This example compares two ways of making the cuDF Java native libraries
available when a JVM starts:

- Default JAR extraction: `NativeDepsLoader` extracts the bundled `.so` files
  from the cuDF JAR into temporary files before loading them.
- Pre-unpacked library directory: the application starts with
  `-Dai.rapids.cudf.lib-native-dir=<dir>`, so `NativeDepsLoader` loads the
  `.so` files directly from a directory and skips per-process JAR extraction.

The example runs a tiny cuDF workload in each mode and prints the loader timing
summary from `NativeDepsLoader` plus the wall-clock time measured by the driver.

## Prerequisites

Build and install the matching cuDF Java artifact into your local Maven
repository before running the example. The driver locates the installed
`ai.rapids:cudf` JAR under `~/.m2/repository` by default, or under
`$MAVEN_REPO` if that environment variable is set.

## Running

From this directory:

```bash
./run_examples.sh
```

The script builds the example JAR if needed, then runs:

1. `Scenario 1: Default JAR extraction`
2. `Scenario 2: Pre-unpacked library directory`

Both scenarios set:

```bash
-Dai.rapids.cudf.lib-log-load-timing=true
```

so the native library extraction and load timings are visible in the output.

To keep the pre-unpacked libraries after the second scenario, pass `-k`:

```bash
./run_examples.sh --keep-unpacked-libs
```

This leaves `unpacked-libs/` in place so a later run can reuse the same files.
Without `-k`, the driver removes that directory after scenario 2.

## Interpreting the Output

The workload prints a line like:

```text
Workload Result: rows=5  first_call_ms=123
```

`first_call_ms` covers the first cuDF Java call in that JVM, which is where
native dependency loading happens. The driver also prints a `Total Wall Time`
line for each scenario.

In the default extraction scenario, the loader timing output includes the cost
of extracting native libraries from the JAR. In the pre-unpacked scenario, the
loader reports that extraction was skipped and loads libraries from the
configured `lib-native-dir`.

## Using the Pre-unpacked Mode

For an application or container image, pre-unpack the cuDF native `.so` files
for the target platform into a directory, then start the JVM with:

```bash
-Dai.rapids.cudf.lib-native-dir=/path/to/cudf/native/libs
```

The directory must contain every native library requested by cuDF Java. The
loader validates the directory up front and leaves those files in place; it
does not delete application-managed pre-unpacked libraries.
