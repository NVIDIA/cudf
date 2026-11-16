# Environment variables

These variables configure libcudf's runtime context and affect applications using the C++, Python, and Java interfaces.
They are read when the context is first initialized. Set them before initializing libcudf; changing the environment afterward does not reconfigure the existing context.

Boolean settings default to disabled. The recognized true values are `ON`, `on`, `1`, `true`, `TRUE`, and `True`.
Other parsed values disable them; use `OFF` or `0` to disable a setting explicitly.

## JIT compilation

| Variable | Default | Description |
| --- | --- | --- |
| `LIBCUDF_JIT_ENABLED` | `OFF` | Select JIT execution for APIs with a runtime-selectable JIT path, including AST column computation and Parquet filter evaluation. APIs explicitly requesting JIT execution do not require this setting. |
| `LIBCUDF_JIT_PCH_HEAP_SIZE_MB` | `512` MiB | NVRTC precompiled-header heap size, expressed as a positive decimal integer in MiB. |

`LIBCUDF_JIT_PCH_HEAP_SIZE_MB` must be non-zero, context initialization fails when the size is too large.

## Kernel cache

| Variable | Default | Description |
| --- | --- | --- |
| `LIBCUDF_KERNEL_CACHE_PATH` | Selected as described below | Root directory for cached compiled code, extracted source bundles, and temporary compilation files. |
| `LIBCUDF_KERNEL_CACHE_PRELOAD` | `OFF` | Preload existing compiled blobs and libraries from the disk cache during context initialization, subject to the in-memory cache limits. This moves loading work to initialization; it does not compile kernels that are absent from the cache. |
| `LIBCUDF_KERNEL_CACHE_DISABLED` | `OFF` | Bypass libcudf's in-memory and on-disk compiled-code cache lookups, forcing compilation or linking instead of reusing those entries. This does not disable NVRTC PCH reuse or CUDA's separate compilation caches. |
| `LIBCUDF_KERNEL_CACHE_CLEAR` | `OFF` | Clear the in-memory and on-disk compiled-code caches during initialization. Later operations can populate the cache again. Clearing occurs after any requested preload. |
| `LIBCUDF_KERNEL_CACHE_LIMIT_PER_PROCESS` | `16384` | Maximum number of compiled blobs and maximum number of loaded libraries retained in memory per process. The same unsigned integer limit applies independently to both caches. It is an entry count, not a byte limit or an on-disk size limit. |
| `LIBCUDF_JIT_DISABLE_CUDA_CACHE` | `OFF` | Pass `--no-cache` to NVRTC and nvJitLink to disable their compilation caches. This is separate from libcudf's compiled-code cache and does not turn off PCH reuse. |

Disabling libcudf cache lookups does not prevent newly compiled artifacts from being written to disk.

### Cache directory selection

libcudf tries these locations in order and uses the first directory it can create
or access with read, write, and execute permissions:

1. `$LIBCUDF_KERNEL_CACHE_PATH`, if set.
2. `$XDG_CACHE_HOME/libcudf`, if `XDG_CACHE_HOME` is set.
3. `$HOME/.cache/libcudf`, if `HOME` is set and names an accessible directory.
4. `$TMPDIR/libcudf`, if `TMPDIR` is set and names an accessible directory.
5. `/tmp/libcudf`.

If an explicit `LIBCUDF_KERNEL_CACHE_PATH` cannot be used, libcudf logs a warning and tries the remaining locations. Initialization fails if none can be used.
The root contains `bundle` for extracted source bundles, `rtcx_cache` for compiled artifacts, and `tmp` for temporary files. A cache directory is still required when compiled-code cache lookups are disabled.

## Diagnostics

| Variable | Default | Description |
| --- | --- | --- |
| `LIBCUDF_JIT_DUMP_CODEGEN` | `OFF` | Print CUDA source generated from row IR for transform execution to standard output. |
| `LIBCUDF_JIT_VERBOSE` | `OFF` | Enable verbose JIT compiler/linker diagnostics. NVRTC warnings are enabled, as are PCH messages on versions supporting PCH. |
| `LIBCUDF_JIT_DUMP_TRACE` | `OFF` | Request compiler/linker timing output: `--time=-` for NVRTC and `-time` for nvJitLink. |
| `LIBCUDF_JIT_DUMP_TIME_PROFILE` | `OFF` | Request NVRTC device compilation time traces using `--fdevice-time-trace=cudf_kernel_<name>_trace`, where `<name>` is the compilation name. Requires an NVRTC version supporting that option. |

Compiler and linker diagnostics are generated only when compilation or linking occurs. A libcudf cache hit can therefore bypass the work that produces them.

## Compression library initialization

| Variable | Default | Description |
| --- | --- | --- |
| `LIBCUDF_NVCOMP_PRELOAD` | `OFF` | Load nvCOMP during context initialization instead of waiting for a later operation to need it. Explicit initialization flags can also request this preload. |
