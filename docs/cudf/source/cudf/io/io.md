# Input / Output

This page contains Input / Output related APIs in cuDF.

## I/O Supported dtypes

The following table lists are compatible cudf types for each supported
IO format.

<div class="special-table-wrapper" style="overflow:auto">

```{eval-rst}
.. table::
    :class: io-supported-types-table special-table
    :widths: 15 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10

    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+-------------------+--------+--------+---------+---------+
    |                       |       CSV       |      Parquet    |       JSON       |       ORC       |  AVRO  |        HDF        |       DLPack    |      Feather      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | Data Type             | Writer | Reader | Writer | Reader | Writer¹ | Reader | Writer | Reader | Reader | Writer² | Reader² | Writer | Reader | Writer² | Reader² |
    +=======================+========+========+========+========+=========+========+========+========+========+=========+=========+========+========+=========+=========+
    | int8                  | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | int16                 | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | int32                 | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | int64                 | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | uint8                 | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ✅     | ❌     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | uint16                | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ✅     | ❌     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | uint32                | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ✅     | ❌     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | uint64                | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | float32               | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | float64               | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | bool                  | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ✅     | ✅     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | str                   | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | category              | ✅     | ❌     | ❌     | ❌     | ❌      | ❌     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | list                  | ❌     | ❌     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ❌     | ❌      | ❌      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | timedelta64[s]        | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | timedelta64[ms]       | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | timedelta64[us]       | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | timedelta64[ns]       | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ❌     | ❌     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | datetime64[s]         | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | datetime64[ms]        | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | datetime64[us]        | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | datetime64[ns]        | ✅     | ✅     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ✅     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | struct                | ❌     | ❌     | ✅     | ✅     | ✅      | ✅     | ✅     | ✅     | ❌     | ✅      | ✅      | ❌     | ❌     | ✅      | ✅      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | decimal32             | ✅     | ✅     | ✅     | ✅     | ✅      | ❌     | ✅     | ✅     | ❌     | ❌      | ❌      | ❌     | ❌     | ❌      | ❌      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | decimal64             | ✅     | ✅     | ✅     | ✅     | ✅      | ❌     | ✅     | ✅     | ❌     | ❌      | ❌      | ❌     | ❌     | ❌      | ❌      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
    | decimal128            | ✅     | ✅     | ✅     | ✅     | ✅      | ❌     | ✅     | ✅     | ❌     | ❌      | ❌      | ❌     | ❌     | ❌      | ❌      |
    +-----------------------+--------+--------+--------+--------+---------+--------+--------+--------+--------+---------+---------+--------+--------+---------+---------+
```

</div>

**Notes:**

- \[¹\] - Not all orientations are GPU-accelerated.
- \[²\] - Not GPU-accelerated.

## KvikIO Integration

cuDF leverages the [KvikIO](https://github.com/rapidsai/kvikio) library for high-performance
I/O features, such as parallel I/O operations and NVIDIA Magnum IO GPUDirect Storage (GDS).

Many IO APIs can use Magnum IO GPUDirect Storage (GDS) library to optimize
IO operations.  GDS enables a direct data path for direct memory access
(DMA) transfers between GPU memory and storage, which avoids a bounce
buffer through the CPU. The SDK is available for download
[here](https://developer.nvidia.com/gpudirect-storage). GDS is also
included in CUDA Toolkit.

Use of GDS in cuDF is controlled by KvikIO's environment variable `KVIKIO_COMPAT_MODE`. It has
3 options (case-insensitive):

- `ON` (aliases: `TRUE`, `YES`, `1`): Enable the compatibility mode, which enforces KvikIO POSIX I/O path.
  This is the default option in cuDF.
- `OFF` (aliases: `FALSE`, `NO`, `0`): Force-enable KvikIO cuFile (the underlying API for GDS) I/O path.
  GDS will be activated if the system requirements for cuFile are met and cuFile is properly
  configured. However, if the system is not suited for cuFile, I/O operations under the `OFF`
  option may error out.
- `AUTO`: Try KvikIO cuFile I/O path first, and fall back to KvikIO POSIX I/O if the system requirements
  for cuFile are not met.

Note that:
- Even if KvikIO cuFile I/O path is taken, it is possible that GDS is still not activated, where cuFile falls back
  to its internal compatibility mode. This will happen, for example, on an ext4 file system whose journaling
  mode has not been explicitly set to `data=ordered`. This may also happen if cuFile's environment variable
  `CUFILE_FORCE_COMPAT_MODE` is set to true. For more details, refer to
  [cuFile compatibility mode](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-compatibility-mode)
  and [cuFile environment variables](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#environment-variables).
- Details of the GDS system requirements can be found in the [GDS documentation](https://docs.nvidia.com/gpudirect-storage/index.html).
- If a KvikIO API call fails for any reason, cuDF throws an exception to propagate the error to the user.

For more information about error handling, compatibility mode, and
tuning parameters in KvikIO see: <https://github.com/rapidsai/kvikio>

### Remote file sizes and HEAD requests

When reading remote files (for example `s3://...` URLs) via `pylibcudf.io.SourceInfo`, KvikIO
may send HEAD requests at open time to probe connectivity and query file size. To skip those
requests when the file size is already known (for example from object-store metadata), pass a
`pylibcudf.io.FilepathSource` with the `size` argument set:

```python
import pylibcudf as plc

content_length = ...  # from external metadata
sources = plc.io.SourceInfo([
    plc.io.FilepathSource("s3://bucket/object.parquet", size=content_length),
])
table = plc.io.parquet.read_parquet(
    plc.io.parquet.ParquetReaderOptions.builder(sources).build()
)
```

Providing an incorrect size avoids the extra HEAD requests but will break footer reads and other
operations that depend on the true file length. Plain string paths in `SourceInfo` preserve the
previous behavior (size queried via KvikIO).

Operations that support the use of GPUDirect Storage:

- {py:func}`cudf.read_avro`
- {py:func}`cudf.read_json`
- {py:func}`cudf.read_parquet`
- {py:func}`cudf.read_orc`
- {py:meth}`cudf.DataFrame.to_csv`
- {py:func}`cudf.DataFrame.to_json`
- {py:meth}`cudf.DataFrame.to_parquet`
- {py:meth}`cudf.DataFrame.to_orc`

## nvCOMP Integration

Some types of compression/decompression can be performed using either
the [nvCOMP library](https://github.com/NVIDIA/nvcomp) or the internal
implementation.

Which implementation is used by default depends on the data format and
the compression type.  Behavior can be influenced through environment
variable `LIBCUDF_NVCOMP_POLICY`.

There are three valid values for the environment variable:

- "STABLE": Only enable the nvCOMP in places where it has been deemed
  stable for production use.
- "ALWAYS": Enable all available uses of nvCOMP, including new,
  experimental combinations.
- "OFF": Disable nvCOMP use whenever possible and use the internal
  implementations instead.

If no value is set, behavior will be the same as the "STABLE" option.

```{eval-rst}
.. table:: Current policy for nvCOMP use for different types
    :widths: 20 20 20 20 20 20 20 20 20 20

    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------------+
    |                       |       CSV       |            Parquet          |       JSON       |             ORC             |     AVRO     |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------------+
    | Compression Type      | Writer | Reader |    Writer    |    Reader    | Writer¹ | Reader |    Writer    |    Reader    |    Reader    |
    +=======================+========+========+==============+==============+=========+========+==============+==============+==============+
    | Snappy                | ❌     | ❌     | Stable       | Stable       | ❌      | ❌     | Stable       | Stable       | Stable       |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------------+
    | ZSTD                  | ❌     | ❌     | Stable       | Stable       | ❌      | ❌     | Stable       | Stable       | ❌           |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------------+
    | DEFLATE               | ❌     | ❌     | ❌           | ❌           | ❌      | ❌     | Stable       | Stable       | ❌           |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------------+
    | LZ4                   | ❌     | ❌     | Stable       | Stable       | ❌      | ❌     | Stable       | Stable       | ❌           |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------------+
    | GZIP                  | ❌     | ❌     | ❌           | Experimental | ❌      | ❌     | ❌           | ❌           | ❌           |
    +-----------------------+--------+--------+--------------+--------------+---------+--------+--------------+--------------+--------------+

```

## Low Memory Considerations

By default, cuDF's parquet and json readers will try to read the entire file in one pass. This can cause problems when dealing with large datasets or when running workloads on GPUs with limited memory.

To better support low memory systems, cuDF provides a "low-memory" reader for parquet and json files. This low memory reader processes data in chunks, leading to lower peak memory usage due to the smaller size of intermediate allocations.

To read a parquet or json file in low memory mode, there are [cuDF options](https://docs.rapids.ai/api/cudf/nightly/cudf/api_docs/options/#api-options) that must be set globally prior to calling the reader. To set those options, call:
- `cudf.set_option("io.parquet.low_memory", True)` for parquet files, or
- `cudf.set_option("io.json.low_memory", True)` for json files.
