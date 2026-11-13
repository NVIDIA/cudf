# Benchmarks

<!-- TODO: add PDS-DS (TPC-DS variant) section, mirroring the PDS-H instructions below -->

## PDS-H (TPC-H variant)

The steps below reproduce the PDS-H benchmark results using the Polars GPU engine.

### Setup

Install `cudf-polars` following the
[NVIDIA CUDA-X installation guide](https://docs.nvidia.com/datascience/install/#install-rapids). For nightly wheels, install with
the `ray` extra (required for multi-GPU benchmarking):

```bash
CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ \
    "cudf-polars-cu${CUDA_MAJOR}[ray]>=0.0.0a0"
```

Because `cudf-polars` pins to a tested range of Polars versions, the nightly wheel will install
the highest Polars version the GPU engine currently supports, which may not be the latest
Polars release.

<!-- TODO: consider adding a [benchmark] pip extra to cudf-polars that includes tpchgen-cli
     so benchmark dependencies can be installed in one step:
     pip install "cudf-polars-cu${CUDA_MAJOR}[ray,benchmark]>=0.0.0a0"
     Requires changes to pyproject.toml and dependencies.yaml. -->

Then install `tpchgen-cli`, a Rust-based TPC-H data generator used to produce the benchmark
dataset as Parquet files:

```bash
pip install tpchgen-cli
```

### Generate data

Set the scale factor once and reuse it across all steps. The following generates SF1000
(scale factor 1000, roughly 1TB of data):

```bash
export SCALE_FACTOR=1000.0
export DATA_PATH="data/tables/scale-${SCALE_FACTOR}"

tpchgen-cli --output-dir="${DATA_PATH}" --format=parquet -s ${SCALE_FACTOR}
```

### Run

**CPU** (`--frontend polars-cpu`, Polars CPU streaming engine):

```bash
python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend polars-cpu \
    --path "${DATA_PATH}"
```

**Single GPU** (`--frontend spmd`, single-process streaming executor, equivalent to `collect(engine="gpu")`):

```bash
python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend spmd \
    --path "${DATA_PATH}"
```

**Multi GPU** (`--frontend ray`, Ray-managed distributed streaming executor):

If running inside a Docker container, increase `/dev/shm` by passing `--shm-size=16g` to
`docker run`. All multi-GPU frontends use UCX for intra-node communication, which relies on
POSIX shared memory (`/dev/shm`) for GPU-to-GPU transfers. Docker's default `/dev/shm` is
64MB, which is far too small and will cause failures on any non-trivial workload.

By default all visible GPUs are used. To select specific devices, set `CUDA_VISIBLE_DEVICES`.
To limit the number of GPUs, use `--num-gpus`:

```bash
# All visible GPUs
python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend ray \
    --path "${DATA_PATH}"

# Specific devices
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend ray \
    --path "${DATA_PATH}"

# Limit to N GPUs
python -m cudf_polars.streaming.benchmarks.pdsh all \
    --frontend ray \
    --num-gpus 4 \
    --path "${DATA_PATH}"
```

### Results

Results are written to `pdsh_results.jsonl` in the current directory by default (override with `-o`).
Each run appends one JSON line using the version-2 benchmark interchange format. Query
timings use the same field names and millisecond units as common benchmark services, but
the runner is standalone: it neither requires nor contacts a service.

```json
{
  "schema_version": 2,
  "run_id": "d45bf21e-c350-4927-80a0-855c888f57cc",
  "run_at": "2026-11-10T18:00:00+00:00",
  "query_engine": {
    "engine_name": "cudf-polars",
    "version": "26.12.0",
    "commit_hash": "abc123"
  },
  "query_logs": [
    {
      "query_name": "1",
      "execution_order": 0,
      "status": "success",
      "runtime_ms": 790.0,
      "extra_info": {"iteration": 0}
    },
    {
      "query_name": "1",
      "execution_order": 1,
      "status": "success",
      "runtime_ms": 550.0,
      "extra_info": {"iteration": 1}
    }
  ],
  "extra_info": {
    "benchmark": {
      "frontend": "spmd",
      "dataset_path": "data/tables/scale-1000.0",
      "scale_factor": 1000
    }
  }
}
```

Runner-specific metadata is retained under namespaced `extra_info` entries, and generated
profiles are described in the `artifacts` manifest. Optional catalog arguments such as
`--sku-name`, `--storage-configuration-name`, and `--benchmark-definition-name` add matching
top-level fields when supplied; they are never required to run a benchmark. Running multiple
frontends with the same `-o` file appends each as a separate line.

### Tuning

The commands above use default settings, which gives a realistic baseline without manual tuning. The most impactful options to adjust are:

| Option | Description |
|--------|-------------|
| `--target-partition-size` | Target IO chunk size in bytes fed to the GPU. The most impactful lever; tune this first if query performance is below expectations. Default: `min(2.5% of smallest GPU memory, 1.5GB)`. |
| `--broadcast-limit` | Maximum table size in bytes for broadcast joins instead of shuffle. Increasing this can significantly speed up join-heavy queries. Default: `min(15% of smallest GPU memory, 16GB)`. |
| `--spill-device-limit` | GPU memory usage percentage before spilling to host. Lower this if hitting out-of-memory errors. Default: `80%`. |
| `--pinned-memory` / `--pinned-initial-pool-size` | Enable a pinned host memory pool for faster CPU-to-GPU transfers. On by default for the `spmd`, `ray`, and `dask` frontends on systems that support it; pass `--no-pinned-memory` to disable. When enabled, the pool starts empty and grows up to 80% of host memory per GPU; set `--pinned-initial-pool-size` (bytes) to pre-allocate capacity upfront. |
