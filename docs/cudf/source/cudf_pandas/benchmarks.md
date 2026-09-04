# Benchmarks

## PDS-H benchmarks

We ran the PDS-H benchmark (a TPC-H variant) to compare `cudf.pandas` on GPU
with pandas on CPU. Here are the results at scale factor 50 (roughly 50 GB of
data):

<figure>

![pdsh-sf50](../_static/cudf_pandas_pdsh_sf50.png)

<figcaption style="text-align: center;">PDS-H (SF50) cumulative runtime for pandas on CPU vs <span
class="title-ref">cudf.pandas</span> on GPU.</figcaption>
</figure>

The steps below reproduce these results.

### Setup

Install `cudf` following the
[RAPIDS installation guide](https://docs.rapids.ai/install/). For nightly wheels:

```bash
CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ \
    "cudf-cu${CUDA_MAJOR}>=0.0.0a0"
```

Then install `tpchgen-cli`, a Rust-based TPC-H data generator used to produce the benchmark
dataset as Parquet files:

```bash
pip install tpchgen-cli
```

### Generate data

Set the scale factor once and reuse it across all steps. The following generates SF50
(scale factor 50, roughly 50GB of data):

```bash
export SCALE_FACTOR=50.0
export DATA_PATH="data/tables/scale-${SCALE_FACTOR}"

tpchgen-cli parquet -o "${DATA_PATH}" -s ${SCALE_FACTOR}
```

`tpchgen-cli` generates Decimal and `datetime.date` columns. pandas cannot use these types
in arithmetic, so convert them to float64 and timestamp before running the benchmark. This
conversion step may not be needed in the future (see [#21204](https://github.com/NVIDIA/cudf/issues/21204)).

```python
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import os

data_path = Path(os.environ["DATA_PATH"])
tables = ["lineitem", "orders", "customer", "supplier", "part", "partsupp", "nation", "region"]

def cast_schema(schema):
    return pa.schema(
        f.with_type(pa.float64()) if pa.types.is_decimal(f.type)
        else f.with_type(pa.timestamp("ms")) if pa.types.is_date(f.type)
        else f
        for f in schema
    )

for table in tables:
    table_path = data_path / f"{table}.parquet"
    parts = [table_path] if table_path.is_file() else sorted(table_path.glob("*.parquet"))
    for part in parts:
        tbl = pq.read_table(part, schema=cast_schema(pq.read_schema(part)))
        pq.write_table(tbl, part)
```

### Run

**CPU** (`--executor cpu`, pandas):

```bash
python -m cudf.pandas._benchmarks.pdsh all \
    --executor cpu \
    --path "${DATA_PATH}"
```

**GPU** (`--executor in-memory`, cudf.pandas):

```bash
python -m cudf.pandas._benchmarks.pdsh all \
    --executor in-memory \
    --path "${DATA_PATH}"
```

### Results

Results are written to `pdsh_results.jsonl` in the current directory by default (override with `-o`).
Each run appends one JSON line containing metadata and a `records` field with per-query,
per-iteration timings:

```json
{
  "query_set": "pdsh",
  "executor": "in-memory",
  "dataset_path": "data/tables/scale-50.0",
  "scale_factor": 50,
  "records": {
    "1": [
      {"query": 1, "iteration": 0, "duration": 0.79, "status": "success"},
      {"query": 1, "iteration": 1, "duration": 0.55, "status": "success"}
    ]
  }
}
```

`duration` is in seconds. Running multiple executors with the same `-o` file appends each as a
separate line, making it easy to compare CPU and GPU results in one file.
