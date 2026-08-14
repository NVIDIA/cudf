# cudf-java smoke tests

Smoke published `ai.rapids:cudf` classifier JARs under a Maven Central-style
classpath (cudf + slf4j only).

## Quick start

Exactly one source mode is required.

```bash
# Local gather tree (must contain ai/rapids/cudf/)
./java/smoke-tests/bin/run.sh --maven-repo-dir /tmp/cudf-java-build/maven-repo

# GitHub Actions gather artifact
./java/smoke-tests/bin/run.sh \
  --artifact-url 'https://github.com/rapidsai/cudf/actions/runs/<run>/artifacts/<id>'

# Installed into ~/.m2
./java/smoke-tests/bin/run.sh --use-maven-home

# Maven Central release (version required)
./java/smoke-tests/bin/run.sh --use-maven-central --version 25.12.0
```

Narrow to one CUDA major:

```bash
./java/smoke-tests/bin/run.sh --maven-repo-dir /tmp/maven-repo --cuda-version 12
```

## Source modes

| Mode | Flag |
|---|---|
| GitHub Actions artifact | `--artifact-url URL` |
| Local Maven-repo tree | `--maven-repo-dir DIR` |
| `~/.m2/repository` | `--use-maven-home` |
| Maven Central | `--use-maven-central` (requires `--version`) |

Shared options:

| Option | Meaning |
|---|---|
| `--version VER` | Pin `ai.rapids:cudf` version. Required for Central. For other modes, if omitted, exactly one version directory must exist under `ai/rapids/cudf/`. |
| `--cuda-version 12\|13` | Narrow classifiers to this CUDA major |

## Classifier selection

Without `--cuda-version`, by host arch:

- `x86_64`: `unclassified`, `cuda12`, `cuda13`
- `aarch64`: `cuda12-arm64`, `cuda13-arm64`

With `--cuda-version N`:

- `x86_64` + 12: `unclassified`, `cuda12`
- `x86_64` + 13: `cuda13`
- `aarch64` + N: `cudaN-arm64`

Missing JARs are warned and skipped. The run fails if no classifier is actually smoke tested, or if any smoke-tested classifier fails.

After each classifier run, `run.sh` also requires the native `CUDF_LOG_WARN` from
`parquetChunkedLoggerSmoke` (`Chunked Parquet reader: a chunk_read_limit`) in the
captured smoke log under `.cache/logs/`.

## Artifact downloads

`--artifact-url` uses [`bin/fetch_bundle.sh`](bin/fetch_bundle.sh) (requires authenticated `gh`).
Downloads land at:

```text
java/smoke-tests/.cache/downloads/runs/<run_id>/artifacts/<artifact_id>/
```

(or under `$CUDF_JAVA_SMOKE_OUTPUT_DIR`). If that destination already exists and
contains `ai/rapids/cudf/`, the fetch warns and reuses it.

## Docker images

`docker/Dockerfile.cuda{12,13}` (Ubuntu 24.04 CUDA runtime + OpenJDK 17 + Maven).
Built on first use; rebuild with `docker build` / `docker rmi` if the Dockerfile changes.
