# Build Jar artifact of cuDF

## Recommended: self-contained release build scripts

The scripts under `java/ci/` build the cuDF Java JAR for every Maven classifier the
same way locally and in CI (GitHub Actions is only a thin wrapper that adds
artifact upload/download). Each script pulls the RAPIDS `ci-conda` build image,
runs the build in a throwaway container, and writes its output to a host
directory. No local `docker build` is required, and no GPU is required to build.

### Prerequisites

1. Docker is installed and the current user can run `docker`.
2. Network access to pull `rapidsai/ci-conda:<rapids_version>-latest`.

### Local one-command shortcut

For local testing only, `java/ci/test_java_build_local.sh` runs Steps 1-3 end-to-end for both CUDA 12 and CUDA 13 on the host architecture.

```bash
./java/ci/test_java_build_local.sh --work-dir /tmp/java-build-test
```

### Step 1 - Build the static libcudf install tree

```bash
./java/ci/build_static_libcudf.sh --output-dir /tmp/libcudf-cuda12 --cuda-version 12.9
```

This produces a static libcudf install tree (`lib/libcudf.a` plus its static
dependencies) under the given output directory. Build outputs are host-user-owned
so plain `rm -rf` works.

### Step 2 - Package the cuDF Java JAR for one classifier

```bash
./java/ci/build_cudf_java_jar.sh \
  --libcudf-dir /tmp/libcudf-cuda12 \
  --output-dir /tmp/jars \
  --cuda-version 12.9
```

This compiles the JNI layer against the static libcudf from Step 1 and emits a
single classifier JAR (e.g. `cudf-26.10.0-SNAPSHOT-cuda12.jar`) plus its POM
into a classifier-named subdirectory under `--output-dir`:

```
/tmp/jars/cuda12/
    cudf-26.10.0-SNAPSHOT-cuda12.jar
    cudf-26.10.0-SNAPSHOT.pom
```

The classifier is derived from `--cuda-version` (major) + host arch (`uname
-m`): `cuda12` / `cuda13` on `x86_64`, `cuda12-arm64` / `cuda13-arm64` on
`aarch64`. Producing the ARM classifiers requires a real `aarch64` host.
Repeat Step 2 for each classifier, pointing `--libcudf-dir` at the matching
static libcudf tree and using the same `--output-dir` (each classifier lands
in its own subdirectory). Concurrent invocations for different classifiers
are safe because each nests its own bind-mount over `/repo/java/target`
inside the container.

### Step 3 - Assemble the Maven repository layout

```bash
./java/ci/assemble_maven_repo.sh \
  --jars-dir /tmp/jars \
  --output-dir /tmp/maven-repo
```

This walks every subdirectory of `--jars-dir` (each subdir name IS the
classifier), gathers the per-classifier JAR and shared POM, derives the
artifact version from the JAR filenames (requiring a single unique version
across subdirs), and lays them out as:

```
/tmp/maven-repo/ai/rapids/cudf/26.10.0-SNAPSHOT/
    cudf-26.10.0-SNAPSHOT-cuda12.jar
    cudf-26.10.0-SNAPSHOT-cuda13.jar
    cudf-26.10.0-SNAPSHOT.pom
```

The set of classifiers is whatever subdirectories are present under
`--jars-dir`. For a local `x86_64`-only run, populate `/tmp/jars/cuda12/`
and `/tmp/jars/cuda13/`. For the full four-way release build, add
`/tmp/jars/cuda12-arm64/` and `/tmp/jars/cuda13-arm64/`.

In GitHub Actions (`.github/workflows/build.yaml`), the `java-build` matrix job
runs Steps 1-2 per (CUDA x arch) entry and uploads each classifier subdir as a
per-entry artifact. The separate `java-gather` job downloads them (with
`merge-multiple: true`, so all subdirs land in a single parent dir), runs
Step 3, and uploads the combined `cudf_java_maven_repo` artifact.

## Legacy: manual Dockerfile.rocky build (obsolete)

> The `java/ci/Dockerfile.rocky` + `java/ci/build-in-docker.sh` flow below is the
> old build path. It is retained for reference but superseded by the
> self-contained scripts above.

### Build the docker image

In the root path of cuDF repo, run below command to build the docker image.
```bash
docker build -f java/ci/Dockerfile.rocky --build-arg CUDA_VERSION=12.9.1 -t cudf-build:12.9.1-devel-rocky8 .
```

The following CUDA versions are supported w/ CUDA Enhanced Compatibility:
* CUDA 12.2+

Change the --build-arg CUDA_VERSION to what you need.
You can replace the tag "cudf-build:12.9.1-devel-rocky8" with another name you like.

### Start the docker then build

Run below command to start a docker container with GPU.
```bash
nvidia-docker run -it cudf-build:12.9.1-devel-rocky8 bash
```

You can download the cuDF repo in the docker container or you can mount it into the container.
Here I choose to download again in the container.
```bash
git clone --recursive https://github.com/rapidsai/cudf.git -b main
```

```bash
cd cudf
export WORKSPACE=`pwd`
source java/ci/env.sh
${sclCMD} "java/ci/build-in-docker.sh"
```

### The output

You can find the cuDF jar in java/target/ like cudf-26.10.0-SNAPSHOT-cuda12.jar.
