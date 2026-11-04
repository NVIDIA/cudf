# cuDF for Java

This project provides Java bindings for NVIDIA cuDF, a library for GPU-accelerated structured
data processing.

The API reference is published with the rest of the cuDF documentation at
https://docs.nvidia.com/cudf/latest/java/.

## Behavior on Systems with Multiple GPUs

The Java bindings work with a single GPU per process. The CUDA runtime assigns the default GPU to
each new operating system thread when the thread first interacts with CUDA. In a multi-threaded
environment, which is typical for Java processes, using a GPU other than the default one can lead to
hard-to-debug problems, including resource leaks, invalid states, application crashes and poor
performance.

To prevent this, the Java bindings remember the device that was used to initialize NVIDIA RMM and,
when needed, automatically set the active device of the calling thread to that device. They do not
restore the previous device when the cuDF call completes. This differs from most CUDA libraries and
can cause unexpected behavior if you mix them with the Java bindings on the same thread.

## Dependency

cuDF for Java is published to Maven Central as [`ai.rapids:cudf`](https://central.sonatype.com/artifact/ai.rapids/cudf):

```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <version>${cudf.version}</version>
</dependency>
```

Replace `${cudf.version}` with a published cuDF version, or define the `cudf.version` property in
your POM.

The jar bundles its native libraries, so it only runs on the platforms it was built for. Each
release includes a jar for each supported CUDA major version and CPU architecture, selected with a
classifier:

| Classifier     | CUDA version | Minimum driver | Architecture |
|----------------|--------------|----------------|--------------|
| `cuda12`       | 12.2+        | 535            | x86_64       |
| `cuda12-arm64` | 12.2+        | 535            | aarch64      |
| `cuda13`       | 13.0+        | 580            | x86_64       |
| `cuda13-arm64` | 13.0+        | 580            | aarch64      |

See [Platform Support](https://docs.nvidia.com/datascience/platform-support/) for the CUDA toolkit
versions, drivers and GPU architectures supported by each release.

The jar without a classifier is the same as the `cuda12` jar. For example, to use the CUDA 13 jar
on x86_64:

```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <classifier>cuda13</classifier>
    <version>${cudf.version}</version>
</dependency>
```

## Build From Source

First build [libcudf](../cpp), and make sure a JDK is installed and available. Pass the following
CMake option to the libcudf build:

```
-DCUDF_LARGE_STRINGS_DISABLED=ON
```

This disables large string support, which the Java bindings do not support yet. See
https://github.com/NVIDIA/cudf/issues/16215.

After building libcudf, build the Java bindings with Maven:

```shell
cd java
mvn clean install
```

The jar gets a classifier based on the CUDA version reported by `nvcc` and the host architecture,
as listed in the [Dependency](#dependency) section.

If your build system has a compatible GPU, the tests use it. Otherwise, many tests are skipped.

### Building Jars Like the Official Releases

A jar built directly on your machine will likely only run in environments similar to the build
machine. To build a jar that is similar to the official releases and runs on all modern Linux
systems, use the scripts described in the [Java CI README](ci/README.md). They build libcudf and
the jar inside the same container image that CI uses, and do not require a GPU.

If you build without these scripts and run into problems, the CMake and Maven settings in
[build_static_libcudf_in_container.sh](ci/build_static_libcudf_in_container.sh) and
[build_cudf_java_jar_in_container.sh](ci/build_cudf_java_jar_in_container.sh) can be a useful
reference.

### Building with a Static libcudf

When statically linking the CUDA runtime, we recommend building libcudf as a static library rather
than a shared library, so that the Java bindings have only a single shared library that uses the
CUDA runtime. To do this, pass `-DBUILD_SHARED_LIBS=OFF` to CMake when building libcudf, and then
pass `-DCUDF_JNI_LIBCUDF_STATIC=ON` to Maven when building the Java bindings.

## Per-Thread Default Stream

The JNI code can be built with *per-thread default stream* (PTDS), which gives each host thread its
own default CUDA stream. This can increase the overlap of data copies and computation between
threads (see this
[blog post](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)).
The official release jars are built with PTDS enabled.

PTDS is set for each compilation unit, so it must be enabled for both libcudf and the JNI code. To
enable it, first build libcudf from the root of the repository, together with the options listed
in [Build From Source](#build-from-source):

```shell
cmake -S cpp -B cpp/build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON
cmake --build cpp/build --parallel
cmake --install cpp/build
```

Then build the jar:

```shell
cd java
mvn clean install -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON
```

## GPUDirect Storage (GDS)

The JNI code can be built with *GPUDirect Storage* (GDS) support, which enables direct copies
between GPU memory and supported file systems (see https://docs.nvidia.com/gpudirect-storage/).

To enable GDS support, first make sure that GDS is installed (see
https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html), and then run:

```shell
cd java
mvn clean install -DUSE_GDS=ON
```
