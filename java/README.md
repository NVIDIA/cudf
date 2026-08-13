# Java API for cudf

This project provides java bindings for cudf, to be able to process large amounts of data on
a GPU. This is still a work in progress so some APIs may change until the 1.0 release.

## Behavior on Systems with Multiple GPUs

The cudf project currently works with a single GPU per process. The CUDA runtime
assigns the default GPU to all new operating system threads when they start to
interact with CUDA. This means that if you use a multi-threaded environment,
like Java processes tend to be, and try to use a non-default GPU for cudf you
can run into hard to debug issues including resource leaks, invalid states,
application crashes, and poor performance.

To prevent this the Java cudf API will remember the device used to initialize
the Rapids Memory Manager (RMM), and automatically set the thread's active
device to it, if needed. It will not set the device back when the cudf call
completes. This is different from most CUDA libraries and can result in
unexpected behavior if you try to mix these libraries using the same thread.

## Dependency

This is a fat jar with the binary dependencies packaged in the jar.  This means the jar will only
run on platforms the jar was compiled for.  When this is in an official Maven repository we will
list the platforms that it is compiled and tested for.  In the mean time you will need to build it
yourself. In official releases there should be no classifier on the jar and it should run against
most modern cuda drivers.

```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <version>${cudf.version}</version>
</dependency>
```

In some cases there may be a classifier to indicate the version of cuda required. See the
[Build From Source](#build-from-source) section below for more information about when this
can happen. No official release of the jar will have a classifier on it.

CUDA 12.2:
```xml
<dependency>
    <groupId>ai.rapids</groupId>
    <artifactId>cudf</artifactId>
    <classifier>cuda12</classifier>
    <version>${cudf.version}</version>
</dependency>
```

## Build From Source

Build [libcudf](../cpp) first, and make sure the JDK is installed and available. Specify
the following cmake options to the libcudf build:
```
-DCUDF_LARGE_STRINGS_DISABLED=ON -DCUDF_USE_ARROW_STATIC=ON -DCUDF_ENABLE_ARROW_S3=OFF
```
These options:
- Disable large string support, see https://github.com/rapidsai/cudf/issues/16215
- Statically link Arrow to libcudf to remove Arrow as a runtime dependency.

After building libcudf, the Java bindings can be built via Maven, e.g.:
```
mvn clean install
```

If you have a compatible GPU on your build system the tests will use it.  If not you will see a
lot of skipped tests.

### Using the Java CI Docker Image

If you are interested in building a Java cudf jar that is similar to the official releases
that can run on all modern Linux systems, see the [Java CI README](ci/README.md) for
instructions on how to build within a Docker environment using devtoolset. Note that
building the jar without the Docker setup and script will likely produce a jar that can
only run in environments similar to that of the build machine.

If you decide to build without Docker and the build script, examining the cmake and Maven
settings in the [Java CI build script](ci/build-in-docker.sh) can be helpful if you are
encountering difficulties during the build.

### Building with a libcudf Archive

When statically linking the CUDA runtime, it is recommended to build cuDF as an archive rather than
a shared library, as this allows the Java bindings to only have a single shared library that uses
the CUDA runtime. To build libcudf as an archive, specify `-DBUILD_SHARED_LIBS=OFF` as a cmake
parameter when building libcudf, then specify `-DCUDF_JNI_LIBCUDF_STATIC=ON` when building the Java
bindings with Maven.

## Per-thread Default Stream

The JNI code can be built with *per-thread default stream* (PTDS), which gives each host thread its
own default CUDA stream, and can potentially increase the overlap of data copying and compute
between different threads (see
[blog post](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)).

Since the PTDS option is for each compilation unit, it should be done at the same time across the
whole codebase. To enable PTDS, first build cuDF:
```shell script
cd src/cudf/cpp/build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON
make -j`nproc`
make install
```

then build the jar:
```shell script
cd src/cudf/java
mvn clean install -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON
```

## GPUDirect Storage (GDS)

The JNI code can be built with *GPUDirect Storage* (GDS) support, which enables direct copying
between GPU device buffers and supported filesystems (see
https://docs.nvidia.com/gpudirect-storage/).

To enable GDS support, first make sure GDS is installed (see
https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html), then run:
```shell script
cd src/cudf/java
mvn clean install -DUSE_GDS=ON
```
