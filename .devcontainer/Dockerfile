# syntax=docker/dockerfile:1.5

ARG BASE
ARG PYTHON_PACKAGE_MANAGER=conda

FROM ${BASE} as pip-base

# libnuma-dev is required for pip devcontainers for cucascade from rapidsmpf
RUN apt update -y \
 && DEBIAN_FRONTEND=noninteractive apt install -y \
    librdkafka-dev \
    libnuma-dev \
 && rm -rf /tmp/* /var/tmp/* /var/cache/apt/* /var/lib/apt/lists/*;

ENV DEFAULT_VIRTUAL_ENV=rapids

FROM ${BASE} as conda-base

ENV DEFAULT_CONDA_ENV=rapids

FROM ${PYTHON_PACKAGE_MANAGER}-base

ARG TARGETARCH

ARG CUDA
ENV CUDAARCHS="RAPIDS"
ENV CUDA_VERSION="${CUDA_VERSION:-${CUDA}}"

ARG PYTHON_PACKAGE_MANAGER
ENV PYTHON_PACKAGE_MANAGER="${PYTHON_PACKAGE_MANAGER}"

ENV PYTHONSAFEPATH="1"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

ENV HISTFILE="/home/coder/.cache/._bash_history"
ENV LIBCUDF_KERNEL_CACHE_PATH="/home/coder/cudf/cpp/build/${PYTHON_PACKAGE_MANAGER}/cuda-${CUDA_VERSION}/latest/libcudf_kernel_cache"

###
# sccache configuration
###
ENV AWS_IDP_URL="https://token.rapids.nvidia.com"
ENV AWS_ROLE_ARN="arn:aws:iam::279114543810:role/rapids-token-sccache-devs"
ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true
ENV SCCACHE_IDLE_TIMEOUT=0

###
# sccache-dist configuration
###
# Enable sccache-dist by default
ENV DEVCONTAINER_UTILS_ENABLE_SCCACHE_DIST=1
# Compile locally if max retries exceeded
ENV SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE=true
# Retry transient errors 4 times (for a total of 5 attempts)
ENV SCCACHE_DIST_MAX_RETRIES=4
# 1hr 59min (to accommodate debug builds)
ENV SCCACHE_DIST_REQUEST_TIMEOUT=7140
ENV SCCACHE_DIST_URL="https://${TARGETARCH}.linux.sccache.rapids.nvidia.com"

# Build as much in parallel as possible
ENV INFER_NUM_DEVICE_ARCHITECTURES=1
ENV MAX_DEVICE_OBJ_TO_COMPILE_IN_PARALLEL=20
