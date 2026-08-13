#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
########################
# cuDF Version Updater #
########################

## Usage
# Primary interface:   bash update-version.sh <new_version> [--run-context=main|release]
# Fallback interface:  [RAPIDS_RUN_CONTEXT=main|release] bash update-version.sh <new_version>
# CLI arguments take precedence over environment variables
# Defaults to main when no run-context is specified


# Parse command line arguments
CLI_RUN_CONTEXT=""
VERSION_ARG=""

for arg in "$@"; do
    case $arg in
        --run-context=*)
            CLI_RUN_CONTEXT="${arg#*=}"
            shift
            ;;
        *)
            if [[ -z "$VERSION_ARG" ]]; then
                VERSION_ARG="$arg"
            fi
            ;;
    esac
done

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG="$VERSION_ARG"

# Determine RUN_CONTEXT with CLI precedence over environment variable, defaulting to main
if [[ -n "$CLI_RUN_CONTEXT" ]]; then
    RUN_CONTEXT="$CLI_RUN_CONTEXT"
    echo "Using run-context from CLI: $RUN_CONTEXT"
elif [[ -n "${RAPIDS_RUN_CONTEXT}" ]]; then
    RUN_CONTEXT="$RAPIDS_RUN_CONTEXT"
    echo "Using run-context from environment: $RUN_CONTEXT"
else
    RUN_CONTEXT="main"
    echo "No run-context provided, defaulting to: $RUN_CONTEXT"
fi

# Validate RUN_CONTEXT value
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context value '${RUN_CONTEXT}'"
    echo "Valid values: main, release"
    exit 1
fi

# Validate version argument
if [[ -z "$NEXT_FULL_TAG" ]]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <new_version> [--run-context=<context>]"
    echo "   or: [RAPIDS_RUN_CONTEXT=<context>] $0 <new_version>"
    echo "Note: Defaults to main when run-context is not specified"
    exit 1
fi

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[2]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

NEXT_UCXX_TAG="$(curl -s https://version.gpuci.io/rapids/"${NEXT_SHORT_TAG}")"

# Need to distutils-normalize the versions for some use cases
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
PATCH_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_PATCH}'))")
NEXT_UCXX_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_UCXX_TAG}'))")

# Set branch references based on RUN_CONTEXT
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    WORKFLOW_BRANCH_REF="main"
    echo "Preparing development branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    WORKFLOW_BRANCH_REF="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting release/${NEXT_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# latest numba-cuda upper bound
NUMBA_CUDA_JSON=$(curl -sL https://pypi.org/pypi/numba-cuda/json) || true
if [[ -n "$NUMBA_CUDA_JSON" ]]; then
  NUMBA_CUDA_VERSIONS=$(echo "$NUMBA_CUDA_JSON" | python -c "
import json, sys
from packaging.version import Version
try:
    data = json.load(sys.stdin)
    releases = [v for v in data.get('releases', []) if not Version(v).is_prerelease]
    latest = max(releases, key=lambda v: Version(v)) if releases else None
    if not latest:
        sys.exit(1)
    v = Version(latest)
    r = v.release
    # exclusive upper bound = next minor (e.g. 0.23.1 -> 0.24.0)
    upper = '.'.join(str(x) for x in (r[0], r[1] + 1, 0))
    print(latest, upper)
except Exception:
    sys.exit(1)
") || NUMBA_CUDA_VERSIONS=""
  if [[ -n "$NUMBA_CUDA_VERSIONS" ]]; then
    LATEST_NUMBA_CUDA="${NUMBA_CUDA_VERSIONS%% *}"
    NUMBA_CUDA_UPPER="${NUMBA_CUDA_VERSIONS##* }"
    CURRENT_LOWER=$(grep -oE 'numba-cuda>=[0-9]+\.[0-9]+\.[0-9][0-9.]*' dependencies.yaml 2>/dev/null | head -1 | cut -d= -f2)
    echo "Updating numba-cuda: lower bound ${CURRENT_LOWER}, upper bound <${NUMBA_CUDA_UPPER} (latest release: ${LATEST_NUMBA_CUDA})"
    NUMBA_CUDA_SPEC=">=${CURRENT_LOWER},<${NUMBA_CUDA_UPPER}"
    for FILE in dependencies.yaml conda/recipes/cudf/recipe.yaml; do
      for f in $FILE; do
        [[ -f "$f" ]] || continue
        sed_runner "s/numba-cuda>=[0-9]\+\.[0-9]\+\.[0-9][0-9.]*/numba-cuda${NUMBA_CUDA_SPEC}/g" "$f"
        sed_runner "s/numba-cuda\[cu12\]>=[0-9]\+\.[0-9]\+\.[0-9][0-9.]*/numba-cuda[cu12]${NUMBA_CUDA_SPEC}/g" "$f"
        sed_runner "s/numba-cuda\[cu13\]>=[0-9]\+\.[0-9]\+\.[0-9][0-9.]*/numba-cuda[cu13]${NUMBA_CUDA_SPEC}/g" "$f"
        sed_runner "s/numba-cuda >=[0-9]\+\.[0-9]\+\.[0-9][0-9.]*/numba-cuda ${NUMBA_CUDA_SPEC}/g" "$f"
      done
    done
  else
    echo "Warning: Could not determine latest numba-cuda version; leaving existing pins unchanged"
  fi
else
  echo "Warning: Could not fetch numba-cuda metadata from PyPI; leaving existing pins unchanged"
fi

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION
# The cudf version file must be a copy, see https://github.com/rapidsai/cudf/pull/18198
echo "${NEXT_FULL_TAG}" > python/cudf/cudf/VERSION
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

DEPENDENCIES=(
  cudf
  cudf-polars
  cudf-streaming
  cudf_kafka
  cugraph
  cuml
  custreamz
  dask-cuda
  dask-cudf
  kvikio
  libcudf
  libcudf-example
  libcudf-streaming
  libcudf-tests
  libcudf_kafka
  libkvikio
  librapidsmpf
  librmm
  pylibcudf
  rapids-dask-dependency
  rapidsmpf
  rmm
)

UCXX_DEPENDENCIES=(
  libucxx
  ucxx
)

for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml python/cudf/cudf_pandas_tests/third_party_integration_tests/dependencies.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}\(\[.*\]\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for FILE in python/*/pyproject.toml; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
done

for DEP in "${UCXX_DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml python/cudf/cudf_pandas_tests/third_party_integration_tests/dependencies.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_UCXX_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for FILE in python/*/pyproject.toml; do
    sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_UCXX_SHORT_TAG_PEP440}\.*,>=0.0.0a0\"/g" "${FILE}"
  done
done

# README.md update
sed_runner "s/version == ${CURRENT_SHORT_TAG}/version == ${NEXT_SHORT_TAG}/g" README.md
sed_runner "s/cudf=${CURRENT_SHORT_TAG}/cudf=${NEXT_SHORT_TAG}/g" README.md
sed_runner "s/cudf=${CURRENT_SHORT_TAG}/cudf=${NEXT_SHORT_TAG}/g" python/cudf_polars/docs/overview.md

# Documentation references
sed_runner "s|release/[0-9]\+\.[0-9]\+|${RAPIDS_BRANCH_NAME}|g" python/cudf_polars/docs/overview.md
sed_runner "s|/blob/\\bmain\\b/|/blob/${RAPIDS_BRANCH_NAME}/|g" python/cudf_polars/docs/overview.md
sed_runner "s|/tree/\\bmain\\b/|/tree/${RAPIDS_BRANCH_NAME}/|g" python/cudf_polars/docs/overview.md

# Libcudf examples update
sed_runner "s|CUDF_TAG release/[0-9]\+\.[0-9]\+|CUDF_TAG ${RAPIDS_BRANCH_NAME}|g" cpp/examples/versions.cmake
sed_runner "s|CUDF_TAG \\bmain\\b|CUDF_TAG ${RAPIDS_BRANCH_NAME}|g" cpp/examples/versions.cmake

# .pre-commit-config.yaml
sed_runner "/rmm-cu[0-9]*==/ s/==.*\*,/==${NEXT_SHORT_TAG_PEP440}.*,/g" .pre-commit-config.yaml

# CI files
for FILE in .github/workflows/*.yaml .github/workflows/*.yml; do
  sed_runner "/shared-workflows/ s|@.*|@${WORKFLOW_BRANCH_REF}|g" "${FILE}"
  sed_runner "s|dask-cuda.git@release/[^\"\s]\+|dask-cuda.git@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  sed_runner "s|dask-cuda.git@\\bmain\\b|dask-cuda.git@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  sed_runner "s|:[0-9]*\\.[0-9]*-|:${NEXT_SHORT_TAG}-|g" "${FILE}"
done

# Java files
NEXT_FULL_JAVA_TAG="${NEXT_SHORT_TAG}.${PATCH_PEP440}-SNAPSHOT"
sed_runner "s|<version>.*-SNAPSHOT</version>|<version>${NEXT_FULL_JAVA_TAG}</version>|g" java/pom.xml
sed_runner "s|cudf-.*-SNAPSHOT|cudf-${NEXT_FULL_JAVA_TAG}|g" java/ci/README.md
sed_runner "s|/ai/rapids/cudf/[0-9]\+\.[0-9]\+\.[0-9]\+-SNAPSHOT/|/ai/rapids/cudf/${NEXT_FULL_JAVA_TAG}/|g" java/ci/README.md

# Java documentation references
sed_runner "s|release/[0-9]\+\.[0-9]\+|${RAPIDS_BRANCH_NAME}|g" java/ci/README.md
sed_runner "s|-b \\bmain\\b|-b ${RAPIDS_BRANCH_NAME}|g" java/ci/README.md

# Other documentation references
sed_runner "s|/blob/release/[0-9]\+\.[0-9]\+/|/blob/${RAPIDS_BRANCH_NAME}/|g" docs/cudf/source/pylibcudf/developer_docs.md
sed_runner "s|/blob/\\bmain\\b/|/blob/${RAPIDS_BRANCH_NAME}/|g" docs/cudf/source/pylibcudf/developer_docs.md

sed_runner "s|/blob/release/[0-9]\+\.[0-9]\+/|/blob/${RAPIDS_BRANCH_NAME}/|g" python/custreamz/README.md
sed_runner "s|/blob/\\bmain\\b/|/blob/${RAPIDS_BRANCH_NAME}/|g" python/custreamz/README.md

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@ghcr.io/rapidsai/cudf/devcontainer:[0-9.]*@ghcr.io/rapidsai/cudf/devcontainer:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/cuda:[0-9.]*@rapidsai/devcontainers/features/cuda:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-[0-9.]*@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done
