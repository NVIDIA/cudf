#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce a cudf CI job locally by launching the same container and script
# used in GitHub Actions. The container stays running for interactive inspection
# and is automatically removed after an idle timeout.
#
# Usage:
#   .agents/skills/reproduce-ci/run.sh <container-image> <ci-script> <pr-number> [--gpu] [--timeout <minutes>]
#
# Examples (derive the version tag from `head -1 VERSION | cut -d. -f1,2`):
#   .agents/skills/reproduce-ci/run.sh rapidsai/ci-conda:<VERSION>-latest ci/test_cmake.sh 22538
#   .agents/skills/reproduce-ci/run.sh rapidsai/ci-conda:<VERSION>-latest ci/test_java.sh 22538 --gpu
#   .agents/skills/reproduce-ci/run.sh rapidsai/citestwheel:<VERSION>-latest "ci/cudf_pandas_scripts/pandas-tests/run.sh pr" 22538 --gpu
#
# To find the container image and script for a job, look in .github/workflows/pr.yaml
# for the job definition's container_image and script fields.
#
# The container is left running after the CI script completes. To inspect interactively:
#   docker exec -it cudf-ci-repro bash
#
# The container will be automatically removed after --timeout minutes of idle
# (no docker exec sessions). Default: 30 minutes.
#
# To clean up manually:
#   docker rm -f cudf-ci-repro

set -euo pipefail

CONTAINER_NAME="cudf-ci-repro"
RAPIDS_VERSION="$(head -1 "$(git rev-parse --show-toplevel)/VERSION" | cut -d. -f1,2)"
if [[ -z "$RAPIDS_VERSION" ]]; then
    echo "Error: Could not determine RAPIDS version from VERSION file" >&2
    exit 1
fi
IDLE_TIMEOUT_MINUTES=30
DRY_RUN="no"

usage() {
    echo "Usage: $0 <container-image> <ci-script> <pr-number> [--gpu] [--timeout <minutes>]"
    echo ""
    echo "Arguments:"
    echo "  container-image  Docker image (e.g., rapidsai/ci-conda:${RAPIDS_VERSION}-latest)"
    echo "  ci-script        CI script to run (e.g., ci/test_cmake.sh)"
    echo "  pr-number        Pull request number"
    echo "  --gpu            Pass --gpus all to docker (for test jobs that need a GPU)"
    echo "  --timeout N      Idle timeout in minutes before container is auto-removed (default: 30)"
    echo "  --dry-run        Print the docker command without executing it"
    echo ""
    echo "Find these values in .github/workflows/pr.yaml under the job's 'with:' block."
    echo ""
    echo "Examples:"
    echo "  $0 rapidsai/ci-conda:${RAPIDS_VERSION}-latest ci/test_cmake.sh 22538"
    echo "  $0 rapidsai/ci-conda:${RAPIDS_VERSION}-latest ci/test_java.sh 22538 --gpu"
    echo "  $0 rapidsai/citestwheel:${RAPIDS_VERSION}-latest \"ci/cudf_pandas_scripts/pandas-tests/run.sh pr\" 22538 --gpu"
    exit 1
}

if [[ $# -lt 3 ]]; then
    usage
fi

CONTAINER_IMAGE="$1"
CI_SCRIPT="$2"
PR_NUMBER="$3"
GPU_NEEDED="no"

shift 3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_NEEDED="yes" ;;
        --dry-run) DRY_RUN="yes" ;;
        --timeout)
            shift
            IDLE_TIMEOUT_MINUTES="$1"
            ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Get GH_TOKEN
GH_TOKEN="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
if [[ -z "$GH_TOKEN" ]]; then
    echo "Warning: GH_TOKEN not set and 'gh auth token' failed."
    echo "Artifact downloads may fail or prompt interactively."
fi

# Determine RAPIDS_SHA from the PR's head commit
RAPIDS_SHA="${RAPIDS_SHA:-}"
if [[ -z "$RAPIDS_SHA" ]]; then
    RAPIDS_SHA=$(gh pr view "$PR_NUMBER" --repo rapidsai/cudf --json commits --jq '.commits[-1].oid' 2>/dev/null || true)
    if [[ -z "$RAPIDS_SHA" ]]; then
        echo "Warning: Could not determine RAPIDS_SHA for PR #${PR_NUMBER}."
        echo "Artifact downloads inside the container may fail."
        echo "Set RAPIDS_SHA manually or ensure 'gh' can access the PR."
    else
        echo "Detected RAPIDS_SHA=${RAPIDS_SHA} from PR #${PR_NUMBER}"
    fi
fi

# Remove existing container with same name if present
DOCKER_ARGS=(
    --pull=always
    --volume "$PWD:/repo"
    --workdir /repo
    --env "RAPIDS_BUILD_TYPE=pull-request"
    --env "RAPIDS_REPOSITORY=rapidsai/cudf"
    --env "RAPIDS_REF_NAME=pull-request/${PR_NUMBER}"
    --name "$CONTAINER_NAME"
    -d
)

if [[ -n "$GH_TOKEN" ]]; then
    DOCKER_ARGS+=(--env "GH_TOKEN=${GH_TOKEN}")
fi

if [[ -n "$RAPIDS_SHA" ]]; then
    DOCKER_ARGS+=(--env "RAPIDS_SHA=${RAPIDS_SHA}")
fi

if [[ "$GPU_NEEDED" == "yes" ]]; then
    DOCKER_ARGS+=(--gpus all)
fi

if [[ "$DRY_RUN" == "yes" ]]; then
    echo "=== Dry-run: docker command (not executing) ==="
    safe_args=()
    i=0
    while [[ $i -lt ${#DOCKER_ARGS[@]} ]]; do
        arg="${DOCKER_ARGS[$i]}"
        if [[ "$arg" == "--env" ]] && [[ $((i + 1)) -lt ${#DOCKER_ARGS[@]} ]]; then
            next_arg="${DOCKER_ARGS[$((i + 1))]}"
            if [[ "$next_arg" == GH_TOKEN=* ]]; then
                safe_args+=("--env" "GH_TOKEN=***REDACTED***")
            else
                safe_args+=("$arg" "$next_arg")
            fi
            i=$((i + 2))
            continue
        elif [[ "$arg" == GH_TOKEN=* ]]; then
            safe_args+=("GH_TOKEN=***REDACTED***")
        else
            safe_args+=("$arg")
        fi
        i=$((i + 1))
    done
    echo "docker run ${safe_args[*]} $CONTAINER_IMAGE tail -f /dev/null"
    exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
fi

echo "=== Reproducing CI job ==="
echo "  PR:        #${PR_NUMBER}"
echo "  SHA:       ${RAPIDS_SHA:-unknown}"
echo "  Image:     ${CONTAINER_IMAGE}"
echo "  Script:    ${CI_SCRIPT}"
echo "  GPU:       ${GPU_NEEDED}"
echo "  Timeout:   ${IDLE_TIMEOUT_MINUTES} minutes"
echo "  Container: ${CONTAINER_NAME}"
echo ""

# Launch container
echo "Launching container..."
docker run "${DOCKER_ARGS[@]}" "$CONTAINER_IMAGE" tail -f /dev/null
echo ""

# Run the CI script non-interactively
echo "Running CI script: ${CI_SCRIPT}"
echo "---"
CI_EXIT_CODE=0
docker exec "$CONTAINER_NAME" bash -c "./${CI_SCRIPT}" || CI_EXIT_CODE=$?
echo "---"
echo ""
if [[ $CI_EXIT_CODE -eq 0 ]]; then
    echo "CI script finished successfully. Container is still running."
else
    echo "CI script FAILED (exit code ${CI_EXIT_CODE}). Container is still running."
fi
echo "It will be automatically removed after ${IDLE_TIMEOUT_MINUTES} minutes of idle."
echo ""
echo "To inspect interactively:"
echo "  docker exec -it ${CONTAINER_NAME} bash"
echo ""
echo "To clean up manually:"
echo "  docker rm -f ${CONTAINER_NAME}"

# Start idle timeout watchdog in background.
# The watchdog polls `docker top` every 60 seconds. If only the `tail -f /dev/null`
# keepalive process remains for IDLE_TIMEOUT_MINUTES consecutive minutes, the
# container is removed.
(
    idle_seconds=0
    interval=60
    timeout_seconds=$((IDLE_TIMEOUT_MINUTES * 60))

    while true; do
        sleep "$interval"

        # If container no longer exists, we're done
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            exit 0
        fi

        # Count processes excluding the keepalive `tail -f /dev/null`
        # docker top output: header line + process lines
        active_procs=$(docker top "$CONTAINER_NAME" -o pid,cmd 2>/dev/null \
            | tail -n +2 \
            | grep -v -c 'tail -f /dev/null' || true)

        if [[ "$active_procs" -eq 0 ]]; then
            idle_seconds=$((idle_seconds + interval))
        else
            idle_seconds=0
        fi

        if [[ "$idle_seconds" -ge "$timeout_seconds" ]]; then
            echo ""
            echo "[reproduce-ci] Container idle for ${IDLE_TIMEOUT_MINUTES} minutes. Removing."
            docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
            exit 0
        fi
    done
) &
disown
