#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../
source ./ci/test_python_common.sh test_python_cudf test_cuml

# Clone cuml at the matching RAPIDS branch to get test files
# (the conda package doesn't ship cuml/tests/)

# This makes sure we read RAPIDS_BRANCH from the top level of the repo, no matter
# what the current working directory is.
RAPIDS_BRANCH="$(cat "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../RAPIDS_BRANCH)"
rapids-logger "Cloning cuml at branch ${RAPIDS_BRANCH}"
git clone https://github.com/rapidsai/cuml.git --branch "${RAPIDS_BRANCH}" --depth 1 /tmp/cuml


CUML_TESTS_DIR=/tmp/cuml/python/cuml/tests

rapids-logger "pytest cuml cuDF-compat subset"

timeout 15m python -m pytest \
  --cache-clear \
  "${CUML_TESTS_DIR}/test_compose.py" \
  "${CUML_TESTS_DIR}/test_kneighbors_classifier.py" \
  "${CUML_TESTS_DIR}/test_kneighbors_regressor.py" \
  "${CUML_TESTS_DIR}/test_label_encoder.py" \
  "${CUML_TESTS_DIR}/test_elastic_net.py" \
  "${CUML_TESTS_DIR}/test_linear_regression.py" \
  "${CUML_TESTS_DIR}/test_logistic_regression.py" \
  "${CUML_TESTS_DIR}/test_ridge.py" \
  "${CUML_TESTS_DIR}/test_metrics.py" \
  "${CUML_TESTS_DIR}/test_one_hot_encoder.py" \
  "${CUML_TESTS_DIR}/test_ordinal_encoder.py" \
  "${CUML_TESTS_DIR}/test_random_forest.py" \
  "${CUML_TESTS_DIR}/test_reflection.py" \
  "${CUML_TESTS_DIR}/test_sgd.py" \
  "${CUML_TESTS_DIR}/test_split.py" \
  "${CUML_TESTS_DIR}/test_svm.py" \
  "${CUML_TESTS_DIR}/test_target_encoder.py" \
  "${CUML_TESTS_DIR}/test_text_feature_extraction.py" \
  "${CUML_TESTS_DIR}/test_train_test_split.py" \
  "${CUML_TESTS_DIR}/test_validation.py"
