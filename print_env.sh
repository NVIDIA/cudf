#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reports relevant environment information useful for diagnosing and
# debugging cuDF issues.
# Usage:
# "./print_env.sh" - prints to stdout
# "./print_env.sh > env.txt" - prints to file "env.txt"

print_env() {
echo "**git***"
if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]; then
git log --decorate -n 1
echo "**git submodules***"
git submodule status --recursive
else
echo "Not inside a git repository"
fi
echo

echo "***OS Information***"
cat /etc/*-release
uname -a
echo

echo "***GPU Information***"
nvidia-smi
echo

echo "***CPU***"
lscpu
echo

echo "***CMake***"
which cmake && cmake --version
echo

echo "***g++***"
which g++ && g++ --version
echo

echo "***nvcc***"
which nvcc && nvcc --version
echo

echo "***Python***"
which python && python -c "import sys; print('Python {0}.{1}.{2}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))"
echo

echo "***Environment Variables***"

printf '%-32s: %s\n' PATH $PATH

printf '%-32s: %s\n' LD_LIBRARY_PATH $LD_LIBRARY_PATH

printf '%-32s: %s\n' CONDA_PREFIX $CONDA_PREFIX

printf '%-32s: %s\n' PYTHON_PATH $PYTHON_PATH

echo


# Print conda packages if conda exists
if type "conda" &> /dev/null; then
echo '***conda packages***'
which conda && conda list
echo
# Print pip packages if pip exists
elif type "pip" &> /dev/null; then
echo "conda not found"
echo "***pip packages***"
which pip && pip list
echo
else
echo "conda not found"
echo "pip not found"
fi
}

echo "<details><summary>Click here to see environment details</summary><pre>"
echo "     "
print_env | while read -r line; do
    echo "     $line"
done
echo "</pre></details>"
