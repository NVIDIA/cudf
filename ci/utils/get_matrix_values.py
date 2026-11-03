# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Extract matrix dimension values for a file key from dependencies.yaml."""

import argparse

import yaml


def get_matrix_values(
    deps_yaml_path: str, file_key: str, matrix_var: str
) -> list[str]:
    with open(deps_yaml_path) as f:
        deps = yaml.safe_load(f)

    file_cfg = deps.get("files", {}).get(file_key)
    if file_cfg is None:
        raise RuntimeError(
            f"File key '{file_key}' not found in {deps_yaml_path}"
        )

    matrix = file_cfg.get("matrix", {})
    values = matrix.get(matrix_var)
    if values is None:
        raise RuntimeError(
            f"Matrix variable '{matrix_var}' not found in file key '{file_key}'"
        )

    return [str(v) for v in values]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract matrix dimension values for a file key in dependencies.yaml"
    )
    parser.add_argument("deps_yaml", help="Path to dependencies.yaml")
    parser.add_argument("file_key", help="File key in the 'files' section")
    parser.add_argument("matrix_var", help="Matrix variable name to extract")
    args = parser.parse_args()

    values = get_matrix_values(args.deps_yaml, args.file_key, args.matrix_var)
    print(" ".join(values))
