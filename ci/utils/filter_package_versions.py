# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import json

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def get_library_specifier(
    deps_yaml_path: str, deps_yaml_key: str, library_name: str
) -> str:
    with open(deps_yaml_path, "r") as f:
        deps = yaml.safe_load(f)

    try:
        # Assumption: files.all.includes exists in dependencies.yaml
        includes = deps["files"]["all"]["includes"]
        if deps_yaml_key not in includes:
            raise KeyError()
    except KeyError:
        raise RuntimeError(f"{deps_yaml_key} not found in dependencies.yaml")

    try:
        # Assumption: library_name is under a "common" section
        pkgs = deps["dependencies"][deps_yaml_key]["common"]
        for entry in pkgs:
            for pkg in entry.get("packages", []):
                if isinstance(pkg, str) and pkg.startswith(library_name):
                    spec = pkg.removeprefix(library_name).strip()
                    if spec:
                        return spec
    except KeyError:
        pass

    raise RuntimeError(
        f"{library_name} specifier not found in dependencies.yaml"
    )


def filter_versions(
    available_versions: list[str], library_range: str
) -> list[str]:
    matching_versions = map(
        Version,
        SpecifierSet(library_range).filter(available_versions),
    )
    grouped_by_major_minor = itertools.groupby(
        sorted(matching_versions), key=lambda v: (v.major, v.minor)
    )
    # Latest patch only
    return [str(max(versions)) for _, versions in grouped_by_major_minor]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Given a library with version specifier in dependencies.yaml, return all, latest patch versions."
    )
    parser.add_argument(
        "deps_yaml",
        help="Path to dependencies.yaml",
    )
    parser.add_argument(
        "deps_yaml_key",
        help="Section key in deps_yaml to search for the library name",
    )
    parser.add_argument(
        "library_name",
        help="Name of the library to filter versions for",
    )
    parser.add_argument(
        "available_versions",
        type=json.loads,
        help="The available versions of the library from pip index.",
    )
    args = parser.parse_args()

    library_range = get_library_specifier(
        args.deps_yaml, args.deps_yaml_key, args.library_name
    )
    versions = filter_versions(args.available_versions, library_range)
    print(" ".join(versions))
