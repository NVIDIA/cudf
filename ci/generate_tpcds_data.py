# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate TPC-DS data at a given scale factor using DuckDB."""

from __future__ import annotations

import argparse
import os

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale", type=float, default=0.01, help="Scale factor."
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("TPCDS_DATA_DIR"),
        help="Output directory. Defaults to TPCDS_DATA_DIR environment variable.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        parser.error("--output-dir is required (or set TPCDS_DATA_DIR).")

    # TODO: switch to the Rust TPC-DS generator
    conn = duckdb.connect()
    conn.execute("INSTALL tpcds")
    conn.execute("LOAD tpcds")
    conn.execute("CALL dsdgen(sf=$1)", [args.scale])
    for table in conn.execute("SHOW TABLES").df()["name"]:
        path = f"{args.output_dir}/{table}.parquet"
        conn.execute(f"COPY {table} TO $1 (FORMAT PARQUET)", [path])


if __name__ == "__main__":
    main()
