# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Merge the per-module summaries produced by ``summarize-test-results.py`` for
several test shards into a single summary.

Each shard runs a disjoint subset of the suite (see the ``--num-shards``
sharding in ``pandas-testing-plugin.py``), so the combined result for a module
is obtained by summing every numeric field (test counts and the GPU/CPU
function-call counters) across the shards. The same module may appear in more
than one shard because sharding is per test, not per module.

Examples
--------
    python merge-results.py shard-0/pr-results.json shard-1/pr-results.json > pr-results.json
"""

import json
import sys

merged: dict[str, dict] = {}
for path in sys.argv[1:]:
    with open(path) as f:
        results = json.load(f)
    for module_name, row in results.items():
        combined = merged.setdefault(module_name, {})
        for key, value in row.items():
            if isinstance(value, bool):
                # No boolean fields are expected; keep the first seen value.
                combined.setdefault(key, value)
            elif isinstance(value, (int, float)):
                combined[key] = combined.get(key, 0) + value
            else:
                combined.setdefault(key, value)

print(json.dumps(merged, indent=4))
