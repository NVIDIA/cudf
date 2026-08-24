# libcudf C++ example using nested data types

This C++ example demonstrates using libcudf APIs to perform operations on
tables containing nested data types (e.g. structs and lists).

Blog post that uses this code: https://developer.nvidia.com/blog/streamline-etl-workflows-with-nested-data-types-in-rapids-libcudf/

The example reads a line-delimited JSON file whose records contain nested
fields, then deduplicates the rows and annotates each unique row with the
number of times it occurs. It showcases the libcudf nested-type row operators
of three kinds:
1. hashing - used to hash inputs of any type
2. equality - used in conjunction with hashing to determine equality for nested types
3. lexicographic - used to create a lexicographical order for nested types so as to enable sorting

The example performs the following steps:
1. `read_json` - Reads the nested-type table from a line-delimited JSON file
2. `count_aggregate` - Uses a groupby aggregation to count duplicate rows (hashing and equality)
3. `join_count` - Joins each row with its duplicate count (hashing and equality)
4. `sort_keys` - Sorts the resulting table by the nested key column (lexicographic)
5. `write_json` - Writes the deduplicated, sorted table to a line-delimited JSON file

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute using the included example.json and the default pool memory resource
build/deduplication
# Execute with explicit arguments: input file, output file, and memory resource ("pool" or "cuda")
build/deduplication example.json output.json pool
```

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
