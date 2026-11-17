# libcudf C++ example for the 1 billion row challenge

This C++ example demonstrates using libcudf APIs to read and process
a table with 1 billion rows. The 1 billion row challenge is described here:
https://github.com/gunnarmorling/1brc

Blog post that uses this code: https://developer.nvidia.com/blog/processing-one-billion-rows-of-data-with-rapids-cudf-pandas-accelerator-mode/

The examples load the 1 billion row text file using the CSV reader.
The file contains around 400 unique city names (string type) along with
random temperature values (float type).
Once loaded, the examples performs groupby aggregations to find the
minimum, maximum, and average temperature for each city.

There are three examples included:
1. `brc.cpp`
   Loads the file in one call to the CSV reader.
   This generally requires a large amount of available GPU memory.
2. `brc_chunks.cpp`
   Loads and processes the file in chunks.
   The number of chunks to use is a parameter to the executable.
3. `brc_pipeline.cpp`
   Loads and processes the file in chunks with separate threads/streams.
   The number of chunks and number of threads to use are parameters to the executable.

An input file can be generated using the instructions from
https://github.com/gunnarmorling/1brc.

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute
build/brc input.txt
# Execute in chunked mode with 25 chunks (default)
build/brc_chunks input.txt 25
# Execute in pipeline mode with 25 chunks and 2 threads (defaults)
build/brc_pipeline input.txt 25 2
```

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
