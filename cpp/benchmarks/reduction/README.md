# LTO reduction benchmarks

Build and run the fixed-loop C++ driver:

    cmake --build <build> --target REDUCE_LTO_BENCH
    <build>/benchmarks/REDUCE_LTO_BENCH --smoke --csv smoke.csv
    <build>/benchmarks/REDUCE_LTO_BENCH --csv results.csv
    python plot_lto.py results.csv --output-dir figures --subtitle 'GPU | CUDA version'

Q13 measures its post-join predicate and four aggregates on resident synthetic
inputs. Demographic/address branch codes represent completed dimension joins;
joins and their preparation are excluded. Compare LTO with cuDF AST JIT and
non-JIT predicate evaluation, retention-mask filtering and separate reductions.
Use --rows N to select a Q13 row count. The measured expression is:

    SELECT AVG(ss_quantity), AVG(ss_ext_sales_price),
           AVG(ss_ext_wholesale_cost), SUM(ss_ext_wholesale_cost)
    FROM post_join_rows
    WHERE d_year = 2001
      AND ((demographic_branch = 1 AND ss_sales_price BETWEEN 100 AND 150)
        OR (demographic_branch = 2 AND ss_sales_price BETWEEN 50 AND 100)
        OR (demographic_branch = 3 AND ss_sales_price BETWEEN 150 AND 200))
      AND ((address_branch = 1 AND ss_net_profit BETWEEN 100 AND 200)
        OR (address_branch = 2 AND ss_net_profit BETWEEN 150 AND 300)
        OR (address_branch = 3 AND ss_net_profit BETWEEN 50 AND 250));

Segmented SUM and fused SUM/MAX cover short segments, long segments and a skewed
distribution with empty segments. Compare public cuDF APIs with LTO using known
and unknown maximum-length hints. LTO returns independent single-row columns;
cuDF returns packed columns. Allocation/finalization costs remain in the timings.

Default Q13 sizes are 1,000, 100,000, 1 million and 10 million rows. Segmented
shapes are 4,096 × 16, 32 × 1,048,576 and 256 × 65,536 rows; the skewed shape
puts 90% of rows in its first segment.

Each case has four warmups and five synchronized steady-clock measurements.
Generation, reference calculations, compilation warmup, validation and output
destruction are outside timing. The driver uses a common 90% device-memory cap.
Smoke mode also checks empty Q13 inputs and all-null aggregates.

GB/s counts unique logical input values, validity bits and segment offsets,
including unselected rows. It is not measured DRAM bandwidth. SVG plots show
medians and observed minimum/maximum ranges, not confidence intervals.
