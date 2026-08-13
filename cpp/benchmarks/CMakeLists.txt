# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

find_package(Threads REQUIRED)

add_library(
  cudf_datagen STATIC common/generate_input.cu common/generate_skewed_data.cpp
                      common/generate_nested_types.cpp
)
target_compile_features(cudf_datagen PUBLIC cxx_std_20 cuda_std_20)

target_compile_options(
  cudf_datagen PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_link_libraries(
  cudf_datagen
  PUBLIC GTest::gmock GTest::gtest nvbench::nvbench Threads::Threads cudf cudf::cudftestutil
         nvtx3::nvtx3-cpp
  PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>
)

target_include_directories(
  cudf_datagen
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>" "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}>"
         "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
)

add_library(
  ndsh_data_generator STATIC
  common/ndsh_data_generator/ndsh_data_generator.cpp common/ndsh_data_generator/table_helpers.cpp
  common/ndsh_data_generator/random_column_generator.cu
)
target_compile_features(ndsh_data_generator PUBLIC cxx_std_20 cuda_std_20)

target_compile_options(
  ndsh_data_generator PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                             "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_link_libraries(
  ndsh_data_generator
  PUBLIC cudf cudf::cudftestutil nvtx3::nvtx3-cpp
  PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>
)

target_include_directories(
  ndsh_data_generator
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>" "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}>"
         "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
)

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# Use an OBJECT library so we only compile these helper source files only once
add_library(
  cudf_benchmark_common OBJECT io/cuio_common.cpp common/table_utilities.cpp
                               common/nvbench_utilities.cpp
)
target_link_libraries(
  cudf_benchmark_common PRIVATE cudf_datagen $<TARGET_NAME_IF_EXISTS:conda_env> GTest::gmock
                                GTest::gtest kvikio::kvikio
)

add_custom_command(
  OUTPUT CUDF_BENCHMARKS
  COMMAND echo Running benchmarks
  COMMAND mkdir -p results
  VERBATIM
  COMMENT "Running cudf benchmarks."
  USES_TERMINAL
)

# This function takes in a benchmark name and benchmark source for nvbench benchmarks and handles
# setting all of the associated properties and linking to build the benchmark
function(ConfigureNVBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN} fixture/nvbench_main.cpp)
  set_target_properties(
    ${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/benchmarks>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
  )
  target_link_libraries(
    ${CMAKE_BENCH_NAME}
    PRIVATE cudf_benchmark_common ndsh_data_generator cudf_datagen nvbench::nvbench
            $<TARGET_NAME_IF_EXISTS:conda_env> cudf::cudftestutil_objects
  )
  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT testing
    DESTINATION bin/benchmarks/libcudf
    EXCLUDE_FROM_ALL
  )
endfunction()

# ##################################################################################################
# * copying benchmarks ----------------------------------------------------------------------------
ConfigureNVBench(
  COPYING_NVBENCH copying/concatenate.cpp copying/copy_if_else.cpp copying/gather.cpp
  copying/scatter.cpp copying/shift.cpp
)

# ##################################################################################################
# * contiguous_split benchmark  -------------------------------------------------------------------
ConfigureNVBench(CONTIGUOUS_SPLIT_NVBENCH contiguous_split/contiguous_split.cpp)

# ##################################################################################################
# * lists scatter benchmark -----------------------------------------------------------------------
ConfigureNVBench(SCATTER_LISTS_NVBENCH lists/copying/scatter_lists.cu)

# ##################################################################################################
# * Other list-related operartions benchmark ------------------------------------------------------
ConfigureNVBench(SET_OPS_NVBENCH lists/set_operations.cpp)

# ##################################################################################################
# * transpose benchmark ---------------------------------------------------------------------------
ConfigureNVBench(TRANSPOSE_NVBENCH transpose/transpose.cpp)

# ##################################################################################################
# * nds-h benchmark --------------------------------------------------------------------------------
ConfigureNVBench(NDSH_Q01_NVBENCH ndsh/q01.cpp ndsh/utilities.cpp)
ConfigureNVBench(NDSH_Q05_NVBENCH ndsh/q05.cpp ndsh/utilities.cpp)
ConfigureNVBench(NDSH_Q06_NVBENCH ndsh/q06.cpp ndsh/utilities.cpp)
ConfigureNVBench(NDSH_Q09_NVBENCH ndsh/q09.cpp ndsh/utilities.cpp)
ConfigureNVBench(NDSH_Q10_NVBENCH ndsh/q10.cpp ndsh/utilities.cpp)

# ##################################################################################################
# * filter benchmark -------------------------------------------------------------------
ConfigureNVBench(FILTER_NVBENCH filter/minmax_filter.cpp)

# ##################################################################################################
# * stream_compaction benchmark -------------------------------------------------------------------
ConfigureNVBench(
  STREAM_COMPACTION_NVBENCH
  stream_compaction/apply_boolean_mask.cpp
  stream_compaction/distinct.cpp
  stream_compaction/stable_distinct.cpp
  stream_compaction/stream_compaction_common.cpp
  stream_compaction/unique.cpp
)

# ##################################################################################################
# * join benchmark --------------------------------------------------------------------------------
ConfigureNVBench(
  JOIN_NVBENCH
  join/conditional_join.cpp
  join/direct_join.cu
  join/distinct_join.cpp
  join/filter_join_indices.cpp
  join/filter_join_indices_jit.cu
  join/generate_input_tables.cu
  join/join.cpp
  join/join_dictionary.cpp
  join/join_heuristics.cpp
  join/join_on_int32.cu
  join/key_remap_build.cpp
  join/left_join.cpp
  join/mixed_join.cpp
  join/multiplicity_join.cpp
  join/sort_merge_join.cpp
)

# ##################################################################################################
# * iterator benchmark ----------------------------------------------------------------------------
ConfigureNVBench(ITERATOR_NVBENCH iterator/iterator.cu)

# ##################################################################################################
# * search benchmark ------------------------------------------------------------------------------
ConfigureNVBench(
  SEARCH_NVBENCH search/contains_scalar.cpp search/contains_table.cpp search/search.cpp
)

# ##################################################################################################
# * sort benchmark --------------------------------------------------------------------------------
ConfigureNVBench(
  SORT_NVBENCH
  sort/rank.cpp
  sort/rank_lists.cpp
  sort/rank_structs.cpp
  sort/segmented_top_k.cpp
  sort/segmented_sort.cpp
  sort/sort.cpp
  sort/sort_lists.cpp
  sort/sort_strings.cpp
  sort/sort_structs.cpp
  sort/top_k.cpp
)

# ##################################################################################################
# * quantiles benchmark
# --------------------------------------------------------------------------------
ConfigureNVBench(QUANTILES_NVBENCH quantiles/quantiles.cpp)

# ##################################################################################################
# * tdigest benchmark
# --------------------------------------------------------------------------------
ConfigureNVBench(TDIGEST_NVBENCH quantiles/tdigest.cpp)

# ##################################################################################################
# * type_dispatcher benchmark ---------------------------------------------------------------------
ConfigureNVBench(TYPE_DISPATCHER_NVBENCH type_dispatcher/type_dispatcher.cu)

# ##################################################################################################
# * reduction benchmark ---------------------------------------------------------------------------
ConfigureNVBench(
  REDUCTION_NVBENCH
  reduction/anyall.cpp
  reduction/dictionary.cpp
  reduction/distinct_count.cpp
  reduction/histogram.cpp
  reduction/minmax.cpp
  reduction/rank.cpp
  reduction/reduce.cpp
  reduction/scan.cpp
  reduction/scan_structs.cpp
  reduction/segmented_reduce.cpp
  reduction/unique_count.cpp
)

# ##################################################################################################
# * replace benchmark ---------------------------------------------------------------------------
ConfigureNVBench(REPLACE_NVBENCH replace/clamp.cpp replace/nans.cpp replace/nulls.cpp)

# ##################################################################################################
# * filling benchmark -----------------------------------------------------------------------------
ConfigureNVBench(FILL_NVBENCH filling/repeat.cpp)

# ##################################################################################################
# * groupby benchmark -----------------------------------------------------------------------------
ConfigureNVBench(
  GROUPBY_NVBENCH
  groupby/group_complex_keys.cpp
  groupby/group_histogram.cpp
  groupby/group_m2_var_std.cpp
  groupby/group_max.cpp
  groupby/group_max_multithreaded.cpp
  groupby/group_no_requests.cpp
  groupby/group_nth.cpp
  groupby/group_nunique.cpp
  groupby/group_rank.cpp
  groupby/group_scan.cpp
  groupby/group_shift.cpp
  groupby/group_struct_keys.cpp
  groupby/group_struct_values.cpp
  groupby/group_sum.cpp
)

# ##################################################################################################
# * hashing benchmark -----------------------------------------------------------------------------
ConfigureNVBench(HASHING_NVBENCH hashing/hash.cpp hashing/partition.cpp)

# ##################################################################################################
# * hybrid scan filters benchmark
# ----------------------------------------------------------------------
ConfigureNVBench(
  HYBRID_SCAN_FILTERS_NVBENCH io/parquet/experimental/hybrid_scan/dict_page_filter.cpp
)

# ##################################################################################################
# * hybrid scan multithread benchmark
# ----------------------------------------------------------------------
ConfigureNVBench(
  HYBRID_SCAN_MULTITHREAD_NVBENCH io/parquet/experimental/hybrid_scan/hybrid_scan_multithread.cpp
  io/parquet/experimental/hybrid_scan/hybrid_scan_composer.cpp
)

# ##################################################################################################
# * interop benchmark ------------------------------------------------------------------------------
ConfigureNVBench(INTEROP_NVBENCH interop/interop.cpp interop/interop_stringview.cpp)
target_link_libraries(INTEROP_NVBENCH PRIVATE nanoarrow::nanoarrow)

# ##################################################################################################
# * merge benchmark -------------------------------------------------------------------------------
ConfigureNVBench(
  MERGE_NVBENCH merge/merge.cpp merge/merge_lists.cpp merge/merge_structs.cpp
  merge/merge_strings.cpp
)

# ##################################################################################################
# * bitmask benchmark ---------------------------------------------------------------------------
ConfigureNVBench(BITMASK_NVBENCH bitmask/bitmask_and.cpp bitmask/set_null_mask.cpp)

# ##################################################################################################
# * parquet writer benchmark ----------------------------------------------------------------------
ConfigureNVBench(
  PARQUET_WRITER_NVBENCH io/parquet/parquet_writer.cpp io/parquet/parquet_writer_chunks.cpp
  io/parquet/parquet_writer_dict.cpp
)

# ##################################################################################################
# * parquet reader benchmark ----------------------------------------------------------------------
ConfigureNVBench(
  PARQUET_READER_NVBENCH io/parquet/parquet_reader_input.cpp io/parquet/parquet_reader_encoding.cpp
  io/parquet/parquet_reader_options.cpp io/parquet/reader_common.cpp
)

# ##################################################################################################
# * parquet reader chunks benchmark
# ------------------------------------------------------------------
ConfigureNVBench(
  PARQUET_READER_CHUNKS_NVBENCH io/parquet/parquet_reader_chunks.cpp io/parquet/reader_common.cpp
)

# ##################################################################################################
# * parquet reader compressed benchmark ------------------------------------------------------------
ConfigureNVBench(
  PARQUET_READER_COMPRESSED_NVBENCH io/parquet/parquet_reader_compressed.cpp
  io/parquet/reader_common.cpp
)

# ##################################################################################################
# * parquet reader strings benchmark
# ------------------------------------------------------------------
ConfigureNVBench(
  PARQUET_READER_STRINGS_NVBENCH io/parquet/parquet_reader_strings.cpp io/parquet/reader_common.cpp
)

# ##################################################################################################
# * parquet reader wide benchmark ------------------------------------------------------------------
ConfigureNVBench(
  PARQUET_READER_WIDE_NVBENCH io/parquet/parquet_reader_wide.cpp io/parquet/reader_common.cpp
)

# ##################################################################################################
# * parquet reader filter benchmark
# ----------------------------------------------------------------------
ConfigureNVBench(PARQUET_READER_FILTER_NVBENCH io/parquet/parquet_reader_filter.cpp)

# ##################################################################################################
# * parquet reader metadata benchmark
# ----------------------------------------------------------------------
ConfigureNVBench(PARQUET_READER_METADATA_NVBENCH io/parquet/parquet_reader_metadata.cpp)

# ##################################################################################################
# * parquet deletion vector benchmark
# ----------------------------------------------------------------------
ConfigureNVBench(
  PARQUET_DELETION_VECTORS_NVBENCH
  io/parquet/experimental/deletion_vectors/parquet_deletion_vectors.cpp
)
target_compile_definitions(
  PARQUET_DELETION_VECTORS_NVBENCH
  PRIVATE DISABLENEON=1 ROARING_DISABLE_X64=1 ROARING_DISABLE_AVX=1
          CROARING_COMPILER_SUPPORTS_AVX512=0
)
target_link_libraries(PARQUET_DELETION_VECTORS_NVBENCH PRIVATE roaring)

# ##################################################################################################
# * parquet multithread reader benchmark
# ----------------------------------------------------------------------
ConfigureNVBench(PARQUET_MULTITHREAD_READER_NVBENCH io/parquet/parquet_reader_multithread.cpp)

# ##################################################################################################
# * cudftable reader benchmark
# --------------------------------------------------------------------------
ConfigureNVBench(CUDFTABLE_READER_NVBENCH io/cudftable/cudftable_reader.cpp)

# ##################################################################################################
# * cudftable writer benchmark
# --------------------------------------------------------------------------
ConfigureNVBench(CUDFTABLE_WRITER_NVBENCH io/cudftable/cudftable_writer.cpp)

# ##################################################################################################
# * orc reader benchmark --------------------------------------------------------------------------
ConfigureNVBench(ORC_READER_NVBENCH io/orc/orc_reader_input.cpp io/orc/orc_reader_options.cpp)

# ##################################################################################################
# * orc multithreaded benchmark
# --------------------------------------------------------------------------
ConfigureNVBench(ORC_MULTITHREADED_NVBENCH io/orc/orc_reader_multithreaded.cpp)

# ##################################################################################################
# * csv reader benchmark --------------------------------------------------------------------------
ConfigureNVBench(CSV_READER_NVBENCH io/csv/csv_reader_input.cpp io/csv/csv_reader_options.cpp)

# ##################################################################################################
# * orc writer benchmark --------------------------------------------------------------------------
ConfigureNVBench(ORC_WRITER_NVBENCH io/orc/orc_writer.cpp io/orc/orc_writer_chunks.cpp)

# ##################################################################################################
# * csv writer benchmark --------------------------------------------------------------------------
ConfigureNVBench(CSV_WRITER_NVBENCH io/csv/csv_writer.cpp)

# ##################################################################################################
# * ast benchmark ---------------------------------------------------------------------------------
ConfigureNVBench(AST_NVBENCH ast/polynomials.cpp ast/transform.cpp)

# ##################################################################################################
# * LTO Fragments ----------------------------------------------------------------------------
rtcx_add_embed(cudf_benchmark_fragments)
add_fragment(cudf_benchmark_fragments FRAGMENT add_f32 SOURCE binaryop/fragments/add_f32.cu)
add_fragment(cudf_benchmark_fragments FRAGMENT mul_f32 SOURCE binaryop/fragments/mul_f32.cu)
rtcx_embed(
  cudf_benchmark_fragments COMPRESSION none OUTPUT_DIRECTORY
  "${CUDF_GENERATED_INCLUDE_DIR}/rtcx_embed"
)

# ##################################################################################################
# * binaryop benchmark ----------------------------------------------------------------------------
ConfigureNVBench(
  BINARYOP_NVBENCH binaryop/binaryop.cpp binaryop/compiled_binaryop.cpp binaryop/polynomials.cpp
  ${cudf_benchmark_fragments_SOURCE_DIR}/cudf_benchmark_fragments.s
)
target_include_directories(BINARYOP_NVBENCH PRIVATE ${cudf_benchmark_fragments_SOURCE_DIR})
add_dependencies(BINARYOP_NVBENCH cudf_benchmark_fragments)

# ##################################################################################################
# * transform benchmark
# ---------------------------------------------------------------------------------
ConfigureNVBench(
  TRANSFORM_NVBENCH transform/encode.cpp transform/polynomials.cpp
  transform/polynomials_concurrent.cpp transform/transform.cpp
)

# ##################################################################################################
# * nvtext benchmark -------------------------------------------------------------------
ConfigureNVBench(
  TEXT_NVBENCH
  text/deduplicate.cpp
  text/edit_distance.cpp
  text/hash_ngrams.cpp
  text/jaccard.cpp
  text/minhash.cpp
  text/ngrams.cpp
  text/normalize.cpp
  text/replace.cpp
  text/subword.cpp
  text/tokenize.cpp
  text/vocab.cpp
)

# ##################################################################################################
# * strings benchmark -------------------------------------------------------------------
ConfigureNVBench(
  STRINGS_NVBENCH
  string/case.cpp
  string/char_types.cpp
  string/combine.cpp
  string/contains.cpp
  string/convert_datetime.cpp
  string/convert_durations.cpp
  string/convert_fixed_point.cpp
  string/convert_numerics.cpp
  string/copy.cpp
  string/copy_if_else.cpp
  string/copy_range.cpp
  string/count.cpp
  string/extract.cpp
  string/factory.cpp
  string/filter.cpp
  string/find.cpp
  string/find_instance.cpp
  string/find_multiple.cpp
  string/intcast.cpp
  string/join_strings.cpp
  string/lengths.cpp
  string/like.cpp
  string/make_strings_column.cu
  string/repeat_strings.cpp
  string/replace.cpp
  string/replace_re.cpp
  string/reverse.cpp
  string/slice.cpp
  string/split.cpp
  string/split_re.cpp
  string/translate.cpp
  string/url_decode.cu
)

# ##################################################################################################
# * strings experimental benchmark ---------------------------------------------------
ConfigureNVBench(STRINGS_EXPERIMENTAL_NVBENCH string/experimental/stringview_compare.cu)
target_link_libraries(STRINGS_EXPERIMENTAL_NVBENCH PRIVATE nanoarrow::nanoarrow)

# ##################################################################################################
# * json benchmark -------------------------------------------------------------------
ConfigureNVBench(JSON_NVBENCH json/json.cu)
ConfigureNVBench(FST_NVBENCH io/fst.cu)
ConfigureNVBench(JSON_READER_NVBENCH io/json/nested_json.cpp io/json/json_reader_input.cpp)
ConfigureNVBench(JSON_READER_OPTION_NVBENCH io/json/json_reader_option.cpp)
ConfigureNVBench(JSON_WRITER_NVBENCH io/json/json_writer.cpp)

# ##################################################################################################
# * io benchmark ---------------------------------------------------------------------
ConfigureNVBench(MULTIBYTE_SPLIT_NVBENCH io/text/multibyte_split.cpp)
target_link_libraries(MULTIBYTE_SPLIT_NVBENCH PRIVATE ZLIB::ZLIB)

# ##################################################################################################
# * decimal benchmark
# ---------------------------------------------------------------------------------
ConfigureNVBench(DECIMAL_NVBENCH decimal/convert_floating.cpp)

# ##################################################################################################
# * reshape benchmark
# ---------------------------------------------------------------------------------
ConfigureNVBench(RESHAPE_NVBENCH reshape/interleave.cpp reshape/table_to_array.cpp)

# ##################################################################################################
# * rolling benchmark
# ---------------------------------------------------------------------------------
ConfigureNVBench(
  ROLLING_NVBENCH rolling/grouped_range_rolling_sum.cu rolling/grouped_rolling_sum.cpp
  rolling/multi_orderby_range_rolling_sum.cpp rolling/range_rolling_sum.cu rolling/rolling_sum.cpp
)

# ##################################################################################################
# * dictionary benchmark
# ---------------------------------------------------------------------------------
ConfigureNVBench(
  DICTIONARY_NVBENCH dictionary/encode.cpp dictionary/set_keys.cpp dictionary/sort.cpp
  dictionary/concatenate.cpp dictionary/match_keys.cpp
)

add_custom_target(
  run_benchmarks
  DEPENDS CUDF_BENCHMARKS
  COMMENT "Custom command for running cudf benchmarks."
)
