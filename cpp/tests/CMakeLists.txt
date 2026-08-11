# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# ##################################################################################################
# enable testing ################################################################################
# ##################################################################################################
enable_testing()

include(rapids-test)
rapids_test_init()
rapids_cmake_install_lib_dir(lib_dir)

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest CMAKE_TEST_NAME)
  set(options)
  set(one_value GPUS PERCENT STREAM_MODE)
  set(multi_value EXTRA_LIBS EXTRA_COMPILE_DEFINITIONS)
  cmake_parse_arguments(_CUDF_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED _CUDF_TEST_GPUS AND NOT DEFINED _CUDF_TEST_PERCENT)
    set(_CUDF_TEST_GPUS 1)
    set(_CUDF_TEST_PERCENT 15)
  endif()

  if(NOT DEFINED _CUDF_TEST_GPUS)
    set(_CUDF_TEST_GPUS 1)
  endif()

  if(NOT DEFINED _CUDF_TEST_PERCENT)
    set(_CUDF_TEST_PERCENT 100)
  endif()

  if(NOT DEFINED _CUDF_TEST_STREAM_MODE)
    set(_CUDF_TEST_STREAM_MODE cudf)
  endif()

  add_executable(${CMAKE_TEST_NAME} ${_CUDF_TEST_UNPARSED_ARGUMENTS})
  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../${lib_dir}"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using cuda::std
               CXX_EXTENSIONS ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )

  target_compile_definitions(
    ${CMAKE_TEST_NAME} PRIVATE THRUST_FORCE_32_BIT_OFFSET_TYPE=1 CCCL_AVOID_SORT_UNROLL=1
                               ${_CUDF_TEST_EXTRA_COMPILE_DEFINITIONS}
  )

  target_link_libraries(
    ${CMAKE_TEST_NAME} PRIVATE cudf::cudftestutil_objects nvtx3::nvtx3-cpp
                               $<TARGET_NAME_IF_EXISTS:conda_env> "${_CUDF_TEST_EXTRA_LIBS}"
  )
  rapids_cuda_set_runtime(${CMAKE_TEST_NAME} USE_STATIC ON)
  rapids_test_add(
    NAME ${CMAKE_TEST_NAME}
    COMMAND ${CMAKE_TEST_NAME}
    GPUS ${_CUDF_TEST_GPUS}
    PERCENT ${_CUDF_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )

  if(CUDF_BUILD_STREAMS_TEST_UTIL)
    set_tests_properties(
      ${CMAKE_TEST_NAME}
      PROPERTIES
        ENVIRONMENT
        "GTEST_CUDF_STREAM_MODE=new_${_CUDF_TEST_STREAM_MODE}_default;LD_PRELOAD=$<TARGET_FILE:cudf_identify_stream_usage_mode_${_CUDF_TEST_STREAM_MODE}>"
    )
  endif()
endfunction()

# ##################################################################################################
# dependencies  ###################################################################################
# ##################################################################################################

# No need to install Arrow libs when only the final test executables are shipped.
set(CUDF_INSTALL_LIBRARY_DEPS OFF)
set(CUDF_EXCLUDE_DEPS_FROM_ALL ON)
set(CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG EXCLUDE_FROM_ALL)
set(CUDF_ENABLE_ARROW_COMPUTE ON)
include(../cmake/thirdparty/get_arrow.cmake)

# ##################################################################################################
# test sources ##################################################################################
# ##################################################################################################

# ##################################################################################################
# * column tests ----------------------------------------------------------------------------------
ConfigureTest(
  COLUMN_TEST
  column/bit_cast_test.cpp
  column/column_device_view_test.cu
  column/column_test.cpp
  column/column_view_device_span_test.cpp
  column/column_view_shallow_test.cpp
  column/compound_test.cu
)

# ##################################################################################################
# * scalar tests ----------------------------------------------------------------------------------
ConfigureTest(SCALAR_TEST scalar/scalar_test.cpp scalar/scalar_device_view_test.cu)

# ##################################################################################################
# * timestamps tests ------------------------------------------------------------------------------
ConfigureTest(TIMESTAMPS_TEST wrappers/timestamps_test.cu)

# ##################################################################################################
# * groupby tests ---------------------------------------------------------------------------------
ConfigureTest(
  GROUPBY_TEST
  groupby/argmin_tests.cpp
  groupby/argmax_tests.cpp
  groupby/bitwise_tests.cpp
  groupby/collect_list_tests.cpp
  groupby/collect_set_tests.cpp
  groupby/correlation_tests.cpp
  groupby/count_scan_tests.cpp
  groupby/count_tests.cpp
  groupby/covariance_tests.cpp
  groupby/groupby_test_util.cpp
  groupby/groups_tests.cpp
  groupby/histogram_tests.cpp
  groupby/host_udf_example_tests.cu
  groupby/host_udf_tests.cpp
  groupby/keys_tests.cpp
  groupby/lists_tests.cpp
  groupby/m2_tests.cpp
  groupby/min_tests.cpp
  groupby/max_scan_tests.cpp
  groupby/max_tests.cpp
  groupby/mean_tests.cpp
  groupby/median_tests.cpp
  groupby/merge_m2_tests.cpp
  groupby/merge_lists_tests.cpp
  groupby/merge_sets_tests.cpp
  groupby/min_scan_tests.cpp
  groupby/nth_element_tests.cpp
  groupby/nunique_tests.cpp
  groupby/product_scan_tests.cpp
  groupby/product_tests.cpp
  groupby/quantile_tests.cpp
  groupby/rank_scan_tests.cpp
  groupby/replace_nulls_tests.cpp
  groupby/shift_tests.cpp
  groupby/std_tests.cpp
  groupby/streaming_groupby_test.cpp
  groupby/structs_tests.cpp
  groupby/sum_of_squares_tests.cpp
  groupby/sum_scan_tests.cpp
  groupby/sum_tests.cpp
  groupby/sum_overflow_tests.cpp
  groupby/tdigest_tests.cpp
  groupby/topk_tests.cpp
  groupby/var_tests.cpp
  GPUS 1
  PERCENT 100
)

# ##################################################################################################
# * join tests ------------------------------------------------------------------------------------
ConfigureTest(
  JOIN_TEST
  join/join_tests.cpp
  join/conditional_join_tests.cu
  join/cross_join_tests.cpp
  join/semi_anti_join_tests.cpp
  join/mixed_join_tests.cu
  join/direct_join_tests.cpp
  join/distinct_join_tests.cpp
  join/key_remapping_tests.cpp
  GPUS 1
  PERCENT 100
)

# ##################################################################################################
# * is_sorted tests -------------------------------------------------------------------------------
ConfigureTest(IS_SORTED_TEST sort/is_sorted_tests.cpp)

# ##################################################################################################
# * datetime tests --------------------------------------------------------------------------------
ConfigureTest(DATETIME_OPS_TEST datetime/datetime_ops_test.cpp)
ConfigureTest(TIMEZONE_TEST datetime/timezone_test.cpp)

# ##################################################################################################
# * hashing tests ---------------------------------------------------------------------------------
ConfigureTest(
  HASHING_TEST
  hashing/md5_test.cpp
  hashing/murmurhash3_x86_32_test.cpp
  hashing/murmurhash3_x64_128_test.cpp
  hashing/sha1_test.cpp
  hashing/sha224_test.cpp
  hashing/sha256_test.cpp
  hashing/sha384_test.cpp
  hashing/sha512_test.cpp
  hashing/xxhash_32_test.cpp
  hashing/xxhash_64_test.cpp
)

# ##################################################################################################
# * partitioning tests ----------------------------------------------------------------------------
ConfigureTest(
  PARTITIONING_TEST partitioning/hash_partition_test.cpp partitioning/round_robin_test.cpp
  partitioning/partition_test.cpp
  GPUS 1
  PERCENT 70
)

# ##################################################################################################
# * quantiles tests -------------------------------------------------------------------------------
ConfigureTest(
  QUANTILES_TEST quantiles/percentile_approx_test.cpp quantiles/quantile_test.cpp
  quantiles/quantiles_test.cpp
  GPUS 1
  PERCENT 70 EXTRA_LIBS ${ARROW_LIBRARIES}
)

# ##################################################################################################
# * reduction tests -------------------------------------------------------------------------------
ConfigureTest(
  REDUCTIONS_TEST
  reductions/approx_distinct_count_tests.cpp
  reductions/bitwise_tests.cpp
  reductions/collect_ops_tests.cpp
  reductions/distinct_count_tests.cpp
  reductions/ewm_tests.cpp
  reductions/host_udf_example_tests.cu
  reductions/list_rank_test.cpp
  reductions/rank_tests.cpp
  reductions/reduction_tests.cpp
  reductions/scan_tests.cpp
  reductions/segmented_reduction_tests.cpp
  reductions/tdigest_tests.cpp
  reductions/unique_count_tests.cpp
  GPUS 1
  PERCENT 70
)

# ##################################################################################################
# * replace tests ---------------------------------------------------------------------------------
ConfigureTest(REPLACE_TEST replace/replace_tests.cpp)

ConfigureTest(REPLACE_NULLS_TEST replace/replace_nulls_tests.cpp)

ConfigureTest(REPLACE_NANS_TEST replace/replace_nans_tests.cpp)

ConfigureTest(NORMALIZE_REPLACE_TEST replace/normalize_replace_tests.cpp)

ConfigureTest(CLAMP_TEST replace/clamp_test.cpp)

# ##################################################################################################
# * fixed_point tests -----------------------------------------------------------------------------
ConfigureTest(FIXED_POINT_TEST fixed_point/fixed_point_tests.cpp fixed_point/fixed_point_tests.cu)

# ##################################################################################################
# * unary tests -----------------------------------------------------------------------------------
ConfigureTest(UNARY_TEST unary/math_ops_test.cpp unary/unary_ops_test.cpp unary/cast_tests.cpp)

# ##################################################################################################
# * round tests -----------------------------------------------------------------------------------
ConfigureTest(ROUND_TEST round/round_tests.cpp)

# ##################################################################################################
# * binary tests ----------------------------------------------------------------------------------
ConfigureTest(
  BINARYOP_TEST
  binaryop/binop-verify-input-test.cpp
  binaryop/binop-null-test.cpp
  binaryop/binop-compiled-test.cpp
  binaryop/binop-compiled-fixed_point-test.cpp
  binaryop/binop-generic-ptx-test.cpp
)

# ##################################################################################################
# * transform tests -------------------------------------------------------------------------
ConfigureTest(
  TRANSFORM_TEST
  transform/integration/unary_transform_test.cpp
  transform/nans_to_null_test.cpp
  transform/mask_to_bools_test.cpp
  transform/bools_to_mask_test.cpp
  transform/row_bit_count_test.cu
  transform/segmented_row_bit_count_test.cu
  transform/one_hot_encode_tests.cpp
)

# ##################################################################################################
# * filter tests -------------------------------------------------------------------------
ConfigureTest(FILTER_TEST filter/filter_test.cpp)

# ##################################################################################################
# * interop tests -------------------------------------------------------------------------
ConfigureTest(
  INTEROP_TEST
  interop/arrow_data_structures_test.cpp
  interop/to_arrow_device_test.cpp
  interop/to_arrow_test.cpp
  interop/to_arrow_host_test.cpp
  interop/from_arrow_test.cpp
  interop/from_arrow_device_test.cpp
  interop/from_arrow_host_test.cpp
  interop/from_arrow_stream_test.cpp
  interop/dlpack_test.cpp
  EXTRA_LIBS
  nanoarrow::nanoarrow
  ${ARROW_LIBRARIES}
)

# ##################################################################################################
# * io tests --------------------------------------------------------------------------------------
ConfigureTest(COMPRESSION_TEST io/comp/comp_test.cpp)
ConfigureTest(ROW_SELECTION_TEST io/row_selection_test.cpp)
ConfigureTest(FILEPATH_SOURCE_TEST io/filepath_source_test.cpp)

ConfigureTest(
  CSV_TEST io/csv_test.cpp
  GPUS 1
  PERCENT 30
)
ConfigureTest(
  ORC_TEST io/orc_chunked_reader_test.cu io/orc_test.cpp
  GPUS 1
  PERCENT 100
)
ConfigureTest(
  PARQUET_TEST
  io/parquet_bloom_filter_test.cu
  io/parquet_chunked_reader_test.cu
  io/parquet_chunked_writer_test.cpp
  io/parquet_common.cpp
  io/parquet_misc_test.cpp
  io/parquet_reader_dict_test.cpp
  io/parquet_reader_test.cpp
  io/parquet_test.cpp
  io/parquet_v2_test.cpp
  io/parquet_writer_test.cpp
  GPUS 1
  PERCENT 100
)
ConfigureTest(
  HYBRID_SCAN_TEST
  io/experimental/hybrid_scan_common.cpp
  io/experimental/hybrid_scan_composer.cpp
  io/experimental/hybrid_scan_filters_test.cpp
  io/experimental/hybrid_scan_multifile_composer.cpp
  io/experimental/hybrid_scan_multifile_filters_test.cpp
  io/experimental/hybrid_scan_multifile_test.cpp
  io/experimental/hybrid_scan_test.cpp
  io/parquet_common.cpp
  io/parquet_test.cpp
)
ConfigureTest(VARIANT_EXTRACT_TEST io/experimental/variant_extract_test.cpp)
ConfigureTest(
  PARQUET_DELETION_VECTORS_TEST
  io/parquet_deletion_vectors_test.cpp
  io/parquet_common.cpp
  io/parquet_test.cpp
  EXTRA_LIBS
  roaring
  EXTRA_COMPILE_DEFINITIONS
  DISABLENEON=1
  ROARING_DISABLE_X64=1
  ROARING_DISABLE_AVX=1
  CROARING_COMPILER_SUPPORTS_AVX512=0
)
ConfigureTest(
  JSON_TEST io/json/json_chunked_reader.cpp io/json/json_test.cpp io/json/json_type_cast_test.cpp
  io/json/json_utils.cu
  GPUS 1
  PERCENT 30
)
ConfigureTest(JSON_WRITER_TEST io/json/json_writer.cpp)
ConfigureTest(NESTED_JSON_TEST io/json/nested_json_test.cpp io/json/json_tree.cpp)
ConfigureTest(MULTIBYTE_SPLIT_TEST io/text/multibyte_split_test.cpp)
ConfigureTest(JSON_QUOTE_NORMALIZATION_TEST io/json/json_quote_normalization_test.cpp)
ConfigureTest(JSON_WHITESPACE_NORMALIZATION_TEST io/json/json_whitespace_normalization_test.cpp)
ConfigureTest(JSON_TREE_CSR_TEST io/json/json_tree_csr.cu)
ConfigureTest(
  DATA_CHUNK_SOURCE_TEST io/text/data_chunk_source_test.cpp
  GPUS 1
  PERCENT 100
)
target_link_libraries(DATA_CHUNK_SOURCE_TEST PRIVATE ZLIB::ZLIB)
ConfigureTest(LOGICAL_STACK_TEST io/fst/logical_stack_test.cu)
ConfigureTest(FST_TEST io/fst/fst_test.cu)
ConfigureTest(CUDFTABLE_TEST io/cudftable_test.cpp)
ConfigureTest(TYPE_INFERENCE_TEST io/type_inference_test.cpp)

# ##################################################################################################
# * sort tests ------------------------------------------------------------------------------------
ConfigureTest(
  SORT_TEST sort/rank_test.cpp sort/segmented_sort_tests.cpp sort/sort_nested_types_tests.cpp
  sort/sort_test.cpp sort/stable_sort_tests.cpp sort/top_k_tests.cpp
  GPUS 1
  PERCENT 70
)

# ##################################################################################################
# * copying tests ---------------------------------------------------------------------------------
ConfigureTest(
  COPYING_TEST
  copying/concatenate_tests.cpp
  copying/copy_if_else_nested_tests.cpp
  copying/copy_range_tests.cpp
  copying/copy_tests.cpp
  copying/gather_list_tests.cpp
  copying/gather_str_tests.cpp
  copying/gather_struct_tests.cpp
  copying/gather_tests.cpp
  copying/get_value_tests.cpp
  copying/pack_tests.cpp
  copying/purge_nonempty_nulls_tests.cpp
  copying/sample_tests.cpp
  copying/scatter_tests.cpp
  copying/scatter_list_tests.cpp
  copying/scatter_list_scalar_tests.cpp
  copying/scatter_struct_tests.cpp
  copying/scatter_struct_scalar_tests.cpp
  copying/segmented_gather_list_tests.cpp
  copying/shift_tests.cpp
  copying/slice_tests.cpp
  copying/split_tests.cpp
  copying/utility_tests.cpp
  copying/reverse_tests.cpp
  GPUS 1
  PERCENT 100
)

# ##################################################################################################
# * utilities tests -------------------------------------------------------------------------------
ConfigureTest(
  UTILITIES_TEST
  utilities_tests/batched_memcpy_tests.cu
  utilities_tests/batched_memset_tests.cu
  utilities_tests/column_debug_tests.cpp
  utilities_tests/column_utilities_tests.cpp
  utilities_tests/column_wrapper_tests.cpp
  utilities_tests/context_tests.cpp
  utilities_tests/default_stream_tests.cpp
  utilities_tests/io_utilities_tests.cpp
  utilities_tests/lists_column_wrapper_tests.cpp
  utilities_tests/memory_resource_tests.cpp
  utilities_tests/pinned_memory_tests.cpp
  utilities_tests/type_check_tests.cpp
  utilities_tests/type_list_tests.cpp
)

# ##################################################################################################
# * span tests -------------------------------------------------------------------------------

ConfigureTest(SPAN_TEST utilities_tests/span_tests.cu)

# ##################################################################################################
# * iterator tests --------------------------------------------------------------------------------
ConfigureTest(
  ITERATOR_TEST
  iterator/indexalator_test.cu
  iterator/offsetalator_test.cu
  iterator/optional_iterator_test_chrono.cu
  iterator/optional_iterator_test_numeric.cu
  iterator/pair_iterator_test_chrono.cu
  iterator/pair_iterator_test_numeric.cu
  iterator/scalar_iterator_test.cu
  iterator/sizes_to_offsets_iterator_test.cu
  iterator/value_iterator.cpp
  iterator/value_iterator_test_chrono.cu
  iterator/value_iterator_test_numeric.cu
  iterator/value_iterator_test_strings.cu
  iterator/value_iterator_test_transform.cu
)

# ##################################################################################################
# * device atomics tests --------------------------------------------------------------------------
ConfigureTest(DEVICE_ATOMICS_TEST device_atomics/device_atomics_test.cu)

# ##################################################################################################
# * transpose tests -------------------------------------------------------------------------------
ConfigureTest(
  TRANSPOSE_TEST transpose/transpose_test.cpp
  GPUS 1
  PERCENT 70
)

# ##################################################################################################
# * table tests -----------------------------------------------------------------------------------
ConfigureTest(TABLE_TEST table/table_tests.cpp table/table_view_tests.cu)

# ##################################################################################################
# * row operator tests ----------------------------------------------------------------------------
ConfigureTest(
  ROW_OPERATOR_TEST
  row_operator/row_operator_tests.cu
  row_operator/nan_tests.cpp
  row_operator/self_comparison_utilities.cu
  row_operator/two_table_comparison_utilities.cu
  row_operator/two_table_equality_utilities.cu
)

# ##################################################################################################
# * sorted-merge tests ----------------------------------------------------------------------------
ConfigureTest(
  MERGE_TEST merge/merge_test.cpp merge/merge_dictionary_test.cpp merge/merge_string_test.cpp
)

# ##################################################################################################
# * stream compaction tests -----------------------------------------------------------------------
ConfigureTest(
  STREAM_COMPACTION_TEST
  stream_compaction/apply_boolean_mask_tests.cpp
  stream_compaction/distinct_tests.cpp
  stream_compaction/drop_nans_tests.cpp
  stream_compaction/drop_nulls_tests.cpp
  stream_compaction/stable_distinct_tests.cpp
  stream_compaction/unique_tests.cpp
)

# ##################################################################################################
# * rolling tests ---------------------------------------------------------------------------------
ConfigureTest(
  ROLLING_TEST
  rolling/collect_ops_test.cpp
  rolling/empty_input_test.cpp
  rolling/grouped_rolling_range_test.cpp
  rolling/grouped_rolling_test.cpp
  rolling/lead_lag_test.cpp
  rolling/nth_element_test.cpp
  rolling/offset_row_window_test.cpp
  rolling/range_comparator_test.cu
  rolling/range_rolling_window_test.cpp
  rolling/range_window_bounds_test.cpp
  rolling/range_window_type_test.cpp
  rolling/rolling_test.cpp
  GPUS 1
  PERCENT 70
)

# ##################################################################################################
# * filling test ----------------------------------------------------------------------------------
ConfigureTest(
  FILLING_TEST filling/fill_tests.cpp filling/repeat_tests.cpp filling/sequence_tests.cpp
)

# ##################################################################################################
# * search test -----------------------------------------------------------------------------------
ConfigureTest(
  SEARCH_TEST search/search_dictionary_test.cpp search/search_list_test.cpp
  search/search_struct_test.cpp search/search_test.cpp
)

# ##################################################################################################
# * reshape test ----------------------------------------------------------------------------------
ConfigureTest(
  RESHAPE_TEST reshape/byte_cast_tests.cpp reshape/interleave_columns_tests.cpp
  reshape/table_to_array_tests.cpp reshape/tile_tests.cpp
)

# ##################################################################################################
# * traits test -----------------------------------------------------------------------------------
ConfigureTest(TRAITS_TEST types/traits_test.cpp)

# ##################################################################################################
# * factories test --------------------------------------------------------------------------------
ConfigureTest(FACTORIES_TEST scalar/factories_test.cpp column/factories_test.cpp)

# ##################################################################################################
# * dispatcher test -------------------------------------------------------------------------------
ConfigureTest(DISPATCHER_TEST types/type_dispatcher_test.cu)

# ##################################################################################################
# * strings test ----------------------------------------------------------------------------------
ConfigureTest(
  STRINGS_TEST
  strings/array_tests.cpp
  strings/attrs_tests.cpp
  strings/booleans_tests.cpp
  strings/case_tests.cpp
  strings/chars_types_tests.cpp
  strings/combine/concatenate_tests.cpp
  strings/combine/join_list_elements_tests.cpp
  strings/combine/join_strings_tests.cpp
  strings/concatenate_tests.cpp
  strings/contains_tests.cpp
  strings/datetime_tests.cpp
  strings/durations_tests.cpp
  strings/extract_tests.cpp
  strings/factories_test.cu
  strings/fill_tests.cpp
  strings/findall_tests.cpp
  strings/find_tests.cpp
  strings/find_multiple_tests.cpp
  strings/fixed_point_tests.cpp
  strings/floats_tests.cpp
  strings/format_lists_tests.cpp
  strings/integers_tests.cpp
  strings/ipv4_tests.cpp
  strings/like_tests.cpp
  strings/pad_tests.cpp
  strings/repeat_strings_tests.cpp
  strings/replace_regex_tests.cpp
  strings/replace_tests.cpp
  strings/reverse_tests.cpp
  strings/slice_tests.cpp
  strings/split_tests.cpp
  strings/strip_tests.cpp
  strings/translate_tests.cpp
  strings/urls_tests.cpp
)

# ##################################################################################################
# * large strings test ----------------------------------------------------------------------------
ConfigureTest(
  LARGE_STRINGS_TEST
  large_strings/concatenate_tests.cpp
  large_strings/case_tests.cpp
  large_strings/json_tests.cpp
  large_strings/large_strings_fixture.cpp
  large_strings/merge_tests.cpp
  large_strings/parquet_tests.cpp
  large_strings/replace_tests.cpp
  large_strings/reshape_tests.cpp
  large_strings/split_strings_tests.cpp
  io/json/json_utils.cu
  GPUS 1
  PERCENT 100
)

# ##################################################################################################
# * json path test --------------------------------------------------------------------------------
ConfigureTest(JSON_PATH_TEST json/json_tests.cpp)

# ##################################################################################################
# * structs test ----------------------------------------------------------------------------------
ConfigureTest(STRUCTS_TEST structs/structs_column_tests.cpp structs/utilities_tests.cpp)

# ##################################################################################################
# * nvtext test -----------------------------------------------------------------------------------
ConfigureTest(
  TEXT_TEST
  text/deduplicate_tests.cpp
  text/edit_distance_tests.cpp
  text/jaccard_tests.cpp
  text/minhash_tests.cpp
  text/ngrams_tests.cpp
  text/ngrams_tokenize_tests.cpp
  text/normalize_tests.cpp
  text/replace_tests.cpp
  text/stemmer_tests.cpp
  text/subword_tests.cpp
  text/tokenize_tests.cpp
)

# ##################################################################################################
# * bitmask tests ---------------------------------------------------------------------------------
ConfigureTest(
  BITMASK_TEST bitmask/valid_if_tests.cu bitmask/set_nullmask_tests.cu bitmask/bitmask_tests.cpp
  bitmask/is_element_valid_tests.cpp
)

# ##################################################################################################
# * dictionary tests ------------------------------------------------------------------------------
ConfigureTest(
  DICTIONARY_TEST
  dictionary/add_keys_test.cpp
  dictionary/decode_test.cpp
  dictionary/encode_test.cpp
  dictionary/factories_test.cpp
  dictionary/fill_test.cpp
  dictionary/gather_test.cpp
  dictionary/remove_keys_test.cpp
  dictionary/scatter_test.cpp
  dictionary/search_test.cpp
  dictionary/set_keys_test.cpp
  dictionary/slice_test.cpp
  dictionary/sort_test.cpp
)

# ##################################################################################################
# * encode tests -----------------------------------------------------------------------------------
ConfigureTest(ENCODE_TEST encode/encode_tests.cpp)

# ##################################################################################################
# * ast tests -------------------------------------------------------------------------------------
ConfigureTest(AST_TEST ast/transform_tests.cpp ast/ast_tree_tests.cpp ast/jit_expressions_tests.cpp)

# ##################################################################################################
# * transform LTO tests
# -------------------------------------------------------------------------------------
rtcx_add_embed(cudf_test_fragments)

add_fragment(
  cudf_test_fragments FRAGMENT bankers_rounding SOURCE transform/fragments/bankers_rounding.cu
)

add_fragment(cudf_test_fragments FRAGMENT distance SOURCE transform/fragments/distance.cu)

add_fragment(cudf_test_fragments FRAGMENT invsqrt SOURCE transform/fragments/invsqrt.cu)

add_fragment(cudf_test_fragments FRAGMENT lehmer_mean SOURCE transform/fragments/lehmer_mean.cu)

add_fragment(
  cudf_test_fragments FRAGMENT sum_of_squares SOURCE transform/fragments/sum_of_squares.cu
)

add_fragment(cudf_test_fragments FRAGMENT to_upper SOURCE transform/fragments/to_upper.cu)

rtcx_embed(
  cudf_test_fragments COMPRESSION none OUTPUT_DIRECTORY "${CUDF_GENERATED_INCLUDE_DIR}/rtcx_embed"
)

ConfigureTest(
  TRANSFORM_LTO_TEST transform/transform_lto_test.cpp
  ${cudf_test_fragments_SOURCE_DIR}/cudf_test_fragments.s
)
add_dependencies(TRANSFORM_LTO_TEST cudf_test_fragments)
target_include_directories(TRANSFORM_LTO_TEST PRIVATE ${cudf_test_fragments_SOURCE_DIR})

# ##################################################################################################
# * lists tests ----------------------------------------------------------------------------------
ConfigureTest(
  LISTS_TEST
  lists/combine/concatenate_list_elements_tests.cpp
  lists/combine/concatenate_rows_tests.cpp
  lists/contains_tests.cpp
  lists/count_elements_tests.cpp
  lists/explode_tests.cpp
  lists/extract_tests.cpp
  lists/reverse_tests.cpp
  lists/sequences_tests.cpp
  lists/set_operations/difference_distinct_tests.cpp
  lists/set_operations/have_overlap_tests.cpp
  lists/set_operations/intersect_distinct_tests.cpp
  lists/set_operations/union_distinct_tests.cpp
  lists/sort_lists_tests.cpp
  lists/stream_compaction/apply_boolean_mask_tests.cpp
  lists/stream_compaction/distinct_tests.cpp
  GPUS 1
  PERCENT 70
)

# ##################################################################################################
# * bin tests ----------------------------------------------------------------------------------
ConfigureTest(LABEL_BINS_TEST labeling/label_bins_tests.cpp)

# ##################################################################################################
# * jit tests ----------------------------------------------------------------------------------
ConfigureTest(JIT_PARSER_TEST jit/parse_ptx_function.cpp)
target_include_directories(JIT_PARSER_TEST PRIVATE "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>")

ConfigureTest(ROW_IR_TEST jit/row_ir.cpp)

# ##################################################################################################
# * stream testing ---------------------------------------------------------------------------------
if(CUDF_BUILD_STREAMS_TEST_UTIL)
  ConfigureTest(
    STREAM_IDENTIFICATION_TEST identify_stream_usage/test_default_stream_identification.cu
  )
endif()

ConfigureTest(STREAM_BINARYOP_TEST streams/binaryop_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_COLUMN_VIEW_TEST streams/column_view_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_CONCATENATE_TEST streams/concatenate_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_COPYING_TEST streams/copying_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_DATETIME_TEST streams/datetime_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_DICTIONARY_TEST streams/dictionary_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_FILLING_TEST streams/filling_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_GROUPBY_TEST streams/groupby_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_HASHING_TEST streams/hash_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_INTEROP streams/interop_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_IO_CSV_TEST streams/io/csv_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_IO_JSON_TEST streams/io/json_test.cpp STREAM_MODE testing)
ConfigureTest(
  STREAM_IO_MULTIBYTE_SPLIT_TEST streams/io/multibyte_split_test.cpp STREAM_MODE testing
)
ConfigureTest(STREAM_IO_ORC_TEST streams/io/orc_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_IO_PARQUET_TEST streams/io/parquet_test.cpp STREAM_MODE testing)
ConfigureTest(
  STREAM_IO_HYBRID_SCAN_TEST streams/io/experimental/hybrid_scan_test.cpp STREAM_MODE testing
)
ConfigureTest(STREAM_JOIN_TEST streams/join_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_LABELING_BINS_TEST streams/labeling_bins_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_LISTS_TEST streams/lists_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_MERGE_TEST streams/merge_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_NULL_MASK_TEST streams/null_mask_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_PARTITIONING_TEST streams/partitioning_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_POOL_TEST streams/pool_test.cu STREAM_MODE testing)
ConfigureTest(STREAM_QUANTILE_TEST streams/quantile_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_REDUCTION_TEST streams/reduction_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_REPLACE_TEST streams/replace_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_RESHAPE_TEST streams/reshape_test.cpp STREAM_MODE testing)
ConfigureTest(
  STREAM_ROARING_BITMAP_TEST
  streams/roaring_bitmap_test.cpp
  STREAM_MODE
  testing
  EXTRA_LIBS
  roaring
  EXTRA_COMPILE_DEFINITIONS
  DISABLENEON=1
  ROARING_DISABLE_X64=1
  ROARING_DISABLE_AVX=1
  CROARING_COMPILER_SUPPORTS_AVX512=0
)
ConfigureTest(STREAM_ROLLING_TEST streams/rolling_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_ROUND_TEST streams/round_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_SCALAR_TEST streams/scalar_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_SEARCH_TEST streams/search_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_SORTING_TEST streams/sorting_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_STREAM_COMPACTION_TEST streams/stream_compaction_test.cpp STREAM_MODE testing)
ConfigureTest(
  STREAM_STRINGS_TEST
  streams/strings/attributes_test.cpp
  streams/strings/case_test.cpp
  streams/strings/combine_test.cpp
  streams/strings/contains_test.cpp
  streams/strings/convert_test.cpp
  streams/strings/extract_test.cpp
  streams/strings/factory_test.cpp
  streams/strings/filter_test.cpp
  streams/strings/find_test.cpp
  streams/strings/replace_test.cpp
  streams/strings/reverse_test.cpp
  streams/strings/split_test.cpp
  streams/strings/strings_tests.cpp
  STREAM_MODE
  testing
)
ConfigureTest(STREAM_TABLE_EQUALITY_TEST streams/table_equality_test.cpp STREAM_MODE testing)
ConfigureTest(
  STREAM_TEXT_TEST
  streams/text/edit_distance_test.cpp
  streams/text/ngrams_test.cpp
  streams/text/replace_test.cpp
  streams/text/stemmer_test.cpp
  streams/text/tokenize_test.cpp
  STREAM_MODE
  testing
)
ConfigureTest(STREAM_TRANSFORM_TEST streams/transform_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_TRANSPOSE_TEST streams/transpose_test.cpp STREAM_MODE testing)
ConfigureTest(STREAM_UNARY_TEST streams/unary_test.cpp STREAM_MODE testing)

# ##################################################################################################
# * roaring bitmap tests ---------------------------------------------------------------------------
ConfigureTest(
  ROARING_BITMAP_TEST
  utilities/roaring_bitmap_test.cpp
  EXTRA_LIBS
  roaring
  EXTRA_COMPILE_DEFINITIONS
  DISABLENEON=1
  ROARING_DISABLE_X64=1
  ROARING_DISABLE_AVX=1
  CROARING_COMPILER_SUPPORTS_AVX512=0
)

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcudf)
