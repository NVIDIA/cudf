/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Doxygen group definitions
 */

// This header is only processed by doxygen and does
// not need to be included in any source file.
// Below are the main groups that doxygen uses to build
// the Modules page in the specified order.
//
// To add a new API to an existing group, just use the
// @ingroup tag to the API's doxygen comment.
// Add a new group by first specifying in the hierarchy below.

/**
 * @defgroup default_stream Default Stream
 * @defgroup memory_resource Memory Resource Management
 * @defgroup cudf_classes Classes
 * @{
 *   @defgroup column_classes Column
 *   @{
 *      @defgroup column_factories Factories
 *      @defgroup column_stream Column stream
 *      @defgroup strings_classes Strings
 *      @defgroup dictionary_classes Dictionary
 *      @defgroup timestamp_classes Timestamp
 *      @defgroup lists_classes Lists
 *      @defgroup structs_classes Structs
 *   @}
 *   @defgroup table_classes Table
 *   @defgroup scalar_classes Scalar
 *   @{
 *      @defgroup scalar_factories Factories
 *   @}
 *   @defgroup fixed_point_classes Fixed Point
 * @}
 * @defgroup column_apis Column and Table
 * @{
 *   @defgroup column_copy Copying
 *   @{
 *     @defgroup copy_concatenate Concatenating
 *     @defgroup copy_gather Gathering
 *     @{
 *        @file cudf/copying.hpp
 *     @}
 *     @defgroup copy_scatter Scattering
 *     @{
 *        @file cudf/copying.hpp
 *     @}
 *     @defgroup copy_slice Slicing
 *     @{
 *        @file cudf/copying.hpp
 *     @}
 *     @defgroup copy_split Splitting
 *     @{
 *        @file cudf/contiguous_split.hpp
 *        @file cudf/copying.hpp
 *     @}
 *     @defgroup copy_shift Shifting
 *     @{
 *        @file cudf/copying.hpp
 *     @}
 *   @}
 *   @defgroup column_nullmask Bitmask Operations
 *   @defgroup column_sort Sorting
 *   @defgroup column_search Searching
 *   @defgroup column_hash Hashing
 *   @defgroup column_merge Merging
 *   @defgroup column_join Joining
 *   @defgroup column_quantiles Quantiles
 *   @defgroup column_reduction Reduction
 *   @defgroup column_aggregation Aggregation
 *   @{
 *     @defgroup aggregation_factories Aggregation Factories
 *     @defgroup aggregation_reduction Reduction
 *     @defgroup aggregation_groupby GroupBy
 *     @defgroup aggregation_rolling Rolling Window
 *   @}
 *   @defgroup column_transformation Transformation
 *   @{
 *     @defgroup transformation_unaryops Unary Operations
 *     @defgroup transformation_binaryops Binary Operations
 *     @defgroup transformation_transform Transform
 *     @defgroup transformation_replace Replacing
 *     @defgroup transformation_fill Filling
 *   @}
 *   @defgroup column_reshape Reshaping
 *   @{
 *     @defgroup reshape_transpose Transpose
 *   @}
 *   @defgroup column_reorder Reordering
 *   @{
 *     @defgroup reorder_partition Partitioning
 *     @defgroup reorder_compact Stream Compaction
 *   @}
 *   @defgroup column_interop Interop
 *   @{
 *     @defgroup interop_dlpack DLPack
 *     @defgroup interop_arrow Arrow
 *   @}
 * @}
 * @defgroup datetime_apis DateTime
 * @{
 *   @defgroup datetime_extract Extracting
 *   @defgroup datetime_compute Compute Day
 * @}
 * @defgroup strings_apis Strings
 * @{
 *   @defgroup strings_case Case
 *   @defgroup strings_types Character Types
 *   @defgroup strings_combine Combining
 *   @defgroup strings_contains Searching
 *   @defgroup strings_convert Converting
 *   @defgroup strings_copy Copying
 *   @defgroup strings_slice Slicing
 *   @defgroup strings_find Finding
 *   @defgroup strings_modify Modifying
 *   @defgroup strings_replace Replacing
 *   @defgroup strings_split Splitting
 *   @defgroup strings_extract Extracting
 *   @defgroup strings_regex Regex
 * @}
 * @defgroup dictionary_apis Dictionary
 * @{
 *   @defgroup dictionary_encode Encoding
 *   @defgroup dictionary_search Searching
 *   @defgroup dictionary_update Updating Keys
 * @}
 * @defgroup io_apis IO
 * @{
 *   @defgroup io_types IO Types
 *   @defgroup io_readers Readers
 *   @defgroup io_writers Writers
 *   @defgroup io_datasources Data Sources
 *   @defgroup io_datasinks Data Sinks
 *   @defgroup io_configuration IO Configuration
 *   @defgroup io_utils IO Utilities
 * @}
 * @defgroup json_apis JSON
 * @{
 *   @defgroup json_object JSON Path
 * @}
 * @defgroup lists_apis Lists
 * @{
 *   @defgroup lists_combine Combining
 *   @defgroup lists_modify Modifying
 *   @defgroup lists_extract Extracting
 *   @defgroup lists_filling Filling
 *   @defgroup lists_contains Searching
 *   @defgroup lists_gather Gathering
 *   @defgroup lists_elements Counting
 *   @defgroup lists_filtering Filtering
 *   @defgroup lists_sort Sorting
 *   @defgroup set_operations Set Operations
 * @}
 * @defgroup nvtext_apis NVText
 * @{
 *   @defgroup nvtext_ngrams NGrams
 *   @defgroup nvtext_normalize Normalizing
 *   @defgroup nvtext_stemmer Stemming
 *   @defgroup nvtext_edit_distance Edit Distance
 *   @defgroup nvtext_tokenize Tokenizing
 *   @defgroup nvtext_replace Replacing
 *   @defgroup nvtext_minhash MinHashing
 *   @defgroup nvtext_jaccard Jaccard Index
 *   @defgroup nvtext_dedup Deduplication
 * @}
 * @defgroup utility_apis Utilities
 * @{
 *   @defgroup utility_types Types
 *   @defgroup utility_dispatcher Type Dispatcher
 *   @defgroup utility_bitmask Bitmask
 *   @defgroup utility_error Exception
 *   @defgroup utility_span Exception
 *   @defgroup utility_roaring_bitmap Roaring Bitmap
 * @}
 * @defgroup labeling_apis Labeling
 * @{
 *   @defgroup label_bins Bin Labeling
 * @}
 * @defgroup expressions Expression Evaluation
 * @defgroup tdigest tdigest APIs
 */
