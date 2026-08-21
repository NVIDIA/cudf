/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <numeric>
#include <sstream>

namespace cudf {

namespace test {

namespace {

std::unique_ptr<column> generate_all_row_indices(size_type num_rows,
                                                 cuda::stream_ref stream,
                                                 cudf::memory_resources mr)
{
  auto indices = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, num_rows, mask_state::UNALLOCATED, stream, mr.get_temporary_mr());
  thrust::sequence(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                   indices->mutable_view().begin<size_type>(),
                   indices->mutable_view().end<size_type>(),
                   0);
  return indices;
}

// generate the rows indices that should be checked for the child column of a list column.
//
// - if we are just checking for equivalence, we can skip any rows that are nulls. this allows
//   things like non-empty rows that have been nullified after creation.  they may actually contain
//   values, but since the row is null they don't matter for equivalency.
//
// - if we are checking for exact equality, we need to check all rows.
//
//   This allows us to differentiate between:
//
//  List<int32_t>:
//    Length : 1
//    Offsets : 0, 4
//    Null count: 1
//    0
//       0, 1, 2, 3
//
//  List<int32_t>:
//    Length : 1
//    Offsets : 0, 0
//    Null count: 1
//    0
//
std::unique_ptr<column> generate_child_row_indices(lists_column_view const& c,
                                                   column_view const& row_indices,
                                                   bool check_exact_equality,
                                                   cuda::stream_ref stream,
                                                   cudf::memory_resources mr)
{
  // if we are checking for exact equality, we should be checking for "unsanitized" data that may
  // be hiding underneath nulls. so check all rows instead of just non-null rows
  if (check_exact_equality) {
    return generate_all_row_indices(c.get_sliced_child(stream).size(), stream, mr);
  }

  // Example input
  // List<int32_t>:
  // Length : 7
  // Offsets : 0, 3, 6, 8, 11, 14, 16, 19
  //                 |     |                        <-- non-null input rows
  // Null count: 5
  // 0010100
  //    1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7
  //                      |  |           |  |  |    <-- child rows of non-null rows
  //
  // Desired output:  [6, 7, 11, 12, 13]

  // compute total # of child row indices we will be emitting.
  auto row_size_iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>([row_indices = row_indices.begin<size_type>(),
                                           validity    = c.null_mask(),
                                           offsets     = c.offsets().begin<int32_t>(),
                                           offset      = c.offset()] __device__(int index) {
      // both null mask and offsets data are not pre-sliced. so we need to add the column offset to
      // every incoming index.
      auto const true_index = row_indices[index] + offset;
      return !validity || cudf::bit_is_set(validity, true_index)
               ? (offsets[true_index + 1] - offsets[true_index])
               : 0;
    }));
  auto const output_size = thrust::reduce(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                                          row_size_iter,
                                          row_size_iter + row_indices.size());
  // no output. done.
  auto result = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED, stream, mr.get_temporary_mr());
  if (output_size == 0) { return result; }

  // for all input rows, what position in the output column they will start at.
  //
  // output_row_start = [0, 0, 0, 2, 2, 5, 5]
  //                           |     |              <-- non-null input rows
  //
  auto output_row_start = cudf::make_fixed_width_column(data_type{type_id::INT32},
                                                        row_indices.size(),
                                                        mask_state::UNALLOCATED,
                                                        stream,
                                                        mr.get_temporary_mr());
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                         row_size_iter,
                         row_size_iter + row_indices.size(),
                         output_row_start->mutable_view().begin<size_type>());

  // fill result column with 1s
  //
  // result = [1, 1, 1, 1, 1]
  //
  thrust::generate(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                   result->mutable_view().begin<size_type>(),
                   result->mutable_view().end<size_type>(),
                   cuda::proclaim_return_type<size_type>([] __device__() { return 1; }));

  // scatter the output row positions into result buffer
  //
  // result = [6, 1, 11, 1, 1]
  //
  auto output_row_iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>(
      [row_indices = row_indices.begin<size_type>(),
       offsets     = c.offsets().begin<int32_t>(),
       offset      = c.offset(),
       first_offset =
         cudf::detail::get_value<int32_t>(c.offsets(), c.offset(), stream)] __device__(int index) {
        auto const true_index = row_indices[index] + offset;
        return offsets[true_index] - first_offset;
      }));
  thrust::scatter_if(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                     output_row_iter,
                     output_row_iter + row_indices.size(),
                     output_row_start->view().begin<size_type>(),
                     row_size_iter,
                     result->mutable_view().begin<size_type>(),
                     [] __device__(auto row_size) { return row_size != 0; });

  // generate keys for each output row
  //
  // result = [1, 1, 2, 2, 2]
  //
  auto keys = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED, stream, mr.get_temporary_mr());
  thrust::generate(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                   keys->mutable_view().begin<size_type>(),
                   keys->mutable_view().end<size_type>(),
                   cuda::proclaim_return_type<size_type>([] __device__() { return 0; }));
  thrust::scatter_if(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                     row_size_iter,
                     row_size_iter + row_indices.size(),
                     output_row_start->view().begin<size_type>(),
                     row_size_iter,
                     keys->mutable_view().begin<size_type>(),
                     [] __device__(auto row_size) { return row_size != 0; });
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                         keys->view().begin<size_type>(),
                         keys->view().end<size_type>(),
                         keys->mutable_view().begin<size_type>());

  // scan by key to generate final child row indices.
  // input
  //    result = [6, 1, 11, 1, 1]
  //    keys   = [1, 1, 2,  2, 2]
  //
  // output
  //    result = [6, 7, 11, 12, 13]
  //
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                                keys->view().begin<size_type>(),
                                keys->view().end<size_type>(),
                                result->view().begin<size_type>(),
                                result->mutable_view().begin<size_type>());
  return result;
}

#define PROP_EXPECT_EQ(a, b)                                \
  do {                                                      \
    if (verbosity == debug_output_level::QUIET) {           \
      if (a != b) { return false; }                         \
    } else {                                                \
      EXPECT_EQ(a, b);                                      \
      if (a != b) {                                         \
        if (verbosity == debug_output_level::FIRST_ERROR) { \
          return false;                                     \
        } else {                                            \
          result = false;                                   \
        }                                                   \
      }                                                     \
    }                                                       \
  } while (0)

template <bool check_exact_equality>
struct column_property_comparator {
  bool compare_common(cudf::column_view const& lhs,
                      cudf::column_view const& rhs,
                      cudf::column_view const& lhs_row_indices,
                      cudf::column_view const& rhs_row_indices,
                      debug_output_level verbosity,
                      cuda::stream_ref stream,
                      cudf::memory_resources mr)
  {
    bool result = true;

    if (check_exact_equality) {
      PROP_EXPECT_EQ(cudf::have_same_types(lhs, rhs), true);
    } else {
      PROP_EXPECT_EQ(cudf::column_types_equivalent(lhs, rhs), true);
    }

    auto const lhs_size = check_exact_equality ? lhs.size() : lhs_row_indices.size();
    auto const rhs_size = check_exact_equality ? rhs.size() : rhs_row_indices.size();
    PROP_EXPECT_EQ(lhs_size, rhs_size);

    if (lhs_size > 0 && check_exact_equality) { PROP_EXPECT_EQ(lhs.nullable(), rhs.nullable()); }

    PROP_EXPECT_EQ(lhs.null_count(), rhs.null_count());

    // equivalent, but not exactly equal columns can have a different number of children if their
    // sizes are both 0. Specifically, empty string columns may or may not have children.
    if (check_exact_equality || (lhs.size() > 0 && lhs.null_count() < lhs.size())) {
      PROP_EXPECT_EQ(lhs.num_children(), rhs.num_children());
    }

    return result;
  }

  template <typename T>
  bool operator()(cudf::column_view const& lhs,
                  cudf::column_view const& rhs,
                  cudf::column_view const& lhs_row_indices,
                  cudf::column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr)
    requires(!std::is_same_v<T, cudf::list_view> && !std::is_same_v<T, cudf::struct_view>)
  {
    return compare_common(lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity, stream, mr);
  }

  template <typename T>
  bool operator()(cudf::column_view const& lhs,
                  cudf::column_view const& rhs,
                  cudf::column_view const& lhs_row_indices,
                  cudf::column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr)
    requires(std::is_same_v<T, cudf::list_view>)
  {
    if (!compare_common(lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity, stream, mr)) {
      return false;
    }

    cudf::lists_column_view lhs_l(lhs);
    cudf::lists_column_view rhs_l(rhs);

    // recurse

    // note: if a column is all nulls (and we are checking for exact equality) or otherwise empty,
    // no indices are generated and no recursion happens
    auto lhs_child_indices =
      generate_child_row_indices(lhs_l, lhs_row_indices, check_exact_equality, stream, mr);
    if (lhs_child_indices->size() > 0) {
      auto lhs_child = lhs_l.get_sliced_child(stream);
      auto rhs_child = rhs_l.get_sliced_child(stream);
      auto rhs_child_indices =
        generate_child_row_indices(rhs_l, rhs_row_indices, check_exact_equality, stream, mr);
      return cudf::type_dispatcher(lhs_child.type(),
                                   column_property_comparator<check_exact_equality>{},
                                   lhs_child,
                                   rhs_child,
                                   *lhs_child_indices,
                                   *rhs_child_indices,
                                   verbosity,
                                   stream,
                                   mr);
    }
    return true;
  }

  template <typename T>
  bool operator()(cudf::column_view const& lhs,
                  cudf::column_view const& rhs,
                  cudf::column_view const& lhs_row_indices,
                  cudf::column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr)
    requires(std::is_same_v<T, cudf::struct_view>)
  {
    if (!compare_common(lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity, stream, mr)) {
      return false;
    }

    structs_column_view l_scv(lhs);
    structs_column_view r_scv(rhs);

    for (size_type i = 0; i < lhs.num_children(); i++) {
      column_view lhs_child = l_scv.get_sliced_child(i, stream);
      column_view rhs_child = r_scv.get_sliced_child(i, stream);
      if (!cudf::type_dispatcher(lhs_child.type(),
                                 column_property_comparator<check_exact_equality>{},
                                 lhs_child,
                                 rhs_child,
                                 lhs_row_indices,
                                 rhs_row_indices,
                                 verbosity,
                                 stream,
                                 mr)) {
        return false;
      }
    }

    return true;
  }
};

template <typename DeviceComparator>
class corresponding_rows_unequal {
 public:
  corresponding_rows_unequal(column_device_view lhs_row_indices_,
                             column_device_view rhs_row_indices_,
                             size_type /*fp_ulps*/,
                             DeviceComparator comp_,
                             column_device_view /*lhs*/,
                             column_device_view /*rhs*/)
    : lhs_row_indices(lhs_row_indices_), rhs_row_indices(rhs_row_indices_), comp(comp_)
  {
  }

  __device__ bool operator()(size_type index)
  {
    using cudf::detail::row::lhs_index_type;
    using cudf::detail::row::rhs_index_type;

    return !comp(lhs_index_type{lhs_row_indices.element<size_type>(index)},
                 rhs_index_type{rhs_row_indices.element<size_type>(index)});
  }

  column_device_view lhs_row_indices;
  column_device_view rhs_row_indices;
  DeviceComparator comp;
};

template <typename DeviceComparator>
class corresponding_rows_not_equivalent {
  column_device_view lhs_row_indices;
  column_device_view rhs_row_indices;
  size_type const fp_ulps;
  DeviceComparator comp;
  column_device_view lhs;
  column_device_view rhs;

 public:
  corresponding_rows_not_equivalent(column_device_view lhs_row_indices_,
                                    column_device_view rhs_row_indices_,
                                    size_type fp_ulps_,
                                    DeviceComparator comp_,
                                    column_device_view lhs_,
                                    column_device_view rhs_)
    : lhs_row_indices(lhs_row_indices_),
      rhs_row_indices(rhs_row_indices_),
      fp_ulps(fp_ulps_),
      comp(comp_),
      lhs(lhs_),
      rhs(rhs_)
  {
  }

  struct typed_element_not_equivalent {
    template <typename T>
    __device__ bool operator()(column_device_view const& lhs,
                               column_device_view const& rhs,
                               size_type lhs_index,
                               size_type rhs_index,
                               size_type fp_ulps)
      requires(std::is_floating_point_v<T>)
    {
      if (lhs.is_valid(lhs_index) and rhs.is_valid(rhs_index)) {
        T const x = lhs.element<T>(lhs_index);
        T const y = rhs.element<T>(rhs_index);

        // Must handle inf and nan separately
        if (cuda::std::isinf(x) || cuda::std::isinf(y)) {
          return x != y;  // comparison of (inf==inf) returns true
        } else if (cuda::std::isnan(x) || cuda::std::isnan(y)) {
          return cuda::std::isnan(x) !=
                 cuda::std::isnan(y);  // comparison of (nan==nan) returns false
        } else {
          T const abs_x_minus_y = cuda::std::abs(x - y);
          return abs_x_minus_y >= cuda::std::numeric_limits<T>::min() &&
                 abs_x_minus_y >
                   cuda::std::numeric_limits<T>::epsilon() * cuda::std::abs(x + y) * fp_ulps;
        }
      } else {
        // if either is null, then the inequality was checked already
        return true;
      }
    }

    template <typename T, typename... Args>
    __device__ bool operator()(Args...)
      requires(not std::is_floating_point_v<T>)
    {
      // Non-floating point inequality is checked already
      return true;
    }
  };

  __device__ bool operator()(size_type index)
  {
    using cudf::detail::row::lhs_index_type;
    using cudf::detail::row::rhs_index_type;

    auto const lhs_index = lhs_row_indices.element<size_type>(index);
    auto const rhs_index = rhs_row_indices.element<size_type>(index);

    if (not comp(lhs_index_type{lhs_index}, rhs_index_type{rhs_index})) {
      return type_dispatcher(
        lhs.type(), typed_element_not_equivalent{}, lhs, rhs, lhs_index, rhs_index, fp_ulps);
    }
    return false;
  }
};

// Stringify the inconsistent values resulted from the comparison of two columns element-wise
std::string stringify_column_differences(cudf::device_span<int const> differences,
                                         column_view const& lhs,
                                         column_view const& rhs,
                                         column_view const& lhs_row_indices,
                                         column_view const& rhs_row_indices,
                                         debug_output_level verbosity,
                                         int depth,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
{
  CUDF_EXPECTS(not differences.empty(), "Shouldn't enter this function if `differences` is empty");
  std::string const depth_str = depth > 0 ? "depth " + std::to_string(depth) + '\n' : "";
  // move the differences to the host.
  auto h_differences = cudf::detail::make_host_vector(differences, stream);
  if (verbosity == debug_output_level::ALL_ERRORS) {
    std::ostringstream buffer;
    buffer << depth_str << "differences:" << std::endl;

    auto source_table = cudf::table_view({lhs, rhs});
    auto diff_column  = fixed_width_column_wrapper<int32_t>(
      h_differences.begin(), h_differences.end(), stream, mr.get_temporary_mr());
    auto diff_table = cudf::gather(source_table,
                                   diff_column,
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   stream,
                                   mr.get_temporary_mr());
    //  Need to pull back the differences
    auto const h_left_strings = to_strings(diff_table->get_column(0), stream, mr);

    auto const h_right_strings = to_strings(diff_table->get_column(1), stream, mr);
    for (size_t i = 0; i < h_differences.size(); ++i)
      buffer << depth_str << "lhs[" << h_differences[i] << "] = " << h_left_strings[i] << ", rhs["
             << h_differences[i] << "] = " << h_right_strings[i] << std::endl;
    return buffer.str();
  } else {
    auto const index = h_differences[0];  // only stringify first difference

    auto const lhs_index = cudf::detail::get_value<size_type>(lhs_row_indices, index, stream);
    auto const rhs_index = cudf::detail::get_value<size_type>(rhs_row_indices, index, stream);
    auto diff_lhs        = cudf::slice(lhs, {lhs_index, lhs_index + 1}).front();
    auto diff_rhs        = cudf::slice(rhs, {rhs_index, rhs_index + 1}).front();
    return depth_str + "first difference: " + "lhs[" + std::to_string(index) +
           "] = " + to_string(diff_lhs, "", stream, mr) + ", rhs[" + std::to_string(index) +
           "] = " + to_string(diff_rhs, "", stream, mr);
  }
}

// non-nested column types
template <typename T, bool check_exact_equality>
struct column_comparator_impl {
  bool operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  size_type fp_ulps,
                  int depth,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr)
  {
    auto d_lhs_row_indices =
      cudf::column_device_view::create(lhs_row_indices, stream, mr.get_temporary_mr());
    auto d_rhs_row_indices =
      cudf::column_device_view::create(rhs_row_indices, stream, mr.get_temporary_mr());

    auto d_lhs = cudf::column_device_view::create(lhs, stream, mr.get_temporary_mr());
    auto d_rhs = cudf::column_device_view::create(rhs, stream, mr.get_temporary_mr());

    auto lhs_tview = table_view{{lhs}};
    auto rhs_tview = table_view{{rhs}};

    auto const comparator = cudf::detail::row::equality::two_table_comparator{
      lhs_tview, rhs_tview, stream, mr.get_temporary_mr()};
    auto const has_nulls = cudf::has_nulls(lhs_tview) or cudf::has_nulls(rhs_tview);

    auto const device_comparator = comparator.equal_to<false>(cudf::nullate::DYNAMIC{has_nulls});

    using ComparatorType =
      std::conditional_t<check_exact_equality,
                         corresponding_rows_unequal<decltype(device_comparator)>,
                         corresponding_rows_not_equivalent<decltype(device_comparator)>>;

    auto differences =
      rmm::device_uvector<int>(lhs_row_indices.size(),
                               stream,
                               mr.get_temporary_mr());  // worst case: everything different
    auto input_iter = cuda::counting_iterator<cudf::size_type>{0};

    auto diff_map =
      rmm::device_uvector<bool>(lhs_row_indices.size(), stream, mr.get_temporary_mr());

    thrust::transform(
      rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
      input_iter,
      input_iter + lhs_row_indices.size(),
      diff_map.begin(),
      ComparatorType(
        *d_lhs_row_indices, *d_rhs_row_indices, fp_ulps, device_comparator, *d_lhs, *d_rhs));

    auto diff_iter = thrust::copy_if(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                                     input_iter,
                                     input_iter + lhs_row_indices.size(),
                                     diff_map.begin(),
                                     differences.begin(),
                                     cuda::std::identity{});

    differences.resize(cuda::std::distance(differences.begin(), diff_iter),
                       stream);  // shrink back down

    if (not differences.is_empty()) {
      if (verbosity != debug_output_level::QUIET) {
        // GTEST_FAIL() does a return that conflicts with our return type. so hide it in a lambda.
        [&]() {
          GTEST_FAIL() << stringify_column_differences(
            differences, lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity, depth, stream, mr);
        }();
      }
      return false;
    }
    return true;
  }
};

// forward declaration for nested-type recursion.
template <bool check_exact_equality>
struct column_comparator;

// specialization for list columns
template <bool check_exact_equality>
struct column_comparator_impl<list_view, check_exact_equality> {
  bool operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  size_type fp_ulps,
                  int depth,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr)
  {
    lists_column_view lhs_l(lhs);
    lists_column_view rhs_l(rhs);

    CUDF_EXPECTS(lhs_row_indices.size() == rhs_row_indices.size(), "List column size mismatch");
    if (lhs_row_indices.is_empty()) { return true; }

    // worst case - everything is different
    rmm::device_uvector<int> differences(lhs_row_indices.size(), stream, mr.get_temporary_mr());

    // compare offsets, taking slicing into account

    // left side
    size_type lhs_shift = cudf::detail::get_value<int32_t>(lhs_l.offsets(), lhs_l.offset(), stream);
    auto lhs_offsets    = cuda::transform_iterator(
      lhs_l.offsets().begin<int32_t>() + lhs_l.offset(),
      cuda::proclaim_return_type<size_type>(
        [lhs_shift] __device__(size_type offset) { return offset - lhs_shift; }));
    auto lhs_valids = cuda::transform_iterator(
      cuda::counting_iterator<cudf::size_type>{0},
      cuda::proclaim_return_type<bool>(
        [mask = lhs_l.null_mask(), offset = lhs_l.offset()] __device__(size_type index) {
          return mask == nullptr ? true : cudf::bit_is_set(mask, index + offset);
        }));

    // right side
    size_type rhs_shift = cudf::detail::get_value<int32_t>(rhs_l.offsets(), rhs_l.offset(), stream);
    auto rhs_offsets    = cuda::transform_iterator(
      rhs_l.offsets().begin<int32_t>() + rhs_l.offset(),
      cuda::proclaim_return_type<size_type>(
        [rhs_shift] __device__(size_type offset) { return offset - rhs_shift; }));
    auto rhs_valids = cuda::transform_iterator(
      cuda::counting_iterator<cudf::size_type>{0},
      cuda::proclaim_return_type<bool>(
        [mask = rhs_l.null_mask(), offset = rhs_l.offset()] __device__(size_type index) {
          return mask == nullptr ? true : cudf::bit_is_set(mask, index + offset);
        }));

    // when checking for equivalency, we can't compare offset values directly, we can only
    // compare lengths of the rows, and only if valid.  as a concrete example, you could have two
    // equivalent columns with the following data:
    //
    // column A
    //    offsets =  [0, 3, 5, 7]
    //    validity = [0, 1, 1, 1]
    //
    // column B
    //   offsets =   [0, 0, 2, 4]
    //   validity =  [0, 1, 1, 1]
    //
    // Row 0 in column A happens to have a positive length, even though the row is null, but column
    // B does not.  So the offsets for the remaining valid rows are fundamentally different even
    // though the row lengths are the same.
    //
    auto input_iter = cuda::counting_iterator<cudf::size_type>{0};
    auto diff_iter  = thrust::copy_if(
      rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
      input_iter,
      input_iter + lhs_row_indices.size(),
      differences.begin(),
      [lhs_offsets,
       rhs_offsets,
       lhs_valids,
       rhs_valids,
       lhs_indices = lhs_row_indices.begin<size_type>(),
       rhs_indices = rhs_row_indices.begin<size_type>()] __device__(size_type index) {
        auto const lhs_index = lhs_indices[index];
        auto const rhs_index = rhs_indices[index];

        // check for validity match
        if (lhs_valids[lhs_index] != rhs_valids[rhs_index]) { return true; }

        // if the row is valid, check that the length of the list is the same. do this
        // for both the equivalency and exact equality checks.
        if (lhs_valids[lhs_index] && ((lhs_offsets[lhs_index + 1] - lhs_offsets[lhs_index]) !=
                                      (rhs_offsets[rhs_index + 1] - rhs_offsets[rhs_index]))) {
          return true;
        }

        // if validity matches -and- is false, we can ignore the actual offset values. this
        // is technically not checking "equal()", but it's how the non-list code path handles it
        if (!lhs_valids[lhs_index]) { return false; }

        // if checking exact equality, compare the actual offset values
        if (check_exact_equality && lhs_offsets[lhs_index] != rhs_offsets[rhs_index]) {
          return true;
        }

        return false;
      });

    differences.resize(cuda::std::distance(differences.begin(), diff_iter),
                       stream);  // shrink back down

    if (not differences.is_empty()) {
      if (verbosity != debug_output_level::QUIET) {
        // GTEST_FAIL() does a return that conflicts with our return type. so hide it in a lambda.
        [&]() {
          GTEST_FAIL() << stringify_column_differences(
            differences, lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity, depth, stream, mr);
        }();
      }
      return false;
    }

    // recurse
    // note: if a column is all nulls (and we are only checking for equivalence) or otherwise empty,
    // no indices are generated and no recursion happens
    auto lhs_child_indices =
      generate_child_row_indices(lhs_l, lhs_row_indices, check_exact_equality, stream, mr);
    if (lhs_child_indices->size() > 0) {
      auto lhs_child = lhs_l.get_sliced_child(stream);
      auto rhs_child = rhs_l.get_sliced_child(stream);
      auto rhs_child_indices =
        generate_child_row_indices(rhs_l, rhs_row_indices, check_exact_equality, stream, mr);
      return cudf::type_dispatcher(lhs_child.type(),
                                   column_comparator<check_exact_equality>{},
                                   lhs_child,
                                   rhs_child,
                                   *lhs_child_indices,
                                   *rhs_child_indices,
                                   verbosity,
                                   fp_ulps,
                                   depth + 1,
                                   stream,
                                   mr);
    }

    return true;
  }
};

template <bool check_exact_equality>
struct column_comparator_impl<struct_view, check_exact_equality> {
  bool operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  size_type fp_ulps,
                  int depth,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr)
  {
    structs_column_view l_scv(lhs);
    structs_column_view r_scv(rhs);

    for (size_type i = 0; i < lhs.num_children(); i++) {
      column_view lhs_child = l_scv.get_sliced_child(i, stream);
      column_view rhs_child = r_scv.get_sliced_child(i, stream);
      if (!cudf::type_dispatcher(lhs_child.type(),
                                 column_comparator<check_exact_equality>{},
                                 lhs_child,
                                 rhs_child,
                                 lhs_row_indices,
                                 rhs_row_indices,
                                 verbosity,
                                 fp_ulps,
                                 depth + 1,
                                 stream,
                                 mr)) {
        return false;
      }
    }
    return true;
  }
};

template <bool check_exact_equality>
struct column_comparator {
  template <typename T>
  bool operator()(column_view const& lhs,
                  column_view const& rhs,
                  column_view const& lhs_row_indices,
                  column_view const& rhs_row_indices,
                  debug_output_level verbosity,
                  size_type fp_ulps,
                  int depth,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr)
  {
    // compare properties
    if (!cudf::type_dispatcher(lhs.type(),
                               column_property_comparator<check_exact_equality>{},
                               lhs,
                               rhs,
                               lhs_row_indices,
                               rhs_row_indices,
                               verbosity,
                               stream,
                               mr)) {
      return false;
    }

    // compare values
    column_comparator_impl<T, check_exact_equality> comparator{};
    return comparator(
      lhs, rhs, lhs_row_indices, rhs_row_indices, verbosity, fp_ulps, depth, stream, mr);
  }
};

void check_non_empty_nulls(column_view const& lhs, column_view const& rhs, cuda::stream_ref stream)
{
  auto check_column_nulls = [stream](column_view const& col, char const* col_name) {
    if (cudf::detail::has_nonempty_nulls(col, stream)) {
      throw std::invalid_argument(col_name + std::string(" column has non-empty nulls"));
    }
  };

  check_column_nulls(lhs, "lhs");
  check_column_nulls(rhs, "rhs");
}

}  // namespace

namespace detail {
/**
 * @copydoc cudf::test::expect_column_properties_equal
 */
bool expect_column_properties_equal(column_view const& lhs,
                                    column_view const& rhs,
                                    debug_output_level verbosity,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  check_non_empty_nulls(lhs, rhs, stream);
  auto lhs_indices = generate_all_row_indices(lhs.size(), stream, mr);
  auto rhs_indices = generate_all_row_indices(rhs.size(), stream, mr);
  return cudf::type_dispatcher(lhs.type(),
                               column_property_comparator<true>{},
                               lhs,
                               rhs,
                               *lhs_indices,
                               *rhs_indices,
                               verbosity,
                               stream,
                               mr);
}

/**
 * @copydoc cudf::test::expect_column_properties_equivalent
 */
bool expect_column_properties_equivalent(column_view const& lhs,
                                         column_view const& rhs,
                                         debug_output_level verbosity,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
{
  check_non_empty_nulls(lhs, rhs, stream);
  auto lhs_indices = generate_all_row_indices(lhs.size(), stream, mr);
  auto rhs_indices = generate_all_row_indices(rhs.size(), stream, mr);
  return cudf::type_dispatcher(lhs.type(),
                               column_property_comparator<false>{},
                               lhs,
                               rhs,
                               *lhs_indices,
                               *rhs_indices,
                               verbosity,
                               stream,
                               mr);
}

/**
 * @copydoc cudf::test::expect_columns_equal
 */
bool expect_columns_equal(cudf::column_view const& lhs,
                          cudf::column_view const& rhs,
                          debug_output_level verbosity,
                          cuda::stream_ref stream,
                          cudf::memory_resources mr)
{
  // TODO: equality row preprocessing (two_table_comparator / preprocessed_table::create) still
  // allocates from the current device resource; pass `mr` through once that path accepts
  // memory_resources so callers need not disable failing current-resource scopes.
  check_non_empty_nulls(lhs, rhs, stream);
  auto lhs_indices = generate_all_row_indices(lhs.size(), stream, mr);
  auto rhs_indices = generate_all_row_indices(rhs.size(), stream, mr);
  return cudf::type_dispatcher(lhs.type(),
                               column_comparator<true>{},
                               lhs,
                               rhs,
                               *lhs_indices,
                               *rhs_indices,
                               verbosity,
                               cudf::test::default_ulp,
                               0,
                               stream,
                               mr);
}

/**
 * @copydoc cudf::test::expect_columns_equivalent
 */
bool expect_columns_equivalent(cudf::column_view const& lhs,
                               cudf::column_view const& rhs,
                               debug_output_level verbosity,
                               size_type fp_ulps,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  check_non_empty_nulls(lhs, rhs, stream);
  auto lhs_indices = generate_all_row_indices(lhs.size(), stream, mr);
  auto rhs_indices = generate_all_row_indices(rhs.size(), stream, mr);
  return cudf::type_dispatcher(lhs.type(),
                               column_comparator<false>{},
                               lhs,
                               rhs,
                               *lhs_indices,
                               *rhs_indices,
                               verbosity,
                               fp_ulps,
                               0,
                               stream,
                               mr);
}

/**
 * @copydoc cudf::test::expect_equal_buffers
 */
void expect_equal_buffers(void const* lhs,
                          void const* rhs,
                          std::size_t size_bytes,
                          cuda::stream_ref stream,
                          cudf::memory_resources mr)
{
  if (size_bytes > 0) {
    EXPECT_NE(nullptr, lhs);
    EXPECT_NE(nullptr, rhs);
  }
  auto typed_lhs = static_cast<char const*>(lhs);
  auto typed_rhs = static_cast<char const*>(rhs);
  EXPECT_TRUE(thrust::equal(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                            typed_lhs,
                            typed_lhs + size_bytes,
                            typed_rhs));
}
}  // namespace detail

/**
 * @copydoc cudf::test::expect_column_empty
 */
void expect_column_empty(cudf::column_view const& col)
{
  EXPECT_EQ(0, col.size());
  EXPECT_EQ(0, col.null_count());
}

/**
 * @copydoc cudf::test::bitmask_to_host
 */
std::vector<bitmask_type> bitmask_to_host(cudf::column_view const& c,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  if (c.nullable()) {
    auto num_bitmasks      = num_bitmask_words(c.size());
    auto [bitmask_span, _] = [&] {
      if (c.offset() == 0) {
        return std::pair{cudf::device_span<bitmask_type const>(c.null_mask(), num_bitmasks),
                         rmm::device_buffer{}};
      }
      auto mask = copy_bitmask(
        c.null_mask(), c.offset(), c.offset() + c.size(), stream, mr.get_temporary_mr());
      return std::pair{cudf::device_span<bitmask_type const>(
                         static_cast<bitmask_type*>(mask.data()), num_bitmasks),
                       std::move(mask)};
    }();
    return cudf::detail::make_std_vector(bitmask_span, stream);
  } else {
    return std::vector<bitmask_type>{};
  }
}

/**
 * @copydoc cudf::test::validate_host_masks
 */
bool validate_host_masks(std::vector<bitmask_type> const& expected_mask,
                         std::vector<bitmask_type> const& got_mask,
                         size_type number_of_elements)
{
  return std::all_of(cuda::counting_iterator<cudf::size_type>{0},
                     cuda::counting_iterator{number_of_elements},
                     [&expected_mask, &got_mask](auto index) {
                       return cudf::bit_is_set(expected_mask.data(), index) ==
                              cudf::bit_is_set(got_mask.data(), index);
                     });
}

template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>*>
std::pair<thrust::host_vector<T>, std::vector<bitmask_type>> to_host(column_view c,
                                                                     cuda::stream_ref stream,
                                                                     cudf::memory_resources mr)
{
  using namespace numeric;
  using Rep = typename T::rep;

  auto col_span       = cudf::device_span<Rep const>(c.begin<Rep>(), c.size());
  auto host_rep_types = cudf::detail::make_host_vector(col_span, stream);

  auto host_fixed_points = thrust::host_vector<T>(c.size());
  std::transform(
    host_rep_types.cbegin(), host_rep_types.cend(), host_fixed_points.begin(), [&](Rep val) {
      return T{scaled_integer<Rep>{val, scale_type{c.type().scale()}}};
    });

  return {std::move(host_fixed_points), bitmask_to_host(c, stream, mr)};
}

template std::pair<thrust::host_vector<numeric::decimal32>, std::vector<bitmask_type>> to_host(
  column_view c, cuda::stream_ref stream, cudf::memory_resources mr);
template std::pair<thrust::host_vector<numeric::decimal64>, std::vector<bitmask_type>> to_host(
  column_view c, cuda::stream_ref stream, cudf::memory_resources mr);
template std::pair<thrust::host_vector<numeric::decimal128>, std::vector<bitmask_type>> to_host(
  column_view c, cuda::stream_ref stream, cudf::memory_resources mr);

namespace {
struct strings_to_host_fn {
  template <typename OffsetType>
  void operator()(thrust::host_vector<std::string>& host_data,
                  char const* chars,
                  cudf::column_view const& offsets,
                  cuda::stream_ref stream)
    requires(std::is_same_v<OffsetType, int32_t> || std::is_same_v<OffsetType, int64_t>)
  {
    auto const h_offsets = cudf::detail::make_std_vector(
      cudf::device_span<OffsetType const>(offsets.data<OffsetType>(), offsets.size()), stream);
    // build std::string vector from chars and offsets
    std::transform(std::begin(h_offsets),
                   std::end(h_offsets) - 1,
                   std::begin(h_offsets) + 1,
                   host_data.begin(),
                   [&](auto start, auto end) { return std::string(chars + start, end - start); });
  }

  template <typename OffsetType>
  void operator()(thrust::host_vector<std::string>&,
                  char const*,
                  cudf::column_view const&,
                  cuda::stream_ref)
    requires(!std::is_same_v<OffsetType, int32_t> && !std::is_same_v<OffsetType, int64_t>)
  {
    CUDF_FAIL("invalid offsets type");
  }
};
}  // namespace

template <>
std::pair<thrust::host_vector<std::string>, std::vector<bitmask_type>> to_host(
  column_view c, cuda::stream_ref stream, cudf::memory_resources mr)
{
  thrust::host_vector<std::string> host_data(c.size());
  if (c.size() > c.null_count()) {
    auto const scv     = strings_column_view(c);
    auto const h_chars = cudf::detail::make_std_vector<char>(
      cudf::device_span<char const>(scv.chars_begin(stream), scv.chars_size(stream)), stream);
    auto offsets =
      cudf::slice(scv.offsets(), {scv.offset(), scv.offset() + scv.size() + 1}).front();
    cudf::type_dispatcher(
      offsets.type(), strings_to_host_fn{}, host_data, h_chars.data(), offsets, stream);
  }
  return {std::move(host_data), bitmask_to_host(c, stream, mr)};
}

large_strings_enabler::large_strings_enabler(bool default_enable)
{
  default_enable ? enable() : disable();
}

large_strings_enabler::~large_strings_enabler() { disable(); }

void large_strings_enabler::enable() { setenv("LIBCUDF_LARGE_STRINGS_ENABLED", "1", 1); }

void large_strings_enabler::disable() { setenv("LIBCUDF_LARGE_STRINGS_ENABLED", "0", 1); }

}  // namespace test
}  // namespace cudf
