/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

template <typename OptionalScalarIterator, typename ReplaceScalarIterator>
struct clamp_strings_fn {
  using string_index_pair = cudf::strings::detail::string_index_pair;
  column_device_view const d_strings;
  OptionalScalarIterator lo_itr;
  ReplaceScalarIterator lo_replace_itr;
  OptionalScalarIterator hi_itr;
  ReplaceScalarIterator hi_replace_itr;

  __device__ string_index_pair operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return string_index_pair{nullptr, 0}; }

    auto const element      = d_strings.element<string_view>(idx);
    auto const d_lo         = (*lo_itr).value_or(element);
    auto const d_hi         = (*hi_itr).value_or(element);
    auto const d_lo_replace = *(*lo_replace_itr);
    auto const d_hi_replace = *(*hi_replace_itr);

    auto d_str = [d_lo, d_lo_replace, d_hi, d_hi_replace, element] {
      if (element < d_lo) { return d_lo_replace; }
      if (d_hi < element) { return d_hi_replace; }
      return element;
    }();

    // ensures an empty string is not converted to a null row
    return !d_str.empty() ? string_index_pair{d_str.data(), d_str.size_bytes()}
                          : string_index_pair{"", 0};
  }
};

template <typename OptionalScalarIterator, typename ReplaceScalarIterator>
std::unique_ptr<cudf::column> clamp_string_column(strings_column_view const& input,
                                                  OptionalScalarIterator lo_itr,
                                                  ReplaceScalarIterator lo_replace_itr,
                                                  OptionalScalarIterator hi_itr,
                                                  ReplaceScalarIterator hi_replace_itr,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto input_device_column = column_device_view::create(input.parent(), stream);
  auto d_input             = *input_device_column;

  auto fn = clamp_strings_fn<OptionalScalarIterator, ReplaceScalarIterator>{
    d_input, lo_itr, lo_replace_itr, hi_itr, hi_replace_itr};
  rmm::device_uvector<cudf::strings::detail::string_index_pair> indices(input.size(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    indices.begin(),
                    fn);

  return cudf::strings::detail::make_strings_column(indices.begin(), indices.end(), stream, mr);
}

template <typename T, typename OptionalIterator, typename ReplaceIterator>
struct clamp_dictionary_fn {
  column_device_view const d_dictionary;
  OptionalIterator lo_itr;
  ReplaceIterator lo_replace_itr;
  OptionalIterator hi_itr;
  ReplaceIterator hi_replace_itr;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_dictionary.is_null(idx)) { return 0; }
    auto const key_index = d_dictionary.element<dictionary32>(idx).value();
    auto const element =
      d_dictionary.child(dictionary_column_view::keys_column_index).element<T>(key_index);
    if (element < (*lo_itr).value_or(element)) { return *lo_replace_itr; }
    if ((*hi_itr).value_or(element) < element) { return *hi_replace_itr; }
    return key_index;
  }
};

template <typename T>
std::unique_ptr<cudf::column> clamp_dictionary_column(dictionary_column_view const& input,
                                                      scalar const& lo,
                                                      scalar const& lo_replace,
                                                      scalar const& hi,
                                                      scalar const& hi_replace,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  // add lo_replace and hi_replace to keys
  auto matched_column = [&] {
    auto matched_view              = dictionary_column_view(input);
    std::unique_ptr<column> result = nullptr;
    auto add_scalar_key            = [&](scalar const& key, scalar const& key_replace) {
      if (key.is_valid(stream)) {
        result = dictionary::detail::add_keys(
          matched_view,
          make_column_from_scalar(key_replace, 1, stream, cudf::get_current_device_resource_ref())
            ->view(),
          stream,
          mr);
        matched_view = dictionary_column_view(result->view());
      }
    };
    add_scalar_key(lo, lo_replace);
    add_scalar_key(hi, hi_replace);
    return result;
  }();
  auto matched_view = dictionary_column_view(matched_column->view());
  auto default_mr   = cudf::get_current_device_resource_ref();

  // get the indexes for lo_replace and for hi_replace
  auto lo_replace_index =
    dictionary::detail::get_index(matched_view, lo_replace, stream, default_mr);
  auto hi_replace_index =
    dictionary::detail::get_index(matched_view, hi_replace, stream, default_mr);
  auto lo_index_itr = cudf::detail::indexalator_factory::make_input_iterator(*lo_replace_index);
  auto hi_index_itr = cudf::detail::indexalator_factory::make_input_iterator(*hi_replace_index);

  auto indices_column = cudf::make_numeric_column(
    matched_view.indices().type(), matched_view.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  auto indices_itr =
    cudf::detail::indexalator_factory::make_output_iterator(indices_column->mutable_view());

  auto lo_itr  = make_optional_iterator<T>(lo, nullate::YES{});
  auto hi_itr  = make_optional_iterator<T>(hi, nullate::YES{});
  auto d_input = column_device_view::create(input.parent(), stream);

  using OptionalIterator = decltype(lo_itr);
  using ReplaceIterator  = decltype(lo_index_itr);

  auto fn = clamp_dictionary_fn<T, OptionalIterator, ReplaceIterator>{
    *d_input, lo_itr, lo_index_itr, hi_itr, hi_index_itr};
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    indices_itr,
                    fn);

  // take the keys from the matched column allocated using mr
  std::unique_ptr<column> keys_column(std::move(matched_column->release().children.back()));

  // create column with keys_column and indices_column
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                input.null_count());
}

template <typename T, typename OptionalScalarIterator, typename ReplaceScalarIterator>
std::unique_ptr<cudf::column> clamper(column_view const& input,
                                      OptionalScalarIterator lo_itr,
                                      ReplaceScalarIterator lo_replace_itr,
                                      OptionalScalarIterator hi_itr,
                                      ReplaceScalarIterator hi_replace_itr,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
  requires(cudf::is_fixed_width<T>())
{
  auto output =
    detail::allocate_like(input, input.size(), mask_allocation_policy::NEVER, stream, mr);
  // mask will not change
  if (input.nullable()) {
    output->set_null_mask(cudf::detail::copy_bitmask(input, stream, mr), input.null_count());
  }

  auto output_device_view =
    cudf::mutable_column_device_view::create(output->mutable_view(), stream);
  auto input_device_view = cudf::column_device_view::create(input, stream);
  auto scalar_zip_itr =
    cuda::make_zip_iterator(cuda::std::make_tuple(lo_itr, lo_replace_itr, hi_itr, hi_replace_itr));

  auto trans =
    cuda::proclaim_return_type<T>([] __device__(auto element_optional, auto scalar_tuple) {
      if (element_optional.has_value()) {
        auto lo_optional = cuda::std::get<0>(scalar_tuple);
        auto hi_optional = cuda::std::get<2>(scalar_tuple);
        if (lo_optional.has_value() and (*element_optional < *lo_optional)) {
          return *(cuda::std::get<1>(scalar_tuple));
        } else if (hi_optional.has_value() and (*element_optional > *hi_optional)) {
          return *(cuda::std::get<3>(scalar_tuple));
        }
        return *element_optional;
      }
      return T{};  // null entry so value is ignored
    });

  auto input_pair_iterator =
    make_optional_iterator<T>(*input_device_view, nullate::DYNAMIC{input.has_nulls()});
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    input_pair_iterator,
                    input_pair_iterator + input.size(),
                    scalar_zip_itr,
                    output_device_view->begin<T>(),
                    trans);

  return output;
}

template <typename T, typename OptionalScalarIterator, typename ReplaceScalarIterator>
std::unique_ptr<cudf::column> clamper(column_view const& input,
                                      OptionalScalarIterator lo_itr,
                                      ReplaceScalarIterator lo_replace_itr,
                                      OptionalScalarIterator hi_itr,
                                      ReplaceScalarIterator hi_replace_itr,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
  requires(std::is_same_v<T, string_view>)
{
  return clamp_string_column(input, lo_itr, lo_replace_itr, hi_itr, hi_replace_itr, stream, mr);
}

}  // namespace

template <typename T, typename OptionalScalarIterator, typename ReplaceScalarIterator>
std::unique_ptr<column> clamp(column_view const& input,
                              OptionalScalarIterator lo_itr,
                              ReplaceScalarIterator lo_replace_itr,
                              OptionalScalarIterator hi_itr,
                              ReplaceScalarIterator hi_replace_itr,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  return clamper<T>(input, lo_itr, lo_replace_itr, hi_itr, hi_replace_itr, stream, mr);
}

struct dispatch_clamp {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     scalar const& lo,
                                     scalar const& lo_replace,
                                     scalar const& hi,
                                     scalar const& hi_replace,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    CUDF_EXPECTS(cudf::have_same_types(input, lo),
                 "mismatching types of scalar and input",
                 cudf::data_type_error);

    if (input.type().id() == type_id::DICTIONARY32) {
      return clamp_dictionary_column<T>(
        dictionary_column_view(input), lo, lo_replace, hi, hi_replace, stream, mr);
    }

    auto lo_itr         = make_optional_iterator<T>(lo, nullate::YES{});
    auto hi_itr         = make_optional_iterator<T>(hi, nullate::YES{});
    auto lo_replace_itr = make_optional_iterator<T>(lo_replace, nullate::NO{});
    auto hi_replace_itr = make_optional_iterator<T>(hi_replace, nullate::NO{});

    return clamp<T>(input, lo_itr, lo_replace_itr, hi_itr, hi_replace_itr, stream, mr);
  }
};

template <>
std::unique_ptr<column> dispatch_clamp::operator()<cudf::list_view>(
  column_view const& input,
  scalar const& lo,
  scalar const& lo_replace,
  scalar const& hi,
  scalar const& hi_replace,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FAIL("clamp for list_view not supported");
}

template <>
std::unique_ptr<column> dispatch_clamp::operator()<struct_view>(column_view const& input,
                                                                scalar const& lo,
                                                                scalar const& lo_replace,
                                                                scalar const& hi,
                                                                scalar const& hi_replace,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::device_async_resource_ref mr)
{
  CUDF_FAIL("clamp for struct_view not supported");
}

template <>
std::unique_ptr<column> dispatch_clamp::operator()<dictionary32>(column_view const&,
                                                                 scalar const&,
                                                                 scalar const&,
                                                                 scalar const&,
                                                                 scalar const&,
                                                                 rmm::cuda_stream_view,
                                                                 rmm::device_async_resource_ref)
{
  CUDF_UNREACHABLE("clamp type-dispatch error");
}

/**
 * @copydoc cudf::clamp(column_view const& input,
                                      scalar const& lo,
                                      scalar const& lo_replace,
                                      scalar const& hi,
                                      scalar const& hi_replace,
                                      rmm::device_async_resource_ref mr);
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> clamp(column_view const& input,
                              scalar const& lo,
                              scalar const& lo_replace,
                              scalar const& hi,
                              scalar const& hi_replace,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    cudf::have_same_types(lo, hi), "mismatching types of limit scalars", cudf::data_type_error);
  CUDF_EXPECTS(cudf::have_same_types(lo_replace, hi_replace),
               "mismatching types of replace scalars",
               cudf::data_type_error);
  CUDF_EXPECTS(cudf::have_same_types(lo, lo_replace),
               "mismatching types of limit and replace scalars",
               cudf::data_type_error);

  if ((not lo.is_valid(stream) and not hi.is_valid(stream)) or (input.is_empty())) {
    // There will be no change
    return std::make_unique<column>(input, stream, mr);
  }

  if (lo.is_valid(stream)) {
    CUDF_EXPECTS(lo_replace.is_valid(stream), "lo_replace can't be null if lo is not null");
  }
  if (hi.is_valid(stream)) {
    CUDF_EXPECTS(hi_replace.is_valid(stream), "hi_replace can't be null if hi is not null");
  }

  auto dispatch_type =
    cudf::is_dictionary(input.type()) ? dictionary_column_view(input).keys().type() : input.type();
  return cudf::type_dispatcher<dispatch_storage_type>(
    dispatch_type, dispatch_clamp{}, input, lo, lo_replace, hi, hi_replace, stream, mr);
}

}  // namespace detail

// clamp input at lo and hi with lo_replace and hi_replace
std::unique_ptr<column> clamp(column_view const& input,
                              scalar const& lo,
                              scalar const& lo_replace,
                              scalar const& hi,
                              scalar const& hi_replace,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::clamp(input, lo, lo_replace, hi, hi_replace, stream, mr);
}

// clamp input at lo and hi
std::unique_ptr<column> clamp(column_view const& input,
                              scalar const& lo,
                              scalar const& hi,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::clamp(input, lo, lo, hi, hi, stream, mr);
}
}  // namespace cudf
