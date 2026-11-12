/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/errc.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/span>
#include <cuda/stream>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <cudf_fragments.hpp>
#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/output_columns.hpp>
#include <jit/reflect.hpp>
#include <jit/row_ir.hpp>
#include <rtcx/rtcx.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cudf {
namespace {

using jit::fixed_width_column;
using jit::mutable_strings_column;
using jit::output_column;
using jit::string_views_column;

using input_column_view = transform_input;

using handle = std::variant<
  std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>,
  std::unique_ptr<mutable_column_device_view, std::function<void(mutable_column_device_view*)>>>;

namespace jit_transform {

void launch(cudf::kernel const& kernel,
            size_type row_size,
            bitmask_type const* stencil,
            column_device_view_core const* input_cols,
            mutable_column_device_view_core const* output_cols,
            int32_t* max_error,
            cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  void* args[] = {&row_size, &stencil, &input_cols, &output_cols, &max_error};
  auto cfg     = kernel.max_occupancy_config(0, 0);
  CUDF_EXPECTS(cfg.block_size % cudf::detail::warp_size == 0,
               "Expected block size to be a multiple of warp size",
               std::runtime_error);
  kernel.launch({cfg.min_grid_size}, {cfg.block_size}, 0, stream, args);
}

std::tuple<rtcx::blob, lto_binary_type, std::string> instantiate_fragment(
  bool is_null_aware,
  bool indexed,
  std::string const& ins,
  std::string const& outs,
  std::span<transform_input_spec const> inputs,
  std::span<transform_output_spec const> outputs)
{
  CUDF_FUNC_RANGE();
  auto name = rtcx::reflect_template(
    "cudf::detail::transform_kernel", rtcx::reflect(is_null_aware), ins, outs);
  auto signature =
    jit::reflect_udf_signature(is_null_aware, inputs, outputs, true, indexed ? 1 : 0);
  return {
    jit::get_udf_kernel_fragment("cudf/cpp/src/transform/jit/kernel.cu", name, signature, indexed),
    lto_binary_type::LTO_IR,
    name};
}

auto to_args(std::span<input_column_view const> inputs,
             std::span<output_column> outputs,
             cuda::stream_ref stream,
             cudf::memory_resources mr)
{
  std::vector<handle> handles;
  auto h_args =
    detail::host_vector<detail::column_device_view_base>({get_pinned_memory_resource(), stream});
  h_args.reserve(inputs.size() + outputs.size());

  for (auto& in : inputs) {
    if (auto const* data = std::get_if<user_data>(&in)) {
      h_args.push_back(column_device_view_core{*data});
    } else if (auto* col = std::get_if<column_view>(&in)) {
      auto handle = column_device_view::create(*col, stream, mr.get_output_mr());
      h_args.push_back(*handle);
      handles.emplace_back(std::move(handle));
    } else {
      auto& scalar = std::get<scalar_column_view>(in);
      auto handle = column_device_view::create(scalar.as_column_view(), stream, mr.get_output_mr());
      h_args.push_back(*handle);
      handles.emplace_back(std::move(handle));
    }
  }

  for (auto& out : outputs) {
    std::visit(
      [&](auto& col) {
        auto handle = col.mutable_view().to_device(stream);
        h_args.push_back(*handle);
        handles.push_back(std::move(handle));
      },
      out);
  }

  auto d_args = detail::make_device_buffer(h_args, stream, mr);

  return std::make_tuple(std::move(d_args), std::move(handles));
}

kernel get_linked_kernel(
  std::optional<std::tuple<std::span<uint8_t const>, lto_binary_type, char const*>> precompiled,
  std::span<transform_input_spec const> inputs,
  std::span<transform_output_spec const> outputs,
  transform_udf const& udf)
{
  bool const is_null_aware   = udf.is_null_aware == null_aware::YES;
  bool const indexed         = udf.indexed;
  auto [in_types, out_types] = jit::reflect(true, inputs, outputs);
  std::span<uint8_t const> kernel_fragment;
  auto binary_type = lto_binary_type::FATBIN;
  rtcx::blob fragment_blob;
  if (precompiled) {
    std::tie(kernel_fragment, binary_type, std::ignore) = *precompiled;
  } else {
    std::tie(fragment_blob, binary_type, std::ignore) =
      instantiate_fragment(is_null_aware, indexed, in_types, out_types, inputs, outputs);
    kernel_fragment = fragment_blob->view();
  }
  auto adapter = get_udf_adapter(
    udf.symbol, "cudf_transform_udf", is_null_aware, inputs, outputs, true, indexed ? 1 : 0);
  std::vector<rtcx::memory_fragment> modules{
    {.data = kernel_fragment, .type = jit::as_rtcx_binary_type(binary_type), .name = nullptr}};
  if (adapter) {
    auto view = adapter->view();
    modules.push_back(
      {.data = view.data, .type = jit::as_rtcx_binary_type(view.type), .name = nullptr});
  }
  for (auto const& fragment : udf.fragments) {
    modules.push_back(
      {.data = fragment.data, .type = jit::as_rtcx_binary_type(fragment.type), .name = nullptr});
  }
  return get_lto_linked_kernel("cudf/cpp/src/transform/jit/kernel.cu", {}, modules);
}

void run(kernel const& kernel,
         size_type row_size,
         bitmask_type const* d_stencil,
         std::span<input_column_view const> inputs,
         std::span<output_column> outputs,
         int32_t* d_max_error,
         cuda::stream_ref stream,
         cudf::memory_resources mr)
{
  auto [cols, handles] = to_args(inputs, outputs, stream, mr.as_temporary());
  auto* input_cols     = reinterpret_cast<column_device_view_core const*>(cols.data());
  auto* output_cols =
    reinterpret_cast<mutable_column_device_view_core const*>(input_cols + inputs.size());
  return launch(kernel, row_size, d_stencil, input_cols, output_cols, d_max_error, stream);
}

void run_lto(
  std::optional<std::tuple<std::span<uint8_t const>, lto_binary_type, char const*>> precompiled,
  size_type row_size,
  bitmask_type const* stencil,
  std::span<input_column_view const> inputs,
  std::span<output_column> outputs,
  int32_t* max_error,
  transform_udf const& udf,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  auto kernel = get_linked_kernel(
    precompiled, jit::make_input_specs(inputs), jit::make_output_specs(outputs), udf);
  run(kernel, row_size, stencil, inputs, outputs, max_error, stream, mr);
}

}  // namespace jit_transform

CUDF_KERNEL void copy_offset_bitmask(bitmask_type* __restrict__ destination,
                                     bitmask_type const* __restrict__ source,
                                     size_type source_begin_bit,
                                     size_type source_end_bit,
                                     size_type number_of_mask_words)
{
  auto const stride = detail::grid_1d::grid_stride();
  for (thread_index_type destination_word_index = detail::grid_1d::global_thread_id();
       destination_word_index < number_of_mask_words;
       destination_word_index += stride) {
    destination[destination_word_index] = detail::get_mask_offset_word(
      source, destination_word_index, source_begin_bit, source_end_bit);
  }
}

size_type inplace_null_mask_and(bitmask_type* null_mask,
                                size_type row_size,
                                std::span<transform_input const> inputs,
                                cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();

  auto is_nullable = null_mask != nullptr;

  if (!is_nullable) { return 0; }

  if (inputs.empty()) {
    // no input, set all to valid
    set_null_mask(null_mask, 0, row_size, true, stream);
    return 0;
  }

  auto has_scalars = std::any_of(inputs.begin(), inputs.end(), [](auto& in) {
    return std::holds_alternative<scalar_column_view>(in);
  });

  if (has_scalars) {
    auto scalar_is_null = std::any_of(inputs.begin(), inputs.end(), [](auto& in) {
      if (auto* scalar = std::get_if<scalar_column_view>(&in)) { return scalar->has_nulls(); }
      return false;
    });

    if (scalar_is_null) {
      // scalar is null, all rows will be null
      set_null_mask(null_mask, 0, row_size, false, stream);
      return row_size;
    }
  }

  auto has_cols = std::any_of(
    inputs.begin(), inputs.end(), [](auto& in) { return std::holds_alternative<column_view>(in); });

  if (!has_cols) {
    // no non-scalar columns, so all rows are valid
    set_null_mask(null_mask, 0, row_size, true, stream);
    return 0;
  }

  // collect the non-scalar nullable columns that contribute to the output nullmask
  std::vector<bitmask_type const*> nullable_masks;
  std::vector<size_type> nullable_offsets;
  std::vector<size_type> nullable_null_counts;

  for (auto& in : inputs) {
    if (auto* c = std::get_if<column_view>(&in)) {
      if (c->nullable()) {
        nullable_masks.push_back(c->null_mask());
        nullable_offsets.push_back(c->offset());
        nullable_null_counts.push_back(c->null_count());
      }
    }
  }

  if (nullable_masks.empty()) {
    // we only have non-nullable columns, so all rows are valid
    set_null_mask(null_mask, 0, row_size, true, stream);
    return 0;
  }

  auto num_words               = num_bitmask_words(row_size);
  auto num_bytes               = num_words * sizeof(bitmask_type);
  constexpr auto bits_per_word = sizeof(bitmask_type) * 8;

  if (nullable_masks.size() == 1) {
    // only 1 mask provided, copy it directly to the output
    auto src_begin = nullable_offsets[0];
    auto src_end   = src_begin + row_size;
    if (src_begin % bits_per_word == 0) {
      CUDF_CUDA_TRY(detail::memcpy_async(
        null_mask, nullable_masks[0] + (src_begin / bits_per_word), num_bytes, stream));
    } else {
      detail::grid_1d config(row_size, 256);
      copy_offset_bitmask<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
        static_cast<bitmask_type*>(null_mask), nullable_masks[0], src_begin, src_end, num_words);
      CUDF_CHECK_CUDA(stream.get());
    }
    return nullable_null_counts[0];
  }

  auto num_valid = detail::inplace_bitmask_and(
    device_span<bitmask_type>{null_mask, static_cast<size_t>(num_words)},
    nullable_masks,
    nullable_offsets,
    row_size,
    stream);

  return row_size - std::min(num_valid, row_size);
}

/**
 * @brief Get the null-mask transformation for the transform operation based on the UDF's parameters
 * and inputs
 *
 * @return input and output null-policies for the UDF kernel
 */
auto get_null_transformation(null_aware is_null_aware,
                             std::span<transform_input const> inputs,
                             std::span<transform_output const> outputs)
{
  auto any_input_nullable = std::any_of(inputs.begin(), inputs.end(), [](auto& in) {
    return std::visit(
      [](auto const& value) {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>, user_data>) {
          return false;
        } else {
          return value.nullable();
        }
      },
      in);
  });

  std::vector<char> output_may_be_nullable;
  for (auto& out : outputs) {
    bool may_eval_nulls = true;
    if (is_null_aware == null_aware::YES) {
      // null-aware UDFs may evaluate nulls unless the output is explicitly marked as all valid
      may_eval_nulls = out.nullability != output_nullability::ALL_VALID;
    } else {
      // null-unaware UDFs may evaluate nulls if any input is nullable unless explicitly marked as
      // not producing nulls
      may_eval_nulls = any_input_nullable && (out.nullability == output_nullability::PRESERVE);
    }

    output_may_be_nullable.push_back(may_eval_nulls);
  }

  return output_may_be_nullable;
}

void perform_checks(null_aware is_null_aware,
                    std::optional<size_type> in_row_size,
                    std::span<transform_input const> inputs,
                    std::span<transform_output const> outputs,
                    std::span<std::unique_ptr<column> const> string_offsets)
{
  CUDF_EXPECTS(std::none_of(outputs.begin(),
                            outputs.end(),
                            [](auto& out) {
                              return !is_fixed_width(out.type) && out.type.id() != type_id::STRING;
                            }),
               "Transforms only support output of fixed-width or string types",
               std::invalid_argument);

  static constexpr auto is_input_value_supported = [](auto const& c) {
    return is_fixed_width(c.type()) || c.type().id() == type_id::STRING || is_dictionary(c.type());
  };
  static constexpr auto is_supported_input_type = [&](auto const& input) {
    return std::visit(
      [&](auto const& value) {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>, user_data>) {
          return true;
        } else {
          auto col = jit::as_column_view(value);
          return is_input_value_supported(col) ||
                 (is_dictionary(col.type()) &&
                  is_input_value_supported(col.child(dictionary_keys_column_index)));
        }
      },
      input);
  };
  CUDF_EXPECTS(
    std::none_of(
      inputs.begin(), inputs.end(), [&](auto const& in) { return !is_supported_input_type(in); }),
    "Transforms only support fixed-width, string, dictionary, or user-data inputs",
    std::invalid_argument);

  if (!in_row_size.has_value()) {
    CUDF_EXPECTS(
      std::any_of(inputs.begin(),
                  inputs.end(),
                  [](auto const& in) { return std::holds_alternative<column_view>(in); }),
      "At least one input of a transform must be a non-scalar column if row size is not provided",
      std::invalid_argument);
  }

  auto row_size = in_row_size.has_value() ? *in_row_size : jit::get_projection_size(inputs);
  CUDF_EXPECTS(std::none_of(inputs.begin(),
                            inputs.end(),
                            [&](auto& in) {
                              if (auto* col = std::get_if<column_view>(&in)) {
                                return col->size() != row_size;
                              }
                              return false;
                            }),
               "All transform input columns must have the same size",
               std::invalid_argument);

  CUDF_EXPECTS(string_offsets.empty() || (string_offsets.size() == outputs.size()),
               "Number of string offsets must be empty or match the number of outputs (with nulls "
               "for each non-string column)",
               std::invalid_argument);

  CUDF_EXPECTS(std::all_of(cuda::counting_iterator(size_t{0}),
                           cuda::counting_iterator(string_offsets.size()),
                           [&](auto i) {
                             if (outputs[i].type.id() == type_id::STRING) { return true; }
                             return string_offsets.empty() || string_offsets[i] == nullptr;
                           }),
               "String offsets must only be provided for string outputs",
               std::invalid_argument);
}

std::optional<std::pair<bitmask_type*, size_type>> make_stencil(
  null_aware is_null_aware,
  size_type row_size,
  std::span<transform_input const> inputs,
  std::span<output_column> outputs,
  cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();

  // null-aware, no stencil needed
  if (is_null_aware != null_aware::NO) { return std::nullopt; }

  std::optional<bitmask_type*> stencil = std::nullopt;

  for (auto& out : outputs) {
    if (auto* mask = std::visit([&](auto& c) { return c.null_mask(); }, out)) { stencil = mask; }
  }

  // no nullable outputs
  if (!stencil.has_value()) { return std::pair<bitmask_type*, size_type>{nullptr, 0}; }

  auto stencil_null_count = inplace_null_mask_and(*stencil, row_size, inputs, stream);

  for (auto& out : outputs) {
    auto* mask = std::visit([&](auto& c) { return c.null_mask(); }, out);

    if (mask != nullptr && mask != *stencil) {
      CUDF_CUDA_TRY(
        detail::memcpy_async(mask, *stencil, bitmask_allocation_size_bytes(row_size), stream));
    }

    auto null_count = (mask == nullptr) ? 0 : stencil_null_count;

    std::visit([&](auto& c) { c.set_null_count(null_count); }, out);
  }

  return std::pair<bitmask_type*, size_type>{*stencil, stencil_null_count};
}

rmm::device_uvector<char> make_chars_buffer(column_view const& offsets_view,
                                            int64_t chars_size,
                                            string_view const* begin,
                                            bitmask_type const* stencil,
                                            size_type size,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  auto offsets = detail::offsetalator_factory::make_input_iterator(offsets_view);
  auto chars   = rmm::device_uvector<char>(chars_size, stream, mr);

  auto srcs = detail::make_counting_transform_iterator(
    size_type{0}, [begin] __device__(size_type idx) -> void const* { return begin[idx].data(); });

  auto src_sizes = detail::make_counting_transform_iterator(
    size_type{0}, [begin, stencil] __device__(size_type idx) -> size_type {
      if (stencil != nullptr && !bit_is_set(stencil, idx)) { return 0; }
      return static_cast<size_type>(begin[idx].size_bytes());
    });

  auto dsts = detail::make_counting_transform_iterator(
    size_type{0}, [offsets, chars = chars.data()] __device__(size_type idx) -> void* {
      return chars + offsets[idx];
    });

  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(
    nullptr, temp_storage_bytes, srcs, dsts, src_sizes, size, stream.get()));
  cuda::device_buffer<std::byte> d_temp_storage(
    stream, cudf::get_current_device_resource_ref(), temp_storage_bytes, cuda::no_init);
  CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(
    d_temp_storage.data(), temp_storage_bytes, srcs, dsts, src_sizes, size, stream.get()));

  return chars;
}

std::unique_ptr<column> make_strings_column(device_span<string_view const> strings,
                                            cuda::device_buffer<cudf::bitmask_type> null_mask,
                                            size_type null_count,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto size = static_cast<size_type>(strings.size());
  if (size == 0) return make_empty_column(type_id::STRING);

  auto stencil = null_mask.data();

  // build offsets column from the strings sizes
  auto sizes = detail::make_counting_transform_iterator(
    size_type{0}, [stencil, strings = strings.data()] __device__(size_type index) -> size_type {
      if (stencil != nullptr && !bit_is_set(stencil, index)) { return 0; }
      return static_cast<size_type>(strings[index].size_bytes());
    });

  auto [offsets, bytes] =
    strings::detail::make_offsets_child_column(sizes, sizes + size, stream, mr);

  auto chars = make_chars_buffer(offsets->view(), bytes, strings.data(), stencil, size, stream, mr);

  return make_strings_column(
    size, std::move(offsets), chars.release(), null_count, std::move(null_mask));
}

auto make_outputs(null_aware is_null_aware,
                  size_type row_size,
                  std::span<transform_input const> inputs,
                  std::span<transform_output const> outputs,
                  std::span<char const> is_output_nullable,
                  std::vector<std::unique_ptr<column>> string_offsets,
                  cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  std::vector<output_column> cols;

  for (size_t i = 0; i < outputs.size(); i++) {
    auto output     = outputs[i];
    auto nullable   = is_output_nullable[i];
    auto null_state = nullable ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED;
    auto null_mask  = create_null_mask(row_size, null_state, stream, mr);

    if (is_fixed_width(output.type)) {
      auto col =
        fixed_width_column::make(output.type, row_size, std::move(null_mask), 0, stream, mr);
      cols.emplace_back(std::move(col));
    } else if (output.type.id() == type_id::STRING) {
      if (string_offsets.empty() || string_offsets[i] == nullptr) {
        auto col = string_views_column::make(row_size, std::move(null_mask), 0, stream, mr);
        cols.emplace_back(std::move(col));
      } else {
        auto chars_size =
          strings::detail::get_offset_value(string_offsets[i]->view(), row_size, stream);
        auto chars = rmm::device_buffer{static_cast<size_t>(chars_size), stream, mr};
        auto col   = mutable_strings_column::make(
          row_size, std::move(chars), std::move(string_offsets[i]), std::move(null_mask), 0);
        cols.emplace_back(std::move(col));
      }
    } else {
      CUDF_UNREACHABLE("Unsupported output type for transform");
    }
  }

  auto stencil = make_stencil(is_null_aware, row_size, inputs, cols, stream);

  return std::make_tuple(std::move(cols), stencil);
}

void update_null_counts(std::span<output_column> outputs,
                        null_aware is_null_aware,
                        size_type row_size,
                        cuda::stream_ref stream)
{
  // update null counts if the function is not null-aware, since we haven't processed nullability
  // ahead of time (as in the non-null-aware case)
  if (is_null_aware == null_aware::YES) {
    std::vector<bitmask_type const*> bitmasks;
    std::vector<int32_t> indices;

    for (size_t i = 0; i < outputs.size(); i++) {
      std::visit(
        [&](auto& c) {
          if (c.nullable()) {
            indices.push_back(i);
            bitmasks.push_back(c.null_mask());
          }
        },
        outputs[i]);
    }

    auto null_counts = batch_null_count(bitmasks, 0, row_size, stream);

    for (size_t i = 0; i < bitmasks.size(); i++) {
      std::visit([&](auto& c) { c.set_null_count(null_counts[i]); }, outputs[indices[i]]);
    }
  }
}

auto finalize_output(fixed_width_column&& c, cuda::stream_ref, rmm::device_async_resource_ref)
{
  return std::move(c._col);
}

auto finalize_output(mutable_strings_column&& c, cuda::stream_ref, rmm::device_async_resource_ref)
{
  return std::move(c._col);
}

auto finalize_output(string_views_column&& c,
                     cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr)
{
  return make_strings_column(
    device_span<string_view const>{c._data.data(), static_cast<size_t>(c._size)},
    std::move(c._null_mask),
    c._null_count,
    stream,
    mr);
}

auto finalize_outputs(null_aware is_null_aware,
                      size_type row_size,
                      std::vector<output_column> outputs,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  update_null_counts(outputs, is_null_aware, row_size, stream);

  std::vector<std::unique_ptr<column>> results;

  for (auto& out : outputs) {
    std::visit([&](auto& c) { results.push_back(finalize_output(std::move(c), stream, mr)); }, out);
  }

  return results;
}

std::unique_ptr<table> execute_transform(kernel const& kernel,
                                         null_aware is_null_aware,
                                         std::optional<size_type> in_row_size,
                                         std::span<transform_input const> inputs,
                                         std::span<transform_output const> outputs,
                                         std::vector<std::unique_ptr<column>> string_offsets,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto row_size = in_row_size.has_value() ? *in_row_size : jit::get_projection_size(inputs);
  auto [columns, stencil] = make_outputs(is_null_aware,
                                         row_size,
                                         inputs,
                                         outputs,
                                         get_null_transformation(is_null_aware, inputs, outputs),
                                         std::move(string_offsets),
                                         stream,
                                         mr);
  cudf::detail::device_scalar<int32_t> error(
    static_cast<int32_t>(errc::SUCCESS), stream, cudf::get_current_device_resource_ref());
  jit_transform::run(kernel,
                     row_size,
                     stencil.has_value() && stencil->second > 0 ? stencil->first : nullptr,
                     inputs,
                     columns,
                     error.data(),
                     stream,
                     mr);
  auto status = static_cast<errc>(error.value(stream));
  if (status != errc::SUCCESS) {
    throw evaluation_error(
      status, std::format("Transform UDF evaluation failed with error `{}`", to_string(status)));
  }
  return std::make_unique<table>(
    finalize_outputs(is_null_aware, row_size, std::move(columns), stream, mr));
}

}  // namespace

std::unique_ptr<column> compute_column_jit(table_view const& table,
                                           ast::expression const& expr,
                                           cudf::lto_fragments fragments,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  std::array<std::reference_wrapper<ast::expression const>, 1> expressions{expr};
  auto cols = compute_table_jit(table, expressions, fragments, stream, mr)->release();
  return std::move(cols[0]);
}

// if we have a matching pre-compiled kernel fragment for the given transform configuration, return
// it to use for LTO linking instead of compiling a new one
std::optional<std::tuple<std::span<uint8_t const>, lto_binary_type, char const*>>
dispatch_lto_kernel_fragment(bool is_null_aware,
                             bool indexed,
                             std::span<transform_input const> inputs,
                             std::span<output_column const> outputs)
{
  // Precompiled fragments use the unindexed UDF signature.
  if (indexed) { return std::nullopt; }

  auto strip_whitespace = [](std::string_view str) {
    std::string result;
    result.reserve(str.size());
    for (char c : str) {
      if (!std::isspace(static_cast<unsigned char>(c))) { result.push_back(c); }
    }
    return result;
  };

  // TODO: better and less error-prone symbol mangling, but this is sufficient for now.

  // the contract here is that CMake and this dispatch function agree on symbol mangling of the
  // reflected kernel name.
  auto [in_types, out_types] = jit::reflect(true, inputs, outputs);
  auto target                = strip_whitespace(rtcx::reflect_template(
    "cudf::detail::transform_kernel", rtcx::reflect(is_null_aware), in_types, out_types));

  for (size_t i = 0; i < std::size(cudf_fragments::transform_kernel_FILE_INDEX); i++) {
    auto file_index = cudf_fragments::transform_kernel_FILE_INDEX[i];
    auto instance   = strip_whitespace(cudf_fragments::transform_kernel_INSTANCE[i]);
    if (target == instance) {
      auto range = cudf_fragments::file_ranges[file_index];
      return std::make_tuple(cudf_fragments::files.subspan(range[0], range[1]),
                             lto_binary_type::FATBIN,
                             cudf_fragments::transform_kernel_INSTANCE[i]);
    }
  }

  return std::nullopt;
}

static std::unique_ptr<table> execute_lto_transform(
  transform_udf const& udf,
  std::span<transform_input const> inputs,
  std::span<transform_output const> outputs,
  std::vector<std::unique_ptr<column>> string_offsets,
  std::optional<size_type> in_row_size,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const is_null_aware = udf.is_null_aware;
  CUDF_EXPECTS(!udf.symbol.empty() && !udf.fragments.empty(),
               "A transform requires an exported symbol and device modules",
               std::invalid_argument);
  perform_checks(is_null_aware, in_row_size, inputs, outputs, string_offsets);
  auto row_size = in_row_size.has_value() ? *in_row_size : jit::get_projection_size(inputs);
  auto [columns, stencil] = make_outputs(is_null_aware,
                                         row_size,
                                         inputs,
                                         outputs,
                                         get_null_transformation(is_null_aware, inputs, outputs),
                                         std::move(string_offsets),
                                         stream,
                                         mr);
  auto precompiled =
    dispatch_lto_kernel_fragment(is_null_aware == null_aware::YES, udf.indexed, inputs, columns);
  cudf::detail::device_scalar<int32_t> error(
    static_cast<int32_t>(errc::SUCCESS), stream, cudf::get_current_device_resource_ref());
  jit_transform::run_lto(precompiled,
                         row_size,
                         stencil.has_value() && stencil->second > 0 ? stencil->first : nullptr,
                         inputs,
                         columns,
                         error.data(),
                         udf,
                         stream,
                         mr);
  auto status = static_cast<errc>(error.value(stream));
  if (status != errc::SUCCESS) {
    throw evaluation_error(
      status, std::format("Transform UDF evaluation failed with error `{}`", to_string(status)));
  }
  return std::make_unique<table>(
    finalize_outputs(is_null_aware, row_size, std::move(columns), stream, mr));
}

std::unique_ptr<table> transform(transform_udf const& udf,
                                 std::span<transform_input const> inputs,
                                 std::span<transform_output const> outputs,
                                 std::vector<std::unique_ptr<column>>&& string_offsets,
                                 std::optional<size_type> row_size,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return execute_lto_transform(
    udf, inputs, outputs, std::move(string_offsets), row_size, stream, mr);
}

std::unique_ptr<table> compute_table_jit(
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  cudf::lto_fragments fragments,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto args     = detail::row_ir::ast_converter::compute_table(detail::row_ir::target::CUDA,
                                                           expressions,
                                                           fragments,
                                                           table,
                                                               {},
                                                           "cudf_transform_udf",
                                                           stream,
                                                           mr);
  auto compiled = compile_cuda_to_lto_ir(args.udf.c_str(), {}, {}, {}, true, "cudf_row_ir");
  std::vector<lto_fragment> modules(args.lto_fragments.begin(), args.lto_fragments.end());
  modules.push_back(compiled.view());
  transform_udf udf{args.udf_expression, modules, args.is_null_aware};
  return execute_lto_transform(
    udf, args.inputs, args.outputs, std::move(args.string_offsets), args.row_size, stream, mr);
}

struct transform_program::impl {
  void validate(std::span<transform_input const> actual_inputs,
                std::span<transform_output const> actual_outputs,
                std::span<std::unique_ptr<column> const> string_offsets) const
  {
    auto actual_input_specs  = jit::make_input_specs(actual_inputs);
    auto actual_output_specs = jit::make_output_specs(actual_outputs, string_offsets);
    auto actual_reflection   = jit::reflect(false, actual_input_specs, actual_output_specs);
    CUDF_EXPECTS(reflection_ == actual_reflection,
                 "Transform program specifications do not match the provided inputs and outputs",
                 std::invalid_argument);
  }

  impl(transform_udf const& udf,
       std::span<transform_input_spec const> inputs,
       std::span<transform_output_spec const> outputs)
    : reflection_{jit::reflect(false, inputs, outputs)},
      is_null_aware_{udf.is_null_aware},
      kernel_{jit_transform::get_linked_kernel(std::nullopt, inputs, outputs, udf)}
  {
  }

  jit::transform_reflection reflection_;
  null_aware is_null_aware_;
  kernel kernel_;
  std::vector<std::unique_ptr<column>> ast_scalar_columns_;
  std::optional<std::vector<std::optional<int32_t>>> ast_input_column_indices_;
  std::vector<data_type> ast_input_types_;
  std::vector<bool> ast_input_nullable_;
  std::vector<transform_output> ast_outputs_;
};

transform_program::transform_program(transform_udf const& udf,
                                     std::span<transform_input const> inputs,
                                     std::span<transform_output const> outputs,
                                     std::span<std::unique_ptr<column> const> string_offsets)
  : transform_program(
      udf, jit::make_input_specs(inputs), jit::make_output_specs(outputs, string_offsets))
{
}

transform_program::transform_program(transform_udf const& udf,
                                     std::span<transform_input_spec const> inputs,
                                     std::span<transform_output_spec const> outputs)
{
  CUDF_EXPECTS(!udf.symbol.empty() && !udf.fragments.empty(),
               "A transform program requires an exported symbol and device modules",
               std::invalid_argument);
  impl_ = std::make_unique<impl>(udf, inputs, outputs);
}

transform_program::transform_program(
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  cudf::lto_fragments fragments,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto args     = detail::row_ir::ast_converter::compute_table(detail::row_ir::target::CUDA,
                                                           expressions,
                                                           fragments,
                                                           table,
                                                               {},
                                                           "cudf_transform_udf",
                                                           stream,
                                                           mr);
  auto compiled = compile_cuda_to_lto_ir(args.udf.c_str(), {}, {}, {}, true, "cudf_row_ir");
  std::vector<lto_fragment> modules(args.lto_fragments.begin(), args.lto_fragments.end());
  modules.push_back(compiled.view());
  impl_ = std::make_unique<impl>(transform_udf{args.udf_expression, modules, args.is_null_aware},
                                 jit::make_input_specs(args.inputs),
                                 jit::make_output_specs(args.outputs, args.string_offsets));
  CUDF_EXPECTS(args.inputs.size() == args.input_column_indices.size(),
               "AST transform input metadata size mismatch");
  for (auto i = std::size_t{0}; i < args.inputs.size(); ++i) {
    auto& input = args.inputs[i];
    CUDF_EXPECTS(
      args.input_column_indices[i].has_value() == std::holds_alternative<column_view>(input),
      "AST transform inputs must be table columns or scalar literals");
    auto view = std::holds_alternative<column_view>(input)
                  ? std::get<column_view>(input)
                  : std::get<scalar_column_view>(input).as_column_view();
    impl_->ast_input_types_.push_back(view.type());
    impl_->ast_input_nullable_.push_back(view.nullable());
    if (auto const* scalar = std::get_if<scalar_column_view>(&input)) {
      // The program must outlive non-owning scalar-column literals in the source AST.
      // TODO: Reuse converter-owned scalar columns to avoid copies while preserving input order.
      impl_->ast_scalar_columns_.push_back(
        std::make_unique<column>(scalar->as_column_view(), stream, mr));
    }
  }
  impl_->ast_input_column_indices_ = std::move(args.input_column_indices);
  impl_->ast_outputs_              = std::move(args.outputs);
}

transform_program::transform_program(transform_program&&)            = default;
transform_program& transform_program::operator=(transform_program&&) = default;
transform_program::~transform_program()                              = default;

std::unique_ptr<table> transform_program::run(std::span<transform_input const> inputs,
                                              std::span<transform_output const> outputs,
                                              std::vector<std::unique_ptr<column>>&& string_offsets,
                                              std::optional<size_type> row_size,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  impl_->validate(inputs, outputs, string_offsets);
  perform_checks(impl_->is_null_aware_, row_size, inputs, outputs, string_offsets);
  return execute_transform(impl_->kernel_,
                           impl_->is_null_aware_,
                           row_size,
                           inputs,
                           outputs,
                           std::move(string_offsets),
                           stream,
                           mr);
}

std::unique_ptr<table> transform_program::run(table_view const& table,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(impl_->ast_input_column_indices_.has_value(),
               "Transform program was not constructed from an AST expression",
               std::invalid_argument);

  std::vector<transform_input> inputs;
  auto scalar_index = std::size_t{0};
  for (auto i = std::size_t{0}; i < impl_->ast_input_column_indices_->size(); ++i) {
    auto& column_index = (*impl_->ast_input_column_indices_)[i];
    if (column_index.has_value()) {
      auto input = table.column(*column_index);
      CUDF_EXPECTS(input.type() == impl_->ast_input_types_[i] &&
                     input.nullable() == impl_->ast_input_nullable_[i],
                   "AST transform program input schema does not match the provided table",
                   std::invalid_argument);
      inputs.emplace_back(input);
    } else {
      CUDF_EXPECTS(scalar_index < impl_->ast_scalar_columns_.size(),
                   "AST transform scalar input metadata mismatch");
      inputs.emplace_back(scalar_column_view{impl_->ast_scalar_columns_[scalar_index++]->view()});
    }
  }
  CUDF_EXPECTS(scalar_index == impl_->ast_scalar_columns_.size(),
               "AST transform scalar input metadata mismatch");
  return run(inputs, impl_->ast_outputs_, {}, table.num_rows(), stream, mr);
}

}  // namespace cudf
