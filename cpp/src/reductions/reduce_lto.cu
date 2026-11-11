/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/string_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/errc.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_layout.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/buffer>
#include <cuda/std/optional>
#include <cuda/std/tuple>

#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/reflect.hpp>
#include <jit/util.hpp>
#include <reductions/jit/reduce.cuh>
#include <rtcx/rtcx.hpp>
#include <runtime/context.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace cudf {
namespace {

enum class reduction_kind : int8_t { scalar = 0, segmented = 1, small = 2 };

constexpr int32_t default_block_size    = 8 * detail::warp_size;
constexpr int32_t max_block_size        = 1024;
constexpr int32_t reduction_table_index = 0;

constexpr size_t small_state_bytes       = 64;
constexpr size_t medium_state_bytes      = 256;
constexpr size_t max_shuffle_state_bytes = 512;

constexpr int32_t single_block_rows_per_thread   = 16;
constexpr int32_t segment_rows_per_thread        = 32;
constexpr int32_t max_blocks_per_segment         = 64;
constexpr size_type small_group_max_segment_size = 128;
constexpr size_t small_group_max_state_bytes     = 256;

constexpr char const* scalar_kernel_id = "cudf/cpp/src/reductions/jit/reduce_kernel.cu";
constexpr char const* segmented_kernel_id =
  "cudf/cpp/src/reductions/jit/segmented_reduce_kernel.cu";
constexpr char const* small_groups_segmented_kernel_id =
  "cudf/cpp/src/reductions/jit/small_groups_segmented_reduce.cu";

constexpr char const* udf_dispatcher_path = "cudf/cpp/src/jit/udf_dispatch.cu";

constexpr char const* udf_dispatch_header    = "cudf/detail/udf_dispatch.cuh";
constexpr char const* kernel_instance_header = "cudf/detail/kernel_instance.cuh";

constexpr char const* scalar_kernel_name    = "::cudf::jit::reduce_kernel";
constexpr char const* segmented_kernel_name = "::cudf::jit::segmented_reduce_kernel";
constexpr char const* small_groups_segmented_kernel_name =
  "::cudf::jit::small_groups_segmented_reduce_kernel";

constexpr char const* kernel_source_path(reduction_kind kind)
{
  switch (kind) {
    case reduction_kind::scalar: return scalar_kernel_id;
    case reduction_kind::segmented: return segmented_kernel_id;
    case reduction_kind::small: return small_groups_segmented_kernel_id;
  }
  CUDF_UNREACHABLE("Unsupported reduction kind");
}

struct value_reflection {
  std::string physical{};
  std::string logical{};
};

struct reduction_launch_config {
  int32_t block_size = default_block_size;
  size_t state_bytes = 0;
  bool warp_reduce   = true;
};

struct kernel_reflection {
  std::vector<value_reflection> input_types{};
  std::vector<value_reflection> output_types{};
  std::vector<value_reflection> state_types{};
  std::string inputs{};
  std::string outputs{};
  std::string state{};
  detail::memory_layout state_layout{};
  int32_t block_size = default_block_size;
  bool warp_reduce   = true;
};

struct callback_reflection {
  std::string_view stage{};
  std::string_view target{};
  std::string_view udf_params_macro{};
  std::string udf_params{};
  std::vector<value_reflection> parameters{};
  std::string symbol{};
};

struct reduction_shape {
  std::optional<size_type> rows = std::nullopt;
  size_type segments            = 1;
};

reduction_shape validate_reduce(std::span<reduce_input const> inputs,
                                std::span<data_type const> states,
                                std::span<data_type const> outputs,
                                std::optional<device_span<size_type const>> offsets = std::nullopt,
                                std::optional<size_type> max_segment_size           = std::nullopt)
{
  auto supported_type = [](data_type type) {
    return type.id() != type_id::EMPTY && (is_fixed_width(type) || type.id() == type_id::STRING);
  };
  CUDF_EXPECTS(std::ranges::all_of(states, supported_type),
               "JIT reduction state types must be fixed-width or strings",
               std::invalid_argument);
  CUDF_EXPECTS(std::ranges::all_of(outputs, supported_type),
               "JIT reduction output types must be fixed-width or strings",
               std::invalid_argument);

  reduction_shape result;

  for (auto const& input : inputs) {
    if (auto const* view = std::get_if<column_view>(&input)) {
      CUDF_EXPECTS(!result.rows || *result.rows == view->size(),
                   "Reduction input lengths must match",
                   std::invalid_argument);
      result.rows = view->size();
    }
  }

  if (!offsets) { return result; }

  CUDF_EXPECTS(!offsets->empty() &&
                 offsets->size() - 1 <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "Offsets must contain num_segments + 1 entries",
               std::invalid_argument);
  CUDF_EXPECTS(!max_segment_size || *max_segment_size >= 0,
               "Maximum segment size must be nonnegative",
               std::invalid_argument);

  result.segments = static_cast<size_type>(offsets->size() - 1);
  return result;
}

std::vector<value_reflection> reflect_value_parameters(std::span<value_reflection const> types,
                                                       std::span<uint8_t const> is_optional)
{
  std::vector<value_reflection> result;
  for (size_t i = 0; i < types.size(); ++i) {
    auto const& type = types[i];
    result.push_back(
      is_optional[i] && type.logical != "void*"
        ? value_reflection{rtcx::reflect_template("cuda::std::optional", type.physical),
                           rtcx::reflect_template("cuda::std::optional", type.logical)}
        : type);
  }
  return result;
}

using jit::as_column_view;

column_view as_column_view(user_data const& value)
{
  return {data_type{type_id::INT8}, 0, value.ptr, nullptr, 0};
}

column_view as_column_view(reduce_input const& input)
{
  return std::visit([](auto const& value) { return as_column_view(value); }, input);
}

std::pair<std::vector<value_reflection>, std::string> reflect_inputs(
  std::span<reduce_input const> inputs, std::span<uint8_t const> is_optional)
{
  std::vector<value_reflection> input_types;
  std::vector<std::string> accessors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (std::holds_alternative<user_data>(inputs[i])) {
      input_types.push_back({"void*", "void*"});
      accessors.push_back(
        rtcx::reflect_template("::cudf::jit::user_data_accessor", rtcx::reflect(i)));
      continue;
    }
    auto scalar = std::holds_alternative<scalar_column_view>(inputs[i]);
    auto input  = jit::make_input_spec(as_column_view(inputs[i]), scalar);
    auto type   = jit::reflect_input_element(input, true);
    input_types.push_back(
      {jit::reflect_input_value_type(input, true), jit::reflect_input_value_type(input, false)});
    accessors.push_back(rtcx::reflect_template("::cudf::jit::column_accessor",
                                               rtcx::reflect(i),
                                               "::cudf::column_device_view_core",
                                               type,
                                               rtcx::reflect(scalar),
                                               rtcx::reflect(reduction_table_index)));
  }

  return {reflect_value_parameters(input_types, is_optional),
          rtcx::reflect_template("::cudf::jit::type_list", accessors)};
}

std::pair<std::vector<value_reflection>, std::string> reflect_outputs(
  std::span<data_type const> outputs, std::span<uint8_t const> is_optional)
{
  std::vector<value_reflection> output_types;
  std::vector<std::string> accessors;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto type = type_to_name(jit::physical_type_of(outputs[i]));
    output_types.push_back({type, type_to_name(outputs[i])});
    accessors.push_back(rtcx::reflect_template(
      "::cudf::jit::column_accessor",
      rtcx::reflect(i),
      jit::reflect_output_column(transform_output_spec{.type = outputs[i].id()}),
      type,
      rtcx::reflect(false),
      rtcx::reflect(reduction_table_index)));
  }
  return {reflect_value_parameters(output_types, is_optional),
          rtcx::reflect_template("::cudf::jit::type_list", accessors)};
}

// Reserve a conservative ABI-independent bound: every field may need alignment padding.
// is_optional contains one flag per state field; nonzero flags select optional storage.
std::tuple<detail::memory_layout, std::vector<value_reflection>> reflect_state(
  std::span<data_type const> states, std::span<uint8_t const> is_optional)
{
  // An empty tuple is still a member object and occupies storage before the error code.
  std::vector<value_reflection> state_types;

  auto layout =
    states.empty() ? detail::memory_layout_of<cuda::std::tuple<>> : detail::memory_layout{};

  for (size_t i = 0; i < states.size(); ++i) {
    auto logical       = states[i];
    auto physical      = jit::physical_type_of(logical);
    auto logical_name  = type_to_name(logical);
    auto physical_name = type_to_name(physical);
    state_types.push_back({physical_name, logical_name});
    auto field_layout = type_dispatcher(physical, [&]<typename T> {
      return is_optional[i] ? detail::memory_layout_of<cuda::std::optional<T>>
                            : detail::memory_layout_of<T>;
    });
    layout            = layout.append(field_layout);
  }

  layout = layout
             // end of tuple struct
             .aligned()
             // append the error code layout
             .append(detail::memory_layout_of<errc>)
             // end of the state struct
             .aligned();

  return {layout, reflect_value_parameters(state_types, is_optional)};
}

constexpr int32_t block_size_from_layout(detail::memory_layout const& layout)
{
  // Select block size based on the layout size.
  if (layout.size <= small_state_bytes) return default_block_size;
  if (layout.size <= medium_state_bytes) return default_block_size / 2;
  if (layout.size <= max_shuffle_state_bytes) return default_block_size / 4;
  return detail::warp_size;
}

bool use_warp_reduce(detail::memory_layout const& layout)
{
  return layout.size <= max_shuffle_state_bytes;
}

kernel_reflection reflect_kernel(std::span<reduce_input const> inputs,
                                 std::span<data_type const> states,
                                 std::span<data_type const> outputs,
                                 std::span<uint8_t const> input_is_optional,
                                 std::span<uint8_t const> state_is_optional,
                                 std::span<uint8_t const> output_is_optional)
{
  auto [input_types, input_accessors]   = reflect_inputs(inputs, input_is_optional);
  auto [output_types, output_accessors] = reflect_outputs(outputs, output_is_optional);
  auto [state_layout, state_types]      = reflect_state(states, state_is_optional);
  std::vector<std::string> physical_states;
  for (auto const& type : state_types) {
    physical_states.push_back(type.physical);
  }

  return kernel_reflection{
    .input_types  = input_types,
    .output_types = output_types,
    .state_types  = state_types,
    .inputs       = input_accessors,
    .outputs      = output_accessors,
    .state        = rtcx::reflect_template("::cudf::jit::reduce::state", physical_states),
    .state_layout = state_layout,
    .block_size   = block_size_from_layout(state_layout),
    .warp_reduce  = use_warp_reduce(state_layout)};
}

kernel_reflection reflect_kernel(std::span<reduce_input const> inputs,
                                 std::span<data_type const> states,
                                 std::span<data_type const> outputs,
                                 bool optional_values)
{
  std::vector<uint8_t> input_is_optional(inputs.size(), optional_values);
  std::vector<uint8_t> output_is_optional(outputs.size(), optional_values);
  std::vector<uint8_t> state_is_optional(states.size(), optional_values);
  return reflect_kernel(
    inputs, states, outputs, input_is_optional, state_is_optional, output_is_optional);
}

template <typename... Args>
std::string reflect_kernel_instance(std::string_view name,
                                    kernel_reflection const& spec,
                                    bool optional_values,
                                    Args const&... args)
{
  return rtcx::reflect_template(name,
                                rtcx::reflect(optional_values),
                                spec.inputs,
                                spec.outputs,
                                rtcx::reflect(spec.block_size),
                                args...);
}

std::string reflect_reduce_kernel(kernel_reflection const& spec, bool optional_values)
{
  return reflect_kernel_instance(
    scalar_kernel_name, spec, optional_values, rtcx::reflect(spec.warp_reduce), spec.state);
}

std::string reflect_segmented_reduce_kernel(kernel_reflection const& spec, bool optional_values)
{
  return reflect_kernel_instance(
    segmented_kernel_name, spec, optional_values, rtcx::reflect(spec.warp_reduce), spec.state);
}

std::string reflect_small_groups_segmented_reduce_kernel(kernel_reflection const& spec,
                                                         bool optional_values,
                                                         int32_t group_size)
{
  return reflect_kernel_instance(small_groups_segmented_kernel_name,
                                 spec,
                                 optional_values,
                                 rtcx::reflect(group_size),
                                 spec.state);
}

std::vector<value_reflection> reflect_pointer_parameters(std::span<value_reflection const> types)
{
  std::vector<value_reflection> result;
  for (auto const& type : types) {
    result.push_back({type.physical + "*", type.logical + "*"});
  }
  return result;
}

std::vector<value_reflection> reflect_callback_parameters(
  bool segmented,
  std::span<value_reflection const> first  = {},
  std::span<value_reflection const> second = {})
{
  std::vector<value_reflection> result;
  if (segmented) { result.push_back({"::cudf::size_type", "::cudf::size_type"}); }
  result.insert(result.end(), first.begin(), first.end());
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

callback_reflection reflect_callback(std::string_view stage,
                                     std::string_view target,
                                     std::string_view udf_params_macro,
                                     std::vector<value_reflection> parameters,
                                     std::string_view symbol)
{
  auto udf_params =
    detail::join_strings(parameters.begin(), parameters.end(), ", ", &value_reflection::physical);
  return {.stage            = stage,
          .target           = target,
          .udf_params_macro = udf_params_macro,
          .udf_params       = std::move(udf_params),
          .parameters       = std::move(parameters),
          .symbol           = std::string{symbol}};
}

callback_reflection reflect_init_callback(std::string_view symbol,
                                          std::span<value_reflection const> state_pointers,
                                          bool segmented)
{
  return reflect_callback("CUDF_REDUCE_INIT",
                          "cudf_reduce_init",
                          "CUDF_REDUCE_INIT_UDF_PARAMS",
                          reflect_callback_parameters(segmented, state_pointers),
                          symbol);
}

callback_reflection reflect_update_callback(std::string_view symbol,
                                            std::span<value_reflection const> inputs,
                                            std::span<value_reflection const> state_pointers,
                                            bool segmented)
{
  auto parameters = reflect_callback_parameters(segmented);
  parameters.push_back({"::cudf::size_type", "::cudf::size_type"});
  parameters.insert(parameters.end(), state_pointers.begin(), state_pointers.end());
  parameters.insert(parameters.end(), inputs.begin(), inputs.end());
  return reflect_callback("CUDF_REDUCE_UPDATE",
                          "cudf_reduce_update",
                          "CUDF_REDUCE_UPDATE_UDF_PARAMS",
                          std::move(parameters),
                          symbol);
}

callback_reflection reflect_filter_callback(std::string_view symbol,
                                            std::span<value_reflection const> inputs,
                                            bool segmented)
{
  auto parameters = reflect_callback_parameters(segmented);
  parameters.push_back({"::cudf::size_type", "::cudf::size_type"});
  parameters.push_back({"bool*", "bool*"});
  parameters.insert(parameters.end(), inputs.begin(), inputs.end());
  return reflect_callback("CUDF_REDUCE_FILTER",
                          "cudf_reduce_filter",
                          "CUDF_REDUCE_FILTER_UDF_PARAMS",
                          std::move(parameters),
                          symbol);
}

callback_reflection reflect_merge_callback(std::string_view symbol,
                                           std::span<value_reflection const> state_pointers,
                                           std::span<value_reflection const> state_values,
                                           bool segmented)
{
  return reflect_callback("CUDF_REDUCE_MERGE",
                          "cudf_reduce_merge",
                          "CUDF_REDUCE_MERGE_UDF_PARAMS",
                          reflect_callback_parameters(segmented, state_pointers, state_values),
                          symbol);
}

callback_reflection reflect_finalize_callback(std::string_view symbol,
                                              std::span<value_reflection const> outputs,
                                              std::span<value_reflection const> state_values,
                                              bool segmented)
{
  return reflect_callback("CUDF_REDUCE_FINALIZE",
                          "cudf_reduce_finalize",
                          "CUDF_REDUCE_FINALIZE_UDF_PARAMS",
                          reflect_callback_parameters(segmented, outputs, state_values),
                          symbol);
}

std::vector<callback_reflection> reflect_callbacks(std::span<value_reflection const> input_types,
                                                   std::span<value_reflection const> output_types,
                                                   std::span<value_reflection const> state_types,
                                                   reduce_udf const& udf,
                                                   bool segmented)
{
  auto outputs        = reflect_pointer_parameters(output_types);
  auto state_pointers = reflect_pointer_parameters(state_types);

  std::vector<callback_reflection> callbacks{
    reflect_init_callback(udf.init, state_pointers, segmented),
    reflect_update_callback(udf.update, input_types, state_pointers, segmented),
    reflect_merge_callback(udf.merge, state_pointers, state_types, segmented),
    reflect_finalize_callback(udf.finalize, outputs, state_types, segmented)};

  if (!udf.filter.empty()) {
    callbacks.push_back(reflect_filter_callback(udf.filter, input_types, segmented));
  }

  return callbacks;
}

std::optional<rtcx::blob> compile_udf_dispatcher(std::span<callback_reflection const> callbacks)
{
  if (std::ranges::all_of(callbacks, [](callback_reflection const& callback) {
        return callback.symbol == callback.target;
      })) {
    return std::nullopt;
  }

  std::string source;
  for (auto const& callback : callbacks) {
    // The early return handles only all-default callbacks. For mixed default/custom symbols,
    // skip matching callbacks to avoid duplicate definitions and wrappers that call themselves.
    if (callback.symbol == callback.target) { continue; }
    auto const parameter_types = detail::join_strings(
      callback.parameters.begin(), callback.parameters.end(), ", ", &value_reflection::logical);
    auto const indices = std::views::iota(size_t{0}, callback.parameters.size());
    auto const declarations =
      detail::join_strings(indices.begin(), indices.end(), ", ", [&](size_t i) {
        return std::format("{} arg{}", callback.parameters[i].logical, i);
      });
    auto const arguments = detail::join_strings(
      indices.begin(), indices.end(), ", ", [](size_t i) { return std::format("arg{}", i); });
    source += std::format(
      R"CUDA(
extern "C" __device__ ::cudf::errc {}({});

extern "C" __device__ ::cudf::errc {}({})
{{
  return {}({});
}}

)CUDA",
      callback.symbol,
      parameter_types,
      callback.target,
      declarations,
      callback.symbol,
      arguments);
  }

  char const* const names[]{udf_dispatch_header};
  char const* const headers[]{source.c_str()};

  return get_kernel_fragment(udf_dispatcher_path, udf_dispatcher_path, names, headers, "");
}

kernel compile(reduction_kind kind,
               kernel_reflection const& spec,
               reduce_udf const& udf,
               std::string const& instance)
{
  auto callbacks = reflect_callbacks(
    spec.input_types, spec.output_types, spec.state_types, udf, kind != reduction_kind::scalar);

  // Physical storage and launch specialization determine the kernel fragment. Logical UDF types
  // and entry-point names occur only in the independently cached dispatcher fragment.

  auto const callbacks_defines = detail::join_strings(
    callbacks.begin(), callbacks.end(), "\n", [](callback_reflection const& callback) {
      return std::format("#define {} {}", callback.udf_params_macro, callback.udf_params);
    });

  auto source = std::format(
    R"CUDA(
#define CUDF_DISPATCH_REDUCE_STATE {}
{}
{}
#define CUDF_KERNEL_INSTANCE {}
  )CUDA",
    spec.state,
    callbacks_defines,
    udf.filter.empty() ? "#define CUDF_DISPATCH_REDUCE_FILTER ::cudf::jit::reduce::select_all{}"
                       : "",
    instance);

  char const* const names[]{kernel_instance_header};
  char const* const headers[]{source.c_str()};

  std::string const path = kernel_source_path(kind);
  auto object            = get_kernel_fragment(path, path, names, headers, instance);
  auto dispatch          = compile_udf_dispatcher(callbacks);

  std::vector<rtcx::memory_fragment> fragments{
    {object->view(), rtcx::binary_type::LTO_IR, nullptr}};

  if (dispatch.has_value()) {
    fragments.push_back({dispatch.value()->view(), rtcx::binary_type::LTO_IR, nullptr});
  }

  for (auto const& fragment : udf.fragments) {
    fragments.push_back({fragment.data, jit::as_rtcx_binary_type(fragment.type), nullptr});
  }

  return get_lto_linked_kernel(path, {}, fragments);
}

struct input_arguments {
  using handle = std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>;

  std::vector<handle> handles;
  cuda::device_buffer<column_device_view_core> views;

  input_arguments(std::span<reduce_input const> inputs,
                  cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
    : views(stream, mr, inputs.size(), cuda::no_init)
  {
    auto host = detail::make_empty_pinned_vector<column_device_view_core>(inputs.size(), stream);

    for (auto const& input : inputs) {
      auto device = column_device_view::create(as_column_view(input), stream, mr);
      host.push_back(*device);
      handles.push_back(std::move(device));
    }

    // Synchronous transfer keeps the pinned staging array alive until the copy completes.
    detail::cuda_memcpy<column_device_view_core>(views, host, stream);
  }
};

struct output_arguments {
  std::vector<std::unique_ptr<column>> columns;
  cuda::device_buffer<mutable_column_device_view_core> views;
  size_type num_segments;

  static std::unique_ptr<column> allocate_output(data_type type,
                                                 bool nullable,
                                                 cuda::stream_ref stream,
                                                 memory_resources mr)
  {
    auto const mask = nullable ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED;
    if (type.id() != type_id::STRING) {
      return make_fixed_width_column(type, 1, mask, stream, mr.get_output_mr());
    }

    // String outputs initially contain a view. Finalization copies its characters into a column.
    return std::make_unique<column>(
      type,
      1,
      rmm::device_buffer{sizeof(string_view), stream, mr.get_temporary_mr()},
      create_null_mask(1, mask, stream, mr.get_output_mr()),
      0);
  }

  static std::unique_ptr<column> finalize_string_output(column& output,
                                                        string_view value,
                                                        cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
  {
    auto const nulls = output.null_count();
    auto const bytes = nulls == 0 ? value.size_bytes() : size_type{0};
    auto offsets     = detail::make_device_uvector(std::vector<size_type>{0, bytes}, stream, mr);
    auto offsets_column =
      std::make_unique<column>(data_type{type_id::INT32},
                               2,
                               offsets.release(),
                               create_null_mask(0, mask_state::UNALLOCATED, stream, mr),
                               0);
    rmm::device_buffer chars(bytes, stream, mr);
    if (bytes != 0) {
      CUDF_CUDA_TRY(detail::memcpy_async(chars.data(), value.data(), bytes, stream));
    }
    auto contents = output.release();
    return make_strings_column(
      1, std::move(offsets_column), std::move(chars), nulls, std::move(*contents.null_mask));
  }

  output_arguments(std::span<data_type const> types,
                   size_type num_segments,
                   bool nullable,
                   cuda::stream_ref stream,
                   memory_resources mr)
    : views(stream, mr.get_temporary_mr(), num_segments * types.size(), cuda::no_init),
      num_segments(num_segments)
  {
    columns.reserve(num_segments * types.size());
    auto host =
      detail::make_empty_pinned_vector<mutable_column_device_view_core>(views.size(), stream);

    for (size_type segment = 0; segment < num_segments; ++segment) {
      for (auto type : types) {
        auto output     = allocate_output(type, nullable, stream, mr);
        auto const view = output->mutable_view();
        host.push_back(mutable_column_device_view::create(
          type, 1, view.head(), view.null_mask(), 0, nullptr, 0));
        columns.push_back(std::move(output));
      }
    }

    if (!host.empty()) {
      detail::cuda_memcpy<mutable_column_device_view_core>(views, host, stream);
    }
  }

  std::vector<std::unique_ptr<column>> finalize(errc const* errors,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
  {
    auto const nullable = !columns.empty() && columns.front()->nullable();
    auto host_masks =
      detail::make_pinned_vector_async<bitmask_type>(nullable ? columns.size() : 0, stream);
    auto const has_strings = std::ranges::any_of(
      columns, [](auto const& col) { return col->type().id() == type_id::STRING; });
    auto host_strings =
      detail::make_pinned_vector_async<string_view>(has_strings ? columns.size() : 0, stream);

    auto host_errors = detail::make_pinned_vector_async(
      device_span<errc const>{errors, static_cast<size_t>(num_segments)}, stream);

    for (size_t col = 0; col < columns.size(); ++col) {
      if (columns[col]->nullable()) {
        detail::cuda_memcpy_async<bitmask_type>(
          host_span<bitmask_type>{host_masks.data() + col, 1},
          device_span<bitmask_type const>{columns[col]->view().null_mask(), 1},
          stream);
      }
      if (columns[col]->type().id() == type_id::STRING) {
        detail::cuda_memcpy_async<string_view>(
          host_span<string_view>{host_strings.data() + col, 1},
          device_span<string_view const>{
            static_cast<string_view const*>(columns[col]->view().head()), 1},
          stream);
      }
    }

    // Synchronize to ensure all asynchronous copies are complete before proceeding.
    detail::sync_stream(stream);

    auto const error = *std::max_element(host_errors.begin(), host_errors.end());

    if (error != errc::SUCCESS) {
      throw evaluation_error(
        error, std::format("Reduction UDF evaluation failed with error `{}`", to_string(error)));
    }

    for (size_t col = 0; col < columns.size(); ++col) {
      size_type const nulls = columns[col]->nullable() && host_masks[col] == 0;
      columns[col]->set_null_count(nulls);
      if (columns[col]->type().id() == type_id::STRING) {
        columns[col] = finalize_string_output(*columns[col], host_strings[col], stream, mr);
      }
    }

    return std::move(columns);
  }
};

void launch_reduce(kernel_ref kernel,
                   reduction_launch_config config,
                   size_type num_rows,
                   column_device_view_core const* inputs,
                   mutable_column_device_view_core const* outputs,
                   errc* error,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(config.block_size >= detail::warp_size && config.block_size <= max_block_size &&
                 (config.block_size & (config.block_size - 1)) == 0,
               "Reduction block size must be a power of two between 32 and 1024",
               std::invalid_argument);
  CUDF_EXPECTS(
    config.state_bytes > 0, "Reduction state size must be positive", std::invalid_argument);
  CUDF_EXPECTS(num_rows >= 0, "Reduction row count must be nonnegative", std::invalid_argument);

  auto occupancy = kernel.max_occupancy_config(0, config.block_size);

  // Use a work-per-thread cutoff to keep small inputs in one block and avoid a merge launch.
  // For larger inputs, cap the grid at the occupancy recommendation; grid-stride iteration
  // covers the remaining rows while bounding partial-state storage and subsequent merge work.
  auto grid = num_rows <= config.block_size * single_block_rows_per_thread
                ? 1
                : std::min<int32_t>(detail::div_rounding_up_safe(num_rows, config.block_size),
                                    occupancy.min_grid_size);

  // Empty inputs still need one block to initialize and finalize the identity state.
  grid = std::max(1, grid);

  cuda::device_buffer<std::byte> partials(stream, mr, grid * config.state_bytes, cuda::no_init);
  cuda::device_buffer<std::byte> result(stream, mr, config.state_bytes, cuda::no_init);
  cuda::device_buffer<std::byte> scratch(
    stream,
    mr,
    config.warp_reduce ? 0 : config.state_bytes * grid * config.block_size,
    cuda::no_init);

  auto launch = [&](jit::reduce::phase phase,
                    size_type rows,
                    int32_t blocks,
                    void const* source,
                    void* destination,
                    bool finalize) {
    kernel.launch_with({static_cast<uint32_t>(blocks)},
                       {static_cast<uint32_t>(config.block_size)},
                       0,
                       stream,
                       phase,
                       rows,
                       inputs,
                       outputs,
                       source,
                       destination,
                       scratch.data(),
                       error,
                       finalize);
  };

  launch(jit::reduce::phase::REDUCE, num_rows, grid, nullptr, partials.data(), grid == 1);

  if (grid > 1) {
    launch(jit::reduce::phase::MERGE, grid, 1, partials.data(), result.data(), true);
  }
}

void launch_segmented_reduce(kernel_ref kernel,
                             reduction_launch_config config,
                             size_type num_segments,
                             size_type const* offsets,
                             size_type max_segment_size,
                             column_device_view_core const* inputs,
                             mutable_column_device_view_core const* outputs,
                             errc* errors,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(config.block_size >= detail::warp_size && config.block_size <= max_block_size &&
                 (config.block_size & (config.block_size - 1)) == 0,
               "Reduction block size must be a power of two between 32 and 1024",
               std::invalid_argument);
  CUDF_EXPECTS(
    config.state_bytes > 0, "Reduction state size must be positive", std::invalid_argument);
  CUDF_EXPECTS(num_segments >= 0 && max_segment_size >= 0,
               "Segment count and maximum segment size must be nonnegative",
               std::invalid_argument);

  if (num_segments == 0) { return; }

  auto occupancy = kernel.max_occupancy_config(0, config.block_size);
  auto const sms = get_context().get_device_properties().num_multiprocessors;

  int32_t blocks_per_segment = 1;

  // Split segments only when one block per segment would leave some SMs without work.
  if (num_segments < sms) {
    // Estimate useful splitting from the length hint, targeting segment_rows_per_thread rows
    // per thread. Shorter segments may need less work than this upper-bound estimate.
    auto const work_blocks =
      detail::div_rounding_up_safe(max_segment_size, config.block_size * segment_rows_per_thread);

    // Give each segment a share of the occupancy-recommended grid so splitting fills that grid
    // without unnecessarily increasing the number of partial states to merge.
    auto const occupancy_blocks =
      detail::div_rounding_up_safe<int32_t>(occupancy.min_grid_size, num_segments);

    // Keep one block for empty or unknown lengths; cap splitting to bound storage and merge cost.
    blocks_per_segment =
      std::clamp(std::min(work_blocks, occupancy_blocks), 1, max_blocks_per_segment);
  }

  auto const tiles = static_cast<int64_t>(num_segments) * blocks_per_segment;
  auto const grid  = std::clamp<int64_t>(tiles, 1, occupancy.min_grid_size);

  cuda::device_buffer<std::byte> partials(stream, mr, tiles * config.state_bytes, cuda::no_init);
  cuda::device_buffer<std::byte> result(
    stream, mr, blocks_per_segment == 1 ? 0 : num_segments * config.state_bytes, cuda::no_init);
  cuda::device_buffer<std::byte> scratch(
    stream,
    mr,
    config.warp_reduce ? 0 : config.state_bytes * grid * config.block_size,
    cuda::no_init);

  auto launch = [&](jit::reduce::phase phase, void const* source, void* destination) {
    kernel.launch_with({static_cast<uint32_t>(grid)},
                       {static_cast<uint32_t>(config.block_size)},
                       0,
                       stream,
                       phase,
                       num_segments,
                       offsets,
                       inputs,
                       outputs,
                       blocks_per_segment,
                       source,
                       destination,
                       scratch.data(),
                       errors,
                       true);
  };

  launch(jit::reduce::phase::REDUCE, nullptr, partials.data());

  if (blocks_per_segment > 1) { launch(jit::reduce::phase::MERGE, partials.data(), result.data()); }
}

void launch_small_groups_segmented_reduce(kernel_ref kernel,
                                          int32_t block_size,
                                          int32_t group_size,
                                          size_type num_segments,
                                          size_type const* offsets,
                                          column_device_view_core const* inputs,
                                          mutable_column_device_view_core const* outputs,
                                          errc* errors,
                                          cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(block_size >= detail::warp_size && block_size <= max_block_size &&
                 (block_size & (block_size - 1)) == 0,
               "Reduction block size must be a power of two between 32 and 1024",
               std::invalid_argument);
  CUDF_EXPECTS(
    group_size >= 1 && group_size <= detail::warp_size && (group_size & (group_size - 1)) == 0,
    "Reduction group size must be a power of two between 1 and 32",
    std::invalid_argument);
  CUDF_EXPECTS(num_segments >= 0, "Segment count must be nonnegative", std::invalid_argument);

  if (num_segments == 0) { return; }

  // No dynamic shared memory is needed. Each group handles one segment at a time, so size a
  // single pass using groups_per_block, then cap the grid at the occupancy recommendation.
  // The kernel's group-stride loop covers any remaining segments without launching more blocks.
  auto occupancy              = kernel.max_occupancy_config(0, block_size);
  auto const groups_per_block = block_size / group_size;
  auto const blocks           = detail::div_rounding_up_safe(num_segments, groups_per_block);
  auto const grid             = std::clamp<int32_t>(blocks, 1, occupancy.min_grid_size);

  kernel.launch_with({static_cast<uint32_t>(grid)},
                     {static_cast<uint32_t>(block_size)},
                     0,
                     stream,
                     num_segments,
                     offsets,
                     inputs,
                     outputs,
                     errors);
}

}  // namespace

std::vector<std::unique_ptr<column>> reduce_lto(reduce_udf const& udf,
                                                null_aware nulls,
                                                std::span<reduce_input const> inputs,
                                                std::span<data_type const> state_types,
                                                std::span<data_type const> types,
                                                cuda::stream_ref stream,
                                                memory_resources mr)
{
  CUDF_FUNC_RANGE();

  auto const optional_values = nulls == null_aware::YES;
  auto shape                 = validate_reduce(inputs, state_types, types);
  auto spec                  = reflect_kernel(inputs, state_types, types, optional_values);
  auto rows                  = shape.rows.value_or(0);
  auto instance              = reflect_reduce_kernel(spec, optional_values);
  auto compiled              = compile(reduction_kind::scalar, spec, udf, instance);

  input_arguments in(inputs, stream, mr.get_temporary_mr());
  output_arguments out(types, 1, true, stream, mr);

  cuda::device_buffer<errc> error(stream, mr.get_temporary_mr(), 1, cuda::no_init);

  launch_reduce(compiled.get(),
                {spec.block_size, spec.state_layout.size, spec.warp_reduce},
                rows,
                in.views.data(),
                out.views.data(),
                error.data(),
                stream,
                mr.get_temporary_mr());

  return out.finalize(error.data(), stream, mr.get_output_mr());
}

std::vector<std::unique_ptr<column>> segmented_reduce_lto(reduce_udf const& udf,
                                                          null_aware nulls,
                                                          std::span<reduce_input const> inputs,
                                                          device_span<size_type const> offsets,
                                                          std::span<data_type const> state_types,
                                                          std::span<data_type const> types,
                                                          std::optional<size_type> max_segment_size,
                                                          cuda::stream_ref stream,
                                                          memory_resources mr)
{
  CUDF_FUNC_RANGE();

  auto const optional_values = nulls == null_aware::YES;
  auto shape = validate_reduce(inputs, state_types, types, offsets, max_segment_size);
  auto spec  = reflect_kernel(inputs, state_types, types, optional_values);

  auto segments = shape.segments;

  if (segments == 0) { return {}; }

  auto longest = max_segment_size.value_or(std::numeric_limits<size_type>::max());

  bool const small = longest <= small_group_max_segment_size &&
                     spec.state_layout.size <= small_group_max_state_bytes;

  int32_t const group = std::bit_ceil<uint32_t>(std::clamp(longest, 1, detail::warp_size));

  auto instance = small ? reflect_small_groups_segmented_reduce_kernel(spec, optional_values, group)
                        : reflect_segmented_reduce_kernel(spec, optional_values);

  auto compiled =
    compile(small ? reduction_kind::small : reduction_kind::segmented, spec, udf, instance);

  input_arguments in(inputs, stream, mr.get_temporary_mr());
  output_arguments out(types, segments, true, stream, mr);
  cuda::device_buffer<errc> errors(stream, mr.get_temporary_mr(), segments, cuda::no_init);

  if (small) {
    launch_small_groups_segmented_reduce(compiled.get(),
                                         spec.block_size,
                                         group,
                                         segments,
                                         offsets.data(),
                                         in.views.data(),
                                         out.views.data(),
                                         errors.data(),
                                         stream);
  } else {
    launch_segmented_reduce(compiled.get(),
                            {spec.block_size, spec.state_layout.size, spec.warp_reduce},
                            segments,
                            offsets.data(),
                            longest,
                            in.views.data(),
                            out.views.data(),
                            errors.data(),
                            stream,
                            mr.get_temporary_mr());
  }

  return out.finalize(errors.data(), stream, mr.get_output_mr());
}

}  // namespace cudf
