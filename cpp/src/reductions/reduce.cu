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
#include <cudf/detail/utilities/strings.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/errc.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_layout.hpp>
#include <cudf/utilities/pinned_memory.hpp>
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
#include <reductions/jit/reduce_kernel.cuh>
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

constexpr int32_t default_block_size    = 8 * detail::warp_size;
constexpr int32_t max_block_size        = 1024;
constexpr int32_t reduction_table_index = 0;

constexpr size_t small_state_bytes       = 64;
constexpr size_t medium_state_bytes      = 256;
constexpr size_t max_shuffle_state_bytes = 512;

constexpr int32_t single_block_rows_per_thread = 16;

constexpr char const* scalar_kernel_id = "cudf/cpp/src/reductions/jit/reduce_kernel.cu";

constexpr char const* udf_dispatcher_path = "cudf/cpp/src/jit/udf_dispatch.cu";

constexpr char const* udf_dispatch_header    = "cudf/detail/udf_dispatch.cuh";
constexpr char const* kernel_instance_header = "cudf/detail/kernel_instance.cuh";

constexpr char const* scalar_kernel_name = "::cudf::jit::reduce_kernel";

struct reduction_launch_config {
  int32_t block_size = default_block_size;
  size_t state_bytes = 0;
  bool warp_reduce   = true;
};

struct kernel_reflection {
  std::vector<std::string> input_types{};
  std::vector<std::string> filter_input_types{};
  std::vector<std::string> output_types{};
  std::vector<std::string> state_types{};
  std::string inputs{};
  std::string filter_inputs{};
  std::string outputs{};
  std::string state{};
  detail::memory_layout state_layout{};
  int32_t block_size = default_block_size;
  bool warp_reduce   = true;
};

struct callback_reflection {
  std::string_view target{};
  std::string_view parameters_macro{};
  std::vector<std::string> parameters{};
  std::string symbol{};
};

struct reduction_shape {
  std::optional<size_type> rows = std::nullopt;
};

reduction_shape validate_reduce(std::span<reduce_input const> inputs,
                                std::span<reduce_input const> filter_inputs,
                                std::span<data_type const> states,
                                std::span<data_type const> outputs)
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

  for (auto args : {inputs, filter_inputs}) {
    for (auto& input : args) {
      if (auto* view = std::get_if<column_view>(&input)) {
        CUDF_EXPECTS(!result.rows || *result.rows == view->size(),
                     "Reduction input lengths must match",
                     std::invalid_argument);
        result.rows = view->size();
      }
    }
  }

  return result;
}

std::vector<std::string> reflect_value_parameters(std::span<std::string const> types,
                                                  std::span<uint8_t const> is_optional)
{
  std::vector<std::string> result;
  for (size_t i = 0; i < types.size(); ++i) {
    auto type = types[i];
    result.push_back(is_optional[i] && type != "cudf::voidptr_t"
                       ? rtcx::reflect_template("cuda::std::optional", type)
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

std::pair<std::vector<std::string>, std::string> reflect_inputs(
  std::span<reduce_input const> inputs, std::span<uint8_t const> is_optional)
{
  std::vector<std::string> input_types;
  std::vector<std::string> accessors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (std::holds_alternative<user_data>(inputs[i])) {
      input_types.push_back("cudf::voidptr_t");
      accessors.push_back(
        rtcx::reflect_template("::cudf::jit::user_data_accessor", rtcx::reflect(i)));
      continue;
    }
    auto scalar = std::holds_alternative<scalar_column_view>(inputs[i]);
    auto input  = jit::make_input_spec(as_column_view(inputs[i]), scalar);
    auto type   = jit::reflect_input_element(input, true);
    input_types.push_back(jit::reflect_input_value_type(input, true));
    accessors.push_back(rtcx::reflect_template("::cudf::jit::column_accessor",
                                               rtcx::reflect(i),
                                               type,
                                               rtcx::reflect(scalar),
                                               rtcx::reflect(reduction_table_index)));
  }

  return {reflect_value_parameters(input_types, is_optional),
          rtcx::reflect_template("::cudf::jit::type_list", accessors)};
}

std::pair<std::vector<std::string>, std::string> reflect_outputs(
  std::span<data_type const> outputs, std::span<uint8_t const> is_optional)
{
  std::vector<std::string> output_types;
  std::vector<std::string> accessors;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto type = type_to_name(jit::physical_type_of(outputs[i]));
    output_types.push_back(type);
    accessors.push_back(rtcx::reflect_template(
      "::cudf::jit::column_accessor",
      rtcx::reflect(i),
      outputs[i].id() == type_id::STRING ? rtcx::reflect_template("cudf::repr_element", type)
                                         : type,
      rtcx::reflect(false),
      rtcx::reflect(reduction_table_index)));
  }
  return {reflect_value_parameters(output_types, is_optional),
          rtcx::reflect_template("::cudf::jit::type_list", accessors)};
}

// Reserve a conservative ABI-independent bound: every field may need alignment padding.
// is_optional contains one flag per state field; nonzero flags select optional storage.
std::tuple<detail::memory_layout, std::vector<std::string>> reflect_state(
  std::span<data_type const> states, std::span<uint8_t const> is_optional)
{
  // An empty tuple is still a member object and occupies storage before the error code.
  std::vector<std::string> state_types;

  auto layout =
    states.empty() ? detail::memory_layout_of<cuda::std::tuple<>> : detail::memory_layout{};

  for (size_t i = 0; i < states.size(); ++i) {
    auto physical = jit::physical_type_of(states[i]);
    state_types.push_back(type_to_name(physical));
    auto field_layout = type_dispatcher(physical, [&]<typename T> {
      return is_optional[i] ? detail::memory_layout_of<cuda::std::optional<T>>
                            : detail::memory_layout_of<T>;
    });
    layout            = layout.append(field_layout);
  }

  layout = layout.aligned().append(detail::memory_layout_of<errc>).aligned();

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
                                 std::span<reduce_input const> filter_inputs,
                                 std::span<data_type const> states,
                                 std::span<data_type const> outputs,
                                 std::span<uint8_t const> input_is_optional,
                                 std::span<uint8_t const> filter_input_is_optional,
                                 std::span<uint8_t const> state_is_optional,
                                 std::span<uint8_t const> output_is_optional)
{
  auto [input_types, input_accessors]   = reflect_inputs(inputs, input_is_optional);
  auto [filter_types, filter_accessors] = reflect_inputs(filter_inputs, filter_input_is_optional);
  auto [output_types, output_accessors] = reflect_outputs(outputs, output_is_optional);
  auto [state_layout, state_types]      = reflect_state(states, state_is_optional);

  return kernel_reflection{
    .input_types        = input_types,
    .filter_input_types = filter_types,
    .output_types       = output_types,
    .state_types        = state_types,
    .inputs             = input_accessors,
    .filter_inputs      = filter_accessors,
    .outputs            = output_accessors,
    .state              = rtcx::reflect_template("::cudf::jit::reduce::state", state_types),
    .state_layout       = state_layout,
    .block_size         = block_size_from_layout(state_layout),
    .warp_reduce        = use_warp_reduce(state_layout)};
}

kernel_reflection reflect_kernel(std::span<reduce_input const> inputs,
                                 std::span<reduce_input const> filter_inputs,
                                 std::span<data_type const> states,
                                 std::span<data_type const> outputs,
                                 bool optional_values)
{
  std::vector<uint8_t> input_is_optional(inputs.size(), optional_values);
  std::vector<uint8_t> filter_input_is_optional(filter_inputs.size(), optional_values);
  std::vector<uint8_t> output_is_optional(outputs.size(), optional_values);
  std::vector<uint8_t> state_is_optional(states.size(), optional_values);
  return reflect_kernel(inputs,
                        filter_inputs,
                        states,
                        outputs,
                        input_is_optional,
                        filter_input_is_optional,
                        state_is_optional,
                        output_is_optional);
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
                                spec.filter_inputs,
                                spec.outputs,
                                rtcx::reflect(spec.block_size),
                                args...);
}

std::string reflect_reduce_kernel(kernel_reflection const& spec, bool optional_values)
{
  return reflect_kernel_instance(
    scalar_kernel_name, spec, optional_values, rtcx::reflect(spec.warp_reduce), spec.state);
}

std::vector<std::string> reflect_pointer_parameters(std::span<std::string const> types)
{
  std::vector<std::string> result;
  for (auto& type : types) {
    result.push_back(type + "*");
  }
  return result;
}

std::vector<std::string> reflect_callback_parameters(bool indexed,
                                                     std::span<std::string const> first,
                                                     std::span<std::string const> second)
{
  std::vector<std::string> result;
  if (indexed) { result.push_back("cudf::size_type"); }
  result.insert(result.end(), first.begin(), first.end());
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

callback_reflection reflect_callback(std::string_view target,
                                     std::string_view parameters_macro,
                                     std::vector<std::string> parameters,
                                     std::string_view symbol)
{
  return {.target           = target,
          .parameters_macro = parameters_macro,
          .parameters       = std::move(parameters),
          .symbol           = std::string{symbol}};
}

std::vector<callback_reflection> reflect_callbacks(std::span<std::string const> input_types,
                                                   std::span<std::string const> filter_input_types,
                                                   std::span<std::string const> output_types,
                                                   std::span<std::string const> state_types,
                                                   reduce_udf const& udf)
{
  auto outputs        = reflect_pointer_parameters(output_types);
  auto state_pointers = reflect_pointer_parameters(state_types);
  std::string const filter_outputs[]{"bool*"};

  return {
    reflect_callback("cudf_reduce_init", "CUDF_REDUCE_INIT_UDF_PARAMS", state_pointers, udf.init),
    reflect_callback("cudf_reduce_update",
                     "CUDF_REDUCE_UPDATE_UDF_PARAMS",
                     reflect_callback_parameters(udf.indexed_update, state_pointers, input_types),
                     udf.update),
    reflect_callback("cudf_reduce_merge",
                     "CUDF_REDUCE_MERGE_UDF_PARAMS",
                     reflect_callback_parameters(false, state_pointers, state_types),
                     udf.merge),
    reflect_callback("cudf_reduce_finalize",
                     "CUDF_REDUCE_FINALIZE_UDF_PARAMS",
                     reflect_callback_parameters(false, outputs, state_types),
                     udf.finalize),
    reflect_callback(
      "cudf_reduce_filter",
      "CUDF_REDUCE_FILTER_UDF_PARAMS",
      reflect_callback_parameters(udf.indexed_filter, filter_outputs, filter_input_types),
      udf.filter.empty() ? "cudf_all_true" : udf.filter)};
}

std::optional<rtcx::blob> compile_udf_dispatcher(std::span<callback_reflection const> callbacks)
{
  if (std::ranges::all_of(callbacks, [](callback_reflection const& callback) {
        return callback.symbol == callback.target;
      })) {
    return std::nullopt;
  }

  std::string source;
  for (auto& callback : callbacks) {
    // The early return handles only all-default callbacks. For mixed default/custom symbols,
    // skip matching callbacks to avoid duplicate definitions and wrappers that call themselves.
    if (callback.symbol == callback.target) { continue; }
    auto parameter_types = detail::string_join(callback.parameters, {}, ", ");
    auto indices         = std::views::iota(size_t{0}, callback.parameters.size());
    auto declarations    = detail::string_join(
      indices, [&](size_t i) { return std::format("{} arg{}", callback.parameters[i], i); }, ", ");
    auto arguments =
      detail::string_join(indices, [](size_t i) { return std::format("arg{}", i); }, ", ");
    if (callback.symbol != "cudf_all_true") {
      source += std::format(R"CUDA(
extern "C" __device__ cudf::errc {}({});
)CUDA",
                            callback.symbol,
                            parameter_types);
    }
    source += std::format(R"CUDA(
extern "C" __device__ cudf::errc {}({})
{{
  return {}({});
}}
)CUDA",
                          callback.target,
                          declarations,
                          callback.symbol,
                          arguments);
  }

  char const* const names[]{udf_dispatch_header};
  char const* const headers[]{source.c_str()};

  return get_kernel_fragment(udf_dispatcher_path, udf_dispatcher_path, names, headers, "");
}

kernel compile(kernel_reflection const& spec, reduce_udf const& udf, std::string const& instance)
{
  auto callbacks = reflect_callbacks(
    spec.input_types, spec.filter_input_types, spec.output_types, spec.state_types, udf);

  // The kernel depends only on canonical physical types and fixed callback entry points.
  // Symbol forwarding is compiled separately; user-supplied UDF fragments are linked unchanged.

  auto parameter_defines = detail::string_join(
    callbacks,
    [](callback_reflection const& callback) {
      return std::format("#define {} {}",
                         callback.parameters_macro,
                         detail::string_join(callback.parameters, {}, ", "));
    },
    "\n");

  auto source = std::format(
    R"CUDA(
#define CUDF_DISPATCH_REDUCE_STATE {}
{}
#define CUDF_REDUCE_UPDATE_INDEXED {}
#define CUDF_REDUCE_FILTER_INDEXED {}
#define CUDF_KERNEL_INSTANCE {}
  )CUDA",
    spec.state,
    parameter_defines,
    udf.indexed_update,
    udf.indexed_filter,
    instance);

  char const* const names[]{kernel_instance_header};
  char const* const headers[]{source.c_str()};

  std::string const path{scalar_kernel_id};
  auto object   = get_kernel_fragment(path, path, names, headers, instance);
  auto dispatch = compile_udf_dispatcher(callbacks);

  std::vector<rtcx::memory_fragment> fragments{
    {object->view(), rtcx::binary_type::LTO_IR, nullptr}};

  if (dispatch.has_value()) {
    fragments.push_back({dispatch.value()->view(), rtcx::binary_type::LTO_IR, nullptr});
  }

  for (auto& fragment : udf.fragments) {
    fragments.push_back({fragment.data, jit::as_rtcx_binary_type(fragment.type), nullptr});
  }

  return get_lto_linked_kernel(path, {}, fragments);
}

// Own all staging and device storage until output metadata has been read back.
struct scope_t {
  using handle = std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>;

  std::vector<handle> handles;
  cuda::device_buffer<column_device_view_core> input_views;
  cuda::device_buffer<mutable_column_device_view_core> output_views;
  std::vector<std::unique_ptr<column>> outputs;
  cuda::host_buffer<column_device_view_core> host_inputs;
  cuda::host_buffer<mutable_column_device_view_core> host_outputs;
  cuda::host_buffer<bitmask_type> host_masks;
  cuda::host_buffer<string_view> host_strings;
  cuda::host_buffer<errc> host_errors;
  cuda::device_buffer<errc> errors;
  cuda::device_buffer<std::byte> partials;
  cuda::device_buffer<std::byte> result;
  cuda::device_buffer<std::byte> scratch;
  size_t filter_offset;
  fallible is_fallible;
  cuda::stream_ref stream;
  memory_resources mr;

  static std::unique_ptr<column> allocate_output(data_type type,
                                                 bool nullable,
                                                 cuda::stream_ref stream,
                                                 memory_resources mr)
  {
    auto mask = nullable ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED;
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
    auto nulls   = output.null_count();
    auto bytes   = nulls == 0 ? value.size_bytes() : size_type{0};
    auto offsets = detail::make_device_uvector(std::vector<size_type>{0, bytes}, stream, mr);
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

  scope_t(std::span<reduce_input const> inputs,
          std::span<reduce_input const> filter_inputs,
          std::span<data_type const> types,
          size_type num_segments,
          bool nullable,
          fallible is_fallible,
          cuda::stream_ref stream,
          memory_resources mr)
    : input_views(
        stream, mr.get_temporary_mr(), inputs.size() + filter_inputs.size(), cuda::no_init),
      output_views(stream, mr.get_temporary_mr(), num_segments * types.size(), cuda::no_init),
      host_inputs(stream, get_pinned_memory_resource(), input_views.size(), cuda::no_init),
      host_outputs(stream, get_pinned_memory_resource(), output_views.size(), cuda::no_init),
      host_masks(
        stream, get_pinned_memory_resource(), nullable ? output_views.size() : 0, cuda::no_init),
      host_strings(
        stream,
        get_pinned_memory_resource(),
        std::ranges::any_of(types, [](auto type) { return type.id() == type_id::STRING; })
          ? output_views.size()
          : 0,
        cuda::no_init),
      host_errors(stream, get_pinned_memory_resource(), num_segments, cuda::no_init),
      errors(stream, mr.get_temporary_mr(), num_segments, cuda::no_init),
      partials(stream, mr.get_temporary_mr()),
      result(stream, mr.get_temporary_mr()),
      scratch(stream, mr.get_temporary_mr()),
      filter_offset(inputs.size()),
      is_fallible(is_fallible),
      stream(stream),
      mr(mr)
  {
    // Complete stream-ordered allocations before initializing the staging buffers on the CPU.
    detail::sync_stream(stream);

    auto append_inputs = [&](auto values, size_t offset) {
      for (auto i : std::views::iota(size_t{0}, values.size())) {
        auto device =
          column_device_view::create(as_column_view(values[i]), stream, mr.get_temporary_mr());
        host_inputs.data()[offset + i] = *device;
        handles.push_back(std::move(device));
      }
    };
    append_inputs(inputs, 0);
    append_inputs(filter_inputs, filter_offset);
    outputs.reserve(output_views.size());

    for (size_type segment = 0; segment < num_segments; ++segment) {
      for (auto type : types) {
        auto output = allocate_output(type, nullable, stream, mr);
        auto view   = output->mutable_view();
        host_outputs.data()[outputs.size()] =
          mutable_column_device_view::create(type, 1, view.head(), view.null_mask(), 0, nullptr, 0);
        outputs.push_back(std::move(output));
      }
    }

    // Staging belongs to this execution, so transfers can remain asynchronous.
    detail::cuda_memcpy_async<column_device_view_core>(
      input_views,
      host_span<column_device_view_core const>{host_inputs.data(), host_inputs.size(), true},
      stream);
    detail::cuda_memcpy_async<mutable_column_device_view_core>(
      output_views,
      host_span<mutable_column_device_view_core const>{
        host_outputs.data(), host_outputs.size(), true},
      stream);
  }

  // Host-side error checking, null counts and string sizes require completed readbacks.
  bool needs_host_readback() const
  {
    return is_fallible == fallible::YES || std::ranges::any_of(outputs, [](auto const& output) {
             return output->nullable() || output->type().id() == type_id::STRING;
           });
  }

  std::vector<std::unique_ptr<column>> finalize()
  {
    detail::cuda_memcpy_async<errc>(host_span<errc>{host_errors.data(), host_errors.size(), true},
                                    device_span<errc const>{errors.data(), errors.size()},
                                    stream);

    for (auto col : std::views::iota(size_t{0}, outputs.size())) {
      if (outputs[col]->nullable()) {
        detail::cuda_memcpy_async<bitmask_type>(
          host_span<bitmask_type>{host_masks.data() + col, 1, true},
          device_span<bitmask_type const>{outputs[col]->view().null_mask(), 1},
          stream);
      }
      if (outputs[col]->type().id() == type_id::STRING) {
        detail::cuda_memcpy_async<string_view>(
          host_span<string_view>{host_strings.data() + col, 1, true},
          device_span<string_view const>{
            static_cast<string_view const*>(outputs[col]->view().head()), 1},
          stream);
      }
    }

    if (needs_host_readback()) { detail::sync_stream(stream); }

    if (is_fallible == fallible::YES) {
      auto error = *std::max_element(host_errors.data(), host_errors.data() + host_errors.size());
      CUDF_EXPECTS(error == errc::SUCCESS,
                   std::format("Reduction UDF evaluation failed with error `{}`", to_string(error)),
                   evaluation_error,
                   error);
    }

    for (auto col : std::views::iota(size_t{0}, outputs.size())) {
      auto num_nulls = (outputs[col]->nullable() && host_masks.data()[col] == 0) ? 1 : 0;
      outputs[col]->set_null_count(num_nulls);
      if (outputs[col]->type().id() == type_id::STRING) {
        outputs[col] = finalize_string_output(
          *outputs[col], host_strings.data()[col], stream, mr.get_output_mr());
      }
    }

    return std::move(outputs);
  }
};

void launch_reduce(kernel_ref kernel,
                   reduction_launch_config config,
                   size_type num_rows,
                   scope_t& scope)
{
  CUDF_FUNC_RANGE();

  auto stream = scope.stream;
  auto mr     = scope.mr.get_temporary_mr();

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

  scope.partials =
    cuda::device_buffer<std::byte>(stream, mr, grid * config.state_bytes, cuda::no_init);
  scope.result  = cuda::device_buffer<std::byte>(stream, mr, config.state_bytes, cuda::no_init);
  scope.scratch = cuda::device_buffer<std::byte>(
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
                       scope.input_views.data(),
                       scope.input_views.data() + scope.filter_offset,
                       scope.output_views.data(),
                       source,
                       destination,
                       scope.scratch.data(),
                       scope.errors.data(),
                       finalize);
  };

  launch(jit::reduce::phase::REDUCE, num_rows, grid, nullptr, scope.partials.data(), grid == 1);

  if (grid > 1) {
    launch(jit::reduce::phase::MERGE, grid, 1, scope.partials.data(), scope.result.data(), true);
  }
}

}  // namespace

std::vector<std::unique_ptr<column>> reduce(reduce_udf const& udf,
                                            std::span<reduce_input const> inputs,
                                            std::span<data_type const> state_types,
                                            std::span<data_type const> types,
                                            std::span<reduce_input const> filter_inputs,
                                            fallible is_fallible,
                                            cuda::stream_ref stream,
                                            memory_resources mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(filter_inputs.empty() || !udf.filter.empty(),
               "A filter symbol is required when filter inputs are specified",
               std::invalid_argument);

  auto optional_values = udf.is_null_aware == null_aware::YES;
  auto shape           = validate_reduce(inputs, filter_inputs, state_types, types);
  auto spec            = reflect_kernel(inputs, filter_inputs, state_types, types, optional_values);
  auto rows            = shape.rows.value_or(0);
  auto instance        = reflect_reduce_kernel(spec, optional_values);
  auto compiled        = compile(spec, udf, instance);

  scope_t scope(inputs, filter_inputs, types, 1, optional_values, is_fallible, stream, mr);

  launch_reduce(
    compiled.get(), {spec.block_size, spec.state_layout.size, spec.warp_reduce}, rows, scope);

  return scope.finalize();
}

}  // namespace cudf
