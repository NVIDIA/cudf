/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/errc.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/stream>

#include <cudf_fragments.hpp>
#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/parser.hpp>
#include <jit/row_ir.hpp>
#include <jit/span.cuh>
#include <jit/util.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <span>
#include <variant>

namespace cudf {
namespace {

column_view as_column_view(scalar_column_view const& scalar) { return scalar.as_column_view(); }

column_view as_column_view(column_view const& column) { return column; }

struct mutable_fixed_width_column_view {
  mutable_column_view _view;

  auto to_device(cuda::stream_ref stream) const
  {
    return mutable_column_device_view::create(_view, stream);
  }
};

struct fixed_width_column {
  std::unique_ptr<column> _col = nullptr;

  static auto make(data_type type,
                   size_type size,
                   rmm::device_buffer null_mask,
                   size_type null_count,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
  {
    return fixed_width_column{
      make_fixed_width_column(type, size, std::move(null_mask), null_count, stream, mr)};
  }

  auto mutable_view() const { return mutable_fixed_width_column_view{_col->mutable_view()}; }

  void set_null_count(size_type count) { _col->set_null_count(count); }

  bool nullable() const { return _col->nullable(); }

  bitmask_type* null_mask() { return _col->mutable_view().null_mask(); }
};

struct mutable_string_views_column_view {
  void* _data{nullptr};
  size_type _size{0};
  bitmask_type const* _null_mask{nullptr};
  size_type _offset{0};
  size_type _null_count{0};

  auto to_device(cuda::stream_ref stream) const
  {
    using view = mutable_column_device_view;
    return std::unique_ptr<view, std::function<void(view*)>>(
      new view{
        view::create(data_type{type_id::EMPTY}, _size, _data, _null_mask, _offset, nullptr, 0)},
      [](auto* p) { delete p; });
  }
};

struct string_views_column {
  rmm::device_buffer _data;
  size_type _size{0};
  rmm::device_buffer _null_mask{};
  size_type _null_count{0};

  static auto make(size_type size,
                   rmm::device_buffer null_mask,
                   size_type null_count,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
  {
    rmm::device_buffer data{static_cast<size_t>(size) * sizeof(string_view), stream, mr};
    return string_views_column{std::move(data), size, std::move(null_mask), null_count};
  }

  auto mutable_view() const
  {
    return mutable_string_views_column_view{
      const_cast<void*>(_data.data()),
      _size,
      static_cast<bitmask_type*>(const_cast<void*>(_null_mask.data())),
      0,
      _null_count};
  }

  void set_null_count(size_type count) { _null_count = count; }

  bool nullable() const { return !_null_mask.is_empty(); }

  bitmask_type* null_mask() { return static_cast<bitmask_type*>(_null_mask.data()); }
};

struct mutable_strings_column_view {
  mutable_column_view _view;

  auto to_device(cuda::stream_ref stream) const
  {
    return mutable_column_device_view::create(_view, stream);
  }
};

struct mutable_strings_column {
  std::unique_ptr<column> _col = nullptr;

  static auto make(size_type size,
                   rmm::device_buffer chars,
                   std::unique_ptr<column> offsets,
                   rmm::device_buffer null_mask,
                   size_type null_count)
  {
    return mutable_strings_column{make_strings_column(
      size, std::move(offsets), std::move(chars), null_count, std::move(null_mask))};
  }

  auto mutable_view() const { return mutable_strings_column_view{_col->mutable_view()}; }

  void set_null_count(size_type count) { _col->set_null_count(count); }

  bool nullable() const { return _col->nullable(); }

  bitmask_type* null_mask() { return _col->mutable_view().null_mask(); }
};

using input_column_view = transform_input;
using output_column = std::variant<fixed_width_column, string_views_column, mutable_strings_column>;
using handle        = std::variant<
         std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>,
         std::unique_ptr<mutable_column_device_view, std::function<void(mutable_column_device_view*)>>>;

namespace jit_transform {

kernel instantiate(bool is_null_aware,
                   bool has_user_data,
                   std::string const& ins,
                   std::string const& outs,
                   std::vector<std::string> const& ptx_input_types,
                   std::vector<std::string> const& ptx_output_types,
                   std::string const& udf,
                   udf_source_type source_type)
{
  CUDF_FUNC_RANGE();
  auto cuda_source = (source_type == udf_source_type::PTX)
                       ? jit::parse_single_function_ptx(
                           udf,
                           "GENERIC_TRANSFORM_OP",
                           jit::build_ptx_params(ptx_output_types, ptx_input_types, has_user_data))
                       : jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  auto kernel = rtcx::reflect_template("cudf::jit::transform_kernel",
                                       rtcx::reflect(is_null_aware),
                                       rtcx::reflect(has_user_data),
                                       ins,
                                       outs);

  return jit::get_udf_kernel("cudf/cpp/src/transform/jit/kernel.cu", kernel, cuda_source);
}

void launch(cudf::kernel const& kernel,
            size_type row_size,
            bitmask_type const* stencil,
            void* user_data,
            column_device_view_core const* input_cols,
            mutable_column_device_view_core const* output_cols,
            int32_t* max_error,
            cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  void* args[] = {&row_size, &stencil, &user_data, &input_cols, &output_cols, &max_error};
  auto cfg     = kernel.max_occupancy_config(0, 0);
  CUDF_EXPECTS(cfg.block_size % cudf::detail::warp_size == 0,
               "Expected block size to be a multiple of warp size",
               std::runtime_error);
  kernel.launch({cfg.min_grid_size}, {cfg.block_size}, 0, stream, args);
}

std::string get_element_type_name(transform_input_spec const& spec, bool use_physical_type);

struct element_type_name_fn {
  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool use_physical_type) const
    requires(is_fixed_width<T>() || std::same_as<T, cudf::string_view>)
  {
    auto type = data_type{spec.type};
    return type_to_name(use_physical_type ? jit::physical_type_of(type) : type);
  }

  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool use_physical_type) const
    requires(std::same_as<T, cudf::dictionary32>)
  {
    return std::format(
      "cudf::dictionary_element<{}, {}>",
      get_element_type_name(spec.children.at(dictionary_indices_column_index), use_physical_type),
      get_element_type_name(spec.children.at(dictionary_keys_column_index), use_physical_type));
  }

  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool) const
    requires(!is_fixed_width<T>() && !std::same_as<T, cudf::string_view> &&
             !std::same_as<T, cudf::dictionary32>)
  {
    CUDF_FAIL("Unsupported type for JIT compilation: " + type_to_name(data_type{spec.type}));
  }
};

std::string get_element_type_name(transform_input_spec const& spec, bool use_physical_type)
{
  return cudf::type_dispatcher(
    data_type{spec.type}, element_type_name_fn{}, spec, use_physical_type);
}

std::string reflect_input_element(transform_input_spec const& spec, bool use_physical_type)
{
  return get_element_type_name(spec, use_physical_type);
}

std::string reflect_output_element(transform_output_spec const& spec, bool use_physical_type)
{
  if (spec.type == type_id::STRING) {
    return spec.has_string_offsets ? "cuda::std::span<char>" : "cudf::string_view";
  }
  return get_element_type_name(transform_input_spec{.type = spec.type}, use_physical_type);
}

std::string reflect_input_value_type(transform_input_spec const& spec, bool use_physical_type)
{
  if (spec.type == type_id::DICTIONARY32) {
    return reflect_input_value_type(spec.children.at(dictionary_keys_column_index),
                                    use_physical_type);
  }
  return reflect_input_element(spec, use_physical_type);
}

std::string reflect_output_value_type(transform_output_spec const& spec, bool use_physical_type)
{
  return reflect_output_element(spec, use_physical_type);
}

std::string reflect_input_column(transform_input_spec const&)
{
  return "cudf::column_device_view_core";
}

std::string reflect_output_column(transform_output_spec const& spec)
{
  if (spec.type == type_id::STRING) {
    return spec.has_string_offsets ? "cudf::jit::mutable_strings_column_device_view"
                                   : "cudf::jit::mutable_vector_device_view";
  }
  return "cudf::mutable_column_device_view_core";
}

auto reflect(std::variant<udf_source_type, lto_binary_type> source_type,
             std::span<transform_input_spec const> inputs,
             std::span<transform_output_spec const> outputs)
{
  std::vector<std::string> in_types;
  bool use_physical_types = std::holds_alternative<lto_binary_type>(source_type);
  for (size_t i = 0; i < inputs.size(); i++) {
    auto& in       = inputs[i];
    auto column    = reflect_input_column(in);
    auto element   = reflect_input_element(in, use_physical_types);
    bool as_scalar = in.is_scalar;
    auto accessor  = rtcx::reflect_template("cudf::jit::column_accessor",
                                           rtcx::reflect(i),
                                           column,
                                           element,
                                           rtcx::reflect(as_scalar),
                                           rtcx::reflect(0));
    in_types.push_back(accessor);
  }

  std::vector<std::string> out_types;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto& out      = outputs[i];
    auto column    = reflect_output_column(out);
    auto element   = reflect_output_element(out, use_physical_types);
    bool as_scalar = false;  // never scalar
    auto accessor  = rtcx::reflect_template("cudf::jit::column_accessor",
                                           rtcx::reflect(i),
                                           column,
                                           element,
                                           rtcx::reflect(as_scalar),
                                           rtcx::reflect(0));

    out_types.push_back(accessor);
  }

  auto ins  = rtcx::reflect_template("cudf::jit::type_list", in_types);
  auto outs = rtcx::reflect_template("cudf::jit::type_list", out_types);

  std::vector<std::string> ptx_in_types;
  std::vector<std::string> ptx_out_types;
  if (std::holds_alternative<udf_source_type>(source_type) &&
      std::get<udf_source_type>(source_type) == udf_source_type::PTX) {
    for (auto& in : inputs) {
      ptx_in_types.push_back(reflect_input_value_type(in, use_physical_types));
    }

    for (auto& out : outputs) {
      ptx_out_types.push_back(reflect_output_value_type(out, use_physical_types));
    }
  }

  return std::make_tuple(ins, outs, ptx_in_types, ptx_out_types);
}

transform_input_spec make_input_spec(column_view const& column, bool is_scalar)
{
  transform_input_spec result{.type = column.type().id(), .is_scalar = is_scalar};
  if (is_dictionary(column.type())) {
    for (size_type i = 0; i < column.num_children(); ++i) {
      result.children.push_back(make_input_spec(column.child(i), false));
    }
  } else if (column.type().id() == type_id::STRING &&
             column.num_children() > strings_column_view::offsets_column_index) {
    result.children.push_back(
      make_input_spec(column.child(strings_column_view::offsets_column_index), false));
  }
  return result;
}

transform_input_spec make_input_spec(input_column_view const& input)
{
  return std::visit(
    [](auto& value) {
      return make_input_spec(as_column_view(value),
                             std::is_same_v<std::decay_t<decltype(value)>, scalar_column_view>);
    },
    input);
}

std::vector<transform_input_spec> make_input_specs(std::span<input_column_view const> inputs)
{
  std::vector<transform_input_spec> result;
  for (auto& input : inputs) {
    result.push_back(make_input_spec(input));
  }
  return result;
}

transform_output_spec make_output_spec(fixed_width_column const& output)
{
  return {.type = output._col->type().id()};
}

transform_output_spec make_output_spec(string_views_column const&)
{
  return {.type = type_id::STRING};
}

transform_output_spec make_output_spec(mutable_strings_column const& output)
{
  auto offsets = output._col->view().child(strings_column_view::offsets_column_index);
  return {.type               = type_id::STRING,
          .has_string_offsets = true,
          .children           = {{.type = offsets.type().id()}}};
}

std::vector<transform_output_spec> make_output_specs(std::span<output_column const> outputs)
{
  std::vector<transform_output_spec> result;
  for (auto& output : outputs) {
    result.push_back(std::visit([](auto& value) { return make_output_spec(value); }, output));
  }
  return result;
}

std::vector<transform_output_spec> make_output_specs(
  std::span<transform_output const> outputs,
  std::span<std::unique_ptr<column> const> string_offsets)
{
  CUDF_EXPECTS(string_offsets.empty() || string_offsets.size() == outputs.size(),
               "Number of string offsets must be empty or match the number of outputs",
               std::invalid_argument);
  std::vector<transform_output_spec> result;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto has_string_offsets = !string_offsets.empty() && string_offsets[i] != nullptr;
    transform_output_spec spec{.type               = outputs[i].type.id(),
                               .nullability        = outputs[i].nullability,
                               .has_string_offsets = has_string_offsets};
    if (has_string_offsets) { spec.children.push_back({.type = string_offsets[i]->type().id()}); }
    result.push_back(std::move(spec));
  }
  return result;
}

auto reflect(std::variant<udf_source_type, lto_binary_type> source_type,
             std::span<input_column_view const> inputs,
             std::span<output_column const> outputs)
{
  auto input_specs  = make_input_specs(inputs);
  auto output_specs = make_output_specs(outputs);
  return reflect(source_type, input_specs, output_specs);
}

std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input_spec const> inputs,
                                  std::span<transform_output_spec const> outputs,
                                  bool use_physical_types)
{
  std::vector<std::string> in_types;

  for (size_t i = 0; i < inputs.size(); i++) {
    auto element = reflect_input_element(inputs[i], use_physical_types);
    in_types.push_back(is_null_aware ? std::format("cuda::std::optional<{}>", element) : element);
  }

  std::vector<std::string> out_types;

  for (size_t i = 0; i < outputs.size(); i++) {
    auto element = reflect_output_element(outputs[i], use_physical_types);
    out_types.push_back(is_null_aware ? std::format("cuda::std::optional<{}> *", element)
                                      : std::format("{} *", element));
  }

  std::vector<std::string> params;
  if (has_user_data) { params.push_back("void*"); }
  params.insert(params.end(), out_types.begin(), out_types.end());
  params.insert(params.end(), in_types.begin(), in_types.end());

  auto joined =
    params.empty()
      ? ""
      : std::accumulate(std::next(params.begin()), params.end(), params[0], [](auto a, auto b) {
          return std::format("{}, {}", a, b);
        });

  return std::format("int({})", joined);
}
std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<input_column_view const> inputs,
                                  std::span<output_column const> outputs,
                                  bool use_physical_types)
{
  auto input_specs  = make_input_specs(inputs);
  auto output_specs = make_output_specs(outputs);
  return reflect_udf_signature(
    is_null_aware, has_user_data, input_specs, output_specs, use_physical_types);
}

std::tuple<rtcx::blob, lto_binary_type, std::string> instantiate_fragment(
  bool is_null_aware,
  bool has_user_data,
  std::string const& ins,
  std::string const& outs,
  std::span<input_column_view const> inputs,
  std::span<output_column const> outputs)
{
  CUDF_FUNC_RANGE();
  // substitutes the `CUDF_KERNEL_INSTANCE` macro
  auto kernel = rtcx::reflect_template("cudf::jit::transform_kernel",
                                       rtcx::reflect(is_null_aware),
                                       rtcx::reflect(has_user_data),
                                       ins,
                                       outs);

  // substitutes the `CUDF_UDF_TYPE` macro
  auto signature = reflect_udf_signature(
    is_null_aware, has_user_data, inputs, outputs, /*use_physical_types=*/true);

  return {jit::get_udf_kernel_fragment("cudf/cpp/src/transform/jit/kernel.cu", kernel, signature),
          lto_binary_type::LTO_IR,
          kernel};
}

auto to_args(std::span<input_column_view const> inputs,
             std::span<output_column const> outputs,
             cuda::stream_ref stream,
             rmm::device_async_resource_ref mr)
{
  std::vector<handle> handles;
  auto h_args =
    detail::host_vector<detail::column_device_view_base>({get_pinned_memory_resource(), stream});
  h_args.reserve(inputs.size() + outputs.size());

  for (auto& in : inputs) {
    if (auto* col = std::get_if<column_view>(&in)) {
      auto handle = column_device_view::create(*col, stream);
      h_args.push_back(*handle);
      handles.emplace_back(std::move(handle));
    } else {
      auto& scalar = std::get<scalar_column_view>(in);
      auto handle  = column_device_view::create(scalar.as_column_view(), stream);
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

  auto d_args = detail::make_device_uvector(h_args, stream, mr);

  return std::make_tuple(std::move(d_args), std::move(handles));
}

kernel get_kernel(bool is_null_aware,
                  bool has_user_data,
                  std::span<input_column_view const> inputs,
                  std::span<output_column const> outputs,
                  std::string const& udf,
                  udf_source_type source_type)
{
  auto [in_types, out_types, ptx_in_types, ptx_out_types] = reflect(source_type, inputs, outputs);
  return instantiate(is_null_aware,
                     has_user_data,
                     in_types,
                     out_types,
                     ptx_in_types,
                     ptx_out_types,
                     udf,
                     source_type);
}

kernel get_kernel(bool is_null_aware,
                  bool has_user_data,
                  std::span<transform_input_spec const> inputs,
                  std::span<transform_output_spec const> outputs,
                  std::string const& udf,
                  udf_source_type source_type)
{
  auto [in_types, out_types, ptx_in_types, ptx_out_types] = reflect(source_type, inputs, outputs);
  return instantiate(is_null_aware,
                     has_user_data,
                     in_types,
                     out_types,
                     ptx_in_types,
                     ptx_out_types,
                     udf,
                     source_type);
}

void run(bool is_null_aware,
         bool has_user_data,
         size_type row_size,
         bitmask_type const* d_stencil,
         void* user_data,
         std::span<input_column_view const> inputs,
         std::span<output_column const> outputs,
         int32_t* d_max_error,
         std::string const& udf,
         udf_source_type source_type,
         cuda::stream_ref stream,
         rmm::device_async_resource_ref mr)
{
  auto kernel = get_kernel(is_null_aware, has_user_data, inputs, outputs, udf, source_type);
  auto [cols, handles] = to_args(inputs, outputs, stream, mr);
  auto* input_cols     = reinterpret_cast<column_device_view_core const*>(cols.data());
  auto* output_cols =
    reinterpret_cast<mutable_column_device_view_core const*>(input_cols + inputs.size());
  return launch(
    kernel, row_size, d_stencil, user_data, input_cols, output_cols, d_max_error, stream);
}

void run(kernel const& kernel,
         size_type row_size,
         bitmask_type const* d_stencil,
         void* user_data,
         std::span<input_column_view const> inputs,
         std::span<output_column const> outputs,
         int32_t* d_max_error,
         cuda::stream_ref stream,
         rmm::device_async_resource_ref mr)
{
  auto [cols, handles] = to_args(inputs, outputs, stream, mr);
  auto* input_cols     = reinterpret_cast<column_device_view_core const*>(cols.data());
  auto* output_cols =
    reinterpret_cast<mutable_column_device_view_core const*>(input_cols + inputs.size());
  return launch(
    kernel, row_size, d_stencil, user_data, input_cols, output_cols, d_max_error, stream);
}

rtcx::binary_type as_rtcx_binary_type(lto_binary_type type)
{
  switch (type) {
    case lto_binary_type::LTO_IR: return rtcx::binary_type::LTO_IR;
    case lto_binary_type::FATBIN: return rtcx::binary_type::FATBIN;
    default:
      CUDF_FAIL(
        std::format("Unrecognized LTO binary type {} for LTO transform", static_cast<int>(type)),
        std::invalid_argument);
  }
}

void run_lto(std::optional<std::tuple<std::span<uint8_t const>, lto_binary_type, char const*>>
               precompiled_kernel_fragment,
             bool is_null_aware,
             bool has_user_data,
             size_type row_size,
             bitmask_type const* d_stencil,
             void* user_data,
             std::span<input_column_view const> inputs,
             std::span<output_column const> outputs,
             int32_t* d_max_error,
             std::span<uint8_t const> udf_binary,
             lto_binary_type source_type,
             cuda::stream_ref stream,
             rmm::device_async_resource_ref mr)
{
  auto [in_types, out_types, ptx_in_types, ptx_out_types] = reflect(source_type, inputs, outputs);

  std::span<uint8_t const> kernel_fragment;
  lto_binary_type kernel_fragment_binary_type = lto_binary_type::FATBIN;
  rtcx::blob fragment_blob                    = nullptr;
  std::string kernel_fragment_id;

  if (precompiled_kernel_fragment.has_value()) {
    std::tie(kernel_fragment, kernel_fragment_binary_type, kernel_fragment_id) =
      *precompiled_kernel_fragment;
  } else {
    std::tie(fragment_blob, kernel_fragment_binary_type, kernel_fragment_id) =
      instantiate_fragment(is_null_aware, has_user_data, in_types, out_types, inputs, outputs);
    kernel_fragment = fragment_blob->view();
  }

  rtcx::memory_fragment memory_fragments[] = {
    {
      .data = kernel_fragment,
      .type = as_rtcx_binary_type(kernel_fragment_binary_type),
      .name = kernel_fragment_id.c_str(),
    },
    {
      .data = udf_binary,
      .type = as_rtcx_binary_type(source_type),
      .name = nullptr  // nullptr = unnamed fragment: the binary will be used to hash the UDF
    }};

  auto kernel = get_lto_linked_kernel("cudf/cpp/src/transform/jit/kernel.cu", {}, memory_fragments);

  auto [cols, handles] = to_args(inputs, outputs, stream, mr);
  auto* input_cols     = reinterpret_cast<column_device_view_core const*>(cols.data());
  auto* output_cols =
    reinterpret_cast<mutable_column_device_view_core const*>(input_cols + inputs.size());
  return launch(
    kernel, row_size, d_stencil, user_data, input_cols, output_cols, d_max_error, stream);
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
    return std::visit([](auto& c) { return c.nullable(); }, in);
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

void perform_checks(std::variant<udf_source_type, lto_binary_type> source_type,
                    null_aware is_null_aware,
                    std::optional<size_type> in_row_size,
                    std::span<transform_input const> inputs,
                    std::span<transform_output const> outputs,
                    std::span<std::unique_ptr<column> const> string_offsets)
{
  if (auto* udf_source = std::get_if<udf_source_type>(&source_type);
      udf_source != nullptr && *udf_source == udf_source_type::PTX) {
    static constexpr auto is_input_value_supported = [](auto const& c) {
      return is_integral(c.type()) || is_floating_point(c.type());
    };
    static constexpr auto is_supported_input_type = [](auto const& c) {
      auto col = std::visit([](auto& c) { return as_column_view(c); }, c);
      return is_input_value_supported(col) ||
             (is_dictionary(col.type()) &&
              is_input_value_supported(col.child(dictionary_keys_column_index)));
    };
    CUDF_EXPECTS(
      std::none_of(
        inputs.begin(), inputs.end(), [](auto const& in) { return !is_supported_input_type(in); }),
      "Transforms with PTX UDFs only support integer, floating-point, and boolean",
      std::invalid_argument);
    CUDF_EXPECTS(std::none_of(outputs.begin(),
                              outputs.end(),
                              [](auto& out) {
                                return !is_integral(out.type) && !is_floating_point(out.type);
                              }),
                 "Transforms with PTX UDFs only support integer, floating-point, and boolean types",
                 std::invalid_argument);
    CUDF_EXPECTS(is_null_aware == null_aware::NO,
                 "PTX UDFs do not support null-aware transformations",
                 std::invalid_argument);
  }

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
  static constexpr auto is_supported_input_type = [&](auto const& c) {
    auto col = std::visit([](auto const& c) { return as_column_view(c); }, c);
    return is_input_value_supported(col) ||
           (is_dictionary(col.type()) &&
            is_input_value_supported(col.child(dictionary_keys_column_index)));
  };
  CUDF_EXPECTS(
    std::none_of(
      inputs.begin(), inputs.end(), [&](auto const& in) { return !is_supported_input_type(in); }),
    "Transforms only support input of fixed-width, string, or dictionary types",
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
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);
  CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(
    d_temp_storage.data(), temp_storage_bytes, srcs, dsts, src_sizes, size, stream.get()));

  return chars;
}

std::unique_ptr<column> make_strings_column(device_span<string_view const> strings,
                                            rmm::device_buffer null_mask,
                                            size_type null_count,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto size = static_cast<size_type>(strings.size());
  if (size == 0) return make_empty_column(type_id::STRING);

  auto stencil = static_cast<bitmask_type const*>(null_mask.data());

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
    device_span<string_view const>{static_cast<string_view const*>(c._data.data()),
                                   static_cast<size_t>(c._size)},
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

std::unique_ptr<table> execute_transform(std::string const& udf,
                                         udf_source_type source_type,
                                         null_aware is_null_aware,
                                         std::optional<size_type> in_row_size,
                                         std::optional<void*> user_data,
                                         std::span<transform_input const> inputs,
                                         std::span<transform_output const> outputs,
                                         std::vector<std::unique_ptr<column>> string_offsets,
                                         kernel const* compiled_kernel,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto row_size = in_row_size.has_value() ? *in_row_size : jit::get_projection_size(inputs);
  auto output_may_be_nullable    = get_null_transformation(is_null_aware, inputs, outputs);
  auto [output_columns, stencil] = make_outputs(is_null_aware,
                                                row_size,
                                                inputs,
                                                outputs,
                                                output_may_be_nullable,
                                                std::move(string_offsets),
                                                stream,
                                                mr);

  auto stencil_arg       = stencil.has_value() ? stencil->first : nullptr;
  auto stencil_has_nulls = stencil.has_value() ? (stencil->second > 0) : false;

  cudf::detail::device_scalar<int32_t> d_max_error(
    static_cast<int32_t>(errc::SUCCESS), stream, cudf::get_current_device_resource_ref());

  if (compiled_kernel == nullptr) {
    jit_transform::run(is_null_aware == null_aware::YES,
                       user_data.has_value(),
                       row_size,
                       stencil_has_nulls ? stencil_arg : nullptr,
                       user_data.value_or(nullptr),
                       inputs,
                       output_columns,
                       d_max_error.data(),
                       udf,
                       source_type,
                       stream,
                       mr);
  } else {
    jit_transform::run(*compiled_kernel,
                       row_size,
                       stencil_has_nulls ? stencil_arg : nullptr,
                       user_data.value_or(nullptr),
                       inputs,
                       output_columns,
                       d_max_error.data(),
                       stream,
                       mr);
  }

  auto error = static_cast<errc>(d_max_error.value(stream));

  switch (error) {
    case errc::SUCCESS: break;
    default:
      throw evaluation_error(
        error, std::format("Transform UDF evaluation failed with error `{}`", to_string(error)));
  }

  auto finalized = finalize_outputs(is_null_aware, row_size, std::move(output_columns), stream, mr);
  return std::make_unique<table>(std::move(finalized));
}

}  // namespace

std::unique_ptr<table> transform(std::string const& udf,
                                 udf_source_type source_type,
                                 null_aware is_null_aware,
                                 std::optional<void*> user_data,
                                 std::span<transform_input const> inputs,
                                 std::span<transform_output const> outputs,
                                 std::vector<std::unique_ptr<column>>&& string_offsets,
                                 std::optional<size_type> row_size,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  perform_checks(source_type, is_null_aware, row_size, inputs, outputs, string_offsets);
  return execute_transform(udf,
                           source_type,
                           is_null_aware,
                           row_size,
                           user_data,
                           inputs,
                           outputs,
                           std::move(string_offsets),
                           nullptr,
                           stream,
                           mr);
}

std::unique_ptr<table> multi_transform(std::string const& udf,
                                       udf_source_type source_type,
                                       null_aware is_null_aware,
                                       std::optional<void*> user_data,
                                       std::span<transform_input const> inputs,
                                       std::span<transform_output const> outputs,
                                       std::vector<std::unique_ptr<column>>&& string_offsets,
                                       std::optional<size_type> row_size,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  return transform(udf,
                   source_type,
                   is_null_aware,
                   user_data,
                   inputs,
                   outputs,
                   std::move(string_offsets),
                   row_size,
                   stream,
                   mr);
}

std::unique_ptr<column> transform_extended(std::span<transform_input const> inputs,
                                           std::string const& udf,
                                           data_type output_type,
                                           udf_source_type source_type,
                                           std::optional<void*> user_data,
                                           null_aware is_null_aware,
                                           std::optional<size_type> row_size,
                                           output_nullability null_policy,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  transform_output outputs[] = {{.type = output_type, .nullability = null_policy}};

  auto result = transform(
    udf, source_type, is_null_aware, user_data, inputs, outputs, {}, row_size, stream, mr);
  auto columns = result->release();
  return std::move(columns.front());
}

std::unique_ptr<column> compute_column_jit(table_view const& table,
                                           ast::expression const& expr,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  std::array<std::reference_wrapper<ast::expression const>, 1> expressions{expr};
  auto args = detail::row_ir::ast_converter::compute_table(
    detail::row_ir::target::CUDA, expressions, table, {}, "compute_operation", stream, mr);
  auto result = transform(args.udf,
                          args.source_type,
                          args.is_null_aware,
                          args.user_data,
                          args.inputs,
                          args.outputs,
                          std::move(args.string_offsets),
                          args.row_size,
                          stream,
                          mr);
  auto cols   = result->release();
  return std::move(cols[0]);
}

std::unique_ptr<table> compute_table_jit(
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto args = detail::row_ir::ast_converter::compute_table(
    detail::row_ir::target::CUDA, expressions, table, {}, "compute_operation", stream, mr);
  return transform(args.udf,
                   args.source_type,
                   args.is_null_aware,
                   args.user_data,
                   args.inputs,
                   args.outputs,
                   std::move(args.string_offsets),
                   args.row_size,
                   stream,
                   mr);
}

// if we have a matching pre-compiled kernel fragment for the given transform configuration, return
// it to use for LTO linking instead of compiling a new one
std::optional<std::tuple<std::span<uint8_t const>, lto_binary_type, char const*>>
dispatch_lto_kernel_fragment(bool is_null_aware,
                             bool has_user_data,
                             std::span<transform_input const> inputs,
                             std::span<output_column const> outputs)
{
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
  auto [in_types, out_types, ptx_in_types, ptx_out_types] =
    jit_transform::reflect(lto_binary_type::FATBIN, inputs, outputs);
  auto target = strip_whitespace(rtcx::reflect_template("cudf::jit::transform_kernel",
                                                        rtcx::reflect(is_null_aware),
                                                        rtcx::reflect(has_user_data),
                                                        in_types,
                                                        out_types));

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

std::unique_ptr<table> transform_lto(std::span<uint8_t const> udf,
                                     lto_binary_type binary_type,
                                     null_aware is_null_aware,
                                     std::optional<void*> user_data,
                                     std::span<transform_input const> inputs,
                                     std::span<transform_output const> outputs,
                                     std::vector<std::unique_ptr<column>>&& string_offsets,
                                     std::optional<size_type> in_row_size,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  perform_checks(binary_type, is_null_aware, in_row_size, inputs, outputs, string_offsets);
  auto row_size = in_row_size.has_value() ? *in_row_size : jit::get_projection_size(inputs);
  auto output_may_be_nullable = get_null_transformation(is_null_aware, inputs, outputs);

  auto [output_columns, stencil] = make_outputs(is_null_aware,
                                                row_size,
                                                inputs,
                                                outputs,
                                                output_may_be_nullable,
                                                std::move(string_offsets),
                                                stream,
                                                mr);
  auto stencil_arg               = stencil.has_value() ? stencil->first : nullptr;
  auto stencil_has_nulls         = stencil.has_value() ? (stencil->second > 0) : false;

  auto precompiled_kernel_fragment = dispatch_lto_kernel_fragment(
    is_null_aware == null_aware::YES, user_data.has_value(), inputs, output_columns);

  cudf::detail::device_scalar<int32_t> d_max_error(
    static_cast<int32_t>(errc::SUCCESS), stream, cudf::get_current_device_resource_ref());

  jit_transform::run_lto(precompiled_kernel_fragment,
                         is_null_aware == null_aware::YES,
                         user_data.has_value(),
                         row_size,
                         stencil_has_nulls ? stencil_arg : nullptr,
                         user_data.value_or(nullptr),
                         inputs,
                         output_columns,
                         d_max_error.data(),
                         udf,
                         binary_type,
                         stream,
                         mr);

  auto error = static_cast<errc>(d_max_error.value(stream));
  switch (error) {
    case errc::SUCCESS: break;
    default:
      throw evaluation_error(
        error, std::format("Transform UDF evaluation failed with error `{}`", to_string(error)));
  }

  auto finalized = finalize_outputs(is_null_aware, row_size, std::move(output_columns), stream, mr);
  return std::make_unique<table>(std::move(finalized));
}

struct transform_program::impl {
  void validate(udf_source_type source_type,
                std::span<transform_input const> actual_inputs,
                std::span<transform_output const> actual_outputs,
                std::span<std::unique_ptr<column> const> string_offsets) const
  {
    auto actual_input_specs  = jit_transform::make_input_specs(actual_inputs);
    auto actual_output_specs = jit_transform::make_output_specs(actual_outputs, string_offsets);
    auto actual_reflection =
      jit_transform::reflect(source_type, actual_input_specs, actual_output_specs);
    CUDF_EXPECTS(reflection_ == actual_reflection,
                 "Transform program specifications do not match the provided inputs and outputs",
                 std::invalid_argument);
  }

  impl(std::string const& udf,
       udf_source_type source_type,
       null_aware is_null_aware,
       std::optional<void*> user_data,
       std::vector<transform_input_spec> inputs,
       std::vector<transform_output_spec> outputs)
    : reflection_{jit_transform::reflect(source_type, inputs, outputs)},
      source_type_{source_type},
      is_null_aware_{is_null_aware},
      user_data_{user_data},
      kernel_{jit_transform::get_kernel(is_null_aware_ == null_aware::YES,
                                        user_data_.has_value(),
                                        inputs,
                                        outputs,
                                        udf,
                                        source_type_)}
  {
  }

  std::tuple<std::string, std::string, std::vector<std::string>, std::vector<std::string>>
    reflection_;
  udf_source_type source_type_;
  null_aware is_null_aware_;
  std::optional<void*> user_data_;
  kernel kernel_;
  std::vector<std::unique_ptr<column>> ast_scalar_columns_;
  std::optional<std::vector<std::optional<int32_t>>> ast_input_column_indices_;
  std::vector<data_type> ast_input_types_;
  std::vector<bool> ast_input_nullable_;
  std::vector<transform_output> ast_outputs_;
};

transform_program::transform_program(std::string const& udf,
                                     udf_source_type source_type,
                                     null_aware is_null_aware,
                                     std::optional<void*> user_data,
                                     std::span<transform_input const> inputs,
                                     std::span<transform_output const> outputs,
                                     std::span<std::unique_ptr<column> const> string_offsets)
  : transform_program(udf,
                      source_type,
                      is_null_aware,
                      user_data,
                      jit_transform::make_input_specs(inputs),
                      jit_transform::make_output_specs(outputs, string_offsets))
{
}

transform_program::transform_program(std::string const& udf,
                                     udf_source_type source_type,
                                     null_aware is_null_aware,
                                     std::optional<void*> user_data,
                                     std::span<transform_input_spec const> inputs,
                                     std::span<transform_output_spec const> outputs)
{
  CUDF_FUNC_RANGE();
  impl_ =
    std::make_unique<impl>(udf,
                           source_type,
                           is_null_aware,
                           user_data,
                           std::vector<transform_input_spec>{inputs.begin(), inputs.end()},
                           std::vector<transform_output_spec>{outputs.begin(), outputs.end()});
}

transform_program::transform_program(
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto args = detail::row_ir::ast_converter::compute_table(
    detail::row_ir::target::CUDA, expressions, table, {}, "compute_operation", stream, mr);
  impl_ =
    std::make_unique<impl>(args.udf,
                           args.source_type,
                           args.is_null_aware,
                           args.user_data,
                           jit_transform::make_input_specs(args.inputs),
                           jit_transform::make_output_specs(args.outputs, args.string_offsets));
  for (auto& input : args.inputs) {
    impl_->ast_input_types_.push_back(std::visit([](auto& view) { return view.type(); }, input));
    impl_->ast_input_nullable_.push_back(
      std::visit([](auto& view) { return view.nullable(); }, input));
  }
  impl_->ast_scalar_columns_       = std::move(args.scalar_columns);
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
  impl_->validate(impl_->source_type_, inputs, outputs, string_offsets);
  perform_checks(
    impl_->source_type_, impl_->is_null_aware_, row_size, inputs, outputs, string_offsets);
  return execute_transform({},
                           impl_->source_type_,
                           impl_->is_null_aware_,
                           row_size,
                           impl_->user_data_,
                           inputs,
                           outputs,
                           std::move(string_offsets),
                           &impl_->kernel_,
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
      inputs.emplace_back(scalar_column_view{impl_->ast_scalar_columns_[scalar_index++]->view()});
    }
  }
  return run(inputs, impl_->ast_outputs_, {}, table.num_rows(), stream, mr);
}

}  // namespace cudf
