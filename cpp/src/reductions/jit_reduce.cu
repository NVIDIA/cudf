/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/errc.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction/jit.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <jit/cache.hpp>
#include <jit/row_ir.hpp>
#include <jit/udf_reflection.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <string_view>

namespace cudf {
namespace {

constexpr char reduce_kernel_source[] = "cudf/cpp/src/reductions/jit/kernel.cu";

column_view const& input_view(column_view const& col) { return col; }
column_view const& input_view(scalar_column_view const& col) { return col.as_column_view(); }

reduce_input_spec make_spec(column_view const& col, bool scalar = false)
{
  reduce_input_spec spec{col.type(), scalar};
  if (is_dictionary(col.type())) {
    for (size_type i = 0; i < col.num_children(); ++i) {
      spec.children.push_back(make_spec(col.child(i)));
    }
  } else if (col.type().id() == type_id::STRING && col.num_children() > 0) {
    spec.children.push_back(make_spec(col.child(strings_column_view::offsets_column_index)));
  }
  return spec;
}

std::vector<reduce_input_spec> make_specs(std::span<reduce_input const> inputs)
{
  std::vector<reduce_input_spec> result;
  for (auto const& input : inputs) {
    result.push_back(std::visit(
      [](auto const& value) {
        return make_spec(input_view(value),
                         std::is_same_v<std::decay_t<decltype(value)>, scalar_column_view>);
      },
      input));
  }
  return result;
}

bool same_schema(reduce_input_spec const& actual, reduce_input_spec const& expected)
{
  if (actual.type != expected.type || actual.is_scalar != expected.is_scalar) { return false; }
  // String views decode their offset width at runtime, and empty strings may omit offsets.
  if (actual.type.id() == type_id::STRING) { return true; }
  return actual.children.size() == expected.children.size() &&
         std::equal(actual.children.begin(),
                    actual.children.end(),
                    expected.children.begin(),
                    [](auto const& a, auto const& b) { return same_schema(a, b); });
}

transform_input_spec reflection_spec(reduce_input_spec const& input)
{
  transform_input_spec spec{.type = input.type.id(), .is_scalar = input.is_scalar};
  for (auto const& child : input.children) {
    spec.children.push_back(reflection_spec(child));
  }
  return spec;
}

std::string join(std::vector<std::string> const& strings)
{
  std::string result;
  for (auto const& s : strings) {
    if (!result.empty()) { result += ", "; }
    result += s;
  }
  return result;
}

struct field_layout {
  template <typename T>
  std::pair<size_t, size_t> operator()() const
    requires(is_fixed_width<T>())
  {
    return {sizeof(T), alignof(T)};
  }

  template <typename T>
  std::pair<size_t, size_t> operator()() const
    requires(!is_fixed_width<T>())
  {
    CUDF_FAIL("Reduction state and outputs must be fixed-width", std::invalid_argument);
  }
};

size_t state_size(std::span<data_type const> types)
{
  size_t bytes     = 0;
  size_t alignment = alignof(int32_t);
  for (auto type : types) {
    auto [size, align] = type_dispatcher(type, field_layout{});
    bytes              = ((bytes + align - 1) / align) * align + size;
    alignment          = std::max(alignment, align);
  }
  bytes = ((bytes + alignof(int32_t) - 1) / alignof(int32_t)) * alignof(int32_t) + sizeof(int32_t);
  return ((bytes + alignment - 1) / alignment) * alignment;
}

void validate_outputs(std::span<reduce_output const> outputs)
{
  CUDF_EXPECTS(!outputs.empty(), "Reduction must have at least one output", std::invalid_argument);
  CUDF_EXPECTS(outputs.size() < static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "Too many reduction outputs",
               std::invalid_argument);
  for (auto const& out : outputs) {
    CUDF_EXPECTS(
      is_fixed_width(out.type), "Reduction outputs must be fixed-width", std::invalid_argument);
    CUDF_EXPECTS(
      out.size >= 0, "Reduction output lengths must be nonnegative", std::invalid_argument);
  }
}

// Use the actual logical types on both sides of the LTO boundary. Unlike transform's precompiled
// fragments, these fragments are specialized to the complete schema, so physical aliases are not
// necessary. This also keeps Row IR arithmetic in its logical type without numeric conversions.
std::string generate_header(std::string const& source,
                            bool lto,
                            std::span<reduce_input_spec const> inputs,
                            std::span<data_type const> states,
                            std::span<reduce_output const> outputs,
                            reduce_options options,
                            size_t state_bytes,
                            detail::row_ir::transform_args const* ast_args)
{
  std::vector<std::string> state_types;
  std::vector<std::string> state_ptr_types;
  std::vector<std::string> state_ptrs;
  std::vector<std::string> state_values;
  std::vector<std::string> state_shuffles;
  std::string code = "namespace cudf::jit {\nstruct reduce_state {\n";
  for (size_t i = 0; i < states.size(); ++i) {
    auto type = type_to_name(states[i]);
    state_types.push_back(type);
    state_ptr_types.push_back(type + "*");
    state_ptrs.push_back(std::format("&field_{}", i));
    state_values.push_back(std::format("field_{}", i));
    state_shuffles.push_back(std::format("reduction_shuffle_down(field_{}, offset)", i));
    code += std::format("{} field_{}{{}};\n", type, i);
  }
  code += std::format(
    "int32_t error{{}};\n"
    "__device__ auto pointers() {{ return cuda::std::tuple{{{}}}; }}\n"
    "__device__ auto values() const {{ return cuda::std::tuple{{{}}}; }}\n"
    "__device__ reduce_state shuffle_down(int offset) const {{\n"
    "return {{{}, reduction_shuffle_down(error, offset)}}; }}\n"
    "}};\nstatic_assert(sizeof(reduce_state) == {});\n"
    "static_assert(cuda::std::is_trivially_copyable_v<reduce_state>);\n"
    "}}\n",
    join(state_ptrs),
    join(state_values),
    join(state_shuffles),
    state_bytes);

  std::vector<std::string> input_types;
  if (ast_args != nullptr) {
    code += "#define CUDF_REDUCE_HAS_MAP 1\n";
    code += ast_args->udf;
    for (auto const& out : ast_args->outputs) {
      input_types.push_back(type_to_name(out.type));
    }
    code += std::format("namespace cudf::jit {{ using reduce_mapped_types = {}; }}\n",
                        rtcx::reflect_template("cudf::jit::type_list", input_types));
  } else {
    code += "#define CUDF_REDUCE_HAS_MAP 0\n";
    for (auto const& in : inputs) {
      input_types.push_back(jit::reflect_input_value_type(reflection_spec(in), false));
    }
  }

  if (!lto) { return code + source; }

  auto declaration = [&](std::string_view name, std::vector<std::string> params) {
    if (options.user_data.has_value()) { params.insert(params.begin(), "void*"); }
    code += std::format("extern \"C\" __device__ int {}({});\n", name, join(params));
  };
  declaration("cudf_reduce_init", state_ptr_types);
  auto params = state_ptr_types;
  params.insert(params.begin(), "cudf::size_type");
  for (auto type : input_types) {
    params.push_back(options.is_null_aware == null_aware::YES
                       ? std::format("cuda::std::optional<{}>", type)
                       : type);
  }
  declaration("cudf_reduce_update", params);
  params = state_ptr_types;
  params.insert(params.end(), state_types.begin(), state_types.end());
  declaration("cudf_reduce_merge", params);
  params = {"cudf::size_type", "cuda::std::span<cudf::size_type const>"};
  for (auto const& out : outputs) {
    params.push_back(std::format("cuda::std::optional<{}>*", type_to_name(out.type)));
  }
  params.insert(params.end(), state_types.begin(), state_types.end());
  declaration("cudf_reduce_finalize", params);
  return code;
}

std::string reflect_accessors(std::span<reduce_input_spec const> inputs)
{
  std::vector<std::string> accessors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    accessors.push_back(
      rtcx::reflect_template("cudf::jit::column_accessor",
                             rtcx::reflect(i),
                             "cudf::column_device_view_core",
                             jit::reflect_input_element(reflection_spec(inputs[i]), false),
                             rtcx::reflect(inputs[i].is_scalar),
                             rtcx::reflect(0)));
  }
  return rtcx::reflect_template("cudf::jit::type_list", accessors);
}

std::string reflect_accessors(std::span<reduce_output const> outputs)
{
  std::vector<std::string> accessors;
  for (size_t i = 0; i < outputs.size(); ++i) {
    accessors.push_back(rtcx::reflect_template("cudf::jit::column_accessor",
                                               rtcx::reflect(i),
                                               "cudf::mutable_column_device_view_core",
                                               type_to_name(outputs[i].type),
                                               rtcx::reflect(false),
                                               rtcx::reflect(0)));
  }
  return rtcx::reflect_template("cudf::jit::type_list", accessors);
}

}  // namespace

struct reduce_program::impl {
  std::vector<reduce_input_spec> inputs_;
  std::vector<reduce_output> outputs_;
  reduce_options options_;
  size_t state_bytes_;
  int block_size_;
  bool use_warp_reduce_;
  kernel kernel_;
  std::optional<detail::row_ir::transform_args> ast_;

  static kernel compile(std::string const& source,
                        std::span<uint8_t const> binary,
                        std::optional<lto_binary_type> binary_type,
                        std::span<reduce_input_spec const> inputs,
                        std::span<data_type const> states,
                        std::span<reduce_output const> outputs,
                        reduce_options options,
                        size_t state_bytes,
                        int block_size,
                        bool use_warp_reduce,
                        detail::row_ir::transform_args const* ast_args)
  {
    CUDF_EXPECTS(
      !states.empty(), "Reduction state must have at least one field", std::invalid_argument);
    validate_outputs(outputs);
    CUDF_EXPECTS(
      options.is_null_aware == null_aware::YES || options.is_null_aware == null_aware::NO,
      "Invalid reduction null-awareness setting",
      std::invalid_argument);
    CUDF_EXPECTS(
      options.errors == error_policy::PROPAGATE || options.errors == error_policy::NULLIFY,
      "Invalid reduction error policy",
      std::invalid_argument);
    auto instance = rtcx::reflect_template("cudf::jit::reduce_kernel",
                                           rtcx::reflect(options.is_null_aware == null_aware::YES),
                                           rtcx::reflect(options.user_data.has_value()),
                                           reflect_accessors(inputs),
                                           reflect_accessors(outputs),
                                           rtcx::reflect(block_size),
                                           rtcx::reflect(use_warp_reduce));
    auto header   = generate_header(
      source, binary_type.has_value(), inputs, states, outputs, options, state_bytes, ast_args);
    auto instance_header  = std::format("#define CUDF_KERNEL_INSTANCE {}\n", instance);
    char const* names[]   = {"cudf/detail/operation_udf.cuh", "cudf/detail/kernel_instance.cuh"};
    char const* headers[] = {header.c_str(), instance_header.c_str()};
    if (!binary_type.has_value()) {
      CUDF_EXPECTS(
        !source.empty(), "Reduction CUDA source must not be empty", std::invalid_argument);
      return get_kernel(reduce_kernel_source, reduce_kernel_source, names, headers, instance);
    }
    CUDF_EXPECTS(!binary.empty(), "Reduction LTO binary must not be empty", std::invalid_argument);
    auto const type = jit::as_rtcx_binary_type(*binary_type);
    auto fragment =
      get_kernel_fragment(reduce_kernel_source, reduce_kernel_source, names, headers, instance);
    // Hash both complete fragment contents; the source/header schema must participate even when
    // two programs link the same user binary.
    rtcx::memory_fragment fragments[] = {
      {.data = fragment->view(), .type = rtcx::binary_type::LTO_IR, .name = nullptr},
      {.data = binary, .type = type, .name = nullptr}};
    return get_lto_linked_kernel(reduce_kernel_source, {}, fragments);
  }

  impl(std::string const& source,
       std::span<uint8_t const> binary,
       std::optional<lto_binary_type> binary_type,
       std::vector<reduce_input_spec> inputs,
       std::span<data_type const> states,
       std::span<reduce_output const> outputs,
       reduce_options options,
       std::optional<detail::row_ir::transform_args> ast = std::nullopt)
    : inputs_(std::move(inputs)),
      outputs_(outputs.begin(), outputs.end()),
      options_(options),
      state_bytes_(state_size(states)),
      block_size_(state_bytes_ <= 64    ? 256
                  : state_bytes_ <= 256 ? 128
                  : state_bytes_ <= 512 ? 64
                                        : 32),
      use_warp_reduce_(state_bytes_ <= 512),
      kernel_(compile(source,
                      binary,
                      binary_type,
                      inputs_,
                      states,
                      outputs_,
                      options_,
                      state_bytes_,
                      block_size_,
                      use_warp_reduce_,
                      ast ? &*ast : nullptr)),
      ast_(std::move(ast))
  {
  }

  std::vector<std::unique_ptr<column>> run(std::span<reduce_input const> inputs,
                                           std::span<reduce_output const> outputs,
                                           std::optional<size_type> row_size,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr) const
  {
    auto const actual_specs = make_specs(inputs);
    CUDF_EXPECTS(actual_specs.size() == inputs_.size() &&
                   std::equal(actual_specs.begin(),
                              actual_specs.end(),
                              inputs_.begin(),
                              [](auto const& a, auto const& b) { return same_schema(a, b); }),
                 "Reduction program input schema mismatch",
                 std::invalid_argument);
    if (outputs.empty()) { outputs = outputs_; }
    validate_outputs(outputs);
    CUDF_EXPECTS(
      outputs.size() == outputs_.size(), "Reduction output count mismatch", std::invalid_argument);
    for (size_t i = 0; i < outputs.size(); ++i) {
      CUDF_EXPECTS(outputs[i].type == outputs_[i].type,
                   "Reduction output type mismatch",
                   std::invalid_argument);
    }
    for (auto const& in : inputs) {
      if (auto const* col = std::get_if<column_view>(&in)) {
        if (!row_size.has_value()) { row_size = col->size(); }
        CUDF_EXPECTS(
          col->size() == *row_size, "Reduction input lengths must match", std::invalid_argument);
      }
    }
    CUDF_EXPECTS(row_size.has_value() && *row_size >= 0,
                 "Reduction requires a nonnegative row count when inputs are all scalars",
                 std::invalid_argument);

    std::vector<std::unique_ptr<column>> results;
    auto h_views = detail::make_empty_pinned_vector<detail::column_device_view_base>(
      inputs.size() + outputs.size(), stream);
    using input_handle =
      std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>;
    using output_handle =
      std::unique_ptr<mutable_column_device_view, std::function<void(mutable_column_device_view*)>>;
    std::vector<input_handle> input_handles;
    std::vector<output_handle> output_handles;
    for (auto const& in : inputs) {
      auto handle = std::visit(
        [&](auto const& col) { return column_device_view::create(input_view(col), stream); }, in);
      h_views.push_back(*handle);
      input_handles.push_back(std::move(handle));
    }
    auto h_sizes       = detail::make_empty_pinned_vector<size_type>(outputs.size(), stream);
    size_type max_size = 0;
    for (auto const& out : outputs) {
      auto col    = make_fixed_width_column(out.type, out.size, mask_state::ALL_NULL, stream, mr);
      auto handle = mutable_column_device_view::create(col->mutable_view(), stream);
      h_views.push_back(*handle);
      output_handles.push_back(std::move(handle));
      results.push_back(std::move(col));
      h_sizes.push_back(out.size);
      max_size = std::max(max_size, out.size);
    }
    auto views  = detail::make_device_uvector_async(h_views, stream, mr);
    auto sizes  = detail::make_device_uvector_async(h_sizes, stream, mr);
    auto status = detail::make_zeroed_device_uvector_async<int32_t>(1 + outputs.size(), stream, mr);
    auto* input_cols = reinterpret_cast<column_device_view_core const*>(views.data());
    auto* output_cols =
      reinterpret_cast<mutable_column_device_view_core const*>(views.data() + inputs.size());

    auto cfg      = kernel_.max_occupancy_config(0, block_size_);
    auto ceil_div = [&](size_type n) {
      return static_cast<int32_t>((static_cast<int64_t>(n) + block_size_ - 1) / block_size_);
    };
    auto grid = std::max<int64_t>(
      1, std::min<int64_t>(ceil_div(*row_size), static_cast<int64_t>(cfg.min_grid_size) * 4));
    rmm::device_buffer a(static_cast<size_t>(grid) * state_bytes_, stream, mr);
    rmm::device_buffer b(state_bytes_, stream, mr);
    rmm::device_buffer scratch(
      use_warp_reduce_ ? 0 : static_cast<size_t>(grid) * block_size_ * state_bytes_, stream, mr);
    auto launch =
      [&](
        int phase, size_type count, int blocks, void const* partials, void* output, bool finalize) {
        kernel_.launch_with({static_cast<uint32_t>(blocks)},
                            {static_cast<uint32_t>(block_size_)},
                            0,
                            stream,
                            phase,
                            count,
                            options_.user_data.value_or(nullptr),
                            input_cols,
                            output_cols,
                            sizes.data(),
                            max_size,
                            partials,
                            output,
                            scratch.data(),
                            status.data(),
                            finalize);
      };
    auto inline_final = max_size <= block_size_;
    launch(0, *row_size, grid, nullptr, a.data(), grid == 1 && inline_final);
    void* current = a.data();
    if (grid > 1) {
      // The row grid is bounded by occupancy, so a single block can merge all partials using
      // its grid-stride loop. This keeps large reductions to two stages.
      launch(1, grid, 1, a.data(), b.data(), inline_final);
      current = b.data();
    }
    if (!inline_final) {
      auto blocks = std::max<int64_t>(1, std::min<int64_t>(ceil_div(max_size), cfg.min_grid_size));
      launch(2, 0, blocks, current, nullptr, false);
    }

    // One host transfer/synchronization reports both errors and all output null counts.
    auto host_status = detail::make_pinned_vector(status, stream);
    auto error       = static_cast<errc>(host_status[0]);
    if (error != errc::SUCCESS && options_.errors == error_policy::PROPAGATE) {
      throw evaluation_error(
        error, std::format("Reduction UDF evaluation failed with error `{}`", to_string(error)));
    }
    for (size_t i = 0; i < results.size(); ++i) {
      if (error != errc::SUCCESS) {
        if (outputs[i].size != 0) {
          set_null_mask(results[i]->mutable_view().null_mask(), 0, outputs[i].size, false, stream);
        }
        results[i]->set_null_count(outputs[i].size);
      } else {
        results[i]->set_null_count(host_status[1 + i]);
      }
    }
    return results;
  }
};

reduce_program::reduce_program(std::string const& udf,
                               std::span<reduce_input_spec const> inputs,
                               std::span<data_type const> states,
                               std::span<reduce_output const> outputs,
                               reduce_options options)
  : impl_(std::make_unique<impl>(udf,
                                 std::span<uint8_t const>{},
                                 std::nullopt,
                                 std::vector<reduce_input_spec>{inputs.begin(), inputs.end()},
                                 states,
                                 outputs,
                                 options))
{
}

reduce_program::reduce_program(std::string const& udf,
                               std::span<reduce_input const> inputs,
                               std::span<data_type const> states,
                               std::span<reduce_output const> outputs,
                               reduce_options options)
  : reduce_program(udf, make_specs(inputs), states, outputs, options)
{
}

reduce_program::reduce_program(std::span<uint8_t const> udf,
                               lto_binary_type binary_type,
                               std::span<reduce_input_spec const> inputs,
                               std::span<data_type const> states,
                               std::span<reduce_output const> outputs,
                               reduce_options options)
  : impl_(std::make_unique<impl>(std::string{},
                                 udf,
                                 binary_type,
                                 std::vector<reduce_input_spec>{inputs.begin(), inputs.end()},
                                 states,
                                 outputs,
                                 options))
{
}

reduce_program::reduce_program(std::span<uint8_t const> udf,
                               lto_binary_type binary_type,
                               std::span<reduce_input const> inputs,
                               std::span<data_type const> states,
                               std::span<reduce_output const> outputs,
                               reduce_options options)
  : reduce_program(udf, binary_type, make_specs(inputs), states, outputs, options)
{
}

reduce_program::reduce_program(
  std::string const& udf,
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  std::span<data_type const> states,
  std::span<reduce_output const> outputs,
  reduce_options options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto args = detail::row_ir::ast_converter::compute_table(
    detail::row_ir::target::CUDA, expressions, table, {}, "cudf_reduce_map", stream, mr, true);
  for (auto& input : args.inputs) {
    if (auto* scalar = std::get_if<scalar_column_view>(&input)) {
      auto owned = std::make_unique<column>(scalar->as_column_view(), stream, mr);
      input      = scalar_column_view{owned->view()};
      args.scalar_columns.push_back(std::move(owned));
    }
  }
  auto specs = make_specs(args.inputs);
  impl_      = std::make_unique<impl>(udf,
                                 std::span<uint8_t const>{},
                                 std::nullopt,
                                 std::move(specs),
                                 states,
                                 outputs,
                                 options,
                                 std::move(args));
}

reduce_program::reduce_program(
  std::span<uint8_t const> udf,
  lto_binary_type binary_type,
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  std::span<data_type const> states,
  std::span<reduce_output const> outputs,
  reduce_options options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto args = detail::row_ir::ast_converter::compute_table(
    detail::row_ir::target::CUDA, expressions, table, {}, "cudf_reduce_map", stream, mr, true);
  for (auto& input : args.inputs) {
    if (auto* scalar = std::get_if<scalar_column_view>(&input)) {
      auto owned = std::make_unique<column>(scalar->as_column_view(), stream, mr);
      input      = scalar_column_view{owned->view()};
      args.scalar_columns.push_back(std::move(owned));
    }
  }
  auto specs = make_specs(args.inputs);
  impl_      = std::make_unique<impl>(
    std::string{}, udf, binary_type, std::move(specs), states, outputs, options, std::move(args));
}

reduce_program::reduce_program(reduce_program&&)            = default;
reduce_program& reduce_program::operator=(reduce_program&&) = default;
reduce_program::~reduce_program()                           = default;

std::vector<std::unique_ptr<column>> reduce_program::run(std::span<reduce_input const> inputs,
                                                         std::span<reduce_output const> outputs,
                                                         std::optional<size_type> row_size,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    impl_ != nullptr, "Cannot run a moved-from reduction program", std::invalid_argument);
  return impl_->run(inputs, outputs, row_size, stream, mr);
}

std::vector<std::unique_ptr<column>> reduce_program::run(table_view const& table,
                                                         std::span<reduce_output const> outputs,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(impl_ != nullptr && impl_->ast_.has_value(),
               "Reduction program has no AST mapping",
               std::invalid_argument);
  std::vector<reduce_input> inputs;
  auto const& args = *impl_->ast_;
  for (size_t i = 0; i < args.inputs.size(); ++i) {
    if (args.input_column_indices[i].has_value()) {
      CUDF_EXPECTS(args.input_table_sources[i] == 0,
                   "Reduction AST supports only LEFT table references",
                   std::invalid_argument);
      CUDF_EXPECTS(*args.input_column_indices[i] < table.num_columns(),
                   "Reduction AST column index out of bounds",
                   std::invalid_argument);
      inputs.emplace_back(table.column(*args.input_column_indices[i]));
    } else {
      inputs.push_back(args.inputs[i]);
    }
  }
  return impl_->run(inputs, outputs, table.num_rows(), stream, mr);
}

std::vector<std::unique_ptr<column>> reduce(std::string const& udf,
                                            std::span<reduce_input const> inputs,
                                            std::span<data_type const> states,
                                            std::span<reduce_output const> outputs,
                                            reduce_options options,
                                            std::optional<size_type> row_size,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  reduce_program program(udf, inputs, states, outputs, options);
  return program.run(inputs, outputs, row_size, stream, mr);
}

std::vector<std::unique_ptr<column>> reduce_lto(std::span<uint8_t const> udf,
                                                lto_binary_type binary_type,
                                                std::span<reduce_input const> inputs,
                                                std::span<data_type const> states,
                                                std::span<reduce_output const> outputs,
                                                reduce_options options,
                                                std::optional<size_type> row_size,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  reduce_program program(udf, binary_type, inputs, states, outputs, options);
  return program.run(inputs, outputs, row_size, stream, mr);
}

std::vector<std::unique_ptr<column>> reduce(
  std::string const& udf,
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  std::span<data_type const> states,
  std::span<reduce_output const> outputs,
  reduce_options options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  reduce_program program(udf, table, expressions, states, outputs, options, stream, mr);
  return program.run(table, outputs, stream, mr);
}

std::vector<std::unique_ptr<column>> reduce_lto(
  std::span<uint8_t const> udf,
  lto_binary_type binary_type,
  table_view const& table,
  std::span<std::reference_wrapper<ast::expression const> const> expressions,
  std::span<data_type const> states,
  std::span<reduce_output const> outputs,
  reduce_options options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  reduce_program program(
    udf, binary_type, table, expressions, states, outputs, options, stream, mr);
  return program.run(table, outputs, stream, mr);
}

}  // namespace cudf
