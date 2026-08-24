/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <nvbench/nvbench.cuh>

#include <cmath>

template <typename TypeParam>
static void bench_scatter_lists(nvbench::state& state, nvbench::type_list<TypeParam>)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  cudf::size_type const base_size{static_cast<cudf::size_type>(state.get_int64("base_size"))};
  cudf::size_type const num_elements_per_row{
    static_cast<cudf::size_type>(state.get_int64("num_elements_per_row"))};
  bool const coalesce = static_cast<bool>(state.get_int64("coalesce"));
  auto const num_rows =
    static_cast<cudf::size_type>(ceil(double(base_size) / num_elements_per_row));

  auto source_base_col = make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                                 base_size,
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  auto target_base_col = make_fixed_width_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                                 base_size,
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  thrust::sequence(rmm::exec_policy_nosync(stream),
                   source_base_col->mutable_view().begin<TypeParam>(),
                   source_base_col->mutable_view().end<TypeParam>());
  thrust::sequence(rmm::exec_policy_nosync(stream),
                   target_base_col->mutable_view().begin<TypeParam>(),
                   target_base_col->mutable_view().end<TypeParam>());

  auto source_offsets = make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  auto target_offsets = make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);

  thrust::sequence(rmm::exec_policy_nosync(stream),
                   source_offsets->mutable_view().begin<int32_t>(),
                   source_offsets->mutable_view().end<int32_t>(),
                   0,
                   num_elements_per_row);
  thrust::sequence(rmm::exec_policy_nosync(stream),
                   target_offsets->mutable_view().begin<int32_t>(),
                   target_offsets->mutable_view().end<int32_t>(),
                   0,
                   num_elements_per_row);

  auto source = make_lists_column(num_rows,
                                  std::move(source_offsets),
                                  std::move(source_base_col),
                                  0,
                                  cudf::create_null_mask(num_rows, cudf::mask_state::UNALLOCATED));
  auto target = make_lists_column(num_rows,
                                  std::move(target_offsets),
                                  std::move(target_base_col),
                                  0,
                                  cudf::create_null_mask(num_rows, cudf::mask_state::UNALLOCATED));

  auto scatter_map   = make_fixed_width_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                             num_rows,
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr);
  auto m_scatter_map = scatter_map->mutable_view();
  thrust::sequence(rmm::exec_policy_nosync(stream),
                   m_scatter_map.begin<cudf::size_type>(),
                   m_scatter_map.end<cudf::size_type>(),
                   num_rows - 1,
                   -1);

  if (not coalesce) {
    thrust::default_random_engine g;
    thrust::shuffle(rmm::exec_policy_nosync(stream),
                    m_scatter_map.begin<cudf::size_type>(),
                    m_scatter_map.end<cudf::size_type>(),
                    g);
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int8_t>(base_size * sizeof(TypeParam) * 2);  // source + scatter_map
  state.add_global_memory_writes<int8_t>(base_size * sizeof(TypeParam));     // target

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    scatter(cudf::table_view{{*source}},
            *scatter_map,
            cudf::table_view{{*target}},
            cudf::get_default_stream(),
            mr);
  });
}

NVBENCH_BENCH_TYPES(bench_scatter_lists, NVBENCH_TYPE_AXES(nvbench::type_list<double>))
  .set_name("scatter_lists")
  .set_type_axes_names({"type"})
  .add_int64_power_of_two_axis("base_size", nvbench::range(10, 25, 1))
  .add_int64_axis("num_elements_per_row", {64, 128, 256, 512, 1024, 2048})
  .add_int64_axis("coalesce", {true, false});
