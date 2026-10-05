/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/strings/udf/case.cuh>
#include <cudf/strings/udf/char_types.cuh>
#include <cudf/strings/udf/replace.cuh>
#include <cudf/strings/udf/search.cuh>
#include <cudf/strings/udf/starts_with.cuh>
#include <cudf/strings/udf/strip.cuh>
#include <cudf/strings/udf/udf_string.cuh>

#include <cooperative_groups.h>
#include <cuda/atomic>

#include <nrt.cuh>

#include <limits>
#include <type_traits>

using namespace cudf::strings::udf;

/**
 * @brief Destructor for a udf_string object.
 *
 * NRT API compatible destructor for udf_string objects.
 *
 * @param udf_str Pointer to the udf_string object to be destructed.
 * @param size Size of the udf_string object (not used).
 * @param dtor_info Additional information for the destructor (not used).
 */
__device__ void udf_str_dtor(void* udf_str, size_t size, void* dtor_info)
{
  auto ptr = reinterpret_cast<udf_string*>(udf_str);
  ptr->~udf_string();
}

__device__ NRT_MemInfo* make_meminfo_for_new_udf_string(udf_string* udf_str)
{
  // only used in the context of this function
  struct mi_str_allocation {
    NRT_MemInfo mi;
    udf_string st;
  };

  mi_str_allocation* mi_and_str = (mi_str_allocation*)NRT_Allocate(sizeof(mi_str_allocation));
  if (mi_and_str != NULL) {
    auto mi_ptr        = &(mi_and_str->mi);
    udf_string* st_ptr = &(mi_and_str->st);

    // udf_str_dtor can destruct the string without knowing the size
    size_t size = 0;
    NRT_MemInfo_init(mi_ptr, st_ptr, size, udf_str_dtor, NULL);

    // copy the udf_string to the allocated heap space
    udf_string* in_str_ptr = reinterpret_cast<udf_string*>(udf_str);
    memcpy(st_ptr, in_str_ptr, sizeof(udf_string));
    return mi_ptr;
  } else {
    __trap();
    return nullptr;
  }
}

// Special decref called only by python after transferring ownership of output strings
// Must reset dtor with one that is part of the current module
extern "C" __device__ void NRT_decref_managed_string(NRT_MemInfo* mi)
{
  mi->dtor = udf_str_dtor;
  NRT_decref(mi);
}

extern "C" __device__ int len(int* nb_retval, void const* str)
{
  auto sv    = reinterpret_cast<cudf::string_view const*>(str);
  *nb_retval = sv->length();
  return 0;
}

extern "C" __device__ int startswith(bool* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = starts_with(*str_view, *substr_view);
  return 0;
}

extern "C" __device__ int endswith(bool* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = ends_with(*str_view, *substr_view);
  return 0;
}

extern "C" __device__ int contains(bool* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = (str_view->find(*substr_view) != cudf::string_view::npos);
  return 0;
}

extern "C" __device__ int find(int* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = str_view->find(*substr_view);
  return 0;
}

extern "C" __device__ int rfind(int* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = str_view->rfind(*substr_view);
  return 0;
}

extern "C" __device__ int eq(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view == *rhs_view);
  return 0;
}

extern "C" __device__ int ne(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view != *rhs_view);
  return 0;
}

extern "C" __device__ int ge(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view >= *rhs_view);
  return 0;
}

extern "C" __device__ int le(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view <= *rhs_view);
  return 0;
}

extern "C" __device__ int gt(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view > *rhs_view);
  return 0;
}

extern "C" __device__ int lt(bool* nb_retval, void const* str, void const* rhs)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);
  auto rhs_view = reinterpret_cast<cudf::string_view const*>(rhs);

  *nb_retval = (*str_view < *rhs_view);
  return 0;
}

extern "C" __device__ int pyislower(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_lower(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisupper(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_upper(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisspace(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_space(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisdecimal(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_decimal(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisnumeric(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_numeric(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisdigit(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_digit(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisalnum(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_alpha_numeric(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyisalpha(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_alpha(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pyistitle(bool* nb_retval, void const* str, std::uintptr_t chars_table)
{
  auto str_view = reinterpret_cast<cudf::string_view const*>(str);

  *nb_retval = is_title(
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(chars_table), *str_view);
  return 0;
}

extern "C" __device__ int pycount(int* nb_retval, void const* str, void const* substr)
{
  auto str_view    = reinterpret_cast<cudf::string_view const*>(str);
  auto substr_view = reinterpret_cast<cudf::string_view const*>(substr);

  *nb_retval = count(*str_view, *substr_view);
  return 0;
}

extern "C" __device__ int udf_string_from_string_view(void** out_meminfo,
                                                      void const* str,
                                                      void* udf_str)
{
  auto str_view_ptr = reinterpret_cast<cudf::string_view const*>(str);
  auto udf_str_ptr  = new (udf_str) udf_string(*str_view_ptr);
  *out_meminfo      = make_meminfo_for_new_udf_string(udf_str_ptr);
  return 0;
}

extern "C" __device__ int string_view_from_udf_string(int* nb_retval,
                                                      void const* udf_str,
                                                      void* str)
{
  auto udf_str_ptr = reinterpret_cast<udf_string const*>(udf_str);
  auto sv_ptr      = new (str) cudf::string_view(*udf_str_ptr);
  return 0;
}

extern "C" __device__ int strip(void** out_meminfo,
                                void* udf_str,
                                void* const* to_strip,
                                void* const* strip_str)
{
  auto to_strip_ptr  = reinterpret_cast<cudf::string_view const*>(to_strip);
  auto strip_str_ptr = reinterpret_cast<cudf::string_view const*>(strip_str);
  auto udf_str_ptr   = new (udf_str) udf_string(strip(*to_strip_ptr, *strip_str_ptr));
  *out_meminfo       = make_meminfo_for_new_udf_string(udf_str_ptr);
  return 0;
}

extern "C" __device__ int lstrip(void** out_meminfo,
                                 void* udf_str,
                                 void* const* to_strip,
                                 void* const* strip_str)
{
  auto to_strip_ptr  = reinterpret_cast<cudf::string_view const*>(to_strip);
  auto strip_str_ptr = reinterpret_cast<cudf::string_view const*>(strip_str);
  auto udf_str_ptr =
    new (udf_str) udf_string(strip(*to_strip_ptr, *strip_str_ptr, cudf::strings::side_type::LEFT));
  *out_meminfo = make_meminfo_for_new_udf_string(udf_str_ptr);
  return 0;
}

extern "C" __device__ int rstrip(void** out_meminfo,
                                 void* udf_str,
                                 void* const* to_strip,
                                 void* const* strip_str)
{
  auto to_strip_ptr  = reinterpret_cast<cudf::string_view const*>(to_strip);
  auto strip_str_ptr = reinterpret_cast<cudf::string_view const*>(strip_str);
  auto udf_str_ptr =
    new (udf_str) udf_string(strip(*to_strip_ptr, *strip_str_ptr, cudf::strings::side_type::RIGHT));
  *out_meminfo = make_meminfo_for_new_udf_string(udf_str_ptr);
  return 0;
}

extern "C" __device__ int upper(void** out_meminfo,
                                void* udf_str,
                                void const* st,
                                std::uintptr_t flags_table,
                                std::uintptr_t cases_table,
                                std::uintptr_t special_table)
{
  auto st_ptr = reinterpret_cast<cudf::string_view const*>(st);

  auto flags_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(flags_table);
  auto cases_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_cases_table_type*>(cases_table);
  auto special_table_ptr =
    reinterpret_cast<cudf::strings::detail::special_case_mapping*>(special_table);

  cudf::strings::udf::chars_tables tables{flags_table_ptr, cases_table_ptr, special_table_ptr};

  auto udf_str_ptr = new (udf_str) udf_string(to_upper(tables, *st_ptr));
  *out_meminfo     = make_meminfo_for_new_udf_string(udf_str_ptr);
  return 0;
}

extern "C" __device__ int lower(void** out_meminfo,
                                void* udf_str,
                                void const* st,
                                std::uintptr_t flags_table,
                                std::uintptr_t cases_table,
                                std::uintptr_t special_table)
{
  auto st_ptr = reinterpret_cast<cudf::string_view const*>(st);

  auto flags_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_flags_table_type*>(flags_table);
  auto cases_table_ptr =
    reinterpret_cast<cudf::strings::detail::character_cases_table_type*>(cases_table);
  auto special_table_ptr =
    reinterpret_cast<cudf::strings::detail::special_case_mapping*>(special_table);

  cudf::strings::udf::chars_tables tables{flags_table_ptr, cases_table_ptr, special_table_ptr};

  auto udf_str_ptr = new (udf_str) udf_string(to_lower(tables, *st_ptr));
  *out_meminfo     = make_meminfo_for_new_udf_string(udf_str_ptr);
  return 0;
}

extern "C" __device__ int concat(void** out_meminfo,
                                 void* udf_str,
                                 void* const* lhs,
                                 void* const* rhs)
{
  auto lhs_ptr = reinterpret_cast<cudf::string_view const*>(lhs);
  auto rhs_ptr = reinterpret_cast<cudf::string_view const*>(rhs);

  udf_string result;
  result.append(*lhs_ptr).append(*rhs_ptr);
  auto udf_str_ptr = new (udf_str) udf_string(std::move(result));
  *out_meminfo     = make_meminfo_for_new_udf_string(udf_str_ptr);
  return 0;
}

extern "C" __device__ int replace(void** out_meminfo,
                                  void* udf_str,
                                  void* const src,
                                  void* const to_replace,
                                  void* const replacement)
{
  auto src_ptr         = reinterpret_cast<cudf::string_view const*>(src);
  auto to_replace_ptr  = reinterpret_cast<cudf::string_view const*>(to_replace);
  auto replacement_ptr = reinterpret_cast<cudf::string_view const*>(replacement);

  auto udf_str_ptr = new (udf_str) udf_string(replace(*src_ptr, *to_replace_ptr, *replacement_ptr));
  *out_meminfo     = make_meminfo_for_new_udf_string(udf_str_ptr);
  return 0;
}

// Groupby Shim Functions
template <typename T>
__device__ bool are_all_nans(cooperative_groups::thread_block const& block,
                             T const* data,
                             int64_t size)
{
  // TODO: to be refactored with CG vote functions once
  // block size is known at build time
  __shared__ int64_t count;

  if (block.thread_rank() == 0) { count = 0; }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    if (not std::isnan(data[idx])) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_block> ref{count};
      ref.fetch_add(1, cuda::std::memory_order_relaxed);
      break;
    }
  }

  block.sync();
  return count == 0;
}

template <typename T, typename AccumT = std::conditional_t<std::is_integral_v<T>, int64_t, T>>
__device__ AccumT device_sum(cooperative_groups::thread_block const& block,
                             T const* data,
                             int64_t size)
{
  __shared__ AccumT block_sum;
  if (block.thread_rank() == 0) { block_sum = 0; }
  block.sync();

  AccumT local_sum = 0;

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_sum += static_cast<AccumT>(data[idx]);
  }

  cuda::atomic_ref<AccumT, cuda::thread_scope_block> ref{block_sum};
  ref.fetch_add(local_sum, cuda::std::memory_order_relaxed);

  block.sync();
  return block_sum;
}

template <typename T, typename AccumT = std::conditional_t<std::is_integral_v<T>, int64_t, T>>
__device__ AccumT BlockSum(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  if constexpr (std::is_floating_point_v<T>) {
    if (are_all_nans(block, data, size)) { return 0; }
  }

  auto block_sum = device_sum<T>(block, data, size);
  return block_sum;
}

template <typename T>
__device__ double BlockMean(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  auto block_sum = device_sum<T>(block, data, size);
  return static_cast<double>(block_sum) / static_cast<double>(size);
}

template <typename T>
__device__ double BlockCoVar(T const* lhs, T const* rhs, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ double block_covar;

  if (block.thread_rank() == 0) { block_covar = 0; }
  block.sync();

  auto block_sum_lhs = device_sum<T>(block, lhs, size);

  auto const mu_l = static_cast<double>(block_sum_lhs) / static_cast<double>(size);
  auto const mu_r = [=]() {
    if (lhs == rhs) {
      // If the lhs and rhs are the same, this is calculating variance.
      // Thus we can assume mu_r = mu_l.
      return mu_l;
    } else {
      auto block_sum_rhs = device_sum<T>(block, rhs, size);
      return static_cast<double>(block_sum_rhs) / static_cast<double>(size);
    }
  }();

  double local_covar = 0;

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_covar += (static_cast<double>(lhs[idx]) - mu_l) * (static_cast<double>(rhs[idx]) - mu_r);
  }

  cuda::atomic_ref<double, cuda::thread_scope_block> ref{block_covar};
  ref.fetch_add(local_covar, cuda::std::memory_order_relaxed);
  block.sync();

  if (block.thread_rank() == 0) { block_covar /= static_cast<double>(size - 1); }
  block.sync();

  return block_covar;
}

template <typename T>
__device__ double BlockVar(T const* data, int64_t size)
{ return BlockCoVar<T>(data, data, size); }

template <typename T>
__device__ double BlockStd(T const* data, int64_t size)
{
  auto const var = BlockVar(data, size);
  return sqrt(var);
}

template <typename T>
__device__ T BlockMax(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  if constexpr (std::is_floating_point_v<T>) {
    if (are_all_nans(block, data, size)) { return std::numeric_limits<T>::quiet_NaN(); }
  }

  auto local_max = cudf::DeviceMax::identity<T>();
  __shared__ T block_max;
  if (block.thread_rank() == 0) { block_max = local_max; }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_max = max(local_max, data[idx]);
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_max};
  ref.fetch_max(local_max, cuda::std::memory_order_relaxed);

  block.sync();

  return block_max;
}

template <typename T>
__device__ T BlockMin(T const* data, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  if constexpr (std::is_floating_point_v<T>) {
    if (are_all_nans(block, data, size)) { return std::numeric_limits<T>::quiet_NaN(); }
  }

  auto local_min = cudf::DeviceMin::identity<T>();

  __shared__ T block_min;
  if (block.thread_rank() == 0) { block_min = local_min; }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    local_min = min(local_min, data[idx]);
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_min};
  ref.fetch_min(local_min, cuda::std::memory_order_relaxed);

  block.sync();

  return block_min;
}

template <typename T>
__device__ int64_t BlockIdxMax(T const* data, int64_t* index, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_max;
  __shared__ int64_t block_idx_max;
  __shared__ bool found_max;

  auto local_max     = cudf::DeviceMax::identity<T>();
  auto local_idx_max = cudf::DeviceMin::identity<int64_t>();

  if (block.thread_rank() == 0) {
    block_max     = local_max;
    block_idx_max = local_idx_max;
    found_max     = false;
  }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const current_data = data[idx];
    if (current_data > local_max) {
      local_max     = current_data;
      local_idx_max = index[idx];
      found_max     = true;
    }
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_max};
  ref.fetch_max(local_max, cuda::std::memory_order_relaxed);
  block.sync();

  if (found_max) {
    if (local_max == block_max) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_block> ref_idx{block_idx_max};
      ref_idx.fetch_min(local_idx_max, cuda::std::memory_order_relaxed);
    }
  } else {
    if (block.thread_rank() == 0) { block_idx_max = index[0]; }
  }
  block.sync();

  return block_idx_max;
}

template <typename T>
__device__ int64_t BlockIdxMin(T const* data, int64_t* index, int64_t size)
{
  auto block = cooperative_groups::this_thread_block();

  __shared__ T block_min;
  __shared__ int64_t block_idx_min;
  __shared__ bool found_min;

  auto local_min     = cudf::DeviceMin::identity<T>();
  auto local_idx_min = cudf::DeviceMin::identity<int64_t>();

  if (block.thread_rank() == 0) {
    block_min     = local_min;
    block_idx_min = local_idx_min;
    found_min     = false;
  }
  block.sync();

  for (int64_t idx = block.thread_rank(); idx < size; idx += block.size()) {
    auto const current_data = data[idx];
    if (current_data < local_min) {
      local_min     = current_data;
      local_idx_min = index[idx];
      found_min     = true;
    }
  }

  cuda::atomic_ref<T, cuda::thread_scope_block> ref{block_min};
  ref.fetch_min(local_min, cuda::std::memory_order_relaxed);
  block.sync();

  if (found_min) {
    if (local_min == block_min) {
      cuda::atomic_ref<int64_t, cuda::thread_scope_block> ref_idx{block_idx_min};
      ref_idx.fetch_min(local_idx_min, cuda::std::memory_order_relaxed);
    }
  } else {
    if (block.thread_rank() == 0) { block_idx_min = index[0]; }
  }
  block.sync();

  return block_idx_min;
}

template <typename T>
__device__ double BlockCorr(T* const lhs_ptr, T* const rhs_ptr, int64_t size)
{
  auto numerator   = BlockCoVar(lhs_ptr, rhs_ptr, size);
  auto denominator = BlockStd(lhs_ptr, size) * BlockStd<T>(rhs_ptr, size);
  if (denominator == 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  } else {
    return numerator / denominator;
  }
}

extern "C" {
#define make_definition(name, cname, type, return_type)                                          \
  __device__ int name##_##cname(return_type* numba_return_value, type* const data, int64_t size) \
  {                                                                                              \
    return_type const res = name<type>(data, size);                                              \
    *numba_return_value   = res;                                                                 \
    __syncthreads();                                                                             \
    return 0;                                                                                    \
  }

make_definition(BlockSum, int32, int32_t, int64_t);
make_definition(BlockSum, int64, int64_t, int64_t);
make_definition(BlockSum, float32, float, float);
make_definition(BlockSum, float64, double, double);

make_definition(BlockMean, int32, int32_t, double);
make_definition(BlockMean, int64, int64_t, double);
make_definition(BlockMean, float32, float, float);
make_definition(BlockMean, float64, double, double);

make_definition(BlockStd, int32, int32_t, double);
make_definition(BlockStd, int64, int64_t, double);
make_definition(BlockStd, float32, float, float);
make_definition(BlockStd, float64, double, double);

make_definition(BlockVar, int64, int64_t, double);
make_definition(BlockVar, int32, int32_t, double);
make_definition(BlockVar, float32, float, float);
make_definition(BlockVar, float64, double, double);

make_definition(BlockMin, int32, int32_t, int32_t);
make_definition(BlockMin, int64, int64_t, int64_t);
make_definition(BlockMin, float32, float, float);
make_definition(BlockMin, float64, double, double);

make_definition(BlockMax, int32, int32_t, int32_t);
make_definition(BlockMax, int64, int64_t, int64_t);
make_definition(BlockMax, float32, float, float);
make_definition(BlockMax, float64, double, double);
#undef make_definition
}

extern "C" {
#define make_definition_idx(name, cname, type)                                   \
  __device__ int name##_##cname(                                                 \
    int64_t* numba_return_value, type* const data, int64_t* index, int64_t size) \
  {                                                                              \
    auto const res      = name<type>(data, index, size);                         \
    *numba_return_value = res;                                                   \
    __syncthreads();                                                             \
    return 0;                                                                    \
  }

make_definition_idx(BlockIdxMin, int32, int32_t);
make_definition_idx(BlockIdxMin, int64, int64_t);
make_definition_idx(BlockIdxMin, float32, float);
make_definition_idx(BlockIdxMin, float64, double);

make_definition_idx(BlockIdxMax, int32, int32_t);
make_definition_idx(BlockIdxMax, int64, int64_t);
make_definition_idx(BlockIdxMax, float32, float);
make_definition_idx(BlockIdxMax, float64, double);
#undef make_definition_idx
}

extern "C" {
#define make_definition_corr(name, cname, type)                                 \
  __device__ int name##_##cname##_##cname(                                      \
    double* numba_return_value, type* const lhs, type* const rhs, int64_t size) \
  {                                                                             \
    double const res    = name<type>(lhs, rhs, size);                           \
    *numba_return_value = res;                                                  \
    __syncthreads();                                                            \
    return 0;                                                                   \
  }

make_definition_corr(BlockCorr, int32, int32_t);
make_definition_corr(BlockCorr, int64, int64_t);

#undef make_definition_corr
}
