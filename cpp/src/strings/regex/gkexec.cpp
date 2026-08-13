/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/regex/glushkov.cuh"
#include "strings/regex/glushkov_regcomp.hpp"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstring>
#include <functional>
#include <memory>
#include <numeric>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Pack a gkprog into a contiguous device buffer and
 *        return a device pointer to the embedded gkprog_device struct.
 *
 * Buffer layout:
 *   [_positions    : num_states × reinst]
 *   [_shift_masks  : GLUSHKOV_MAX_SHIFTS × glushkov_state_t]
 *   [_shift_amounts: GLUSHKOV_MAX_SHIFTS × uint8_t + padding]
 *   [_reach_ascii  : GLUSHKOV_ASCII_TABLE_SIZE × glushkov_state_t]
 *   [_exception_successors : num_states × glushkov_state_t + padding]
 *   [_classes : classes_count × reclass_device + variable-length literals]
 */
std::unique_ptr<gkprog_device, std::function<void(gkprog_device*)>> gkprog_device::create(
  gkprog const& h_gp, rmm::cuda_stream_view stream)
{
  auto const num_states  = h_gp.num_states;
  auto const classes_cnt = static_cast<int32_t>(h_gp.classes.size());

  auto const smasks_size      = GLUSHKOV_MAX_SHIFTS * sizeof(glushkov_state_t);
  auto const reach_ascii_size = GLUSHKOV_ASCII_TABLE_SIZE * sizeof(glushkov_state_t);
  auto const cls_size         = std::transform_reduce(
    h_gp.classes.begin(),
    h_gp.classes.end(),
    static_cast<std::size_t>(classes_cnt) * sizeof(reclass_device),
    std::plus<std::size_t>{},
    [](auto const& cls) { return cls.literals.size() * sizeof(reclass_range); });

  // compute size of each section
  auto pos_size   = num_states * sizeof(reinst);
  auto samts_size = GLUSHKOV_MAX_SHIFTS * sizeof(uint8_t);
  auto exc_size   = num_states * sizeof(glushkov_state_t);

  // make sure each section is aligned for the subsequent section's data type
  auto const memsize =
    cudf::util::round_up_safe(pos_size, alignof(glushkov_state_t)) + smasks_size +
    cudf::util::round_up_safe(samts_size, alignof(glushkov_state_t)) + reach_ascii_size +
    cudf::util::round_up_safe(exc_size, alignof(reclass_device)) +
    cudf::util::round_up_safe(cls_size, sizeof(char32_t));

  // allocate memory to store all the prog data in a flat contiguous buffer
  auto h_buffer = cudf::detail::make_host_vector<u_char>(memsize, stream);
  auto h_ptr    = h_buffer.data();
  auto d_buffer = std::make_unique<rmm::device_uvector<u_char>>(memsize, stream);
  auto d_ptr    = d_buffer->data();

  // create our device object; this is managed separately and returned to the caller.
  // both allocations are held by unique_ptrs so they are freed if anything below throws;
  // ownership is transferred to the returned unique_ptr's deleter only once we cannot throw.
  auto d_prog              = std::make_unique<gkprog_device>();
  d_prog->num_states       = num_states;
  d_prog->shift_count      = h_gp.shift_count;
  d_prog->first_set        = h_gp.first_set;
  d_prog->accept_mask      = h_gp.accept_mask;
  d_prog->exception_mask   = h_gp.exception_mask;
  d_prog->startchar        = h_gp.startchar;
  d_prog->_codepoint_flags = get_character_flags_table(stream);
  d_prog->_prog_size       = memsize;

  // copy the positions array (fixed-sized structs)
  std::memcpy(h_ptr, h_gp.pos_insts.data(), pos_size);
  d_prog->_positions = reinterpret_cast<reinst const*>(d_ptr);

  // advance to next section; align for glushkov_state_t
  pos_size = cudf::util::round_up_safe(pos_size, alignof(glushkov_state_t));
  h_ptr += pos_size;
  d_ptr += pos_size;

  // copy shift_masks
  std::memcpy(h_ptr, h_gp.shift_masks.data(), smasks_size);
  d_prog->_shift_masks = reinterpret_cast<glushkov_state_t const*>(d_ptr);
  h_ptr += smasks_size;
  d_ptr += smasks_size;

  // copy shift_amounts (uint8_t; no alignment padding needed after glushkov_state_t[])
  std::memcpy(h_ptr, h_gp.shift_amounts.data(), samts_size);
  d_prog->_shift_amounts = reinterpret_cast<uint8_t const*>(d_ptr);

  // advance to next section; align for glushkov_state_t
  samts_size = cudf::util::round_up_safe(samts_size, alignof(glushkov_state_t));
  h_ptr += samts_size;
  d_ptr += samts_size;

  // copy reach_ascii table (GLUSHKOV_ASCII_TABLE_SIZE × glushkov_state_t = 1 KB)
  std::memcpy(h_ptr, h_gp.reach_ascii.data(), reach_ascii_size);
  d_prog->_reach_ascii = reinterpret_cast<glushkov_state_t const*>(d_ptr);
  h_ptr += reach_ascii_size;
  d_ptr += reach_ascii_size;

  // copy exception_successors
  std::memcpy(h_ptr, h_gp.exception_successors.data(), exc_size);
  d_prog->_exception_successors = reinterpret_cast<glushkov_state_t const*>(d_ptr);

  // advance to next section; align for reclass_device (alignas(16))
  exc_size = cudf::util::round_up_safe(exc_size, alignof(reclass_device));
  h_ptr += exc_size;
  d_ptr += exc_size;

  // copy classes into flat memory: [class1, class2, ...][literals arrays]
  auto* classes    = reinterpret_cast<reclass_device*>(h_ptr);
  d_prog->_classes = reinterpret_cast<reclass_device const*>(d_ptr);
  auto h_end       = h_ptr + static_cast<std::size_t>(classes_cnt) * sizeof(reclass_device);
  auto d_end       = d_ptr + static_cast<std::size_t>(classes_cnt) * sizeof(reclass_device);
  for (int32_t ci = 0; ci < classes_cnt; ++ci) {
    auto const& src      = h_gp.classes[ci];
    *classes++           = reclass_device{src.builtins,
                                static_cast<int32_t>(src.literals.size()),
                                reinterpret_cast<reclass_range const*>(d_end)};
    auto const lit_bytes = src.literals.size() * sizeof(reclass_range);
    std::memcpy(h_end, src.literals.data(), lit_bytes);
    h_end += lit_bytes;
    d_end += lit_bytes;
  }

  // copy flat buffer to device
  cudf::detail::cuda_memcpy_async<u_char>(*d_buffer, h_buffer, stream);

  // create a deleter to free both device buffers
  auto deleter = [d_buffer = d_buffer.get()](gkprog_device* t) {
    t->destroy();
    delete d_buffer;
  };

  stream.synchronize();  // wait for h_buffer to finish copying

  auto result =
    std::unique_ptr<gkprog_device, std::function<void(gkprog_device*)>>(d_prog.get(), deleter);
  // ownership now belongs to result's deleter; release from the guarding unique_ptrs
  d_prog.release();
  d_buffer.release();
  return result;
}

void gkprog_device::destroy() { delete this; }

int32_t gkprog_device::compute_shared_memory_size() const
{
  return static_cast<int32_t>(sizeof(glushkov_shmem_cache));
}

std::pair<std::size_t, int32_t> gkprog_device::compute_strided_working_memory(int32_t rows,
                                                                              int32_t,
                                                                              std::size_t) const
{
  return std::make_pair(0, rows);
}

void gkprog_device::set_working_memory(void*, int32_t thread_count, int32_t)
{
  _thread_count = thread_count;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
