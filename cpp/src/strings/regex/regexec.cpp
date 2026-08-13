/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/regex/regcomp.h"
#include "strings/regex/regex.cuh"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <functional>
#include <numeric>

namespace cudf {
namespace strings {
namespace detail {

// Copy reprog primitive values
reprog_device::reprog_device(reprog const& prog)
  : _startinst_id{prog.get_start_inst()},
    _num_capturing_groups{prog.groups_count()},
    _insts_count{prog.insts_count()},
    _starts_count{prog.starts_count()},
    _classes_count{prog.classes_count()},
    _max_insts{prog.insts_count()}
{
}

std::unique_ptr<reprog_device, std::function<void(reprog_device*)>> reprog_device::create(
  reprog const& h_prog, rmm::cuda_stream_view stream)
{
  // compute size to hold all the member data
  auto const insts_count   = h_prog.insts_count();
  auto const classes_count = h_prog.classes_count();
  auto const starts_count  = h_prog.starts_count();

  // compute size of each section
  auto insts_size    = insts_count * sizeof(_insts[0]);
  auto startids_size = starts_count * sizeof(_startinst_ids[0]);
  auto classes_size =
    std::transform_reduce(h_prog.classes_data(),
                          h_prog.classes_data() + h_prog.classes_count(),
                          classes_count * sizeof(_classes[0]),
                          std::plus<std::size_t>{},
                          [](auto& cls) { return cls.literals.size() * sizeof(reclass_range); });
  // make sure each section is aligned for the subsequent section's data type
  auto const memsize = cudf::util::round_up_safe(insts_size, sizeof(_startinst_ids[0])) +
                       cudf::util::round_up_safe(startids_size, sizeof(_classes[0])) +
                       cudf::util::round_up_safe(classes_size, sizeof(char32_t));

  // allocate memory to store all the prog data in a flat contiguous buffer
  auto h_buffer =
    cudf::detail::make_host_vector<u_char>(memsize, stream);  // copy everything into here;
  auto h_ptr    = h_buffer.data();                            // this is our running host ptr;
  auto d_buffer = new rmm::device_uvector<u_char>(memsize, stream);  // output device memory;
  auto d_ptr    = d_buffer->data();                                  // running device pointer

  // create our device object; this is managed separately and returned to the caller
  auto* d_prog = new reprog_device(h_prog);

  d_prog->_codepoint_flags = get_character_flags_table(stream);

  // copy the instructions array first (fixed-sized structs)
  memcpy(h_ptr, h_prog.insts_data(), insts_size);
  d_prog->_insts = reinterpret_cast<reinst*>(d_ptr);

  // point to the end for the next section
  insts_size = cudf::util::round_up_safe(insts_size, sizeof(_startinst_ids[0]));
  h_ptr += insts_size;
  d_ptr += insts_size;
  // copy the startinst_ids next
  memcpy(h_ptr, h_prog.starts_data(), startids_size);
  d_prog->_startinst_ids = reinterpret_cast<int32_t*>(d_ptr);

  // next section; align the size for next data type
  startids_size = cudf::util::round_up_safe(startids_size, sizeof(_classes[0]));
  h_ptr += startids_size;
  d_ptr += startids_size;
  // copy classes into flat memory: [class1,class2,...][char32 arrays]
  auto classes     = reinterpret_cast<reclass_device*>(h_ptr);
  d_prog->_classes = reinterpret_cast<reclass_device*>(d_ptr);
  // get pointer to the end to handle variable length data
  auto h_end = h_ptr + (classes_count * sizeof(reclass_device));
  auto d_end = d_ptr + (classes_count * sizeof(reclass_device));
  // place each class and append the variable length data
  for (int32_t idx = 0; idx < classes_count; ++idx) {
    auto const& h_class = h_prog.class_at(idx);
    reclass_device const d_class{h_class.builtins,
                                 static_cast<int32_t>(h_class.literals.size()),
                                 reinterpret_cast<reclass_range*>(d_end)};
    *classes++ = d_class;
    memcpy(h_end, h_class.literals.data(), h_class.literals.size() * sizeof(reclass_range));
    h_end += h_class.literals.size() * sizeof(reclass_range);
    d_end += h_class.literals.size() * sizeof(reclass_range);
  }

  // initialize the rest of the elements
  d_prog->_max_insts = insts_count;
  d_prog->_prog_size = memsize + sizeof(reprog_device);
  d_prog->_empty_match_possible =
    (h_prog.compute_match_flags() == cudf::strings::detail::match_flags::EMPTY_MATCH);

  // copy flat prog to device memory
  cudf::detail::cuda_memcpy_async<u_char>(*d_buffer, h_buffer, stream);

  // build deleter to cleanup device memory
  auto deleter = [d_buffer](reprog_device* t) {
    t->destroy();
    delete d_buffer;
  };

  stream.synchronize();  // wait for h_buffer to finish copying
  return std::unique_ptr<reprog_device, std::function<void(reprog_device*)>>(d_prog, deleter);
}

void reprog_device::destroy() { delete this; }

std::size_t reprog_device::working_memory_size(int32_t num_threads) const
{
  return compute_working_memory_size(num_threads, insts_counts());
}

std::pair<std::size_t, int32_t> reprog_device::compute_strided_working_memory(
  int32_t rows, int32_t min_rows, std::size_t requested_max_size) const
{
  auto thread_count = rows;
  auto buffer_size  = working_memory_size(thread_count);
  while ((buffer_size > requested_max_size) && (thread_count > min_rows)) {
    thread_count = thread_count / 2;
    buffer_size  = working_memory_size(thread_count);
  }
  // clamp to min_rows but only if rows is greater than min_rows
  if (rows > min_rows && thread_count < min_rows) {
    thread_count = min_rows;
    buffer_size  = working_memory_size(thread_count);
  }
  return std::make_pair(buffer_size, thread_count);
}

void reprog_device::set_working_memory(void* buffer, int32_t thread_count, int32_t max_insts)
{
  _buffer       = buffer;
  _thread_count = thread_count;
  _max_insts    = max_insts > 0 ? max_insts : _insts_count;
}

int32_t reprog_device::compute_shared_memory_size() const
{
  return _prog_size < MAX_SHARED_MEM ? static_cast<int32_t>(_prog_size) : 0;
}

std::size_t reprog_device::compute_working_memory_size(int32_t num_threads, int32_t insts_count)
{
  return relist::alloc_size(insts_count, num_threads) * 2;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
