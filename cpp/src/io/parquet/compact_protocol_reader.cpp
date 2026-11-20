/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compact_protocol_reader.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <future>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudf::io::parquet::detail {
namespace {

// Shared overread/schema-init diagnostics for the decode helpers below.
constexpr char const* overread_message =
  "Parquet footer is truncated or corrupt (read past end of buffer)";
constexpr char const* cannot_init_schema_message = "Cannot initialize schema";

std::string field_type_string(FieldType type)
{
  switch (type) {
    case FieldType::BOOLEAN_TRUE: return "bool(true)";
    case FieldType::BOOLEAN_FALSE: return "bool(false)";
    case FieldType::I8: return "int8";
    case FieldType::I16: return "int16";
    case FieldType::I32: return "int32";
    case FieldType::I64: return "int64";
    case FieldType::DOUBLE: return "double";
    case FieldType::BINARY: return "binary";
    case FieldType::LIST: return "list";
    case FieldType::SET: return "set";
    case FieldType::MAP: return "map";
    case FieldType::STRUCT: return "struct";
    case FieldType::UUID: return "UUID";
    default: return "unknown(" + std::to_string(static_cast<uint8_t>(type)) + ")";
  }
}

void assert_field_type(int type, FieldType expected)
{
  CUDF_EXPECTS(type == static_cast<int>(expected),
               "expected " + field_type_string(expected) + " field, got " +
                 field_type_string(static_cast<FieldType>(type)) + " field instead");
}

void assert_bool_field_type(int type)
{
  auto const field_type = static_cast<FieldType>(type);
  CUDF_EXPECTS(field_type == FieldType::BOOLEAN_TRUE || field_type == FieldType::BOOLEAN_FALSE,
               "expected bool field, got " + field_type_string(field_type) + " field instead");
}

template <int index>
struct FunctionSwitchImpl {
  template <typename... Operator>
  static inline void run(CompactProtocolReader* cpr,
                         int field_type,
                         int const& field,
                         std::tuple<Operator...>& ops)
  {
    if (field == std::get<index>(ops).field()) {
      std::get<index>(ops)(cpr, field_type);
    } else {
      FunctionSwitchImpl<index - 1>::run(cpr, field_type, field, ops);
    }
  }
};

template <>
struct FunctionSwitchImpl<0> {
  template <typename... Operator>
  static inline void run(CompactProtocolReader* cpr,
                         int field_type,
                         int const& field,
                         std::tuple<Operator...>& ops)
  {
    if (field == std::get<0>(ops).field()) {
      std::get<0>(ops)(cpr, field_type);
    } else {
      cpr->skip_struct_field(field_type);
    }
  }
};

template <typename... Operator>
inline void function_builder(CompactProtocolReader* cpr, std::tuple<Operator...>& op)
{
  constexpr int index = std::tuple_size<std::tuple<Operator...>>::value - 1;
  int field           = 0;
  while (true) {
    int const current_byte = cpr->getb();
    if (!current_byte) { break; }
    int const field_delta = current_byte >> 4;
    int const field_type  = current_byte & 0xf;
    field                 = field_delta ? field + field_delta : cpr->get_i16();
    FunctionSwitchImpl<index>::run(cpr, field_type, field, op);
  }
}

}  // namespace

/**
 * @brief Base class for parquet field functors.
 *
 * Holds the field value used by all of the specialized functors.
 */
class parquet_field {
 private:
  int _field_val;

 protected:
  parquet_field(int f) : _field_val(f) {}

 public:
  virtual ~parquet_field() = default;
  [[nodiscard]] int field() const { return _field_val; }
};

/**
 * @brief Abstract base class for list functors.
 */
template <typename T, FieldType EXPECTED_ELEM_TYPE>
class parquet_field_list : public parquet_field {
 private:
  using read_func_type = std::function<void(uint32_t, CompactProtocolReader*)>;
  read_func_type _read_value;

 protected:
  std::vector<T>& val;

  void bind_read_func(read_func_type fn) { _read_value = fn; }

  parquet_field_list(int f, std::vector<T>& v) : parquet_field(f), val(v) {}

 public:
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, FieldType::LIST)) { return false; }
    auto const [t, n] = cpr->get_listh();
    // An empty list has no elements, so its wire element-type nibble is immaterial (some writers,
    // e.g. fastparquet, stamp 0); accept without validating the element type, matching Thrift.
    if (n == 0) {
      val.clear();
      return true;
    }
    // Fail before any per-element work on a count that cannot fit the remaining bytes.
    CUDF_EXPECTS(std::cmp_less_equal(n, cpr->m_end - cpr->m_cur),
                 "Parquet footer list size exceeds remaining buffer");
    if (not cpr->check_list_element_type(t, EXPECTED_ELEM_TYPE, n)) { return false; }
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      _read_value(i, cpr);
    }
    return true;
  }
};

/**
 * @brief Functor to set value to bool read from CompactProtocolReader
 *
 * bool doesn't actually encode a value, we just use the field type to indicate true/false
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
class parquet_field_bool : public parquet_field {
  bool& val;

 public:
  parquet_field_bool(int f, bool& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    auto const wire_type = static_cast<FieldType>(field_type);
    if (wire_type != FieldType::BOOLEAN_TRUE && wire_type != FieldType::BOOLEAN_FALSE) {
      if (cpr->should_throw_on_type_mismatch()) { assert_bool_field_type(field_type); }
      cpr->skip_struct_field(field_type);
      return false;
    }
    val = wire_type == FieldType::BOOLEAN_TRUE;
    return true;
  }
};

/**
 * @brief Functor to read a vector of booleans from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
struct parquet_field_bool_list : public parquet_field_list<bool, FieldType::BOOLEAN_TRUE> {
  parquet_field_bool_list(int f, std::vector<bool>& v) : parquet_field_list(f, v)
  {
    auto const read_value = [&val = v](uint32_t i, CompactProtocolReader* cpr) {
      auto const current_byte = cpr->getb();
      assert_bool_field_type(current_byte);
      CUDF_EXPECTS(i < val.size(), "Index out of bounds");
      val[i] = current_byte == static_cast<int>(FieldType::BOOLEAN_TRUE);
    };
    bind_read_func(read_value);
  }
};

/**
 * @brief Base type for a functor that reads an integer from CompactProtocolReader
 *
 * Assuming signed ints since the parquet spec does not use unsigned ints anywhere.
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
template <typename T, FieldType EXPECTED_TYPE>
class parquet_field_int : public parquet_field {
  static constexpr bool is_byte = std::is_same_v<T, int8_t>;

  T& val;

 public:
  parquet_field_int(int f, T& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, EXPECTED_TYPE)) { return false; }
    if constexpr (is_byte) {
      val = cpr->getb();
    } else {
      val = cpr->get_zigzag<T>();
    }
    return true;
  }
};

using parquet_field_int8  = parquet_field_int<int8_t, FieldType::I8>;
using parquet_field_int16 = parquet_field_int<int16_t, FieldType::I16>;
using parquet_field_int32 = parquet_field_int<int32_t, FieldType::I32>;
using parquet_field_int64 = parquet_field_int<int64_t, FieldType::I64>;

/**
 * @brief Functor to read a vector of integers from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
template <typename T, FieldType EXPECTED_TYPE>
struct parquet_field_int_list : public parquet_field_list<T, EXPECTED_TYPE> {
  parquet_field_int_list(int f, std::vector<T>& v) : parquet_field_list<T, EXPECTED_TYPE>(f, v)
  {
    auto const read_value = [&val = v](uint32_t i, CompactProtocolReader* cpr) {
      CUDF_EXPECTS(i < val.size(), "Index out of bounds");
      val[i] = cpr->get_zigzag<T>();
    };
    this->bind_read_func(read_value);
  }
};

using parquet_field_int64_list = parquet_field_int_list<int64_t, FieldType::I64>;

/**
 * @brief Functor to read a string from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
class parquet_field_string : public parquet_field {
  std::string& val;

 public:
  parquet_field_string(int f, std::string& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, FieldType::BINARY)) { return false; }
    auto const n = cpr->get_u32();
    CUDF_EXPECTS(std::cmp_less_equal(n, cpr->m_end - cpr->m_cur), "string length mismatch");

    val.assign(reinterpret_cast<char const*>(cpr->m_cur), n);
    cpr->m_cur += n;
    return true;
  }
};

/**
 * @brief Functor to read a vector of strings from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
class parquet_field_string_list : public parquet_field_list<std::string, FieldType::BINARY> {
 public:
  parquet_field_string_list(int f, std::vector<std::string>& v) : parquet_field_list(f, v)
  {
    auto const read_value = [&val = v](uint32_t i, CompactProtocolReader* cpr) {
      auto const l = cpr->get_u32();
      CUDF_EXPECTS(std::cmp_less_equal(l, cpr->m_end - cpr->m_cur), "string length mismatch");
      CUDF_EXPECTS(i < val.size(), "Index out of bounds");

      val[i].assign(reinterpret_cast<char const*>(cpr->m_cur), l);
      cpr->m_cur += l;
    };
    bind_read_func(read_value);
  }
};

/**
 * @brief Functor to set value to enum read from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
template <typename Enum>
class parquet_field_enum : public parquet_field {
  Enum& val;

 public:
  parquet_field_enum(int f, Enum& v) : parquet_field(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, FieldType::I32)) { return false; }
    val = static_cast<Enum>(cpr->get_i32());
    return true;
  }
};

/**
 * @brief Functor to read a vector of enums from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
template <typename Enum>
struct parquet_field_enum_list : public parquet_field_list<Enum, FieldType::I32> {
  parquet_field_enum_list(int f, std::vector<Enum>& v)
    : parquet_field_list<Enum, FieldType::I32>(f, v)
  {
    auto const read_value = [&val = v](uint32_t i, CompactProtocolReader* cpr) {
      CUDF_EXPECTS(i < val.size(), "Index out of bounds");
      val[i] = static_cast<Enum>(cpr->get_i32());
    };
    this->bind_read_func(read_value);
  }
};

/**
 * @brief Functor to read a structure from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
template <typename T>
class parquet_field_struct : public parquet_field {
  T& val;

 public:
  parquet_field_struct(int f, T& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, FieldType::STRUCT)) { return false; }
    cpr->read(&val);
    return true;
  }
};

/**
 * @brief Functor to read a LogicalType and its payload from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
class parquet_field_logical_type : public parquet_field {
  cuda::std::optional<LogicalType>& val;
  LogicalTypePayload& payload;

 public:
  parquet_field_logical_type(int f, cuda::std::optional<LogicalType>& v, LogicalTypePayload& p)
    : parquet_field(f), val(v), payload(p)
  {
  }

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, FieldType::STRUCT)) { return false; }
    LogicalType v;
    cpr->read(&v, &payload);
    val = v;
    return true;
  }
};

/**
 * @brief Functor to read optional structures in unions
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
template <typename E, typename T>
class parquet_field_union_struct : public parquet_field {
  E& enum_val;
  cuda::std::optional<T>& val;  // union structs are always wrapped in std::optional

 public:
  parquet_field_union_struct(int f, E& ev, cuda::std::optional<T>& v)
    : parquet_field(f), enum_val(ev), val(v)
  {
  }

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    T v;
    if (not parquet_field_struct<T>{field(), v}(cpr, field_type)) { return false; }
    val      = v;
    enum_val = static_cast<E>(field());
    return true;
  }
};

/**
 * @brief Functor to read empty structures in unions
 *
 * Added to avoid having to define read() functions for empty structs contained in unions.
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
template <typename E>
class parquet_field_union_enumerator : public parquet_field {
  E& val;

 public:
  parquet_field_union_enumerator(int f, E& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    // Union members are structs; a non-struct wire type means this arm is absent, so skip it.
    bool const matched = field_type == static_cast<int>(FieldType::STRUCT);
    if (not matched && cpr->should_throw_on_type_mismatch()) {
      assert_field_type(field_type, FieldType::STRUCT);
    }
    cpr->skip_struct_field(field_type);
    if (matched) { val = static_cast<E>(field()); }
    return matched;
  }
};

/**
 * @brief Default element reader of `parquet_field_struct_list`: decodes the whole struct
 */
struct read_struct_element {
  template <typename T>
  void operator()(CompactProtocolReader& cpr, T& elem, uint32_t) const
  {
    cpr.read(&elem);
  }
};

/**
 * @brief Functor to read a vector of structures from CompactProtocolReader
 *
 * If a selection `mask` is given, only the selected elements are decoded (in order) and the rest
 * are skipped on the wire. Each decoded element is read with `ElementReader`, which also receives
 * the element's index in the encoded list. Uses parallel processing when 512 or more elements are
 * decoded.
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
template <typename T, typename ElementReader = read_struct_element>
class parquet_field_struct_list : public parquet_field {
  std::vector<T>& val;
  std::vector<bool> const* mask;
  ElementReader read_element;

 public:
  parquet_field_struct_list(int f,
                            std::vector<T>& v,
                            std::vector<bool> const* m = nullptr,
                            ElementReader r            = {})
    : parquet_field(f), val(v), mask(m), read_element(std::move(r))
  {
  }

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, FieldType::LIST)) { return false; }
    auto const [t, n] = cpr->get_listh();
    // An empty list has no elements, so its wire element-type nibble is immaterial (some writers,
    // e.g. fastparquet, stamp 0); accept without validating the element type, matching Thrift.
    if (n == 0) {
      val.clear();
      return true;
    }
    // Fail before any per-element work on a count that cannot fit the remaining bytes.
    CUDF_EXPECTS(std::cmp_less_equal(n, cpr->m_end - cpr->m_cur),
                 "Parquet footer list size exceeds remaining buffer");
    if (not cpr->check_list_element_type(t, FieldType::STRUCT, n)) { return false; }
    CUDF_EXPECTS(mask == nullptr or mask->size() == n,
                 "Parquet footer list size does not match its selection");

    auto const is_selected = [&](uint32_t i) { return mask == nullptr or (*mask)[i]; };
    auto const num_selected =
      mask == nullptr ? n : static_cast<uint32_t>(std::count(mask->begin(), mask->end(), true));
    auto const skip_element = [&] { cpr->skip_struct_field(static_cast<int>(FieldType::STRUCT)); };
    val.resize(num_selected);
    if constexpr (requires { read_element.prepare(num_selected); }) {
      read_element.prepare(num_selected);
    }

    constexpr uint32_t parallel_threshold = 512;
    if (num_selected >= parallel_threshold) {
      // Propagate the reader's strict/lenient mode and column chunk policy to the parallel
      // per-struct sub-readers below.
      auto const mode   = cpr->m_mismatch_policy;
      auto const policy = cpr->m_chunk_policy;
      auto const num_tasks =
        std::min<uint32_t>(num_selected, cudf::detail::host_worker_pool().get_thread_count() * 2);
      auto const items_per_task = num_selected / num_tasks;
      auto const remainder      = num_selected % num_tasks;

      // Size once up front, then fill by index below: workers read already-filled slots while the
      // main thread fills later ones, so a size-mutating emplace_back would be a data race.
      std::vector<cudf::host_span<uint8_t const>> all_ranges(num_selected);
      std::vector<uint32_t> list_indices(num_selected);

      // Declared after all_ranges so that every task is waited on before all_ranges dies on any
      // exit path (host-pool futures do not block on destruction)
      std::vector<std::future<void>> tasks;
      cudf::detail::futures_wait_guard tasks_guard{tasks};
      tasks.reserve(num_tasks);

      uint32_t struct_idx = 0;  // index into the selected elements
      uint32_t list_idx   = 0;  // index into the encoded list
      for (uint32_t task_id = 0; task_id < num_tasks; ++task_id) {
        auto const task_size = items_per_task + (task_id < remainder ? 1 : 0);
        auto const start_idx = std::min(num_selected, struct_idx);
        auto const end_idx   = std::min(num_selected, start_idx + task_size);

        if (start_idx >= end_idx) { break; }

        // Collect struct ranges for the next task, skipping unselected elements in between
        for (auto i = start_idx; i < end_idx; ++i, ++list_idx) {
          while (not is_selected(list_idx)) {
            skip_element();
            ++list_idx;
          }
          uint8_t const* const start = cpr->m_cur;
          skip_element();
          all_ranges[i] =
            cudf::host_span<uint8_t const>{start, static_cast<size_t>(cpr->m_cur - start)};
          list_indices[i] = list_idx;
        }

        // Launch task immediately to parse the structs in parallel while the main thread collects
        // the remaining ranges
        tasks.emplace_back(
          cudf::detail::host_worker_pool().submit_task([&val          = this->val,
                                                        &read_element = this->read_element,
                                                        &all_ranges,
                                                        &list_indices,
                                                        start_idx,
                                                        end_idx,
                                                        mode,
                                                        policy]() {
            CompactProtocolReader local_cpr{nullptr, 0, mode};
            local_cpr.set_column_chunk_policy(policy);
            // end_idx alone bounds the loop: all_ranges[start_idx, end_idx) is fully populated
            // before this task launches, so no concurrent read of all_ranges.size() is needed.
            for (size_t i = start_idx; i < end_idx; ++i) {
              local_cpr.init(all_ranges[i].data(), all_ranges[i].size());
              read_element(local_cpr, val[i], list_indices[i]);
            }
          }));

        struct_idx = end_idx;
      }
      // Skip the unselected elements after the last selected one
      for (; list_idx < n; ++list_idx) {
        skip_element();
      }
      // Rethrows the first failure (e.g. a strict-mode mismatch); the guard waits for the
      // remaining tasks, whose lambdas reference all_ranges/val, while unwinding
      for (auto& task : tasks) {
        task.get();
      }
    } else {
      // For small numbers of elements, use sequential processing to avoid overhead
      for (uint32_t i = 0, j = 0; i < n; ++i) {
        if (is_selected(i)) {
          read_element(*cpr, val[j++], i);
        } else {
          skip_element();
        }
      }
    }
    return true;
  }
};

/**
 * @brief Functor to skip a list of structures, recording the byte offset of each element
 *
 * `offsets` receives the offset (relative to `base`) of each element, followed by the offset just
 * past the last element.
 */
class parquet_field_struct_list_offsets : public parquet_field {
  std::vector<uint64_t>& offsets;
  uint8_t const* base;

 public:
  parquet_field_struct_list_offsets(int f, std::vector<uint64_t>& o, uint8_t const* b)
    : parquet_field(f), offsets(o), base(b)
  {
  }

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, FieldType::LIST)) { return false; }
    auto const [t, n]         = cpr->get_listh();
    auto const current_offset = [&] { return static_cast<uint64_t>(cpr->m_cur - base); };
    if (n == 0) {
      offsets.assign(1, current_offset());
      return true;
    }
    CUDF_EXPECTS(std::cmp_less_equal(n, cpr->m_end - cpr->m_cur),
                 "Parquet footer list size exceeds remaining buffer");
    if (not cpr->check_list_element_type(t, FieldType::STRUCT, n)) { return false; }
    offsets.resize(n + 1);
    for (uint32_t i = 0; i < n; ++i) {
      offsets[i] = current_offset();
      cpr->skip_struct_field(static_cast<int>(FieldType::STRUCT));
    }
    offsets[n] = current_offset();
    return true;
  }
};

/**
 * @brief Functor that runs `prepare` right before decoding a field with `FieldFunctor`
 *
 * @return The result of `FieldFunctor`
 */
template <typename FieldFunctor, typename Prepare>
class parquet_field_prepared : public parquet_field {
  FieldFunctor functor;
  Prepare prepare;

 public:
  parquet_field_prepared(FieldFunctor f, Prepare p)
    : parquet_field(f.field()), functor(std::move(f)), prepare(std::move(p))
  {
  }

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    prepare();
    return functor(cpr, field_type);
  }
};

/**
 * @brief Functor that decodes a field with `FieldFunctor` if enabled, and skips it otherwise
 *
 * @return true if the field was read or skipped, false if skipped on a type mismatch
 */
template <typename FieldFunctor>
class parquet_field_if : public parquet_field {
  bool enabled;
  FieldFunctor functor;

 public:
  parquet_field_if(bool e, FieldFunctor f) : parquet_field(f.field()), enabled(e), functor(f) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not enabled) {
      cpr->skip_struct_field(field_type);
      return true;
    }
    return functor(cpr, field_type);
  }
};

/**
 * @brief Functor to read a binary from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
class parquet_field_binary : public parquet_field {
  std::vector<uint8_t>& val;

 public:
  parquet_field_binary(int f, std::vector<uint8_t>& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, FieldType::BINARY)) { return false; }
    auto const n = cpr->get_u32();
    CUDF_EXPECTS(std::cmp_less_equal(n, cpr->m_end - cpr->m_cur), "binary length mismatch");

    val.assign(cpr->m_cur, cpr->m_cur + n);
    cpr->m_cur += n;
    return true;
  }
};

/**
 * @brief Functor to read a vector of binaries from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
class parquet_field_binary_list
  : public parquet_field_list<std::vector<uint8_t>, FieldType::BINARY> {
 public:
  parquet_field_binary_list(int f, std::vector<std::vector<uint8_t>>& v) : parquet_field_list(f, v)
  {
    auto const read_value = [&val = v](uint32_t i, CompactProtocolReader* cpr) {
      auto const l = cpr->get_u32();
      CUDF_EXPECTS(std::cmp_less_equal(l, cpr->m_end - cpr->m_cur), "binary length mismatch");
      CUDF_EXPECTS(i < val.size(), "Index out of bounds");

      val[i].assign(cpr->m_cur, cpr->m_cur + l);
      cpr->m_cur += l;
    };
    bind_read_func(read_value);
  }
};

/**
 * @brief Functor to read a struct from CompactProtocolReader
 *
 * @return true if the field was read, false if skipped on a type mismatch
 */
class parquet_field_struct_blob : public parquet_field {
  std::vector<uint8_t>& val;

 public:
  parquet_field_struct_blob(int f, std::vector<uint8_t>& v) : parquet_field(f), val(v) {}
  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    if (not cpr->check_field_type(field_type, FieldType::STRUCT)) { return false; }
    uint8_t const* const start = cpr->m_cur;
    cpr->skip_struct_field(field_type);
    if (cpr->m_cur > start) { val.assign(start, cpr->m_cur - 1); }
    return true;
  }
};

/**
 * @brief functor to wrap functors for optional fields
 */
template <typename T, typename FieldFunctor, typename OptionalType = std::optional<T>>
class parquet_field_optional : public parquet_field {
  OptionalType& val;

 public:
  parquet_field_optional(int f, OptionalType& v) : parquet_field(f), val(v) {}

  inline bool operator()(CompactProtocolReader* cpr, int field_type)
  {
    T v;
    if (not FieldFunctor{field(), v}(cpr, field_type)) { return false; }
    val = v;
    return true;
  }
};

/**
 * @brief Skips a single value of the given wire type, matching Thrift TCompactProtocol::skip
 *
 * @param[in] t Compact-protocol field type of the value to skip
 * @param[in] depth Current struct/container nesting level
 */
void CompactProtocolReader::skip_struct_field(int t, int depth)
{
  auto const t_enum = static_cast<FieldType>(t);
  switch (t_enum) {
    case FieldType::BOOLEAN_TRUE:
    case FieldType::BOOLEAN_FALSE: break;  // a bool struct-field value lives in the type nibble
    case FieldType::I16:
    case FieldType::I32:
    case FieldType::I64: get_u64(); break;
    case FieldType::I8: skip_bytes(1); break;
    case FieldType::DOUBLE: skip_bytes(8); break;
    case FieldType::BINARY: skip_bytes(get_u32()); break;
    case FieldType::UUID: skip_bytes(16); break;
    case FieldType::LIST:
    case FieldType::SET: {
      auto const [element_type, n] = get_listh();
      // Reject a count that cannot fit the remaining bytes (each element >= 1 byte), bounding the
      // skip loop against a malformed size prefix.
      CUDF_EXPECTS(std::cmp_less_equal(n, m_end - m_cur),
                   "Parquet footer list size exceeds remaining buffer");
      CUDF_EXPECTS(depth <= 10, "struct nesting too deep");
      auto const et = static_cast<FieldType>(element_type);
      if (et == FieldType::BOOLEAN_TRUE || et == FieldType::BOOLEAN_FALSE) {
        // Bool list/set elements are one byte each (unlike a bool struct field, whose value is in
        // the type nibble), so skip them directly instead of recursing to the zero-byte bool case.
        skip_bytes(n);
      } else {
        for (uint32_t i = 0; i < n; i++) {
          skip_struct_field(element_type, depth + 1);
        }
      }
    } break;
    case FieldType::MAP: {
      // Compact map: a varint size, then (only when non-empty) one byte packing the key type (high
      // nibble) and value type (low nibble), followed by size key/value pairs.
      auto const n = get_u32();
      if (n != 0) {
        CUDF_EXPECTS(std::cmp_less_equal(n, m_end - m_cur),
                     "Parquet footer map size exceeds remaining buffer");
        CUDF_EXPECTS(depth <= 10, "struct nesting too deep");
        int const kv_types = getb();
        int const key_type = kv_types >> 4;
        int const val_type = kv_types & 0xf;
        // A bool map key/value is one byte each (unlike a bool struct field, whose value is in the
        // type nibble), so skip it directly instead of recursing to the zero-byte bool case.
        auto const skip_map_elem = [&](int elem_type) {
          auto const e = static_cast<FieldType>(elem_type);
          if (e == FieldType::BOOLEAN_TRUE || e == FieldType::BOOLEAN_FALSE) {
            skip_bytes(1);
          } else {
            skip_struct_field(elem_type, depth + 1);
          }
        };
        for (uint32_t i = 0; i < n; i++) {
          skip_map_elem(key_type);
          skip_map_elem(val_type);
        }
      }
    } break;
    case FieldType::STRUCT:
      for (;;) {
        int const c          = getb();
        int const field_type = c & 0xf;
        if (c == 0) { break; }               // end of struct
        if ((c & 0xf0) == 0) { get_i16(); }  // field id is not a delta
        CUDF_EXPECTS(depth <= 10, "struct nesting too deep");
        skip_struct_field(field_type, depth + 1);
      }
      break;
    default: CUDF_FAIL("Unsupported Parquet Thrift field type encountered while skipping");
  }
}

bool CompactProtocolReader::check_field_type(int type, FieldType expected)
{
  if (type == static_cast<int>(expected)) { return true; }
  if (should_throw_on_type_mismatch()) { assert_field_type(type, expected); }
  skip_struct_field(type);
  return false;
}

bool CompactProtocolReader::check_list_element_type(int type, FieldType expected, uint32_t count)
{
  auto const is_bool = [](FieldType t) {
    return t == FieldType::BOOLEAN_TRUE || t == FieldType::BOOLEAN_FALSE;
  };
  auto const et = static_cast<FieldType>(type);
  // A bool list may carry either nibble (true/false); both are valid element types.
  if (type == static_cast<int>(expected) || (is_bool(expected) && is_bool(et))) { return true; }
  if (should_throw_on_type_mismatch()) {
    if (is_bool(expected)) {
      assert_bool_field_type(type);
    } else {
      assert_field_type(type, expected);
    }
  }
  // Discard the element payloads so the walk resumes at the next field. Bool elements are one
  // byte each (the value lives in the type nibble); callers bound `count` against the buffer.
  if (is_bool(et)) {
    skip_bytes(count);
  } else {
    for (uint32_t i = 0; i < count; ++i) {
      skip_struct_field(type);
    }
  }
  return false;
}

template <typename RowGroupsField>
void CompactProtocolReader::read_file_metadata(FileMetaData* f, RowGroupsField row_groups_field)
{
  using optional_list_column_order =
    parquet_field_optional<std::vector<ColumnOrder>, parquet_field_struct_list<ColumnOrder>>;
  using optional_struct_blob =
    parquet_field_optional<std::vector<uint8_t>, parquet_field_struct_blob>;
  using optional_binary = parquet_field_optional<std::vector<uint8_t>, parquet_field_binary>;
  auto op               = std::make_tuple(parquet_field_int32(1, f->version),
                            parquet_field_struct_list(2, f->schema),
                            parquet_field_int64(3, f->num_rows),
                            std::move(row_groups_field),
                            parquet_field_struct_list(5, f->key_value_metadata),
                            parquet_field_string(6, f->created_by),
                            optional_list_column_order(7, f->column_orders),
                            optional_struct_blob(8, f->encryption_algorithm),
                            optional_binary(9, f->footer_signing_key_metadata));
  function_builder(this, op);
  // Callers can check overread() after schema-init so a footer with no schema reports the
  // specific "Cannot initialize schema", not this generic error.
}

void CompactProtocolReader::read(FileMetaData* f)
{
  read_file_metadata(f, parquet_field_struct_list(4, f->row_groups));
}

void CompactProtocolReader::read(SchemaElement* s)
{
  using optional_converted_type =
    parquet_field_optional<ConvertedType, parquet_field_enum<ConvertedType>>;
  auto op =
    std::make_tuple(parquet_field_enum<Type>(1, s->type),
                    parquet_field_int32(2, s->type_length),
                    parquet_field_enum<FieldRepetitionType>(3, s->repetition_type),
                    parquet_field_string(4, s->name),
                    parquet_field_int32(5, s->num_children),
                    optional_converted_type(6, s->converted_type),
                    parquet_field_int32(7, s->decimal_scale),
                    parquet_field_int32(8, s->decimal_precision),
                    parquet_field_optional<int32_t, parquet_field_int32>(9, s->field_id),
                    parquet_field_logical_type(10, s->logical_type, s->logical_type_payload));
  function_builder(this, op);
}

void CompactProtocolReader::read(LogicalType* l, LogicalTypePayload* payload)
{
  // Clear the destination so only the parsed union member is engaged.
  *payload = LogicalTypePayload{};
  auto op  = std::make_tuple(
    parquet_field_union_enumerator(1, l->type),
    parquet_field_union_enumerator(2, l->type),
    parquet_field_union_enumerator(3, l->type),
    parquet_field_union_enumerator(4, l->type),
    parquet_field_union_struct<LogicalType::Type, DecimalType>(5, l->type, l->decimal_type),
    parquet_field_union_enumerator(6, l->type),
    parquet_field_union_struct<LogicalType::Type, TimeType>(7, l->type, l->time_type),
    parquet_field_union_struct<LogicalType::Type, TimestampType>(8, l->type, l->timestamp_type),
    parquet_field_union_struct<LogicalType::Type, IntType>(10, l->type, l->int_type),
    parquet_field_union_enumerator(11, l->type),
    parquet_field_union_enumerator(12, l->type),
    parquet_field_union_enumerator(13, l->type),
    parquet_field_union_enumerator(14, l->type),  // UUID
    parquet_field_union_enumerator(15, l->type),  // FLOAT16
    parquet_field_union_struct<LogicalType::Type, VariantType>(16, l->type, payload->variant_type),
    parquet_field_union_struct<LogicalType::Type, GeometryType>(
      17, l->type, payload->geometry_type),
    parquet_field_union_struct<LogicalType::Type, GeographyType>(
      18, l->type, payload->geography_type),
    parquet_field_union_enumerator(19, l->type));  // FILE
  function_builder(this, op);
}

void CompactProtocolReader::read(GeometryType* g)
{
  // Thrift field 1 is the `crs` string; only its PRESENCE is recorded, mirroring GeographyType.
  std::optional<std::string> crs;
  auto op = std::make_tuple(parquet_field_optional<std::string, parquet_field_string>(1, crs));
  function_builder(this, op);
  g->has_crs = crs.has_value();
}

void CompactProtocolReader::read(VariantType* v)
{
  using optional_version = parquet_field_optional<int8_t, parquet_field_int8>;
  auto op                = std::make_tuple(optional_version(1, v->specification_version));
  function_builder(this, op);
}

void CompactProtocolReader::read(GeographyType* g)
{
  using optional_algorithm = parquet_field_optional<EdgeInterpolationAlgorithm,
                                                    parquet_field_enum<EdgeInterpolationAlgorithm>>;
  // Thrift field 1 is the `crs` string; only its PRESENCE is recorded while `algorithm` (id 2)
  // round-trips.
  std::optional<std::string> crs;
  auto op = std::make_tuple(parquet_field_optional<std::string, parquet_field_string>(1, crs),
                            optional_algorithm(2, g->algorithm));
  function_builder(this, op);
  g->has_crs = crs.has_value();
}

void CompactProtocolReader::read(DecimalType* d)
{
  auto op = std::make_tuple(parquet_field_int32(1, d->scale), parquet_field_int32(2, d->precision));
  function_builder(this, op);
}

void CompactProtocolReader::read(TimeType* t)
{
  auto op =
    std::make_tuple(parquet_field_bool(1, t->isAdjustedToUTC), parquet_field_struct(2, t->unit));
  function_builder(this, op);
}

void CompactProtocolReader::read(TimestampType* t)
{
  auto op =
    std::make_tuple(parquet_field_bool(1, t->isAdjustedToUTC), parquet_field_struct(2, t->unit));
  function_builder(this, op);
}

void CompactProtocolReader::read(TimeUnit* u)
{
  auto op = std::make_tuple(parquet_field_union_enumerator(1, u->type),
                            parquet_field_union_enumerator(2, u->type),
                            parquet_field_union_enumerator(3, u->type));
  function_builder(this, op);
}

void CompactProtocolReader::read(IntType* i)
{
  auto op = std::make_tuple(parquet_field_int8(1, i->bitWidth), parquet_field_bool(2, i->isSigned));
  function_builder(this, op);
}

void CompactProtocolReader::read(RowGroup* r)
{
  if (m_chunk_offsets != nullptr) {
    read_row_group(r, parquet_field_struct_list_offsets(1, *m_chunk_offsets, m_offsets_base));
  } else {
    read_row_group(r, parquet_field_struct_list(1, r->columns, m_column_mask));
  }
}

template <typename ColumnsField>
void CompactProtocolReader::read_row_group(RowGroup* r, ColumnsField columns_field)
{
  using optional_i16 = parquet_field_optional<int16_t, parquet_field_int16>;
  using optional_i64 = parquet_field_optional<int64_t, parquet_field_int64>;
  using optional_list_sorting_column =
    parquet_field_optional<std::vector<SortingColumn>, parquet_field_struct_list<SortingColumn>>;

  auto op = std::make_tuple(std::move(columns_field),
                            parquet_field_int64(2, r->total_byte_size),
                            parquet_field_int64(3, r->num_rows),
                            optional_list_sorting_column(4, r->sorting_columns),
                            optional_i64(5, r->file_offset),
                            optional_i64(6, r->total_compressed_size),
                            optional_i16(7, r->ordinal));
  function_builder(this, op);
}

void CompactProtocolReader::read(ColumnChunk* c)
{
  using optional_struct_blob =
    parquet_field_optional<std::vector<uint8_t>, parquet_field_struct_blob>;
  using optional_binary = parquet_field_optional<std::vector<uint8_t>, parquet_field_binary>;
  auto op               = std::make_tuple(parquet_field_string(1, c->file_path),
                            parquet_field_int64(2, c->file_offset),
                            parquet_field_struct(3, c->meta_data),
                            parquet_field_int64(4, c->offset_index_offset),
                            parquet_field_int32(5, c->offset_index_length),
                            parquet_field_int64(6, c->column_index_offset),
                            parquet_field_int32(7, c->column_index_length),
                            optional_struct_blob(8, c->crypto_metadata),
                            optional_binary(9, c->encrypted_column_metadata));
  function_builder(this, op);
}

void CompactProtocolReader::read(BloomFilterAlgorithm* alg)
{
  auto op = std::make_tuple(parquet_field_union_enumerator(1, alg->algorithm));
  function_builder(this, op);
}

void CompactProtocolReader::read(BloomFilterHash* hash)
{
  auto op = std::make_tuple(parquet_field_union_enumerator(1, hash->hash));
  function_builder(this, op);
}

void CompactProtocolReader::read(BloomFilterCompression* comp)
{
  auto op = std::make_tuple(parquet_field_union_enumerator(1, comp->compression));
  function_builder(this, op);
}

void CompactProtocolReader::read(BloomFilterHeader* bf)
{
  auto op = std::make_tuple(parquet_field_int32(1, bf->num_bytes),
                            parquet_field_struct(2, bf->algorithm),
                            parquet_field_struct(3, bf->hash),
                            parquet_field_struct(4, bf->compression));
  function_builder(this, op);
}

void CompactProtocolReader::read(ColumnChunkMetaData* c)
{
  using optional_size_statistics =
    parquet_field_optional<SizeStatistics, parquet_field_struct<SizeStatistics>>;
  using optional_list_enc_stats =
    parquet_field_optional<std::vector<PageEncodingStats>,
                           parquet_field_struct_list<PageEncodingStats>>;
  using optional_i64 = parquet_field_optional<int64_t, parquet_field_int64>;
  using optional_i32 = parquet_field_optional<int32_t, parquet_field_int32>;
  auto op            = std::make_tuple(
    parquet_field_enum<Type>(1, c->type),
    parquet_field_enum_list(2, c->encodings),
    parquet_field_string_list(3, c->path_in_schema),
    parquet_field_enum<Compression>(4, c->codec),
    parquet_field_int64(5, c->num_values),
    parquet_field_int64(6, c->total_uncompressed_size),
    parquet_field_int64(7, c->total_compressed_size),
    parquet_field_int64(9, c->data_page_offset),
    parquet_field_int64(10, c->index_page_offset),
    parquet_field_int64(11, c->dictionary_page_offset),
    parquet_field_if(m_chunk_policy.statistics, parquet_field_struct(12, c->statistics)),
    parquet_field_if(m_chunk_policy.encoding_stats, optional_list_enc_stats(13, c->encoding_stats)),
    optional_i64(14, c->bloom_filter_offset),
    optional_i32(15, c->bloom_filter_length),
    parquet_field_if(m_chunk_policy.size_statistics,
                     optional_size_statistics(16, c->size_statistics)));
  function_builder(this, op);
}

void CompactProtocolReader::read(PageHeader* p)
{
  auto op = std::make_tuple(parquet_field_enum<PageType>(1, p->type),
                            parquet_field_int32(2, p->uncompressed_page_size),
                            parquet_field_int32(3, p->compressed_page_size),
                            parquet_field_struct(5, p->data_page_header),
                            parquet_field_struct(7, p->dictionary_page_header),
                            parquet_field_struct(8, p->data_page_header_v2));
  function_builder(this, op);
}

void CompactProtocolReader::read(DataPageHeader* d)
{
  auto op = std::make_tuple(parquet_field_int32(1, d->num_values),
                            parquet_field_enum<Encoding>(2, d->encoding),
                            parquet_field_enum<Encoding>(3, d->definition_level_encoding),
                            parquet_field_enum<Encoding>(4, d->repetition_level_encoding));
  function_builder(this, op);
}

void CompactProtocolReader::read(DictionaryPageHeader* d)
{
  auto op = std::make_tuple(parquet_field_int32(1, d->num_values),
                            parquet_field_enum<Encoding>(2, d->encoding));
  function_builder(this, op);
}

void CompactProtocolReader::read(DataPageHeaderV2* d)
{
  auto op = std::make_tuple(parquet_field_int32(1, d->num_values),
                            parquet_field_int32(2, d->num_nulls),
                            parquet_field_int32(3, d->num_rows),
                            parquet_field_enum<Encoding>(4, d->encoding),
                            parquet_field_int32(5, d->definition_levels_byte_length),
                            parquet_field_int32(6, d->repetition_levels_byte_length),
                            parquet_field_bool(7, d->is_compressed));
  function_builder(this, op);
}

void CompactProtocolReader::read(KeyValue* k)
{
  auto op = std::make_tuple(parquet_field_string(1, k->key),
                            parquet_field_optional<std::string, parquet_field_string>(2, k->value));
  function_builder(this, op);
}

void CompactProtocolReader::read(PageLocation* p)
{
  auto op = std::make_tuple(parquet_field_int64(1, p->offset),
                            parquet_field_int32(2, p->compressed_page_size),
                            parquet_field_int64(3, p->first_row_index));
  function_builder(this, op);
}

void CompactProtocolReader::read(OffsetIndex* o)
{
  using optional_list_i64 = parquet_field_optional<std::vector<int64_t>, parquet_field_int64_list>;

  auto op = std::make_tuple(parquet_field_struct_list(1, o->page_locations),
                            optional_list_i64(2, o->unencoded_byte_array_data_bytes));
  function_builder(this, op);
}

void CompactProtocolReader::read(SizeStatistics* s)
{
  using optional_i64      = parquet_field_optional<int64_t, parquet_field_int64>;
  using optional_list_i64 = parquet_field_optional<std::vector<int64_t>, parquet_field_int64_list>;

  auto op = std::make_tuple(optional_i64(1, s->unencoded_byte_array_data_bytes),
                            optional_list_i64(2, s->repetition_level_histogram),
                            optional_list_i64(3, s->definition_level_histogram));
  function_builder(this, op);
}

void CompactProtocolReader::read(ColumnIndex* c)
{
  using optional_list_i64 = parquet_field_optional<std::vector<int64_t>, parquet_field_int64_list>;

  auto op = std::make_tuple(parquet_field_bool_list(1, c->null_pages),
                            parquet_field_binary_list(2, c->min_values),
                            parquet_field_binary_list(3, c->max_values),
                            parquet_field_enum<BoundaryOrder>(4, c->boundary_order),
                            optional_list_i64(5, c->null_counts),
                            optional_list_i64(6, c->repetition_level_histogram),
                            optional_list_i64(7, c->definition_level_histogram),
                            optional_list_i64(8, c->nan_counts));
  function_builder(this, op);
}

void CompactProtocolReader::read(Statistics* s)
{
  using optional_binary = parquet_field_optional<std::vector<uint8_t>, parquet_field_binary>;
  using optional_int64  = parquet_field_optional<int64_t, parquet_field_int64>;
  using optional_bool   = parquet_field_optional<bool, parquet_field_bool>;

  auto op = std::make_tuple(optional_binary(1, s->max),
                            optional_binary(2, s->min),
                            optional_int64(3, s->null_count),
                            optional_int64(4, s->distinct_count),
                            optional_binary(5, s->max_value),
                            optional_binary(6, s->min_value),
                            optional_bool(7, s->is_max_value_exact),
                            optional_bool(8, s->is_min_value_exact),
                            optional_int64(9, s->nan_count));
  function_builder(this, op);
}

void CompactProtocolReader::read(ColumnOrder* c)
{
  auto op = std::make_tuple(parquet_field_union_enumerator(1, c->type),
                            parquet_field_union_enumerator(2, c->type),
                            parquet_field_union_enumerator(3, c->type));
  function_builder(this, op);
}

void CompactProtocolReader::read(PageEncodingStats* s)
{
  auto op = std::make_tuple(parquet_field_enum<PageType>(1, s->page_type),
                            parquet_field_enum<Encoding>(2, s->encoding),
                            parquet_field_int32(3, s->count));
  function_builder(this, op);
}

void CompactProtocolReader::read(SortingColumn* s)
{
  auto op = std::make_tuple(parquet_field_int32(1, s->column_idx),
                            parquet_field_bool(2, s->descending),
                            parquet_field_bool(3, s->nulls_first));
  function_builder(this, op);
}

namespace {

/**
 * @brief Whether a non-root schema element is a leaf column, i.e. one with column chunks
 *
 * Groups without children (empty structs) have no column chunks.
 */
[[nodiscard]] bool is_leaf_column(SchemaElement const& element)
{
  return element.num_children == 0 and element.type != Type::UNDEFINED;
}

/**
 * @brief Returns the schema index of each leaf column, in leaf (schema pre-order) order
 */
[[nodiscard]] std::vector<int> compute_leaf_schema_indices(std::vector<SchemaElement> const& schema)
{
  std::vector<int> leaves;
  for (std::size_t i = 1; i < schema.size(); ++i) {
    if (is_leaf_column(schema[i])) { leaves.push_back(static_cast<int>(i)); }
  }
  return leaves;
}

/**
 * @brief Whether `path` is the path from the root to schema element `schema_idx`
 *
 * Requires an initialized `parent_idx` for each schema element. An empty path matches any element.
 */
[[nodiscard]] bool path_matches_schema(std::vector<SchemaElement> const& schema,
                                       std::vector<std::string> const& path,
                                       int schema_idx)
{
  if (path.empty()) { return true; }
  auto idx = schema_idx;
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    if (idx <= 0 or schema[idx].name != *it) { return false; }
    idx = schema[idx].parent_idx;
  }
  return idx == 0;
}

/**
 * @brief Searches the schema for the element named by `path`
 *
 * Each path component is searched among the children of the previous one, starting after
 * `current_schema_index` and wrapping around. On success, `current_schema_index` is updated.
 *
 * @return The schema index, or -1 if not found
 */
[[nodiscard]] int find_schema_index_by_path(std::vector<SchemaElement> const& schema,
                                            std::vector<std::string> const& path,
                                            int& current_schema_index)
{
  int parent = 0;  // root of schema
  for (auto const& name : path) {
    auto const matches = [&](auto const& e) { return e.parent_idx == parent && e.name == name; };
    auto const mid     = schema.cbegin() + current_schema_index + 1;
    auto it            = std::find_if(mid, schema.cend(), matches);
    if (it == schema.cend()) {
      it = std::find_if(schema.cbegin(), mid, matches);
      if (it == mid) { return -1; }
    }
    current_schema_index = static_cast<int>(std::distance(schema.cbegin(), it));
    parent               = current_schema_index;
  }
  return current_schema_index;
}

}  // namespace

/**
 * @brief Constructs the schema from the file-level metadata
 *
 * Column chunks are mapped to leaf columns by position, as the Parquet spec requires. A chunk whose
 * `path_in_schema` does not name its positional leaf column is mapped by searching for its path.
 *
 * @param[in] md File metadata that was previously parsed
 *
 * @return True if schema constructed completely, false otherwise
 */
bool CompactProtocolReader::InitSchema(FileMetaData* md)
{
  if (static_cast<std::size_t>(WalkSchema(md)) != md->schema.size()) { return false; }

  auto const leaves = compute_leaf_schema_indices(md->schema);
  for (auto& row_group : md->row_groups) {
    auto const is_positional = row_group.columns.size() == leaves.size();
    int current_schema_index = 0;
    for (std::size_t c = 0; c < row_group.columns.size(); ++c) {
      auto& column     = row_group.columns[c];
      auto const& path = column.meta_data.path_in_schema;
      if (is_positional and path_matches_schema(md->schema, path, leaves[c])) {
        column.schema_idx    = leaves[c];
        current_schema_index = leaves[c];
        continue;
      }
      if (path.empty()) { continue; }
      auto const schema_idx = find_schema_index_by_path(md->schema, path, current_schema_index);
      if (schema_idx < 0) { return false; }
      column.schema_idx = schema_idx;
    }
  }

  return true;
}

/**
 * @brief Populates each node in the schema tree
 *
 * @param[out] md File metadata
 * @param[in] idx Current node index
 * @param[in] parent_idx Parent node index
 * @param[in] max_def_level Max definition level
 * @param[in] max_rep_level Max repetition level
 *
 * @return The node index that was populated
 */
int CompactProtocolReader::WalkSchema(
  FileMetaData* md, int idx, int parent_idx, int max_def_level, int max_rep_level)
{
  if (idx >= 0 && (size_t)idx < md->schema.size()) {
    SchemaElement* e = &md->schema[idx];
    if (e->repetition_type == FieldRepetitionType::OPTIONAL) {
      ++max_def_level;
    } else if (e->repetition_type == FieldRepetitionType::REPEATED) {
      ++max_def_level;
      ++max_rep_level;
    }
    e->max_definition_level = max_def_level;
    e->max_repetition_level = max_rep_level;
    e->parent_idx           = parent_idx;

    parent_idx = idx;
    ++idx;
    if (e->num_children > 0) {
      for (int i = 0; i < e->num_children; i++) {
        e->children_idx.push_back(idx);
        int const idx_old = idx;
        idx               = WalkSchema(md, idx, parent_idx, max_def_level, max_rep_level);
        if (idx <= idx_old) { break; }  // Error
      }
    }
    return idx;
  } else {
    // Error
    return -1;
  }
}

void decode_footer_bytes(std::span<uint8_t const> footer_bytes,
                         FileMetaData* metadata,
                         experimental::thrift_mismatch_policy mode)
{
  CompactProtocolReader reader{footer_bytes.data(), footer_bytes.size(), mode};
  reader.read(metadata);
  CUDF_EXPECTS(not reader.overread(), overread_message);
}

void decode_footer_and_init_schema(std::span<uint8_t const> footer_bytes, FileMetaData* metadata)
{
  CompactProtocolReader reader{footer_bytes.data(), footer_bytes.size()};
  reader.read(metadata);
  // Schema-init failure (e.g. empty input) reports the specific message, not generic overread.
  auto const is_schema_initialized = reader.InitSchema(metadata);
  CUDF_EXPECTS(is_schema_initialized, cannot_init_schema_message);
  CUDF_EXPECTS(not reader.overread(), overread_message);
}

namespace {

/**
 * @brief Row group reader that skips the column chunks, recording their byte offsets instead
 */
struct read_row_group_chunk_offsets {
  std::vector<std::vector<uint64_t>>* chunk_offsets;
  uint8_t const* base;

  void prepare(uint32_t num_row_groups) const { chunk_offsets->resize(num_row_groups); }

  void operator()(CompactProtocolReader& cpr, RowGroup& row_group, uint32_t index) const
  {
    cpr.set_column_chunk_offsets(&(*chunk_offsets)[index], base);
    cpr.read(&row_group);
    cpr.set_column_chunk_offsets(nullptr, nullptr);
  }
};

}  // namespace

void decode_footer_deferring_column_chunks(std::span<uint8_t const> footer_bytes,
                                           FileMetaData* metadata,
                                           std::vector<std::vector<uint64_t>>* chunk_offsets,
                                           std::vector<int>* leaf_schema_indices)
{
  CompactProtocolReader reader{footer_bytes.data(), footer_bytes.size()};
  chunk_offsets->clear();
  reader.read_file_metadata(
    metadata,
    parquet_field_struct_list(4,
                              metadata->row_groups,
                              nullptr,
                              read_row_group_chunk_offsets{chunk_offsets, footer_bytes.data()}));
  // Row groups without a `columns` field have no column chunks
  chunk_offsets->resize(metadata->row_groups.size());
  // Schema-init failure (e.g. empty input) reports the specific message, not generic overread.
  auto const is_schema_initialized = reader.InitSchema(metadata);
  CUDF_EXPECTS(is_schema_initialized, cannot_init_schema_message);
  CUDF_EXPECTS(not reader.overread(), overread_message);
  *leaf_schema_indices = compute_leaf_schema_indices(metadata->schema);
}

bool decode_column_chunks(std::span<uint8_t const> footer_bytes,
                          std::span<std::vector<uint64_t> const> chunk_offsets,
                          std::span<int const> leaf_schema_indices,
                          std::span<cudf::size_type const> row_groups,
                          std::span<cudf::size_type const> leaf_columns,
                          FileMetaData* metadata)
{
  auto const num_leaves = leaf_schema_indices.size();
  for (auto const row_group : row_groups) {
    if (chunk_offsets[row_group].size() != num_leaves + 1) { return false; }
  }
  for (auto const row_group : row_groups) {
    auto& columns = metadata->row_groups[row_group].columns;
    columns.clear();
    columns.resize(leaf_columns.size());
  }

  auto const num_selected_leaves = leaf_columns.size();
  auto const num_chunks          = row_groups.size() * num_selected_leaves;
  if (num_chunks == 0) { return true; }

  // Decodes the selected chunks [begin, end), in row-group-major order. Returns false on a path
  // mismatch. Only leaf names are compared: the schema may already have been sanitized, which
  // re-parents some elements.
  auto const decode_chunks = [&](std::size_t begin, std::size_t end) {
    CompactProtocolReader reader;
    for (auto i = begin; i < end; ++i) {
      auto const row_group = row_groups[i / num_selected_leaves];
      auto const column    = i % num_selected_leaves;
      auto const leaf      = leaf_columns[column];
      auto const& offsets  = chunk_offsets[row_group];
      reader.init(footer_bytes.data() + offsets[leaf], offsets[leaf + 1] - offsets[leaf]);
      auto& chunk = metadata->row_groups[row_group].columns[column];
      reader.read(&chunk);
      CUDF_EXPECTS(not reader.overread(), overread_message);
      auto const schema_idx = leaf_schema_indices[leaf];
      auto const& path      = chunk.meta_data.path_in_schema;
      if (not path.empty() and path.back() != metadata->schema[schema_idx].name) { return false; }
      chunk.schema_idx = schema_idx;
    }
    return true;
  };

  constexpr std::size_t parallel_threshold = 512;
  if (num_chunks < parallel_threshold) { return decode_chunks(0, num_chunks); }

  auto const num_tasks = std::min<std::size_t>(
    num_chunks / (parallel_threshold / 2), cudf::detail::host_worker_pool().get_thread_count() * 2);
  auto const chunks_per_task = num_chunks / num_tasks;
  auto const remainder       = num_chunks % num_tasks;

  std::vector<std::future<bool>> tasks;
  cudf::detail::futures_wait_guard tasks_guard{tasks};
  tasks.reserve(num_tasks);
  std::size_t begin = 0;
  for (std::size_t task_id = 0; task_id < num_tasks; ++task_id) {
    auto const end = begin + chunks_per_task + (task_id < remainder ? 1 : 0);
    tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(
      [&decode_chunks, begin, end] { return decode_chunks(begin, end); }));
    begin = end;
  }
  bool all_matched = true;
  for (auto& task : tasks) {
    all_matched = task.get() and all_matched;
  }
  return all_matched;
}

namespace {

/**
 * @brief Returns the number of leaf columns in a footer schema (element 0 is the root)
 */
[[nodiscard]] cudf::size_type count_leaf_columns(std::vector<SchemaElement> const& schema)
{
  if (schema.empty()) { return 0; }
  return static_cast<cudf::size_type>(std::count_if(
    schema.begin() + 1, schema.end(), [](auto const& e) { return is_leaf_column(e); }));
}

/**
 * @brief Sorts and deduplicates a selection of indices, checking for negative indices
 */
[[nodiscard]] std::optional<std::vector<cudf::size_type>> normalize_selection(
  std::optional<std::vector<cudf::size_type>> const& selection, std::string const& name)
{
  if (not selection.has_value()) { return std::nullopt; }
  auto indices = *selection;
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  CUDF_EXPECTS(indices.empty() or indices.front() >= 0,
               "Negative " + name + " index in Parquet footer selection",
               std::out_of_range);
  return indices;
}

/**
 * @brief Checks that the largest selected index is below `count`
 */
void check_selection_range(std::optional<std::vector<cudf::size_type>> const& selection,
                           cudf::size_type count,
                           std::string const& name)
{
  CUDF_EXPECTS(not selection.has_value() or selection->empty() or selection->back() < count,
               "Out of range " + name + " index in Parquet footer selection",
               std::out_of_range);
}

}  // namespace

experimental::partial_file_metadata decode_footer_bytes(
  std::span<uint8_t const> footer_bytes, experimental::footer_decode_options const& options)
{
  auto const row_groups   = normalize_selection(options.row_groups, "row group");
  auto const leaf_columns = normalize_selection(options.leaf_columns, "leaf column");

  experimental::partial_file_metadata result;
  auto& metadata = result.metadata;

  CompactProtocolReader reader{footer_bytes.data(), footer_bytes.size(), options.mismatch_policy};
  reader.set_column_chunk_policy({.statistics      = options.decode_statistics,
                                  .encoding_stats  = options.decode_encoding_stats,
                                  .size_statistics = options.decode_size_statistics});

  if (not row_groups.has_value() and not leaf_columns.has_value()) {
    reader.read(&metadata);
  } else {
    // Column masks indexed by leaf column, built from the schema once the row groups are reached
    std::vector<bool> selected_columns_mask;
    std::vector<bool> no_columns_mask;
    auto const prepare_masks = [&] {
      CUDF_EXPECTS(not metadata.schema.empty(),
                   "Selective Parquet footer decode requires the schema to precede the row groups");
      auto const num_leaves = count_leaf_columns(metadata.schema);
      check_selection_range(leaf_columns, num_leaves, "leaf column");
      no_columns_mask.assign(num_leaves, false);
      if (leaf_columns.has_value()) {
        selected_columns_mask.assign(num_leaves, false);
        for (auto const col : *leaf_columns) {
          selected_columns_mask[col] = true;
        }
      }
    };
    // Decodes the scalar fields of every row group and the selected column chunks of the selected
    // row groups. Runs on parallel sub-readers for 512 or more row groups.
    auto const read_row_group = [&](CompactProtocolReader& cpr, RowGroup& row_group, uint32_t idx) {
      auto const is_selected =
        not row_groups.has_value() or
        std::binary_search(
          row_groups->begin(), row_groups->end(), static_cast<cudf::size_type>(idx));
      cpr.set_column_mask(not is_selected            ? &no_columns_mask
                          : leaf_columns.has_value() ? &selected_columns_mask
                                                     : nullptr);
      cpr.read(&row_group);
      cpr.set_column_mask(nullptr);
    };
    reader.read_file_metadata(
      &metadata,
      parquet_field_prepared(
        parquet_field_struct_list(4, metadata.row_groups, nullptr, read_row_group), prepare_masks));
  }
  CUDF_EXPECTS(not reader.overread(), overread_message);

  auto const num_row_groups = static_cast<cudf::size_type>(metadata.row_groups.size());
  auto const num_leaves     = count_leaf_columns(metadata.schema);
  check_selection_range(row_groups, num_row_groups, "row group");
  check_selection_range(leaf_columns, num_leaves, "leaf column");

  auto const all_indices = [](cudf::size_type count) {
    std::vector<cudf::size_type> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  };
  result.row_groups   = row_groups.value_or(all_indices(num_row_groups));
  result.leaf_columns = leaf_columns.value_or(all_indices(num_leaves));
  return result;
}

}  // namespace cudf::io::parquet::detail
