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
#include <tuple>
#include <utility>

namespace cudf::io::parquet::detail {
namespace {

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
  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_field_type(field_type, FieldType::LIST);
    auto const [t, n] = cpr->get_listh();
    if constexpr (cuda::std::is_same_v<T, bool>) {
      assert_bool_field_type(t);
    } else {
      assert_field_type(t, EXPECTED_ELEM_TYPE);
    }
    // Reject a declared count that cannot fit the remaining bytes (each element is >= 1 byte),
    // guarding against a malformed/hostile size prefix requesting an enormous allocation.
    CUDF_EXPECTS(std::cmp_less_equal(n, cpr->m_end - cpr->m_cur),
                 "Parquet footer list size exceeds remaining buffer");
    val.resize(n);
    for (uint32_t i = 0; i < n; i++) {
      _read_value(i, cpr);
    }
  }
};

/**
 * @brief Functor to set value to bool read from CompactProtocolReader
 *
 * bool doesn't actually encode a value, we just use the field type to indicate true/false
 *
 * @return True if field type is not bool
 */
class parquet_field_bool : public parquet_field {
  bool& val;

 public:
  parquet_field_bool(int f, bool& v) : parquet_field(f), val(v) {}

  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_bool_field_type(field_type);
    val = field_type == static_cast<int>(FieldType::BOOLEAN_TRUE);
  }
};

/**
 * @brief Functor to read a vector of booleans from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * bool fails
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
 * @return True if there is a type mismatch
 */
template <typename T, FieldType EXPECTED_TYPE>
class parquet_field_int : public parquet_field {
  static constexpr bool is_byte = std::is_same_v<T, int8_t>;

  T& val;

 public:
  parquet_field_int(int f, T& v) : parquet_field(f), val(v) {}

  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_field_type(field_type, EXPECTED_TYPE);
    if constexpr (is_byte) {
      val = cpr->getb();
    } else {
      val = cpr->get_zigzag<T>();
    }
  }
};

using parquet_field_int8  = parquet_field_int<int8_t, FieldType::I8>;
using parquet_field_int16 = parquet_field_int<int16_t, FieldType::I16>;
using parquet_field_int32 = parquet_field_int<int32_t, FieldType::I32>;
using parquet_field_int64 = parquet_field_int<int64_t, FieldType::I64>;

/**
 * @brief Functor to read a vector of integers from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading an
 * integer fails
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
 * @return True if field type mismatches or if size of string exceeds bounds
 * of the CompactProtocolReader
 */
class parquet_field_string : public parquet_field {
  std::string& val;

 public:
  parquet_field_string(int f, std::string& v) : parquet_field(f), val(v) {}

  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_field_type(field_type, FieldType::BINARY);
    auto const n = cpr->get_u32();
    CUDF_EXPECTS(std::cmp_less(n, cpr->m_end - cpr->m_cur), "string length mismatch");

    val.assign(reinterpret_cast<char const*>(cpr->m_cur), n);
    cpr->m_cur += n;
  }
};

/**
 * @brief Functor to read a vector of strings from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * string fails
 */
class parquet_field_string_list : public parquet_field_list<std::string, FieldType::BINARY> {
 public:
  parquet_field_string_list(int f, std::vector<std::string>& v) : parquet_field_list(f, v)
  {
    auto const read_value = [&val = v](uint32_t i, CompactProtocolReader* cpr) {
      auto const l = cpr->get_u32();
      CUDF_EXPECTS(std::cmp_less(l, cpr->m_end - cpr->m_cur), "string length mismatch");
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
 * @return True if field type is not int32
 */
template <typename Enum>
class parquet_field_enum : public parquet_field {
  Enum& val;

 public:
  parquet_field_enum(int f, Enum& v) : parquet_field(f), val(v) {}
  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_field_type(field_type, FieldType::I32);
    val = static_cast<Enum>(cpr->get_i32());
  }
};

/**
 * @brief Functor to read a vector of enums from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading an
 * enum fails
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
 * @return True if field types mismatch or if the process of reading a
 * struct fails
 */
template <typename T>
class parquet_field_struct : public parquet_field {
  T& val;

 public:
  parquet_field_struct(int f, T& v) : parquet_field(f), val(v) {}

  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_field_type(field_type, FieldType::STRUCT);
    cpr->read(&val);
  }
};

/**
 * @brief Functor to read optional structures in unions
 *
 * @return True if field types mismatch
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

  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    T v;
    parquet_field_struct<T>{field(), v}(cpr, field_type);
    val      = v;
    enum_val = static_cast<E>(field());
  }
};

/**
 * @brief Functor to read empty structures in unions
 *
 * Added to avoid having to define read() functions for empty structs contained in unions.
 *
 * @return True if field types mismatch
 */
template <typename E>
class parquet_field_union_enumerator : public parquet_field {
  E& val;

 public:
  parquet_field_union_enumerator(int f, E& v) : parquet_field(f), val(v) {}

  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_field_type(field_type, FieldType::STRUCT);
    cpr->skip_struct_field(field_type);
    val = static_cast<E>(field());
  }
};

/**
 * @brief Functor to read a vector of structures from CompactProtocolReader
 *
 * Uses parallel processing when there are more than 512 elements
 *
 * @return True if field types mismatch or if the process of reading a
 * struct fails
 */
template <typename T>
class parquet_field_struct_list : public parquet_field {
  std::vector<T>& val;

 public:
  parquet_field_struct_list(int f, std::vector<T>& v) : parquet_field(f), val(v) {}

  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_field_type(field_type, FieldType::LIST);
    auto const [t, n] = cpr->get_listh();
    assert_field_type(t, FieldType::STRUCT);
    // Reject a declared count that cannot fit the remaining bytes (each struct is >= 1 byte),
    // guarding against a malformed/hostile size prefix requesting an enormous allocation.
    CUDF_EXPECTS(std::cmp_less_equal(n, cpr->m_end - cpr->m_cur),
                 "Parquet footer list size exceeds remaining buffer");
    val.resize(n);

    constexpr uint32_t parallel_threshold = 512;
    if (n >= parallel_threshold) {
      auto const num_tasks =
        std::min<uint32_t>(n, cudf::detail::host_worker_pool().get_thread_count() * 2);
      auto const items_per_task = n / num_tasks;
      auto const remainder      = n % num_tasks;

      std::vector<std::future<void>> tasks;
      tasks.reserve(num_tasks);

      std::vector<cudf::host_span<uint8_t const>> all_ranges;
      all_ranges.reserve(n);

      uint32_t struct_idx = 0;
      for (uint32_t task_id = 0; task_id < num_tasks; ++task_id) {
        auto const task_size = items_per_task + (task_id < remainder ? 1 : 0);
        auto const start_idx = std::min(n, struct_idx);
        auto const end_idx   = std::min(n, start_idx + task_size);

        if (start_idx >= end_idx) { break; }

        // Collect struct ranges for the next task
        for (auto i = start_idx; i < end_idx; ++i) {
          uint8_t const* const start = cpr->m_cur;
          cpr->skip_struct_field(static_cast<int>(FieldType::STRUCT));
          all_ranges.emplace_back(start, static_cast<size_t>(cpr->m_cur - start));
        }

        // Launch task immediately to parse the structs in parallel while the main thread collects
        // the remaining ranges
        tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(
          [&val = this->val, &all_ranges, start_idx, end_idx]() {
            CompactProtocolReader local_cpr;
            for (size_t i = start_idx; i < end_idx && i < all_ranges.size(); ++i) {
              local_cpr.init(all_ranges[i].data(), all_ranges[i].size());
              local_cpr.read(&val[i]);
            }
          }));

        struct_idx = end_idx;
      }

      for (auto& task : tasks) {
        task.get();
      }
    } else {
      // For small numbers of elements, use sequential processing to avoid overhead
      std::for_each(val.begin(), val.end(), [&cpr](auto& elem) { cpr->read(&elem); });
    }
  }
};

/**
 * @brief Functor to read a binary from CompactProtocolReader
 *
 * @return True if field type mismatches or if size of binary exceeds bounds
 * of the CompactProtocolReader
 */
class parquet_field_binary : public parquet_field {
  std::vector<uint8_t>& val;

 public:
  parquet_field_binary(int f, std::vector<uint8_t>& v) : parquet_field(f), val(v) {}

  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_field_type(field_type, FieldType::BINARY);
    auto const n = cpr->get_u32();
    CUDF_EXPECTS(std::cmp_less_equal(n, cpr->m_end - cpr->m_cur), "binary length mismatch");

    val.assign(cpr->m_cur, cpr->m_cur + n);
    cpr->m_cur += n;
  }
};

/**
 * @brief Functor to read a vector of binaries from CompactProtocolReader
 *
 * @return True if field types mismatch or if the process of reading a
 * binary fails
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
 * @return True if field type mismatches
 */
class parquet_field_struct_blob : public parquet_field {
  std::vector<uint8_t>& val;

 public:
  parquet_field_struct_blob(int f, std::vector<uint8_t>& v) : parquet_field(f), val(v) {}
  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    assert_field_type(field_type, FieldType::STRUCT);
    uint8_t const* const start = cpr->m_cur;
    cpr->skip_struct_field(field_type);
    if (cpr->m_cur > start) { val.assign(start, cpr->m_cur - 1); }
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

  inline void operator()(CompactProtocolReader* cpr, int field_type)
  {
    T v;
    FieldFunctor{field(), v}(cpr, field_type);
    val = v;
  }
};

/**
 * @brief Skips the number of bytes according to the specified struct type
 *
 * @param[in] t Struct type enumeration
 * @param[in] depth Level of struct nesting
 *
 * @return True if the struct type is recognized, false otherwise
 */
void CompactProtocolReader::skip_struct_field(int t, int depth)
{
  auto const t_enum = static_cast<FieldType>(t);
  switch (t_enum) {
    case FieldType::BOOLEAN_TRUE:
    case FieldType::BOOLEAN_FALSE: break;
    case FieldType::I16:
    case FieldType::I32:
    case FieldType::I64: get_u64(); break;
    case FieldType::I8: skip_bytes(1); break;
    case FieldType::DOUBLE: skip_bytes(8); break;
    case FieldType::BINARY: skip_bytes(get_u32()); break;
    case FieldType::LIST:
    case FieldType::SET: {
      auto const [t, n] = get_listh();
      // Reject a declared count that cannot fit the remaining bytes (each element is >= 1 byte),
      // bounding the skip loop against a malformed/hostile size prefix.
      CUDF_EXPECTS(std::cmp_less_equal(n, m_end - m_cur),
                   "Parquet footer list size exceeds remaining buffer");
      CUDF_EXPECTS(depth <= 10, "struct nesting too deep");
      for (uint32_t i = 0; i < n; i++) {
        skip_struct_field(t, depth + 1);
      }
    } break;
    case FieldType::STRUCT:
      for (;;) {
        int const c = getb();
        t           = c & 0xf;
        if (c == 0) { break; }               // end of struct
        if ((c & 0xf0) == 0) { get_i16(); }  // field id is not a delta
        CUDF_EXPECTS(depth <= 10, "struct nesting too deep");
        skip_struct_field(t, depth + 1);
      }
      break;
    default: break;
  }
}

void CompactProtocolReader::read(FileMetaData* f)
{
  using optional_list_column_order =
    parquet_field_optional<std::vector<ColumnOrder>, parquet_field_struct_list<ColumnOrder>>;
  auto op = std::make_tuple(parquet_field_int32(1, f->version),
                            parquet_field_struct_list(2, f->schema),
                            parquet_field_int64(3, f->num_rows),
                            parquet_field_struct_list(4, f->row_groups),
                            parquet_field_struct_list(5, f->key_value_metadata),
                            parquet_field_string(6, f->created_by),
                            optional_list_column_order(7, f->column_orders));
  function_builder(this, op);
  // A well-formed footer never reads past its final stop byte; a set flag means the input was
  // truncated or corrupt, so fail rather than return structurally-invalid metadata.
  CUDF_EXPECTS(not m_overread, "Parquet footer is truncated or corrupt (read past end of buffer)");
}

void CompactProtocolReader::read(SchemaElement* s)
{
  using optional_converted_type =
    parquet_field_optional<ConvertedType, parquet_field_enum<ConvertedType>>;
  using optional_logical_type = parquet_field_optional<LogicalType,
                                                       parquet_field_struct<LogicalType>,
                                                       cuda::std::optional<LogicalType>>;
  auto op                     = std::make_tuple(parquet_field_enum<Type>(1, s->type),
                            parquet_field_int32(2, s->type_length),
                            parquet_field_enum<FieldRepetitionType>(3, s->repetition_type),
                            parquet_field_string(4, s->name),
                            parquet_field_int32(5, s->num_children),
                            optional_converted_type(6, s->converted_type),
                            parquet_field_int32(7, s->decimal_scale),
                            parquet_field_int32(8, s->decimal_precision),
                            parquet_field_optional<int32_t, parquet_field_int32>(9, s->field_id),
                            optional_logical_type(10, s->logical_type));
  function_builder(this, op);
}

void CompactProtocolReader::read(LogicalType* l)
{
  auto op = std::make_tuple(
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
    parquet_field_union_enumerator(16, l->type));
  function_builder(this, op);
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
  using optional_i16 = parquet_field_optional<int16_t, parquet_field_int16>;
  using optional_i64 = parquet_field_optional<int64_t, parquet_field_int64>;
  using optional_list_sorting_column =
    parquet_field_optional<std::vector<SortingColumn>, parquet_field_struct_list<SortingColumn>>;

  auto op = std::make_tuple(parquet_field_struct_list(1, r->columns),
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
  auto op = std::make_tuple(parquet_field_string(1, c->file_path),
                            parquet_field_int64(2, c->file_offset),
                            parquet_field_struct(3, c->meta_data),
                            parquet_field_int64(4, c->offset_index_offset),
                            parquet_field_int32(5, c->offset_index_length),
                            parquet_field_int64(6, c->column_index_offset),
                            parquet_field_int32(7, c->column_index_length));
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
  auto op            = std::make_tuple(parquet_field_enum<Type>(1, c->type),
                            parquet_field_enum_list(2, c->encodings),
                            parquet_field_string_list(3, c->path_in_schema),
                            parquet_field_enum<Compression>(4, c->codec),
                            parquet_field_int64(5, c->num_values),
                            parquet_field_int64(6, c->total_uncompressed_size),
                            parquet_field_int64(7, c->total_compressed_size),
                            parquet_field_int64(9, c->data_page_offset),
                            parquet_field_int64(10, c->index_page_offset),
                            parquet_field_int64(11, c->dictionary_page_offset),
                            parquet_field_struct(12, c->statistics),
                            optional_list_enc_stats(13, c->encoding_stats),
                            optional_i64(14, c->bloom_filter_offset),
                            optional_i32(15, c->bloom_filter_length),
                            optional_size_statistics(16, c->size_statistics));
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
  auto op = std::make_tuple(parquet_field_string(1, k->key), parquet_field_string(2, k->value));
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
                            optional_list_i64(7, c->definition_level_histogram));
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
                            optional_bool(8, s->is_min_value_exact));
  function_builder(this, op);
}

void CompactProtocolReader::read(ColumnOrder* c)
{
  auto op = std::make_tuple(parquet_field_union_enumerator<ColumnOrder::Type>(1, c->type));
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

/**
 * @brief Constructs the schema from the file-level metadata
 *
 * @param[in] md File metadata that was previously parsed
 *
 * @return True if schema constructed completely, false otherwise
 */
bool CompactProtocolReader::InitSchema(FileMetaData* md)
{
  if (static_cast<std::size_t>(WalkSchema(md)) != md->schema.size()) { return false; }

  /* Inside FileMetaData, there is a std::vector of RowGroups and each RowGroup contains a
   * a std::vector of ColumnChunks. Each ColumnChunk has a member ColumnMetaData, which contains
   * a std::vector of std::strings representing paths. The purpose of the code below is to set the
   * schema_idx of each column of each row to it corresponding row_group. This is effectively
   * mapping the columns to the schema.
   */
  for (auto& row_group : md->row_groups) {
    int current_schema_index = 0;
    for (auto& column : row_group.columns) {
      int parent = 0;  // root of schema
      for (auto const& path : column.meta_data.path_in_schema) {
        auto const it = [&] {
          // find_if starting at (current_schema_index + 1) and then wrapping
          auto const schema = [&](auto const& e) {
            return e.parent_idx == parent && e.name == path;
          };
          auto const mid = md->schema.cbegin() + current_schema_index + 1;
          auto const it  = std::find_if(mid, md->schema.cend(), schema);
          if (it != md->schema.cend()) { return it; }
          return std::find_if(md->schema.cbegin(), mid, schema);
        }();
        if (it == md->schema.cend()) { return false; }
        current_schema_index = std::distance(md->schema.cbegin(), it);
        column.schema_idx    = current_schema_index;
        parent               = current_schema_index;
      }
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

}  // namespace cudf::io::parquet::detail
