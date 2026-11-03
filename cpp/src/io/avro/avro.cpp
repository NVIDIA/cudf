/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "avro.hpp"

#include <array>
#include <unordered_map>

namespace cudf {
namespace io {
namespace avro {

template <>
uint64_t container::get_encoded()
{
  uint64_t val = 0;
  for (auto len = 0; len < 64; len += 7) {
    // 64-bit int since shift left is upto 64.
    uint64_t const byte = get_raw<uint8_t>();
    val |= (byte & 0x7f) << len;
    if (byte < 0x80) break;
  }
  return val;
}

template <>
int64_t container::get_encoded()
{
  auto const uval = get_encoded<uint64_t>();
  return (int64_t)((uval >> 1u) ^ -(int64_t)(uval & 1));
}

template <>
std::string container::get_encoded()
{
  auto const len = [&] {
    auto const len = get_encoded<uint64_t>();
    return (len & 1) || (m_cur >= m_end) ? 0
                                         : std::min(len >> 1, static_cast<uint64_t>(m_end - m_cur));
  }();
  auto const s = reinterpret_cast<char const*>(m_cur);
  m_cur += len;
  return std::string(s, len);
}

/**
 * @brief AVRO file metadata parser
 *
 * @param[out] md parsed avro file metadata
 * @param[in] max_num_rows maximum number of rows
 * @param[in] first_row drop blocks below first_row
 *
 * @returns true if successful, false if error
 */
bool container::parse(file_metadata* md, size_t max_num_rows, size_t first_row)
{
  constexpr uint32_t avro_magic = (('O' << 0) | ('b' << 8) | ('j' << 16) | (0x01 << 24));

  uint32_t sig4 = get_raw<uint8_t>();
  sig4 |= get_raw<uint8_t>() << 8;
  sig4 |= get_raw<uint8_t>() << 16;
  sig4 |= get_raw<uint8_t>() << 24;
  if (sig4 != avro_magic) { return false; }
  for (;;) {
    auto md_items_signed = get_encoded<int64_t>();
    if (md_items_signed == 0) { break; }
    if (md_items_signed < 0) {
      // A negative count means a block's byte size follows. Read it and discard it.
      [[maybe_unused]] auto const md_block_size = get_encoded<int64_t>();
      md_items_signed                           = -md_items_signed;
    }
    // Check that the claimed item count can fit in the remaining input
    if (md_items_signed > (m_end - m_cur) / 2) { return false; }
    auto const num_md_items = static_cast<uint32_t>(md_items_signed);
    for (uint32_t i = 0; i < num_md_items; i++) {
      auto const key   = get_encoded<std::string>();
      auto const value = get_encoded<std::string>();
      if (key == "avro.codec") {
        md->codec = value;
      } else if (key == "avro.schema") {
        schema_parser sp;
        if (!sp.parse(md->schema, value)) { return false; }
      } else {
        // printf("\"%s\" = \"%s\"\n", key.c_str(), value.c_str());
        md->user_data.emplace(key, value);
      }
    }
  }
  // Save the first sync markers in the metadata; we compare them to other
  // sync markers that should be present at the end of a block.  If they
  // differ, the data should be interpreted as corrupted.
  md->sync_marker[0] = get_raw<uint64_t>();
  md->sync_marker[1] = get_raw<uint64_t>();

  // Initialize remaining metadata fields.
  md->metadata_size  = m_cur - m_base;
  md->skip_rows      = first_row;
  md->total_num_rows = 0;

  // Enumerate the blocks in this file.  Each block starts with a count of
  // objects (rows) in the block (uint64_t), and then the total size in bytes
  // of the block (uint64_t).  We walk each block and do the following:
  //    1. Capture the total number of rows present across all blocks.
  //    2. Add each block to the metadata's list of blocks.
  //    3. Handle the case where we've been asked to skip or limit rows.
  //    4. Verify sync markers at the end of each block.
  //
  // A row offset is also maintained, and added to each block.  This reflects
  // the absolute offset that needs to be added to any given row in order to
  // get the row's index within the destination array.  See `dst_row` in
  // `avro_decode_row()` for more information.
  //
  // N.B. "object" and "row" are used interchangeably here; "object" is
  //      avro nomenclature, "row" is ours.
  //
  // N.B. If we're skipping rows, we ignore blocks (i.e. don't add them to
  //      md->block_list) that precede the block containing the first row
  //      we're interested in.
  //

  // Number of rows in the current block.
  uint32_t num_rows = 0;

  // Absolute row offset of the current block relative to all blocks selected by
  // the skip rows/limit rows constraints, if any.  Otherwise, absolute row
  // offset relative to all blocks.
  uint32_t row_offset = 0;

  // Maximum block size in bytes encountered whilst processing all blocks
  // selected by the skip rows/limit rows constraints, if any.  Otherwise,
  // maximum block size across all blocks.
  uint32_t max_block_size = 0;

  // Accumulates the total number of rows across all blocks selected by the skip
  // rows/limit rows constraints, if any.  Otherwise, total number of rows across
  // all blocks.
  size_t total_object_count = 0;

  // N.B. The 18 below is (presumably) intended to account for the two 64-bit
  //      object count and block size integers (16 bytes total), and then an
  //      additional two bytes to represent the smallest possible row size.
  while (m_cur + 18 < m_end && total_object_count < max_num_rows) {
    auto const object_count = static_cast<uint32_t>(get_encoded<int64_t>());
    auto const block_size   = static_cast<uint32_t>(get_encoded<int64_t>());
    auto const next_end     = m_cur + block_size + 16;
    // Abort on terminal conditions.  We keep these as separate lines instead of
    // combining them into a single if in order to facilitate setting specific
    // line breakpoints in the debugger.
    if (block_size <= 0) { return false; }
    if (object_count <= 0) { return false; }
    if (next_end > m_end) { return false; }

    // Update our total row count.  This is only captured for information
    // purposes.
    md->total_num_rows += object_count;

    if (object_count <= first_row) {
      // We've been asked to skip rows, and we haven't yet reached our desired
      // number of rows to skip.  Subtract this block's rows (`object_count`)
      // from the remaining rows to skip (`first_row`).  Do not add this block
      // to our block list.
      first_row -= object_count;
    } else {
      // Either we weren't asked to skip rows, or we were, but we've already hit
      // our target number of rows to skip.  Add this block to our block list.
      max_block_size = std::max(max_block_size, block_size);
      total_object_count += object_count;
      if (!md->block_list.size()) {
        // This is the first block, so add it to our list with the current value
        // of `first_row`, which will reflect the number of rows to skip *in
        // this block*.
        m_start = m_cur;
        total_object_count -= first_row;
        num_rows = total_object_count;
        CUDF_EXPECTS(row_offset == 0, "Invariant check failed: row_offset != 0");
        if ((max_num_rows > 0) && (max_num_rows < total_object_count)) { num_rows = max_num_rows; }
        md->block_list.emplace_back(m_cur - m_base, block_size, row_offset, first_row, num_rows);
        first_row = 0;
        row_offset += num_rows;
      } else {
        // Not our first block; `first_row` should always be zero here.
        CUDF_EXPECTS(first_row == 0, "Invariant check failed: first_row != 0");

        num_rows = object_count;
        if ((max_num_rows > 0) && (max_num_rows < total_object_count)) {
          num_rows -= (total_object_count - max_num_rows);
        }

        md->block_list.emplace_back(m_cur - m_base, block_size, row_offset, first_row, num_rows);
        row_offset += num_rows;
      }
    }
    m_cur += block_size;
    // Read the next sync markers and ensure they match the first ones we
    // encountered.  If they don't, we have to assume the data is corrupted,
    // and thus, we terminate processing immediately.
    std::array const sync_marker = {get_raw<uint64_t>(), get_raw<uint64_t>()};
    bool const valid_sync_markers =
      ((sync_marker[0] == md->sync_marker[0]) && (sync_marker[1] == md->sync_marker[1]));
    if (!valid_sync_markers) { return false; }
  }
  md->max_block_size = max_block_size;
  // N.B. `total_object_count` has skip_rows applied to it at this point, i.e.
  //      it represents the number of rows that will be returned *after* rows
  //      have been skipped (if requested).
  if ((max_num_rows <= 0) || (max_num_rows > total_object_count)) {
    md->num_rows = total_object_count;
  } else {
    md->num_rows = max_num_rows;
  }
  md->total_data_size = m_cur - (m_base + md->metadata_size);
  CUDF_EXPECTS(m_cur > m_start, "Invariant check failed: `m_cur > m_start` is false.");
  md->selected_data_size = m_cur - m_start;
  // Extract columns
  for (size_t i = 0; i < md->schema.size(); i++) {
    type_kind_e const kind                = md->schema[i].kind;
    logicaltype_kind_e const logical_kind = md->schema[i].logical_kind;

    bool const is_supported_kind = ((kind > type_null) && (kind < type_record));
    if (is_supported_logical_type(logical_kind) || is_supported_kind) {
      column_desc col;
      int parent_idx       = md->schema[i].parent_idx;
      col.schema_data_idx  = (int32_t)i;
      col.schema_null_idx  = -1;
      col.parent_union_idx = -1;
      col.name             = md->schema[i].name;
      if (parent_idx >= 0) {
        while (parent_idx >= 0) {
          if (md->schema[parent_idx].kind == type_union) {
            std::size_t pos = parent_idx + 1;
            for (int num_children = md->schema[parent_idx].num_children; num_children > 0;
                 --num_children) {
              int skip = 1;
              if (pos == i) {
                // parent_idx will always be pointing to our immediate parent
                // union at this point.
                col.parent_union_idx = parent_idx;
              } else if (md->schema[pos].kind == type_null) {
                col.schema_null_idx = pos;
                break;
              }
              do {
                skip = skip + md->schema[pos].num_children - 1;
                pos++;
              } while (skip != 0);
            }
          }
          // We want to "inherit" the column name from our parent union's
          // name, as long as we're not dealing with the root (parent_idx == 0)
          // or array entries.
          if ((parent_idx != 0 && md->schema[parent_idx].kind != type_array) ||
              col.name.length() == 0) {
            if (col.name.length() > 0) { col.name.insert(0, 1, '.'); }
            col.name.insert(0, md->schema[parent_idx].name);
          }
          parent_idx = md->schema[parent_idx].parent_idx;
        }
      }
      md->columns.emplace_back(std::move(col));
    }
  }
  return true;
}

/**
 * @brief Parser state
 */
enum json_state_e {
  state_attrname = 0,
  state_attrcolon,
  state_attrvalue,
  state_attrvalue_last,
  state_nextattr,
  state_nextsymbol,
};

enum attrtype_e {
  attrtype_none = -1,
  attrtype_type = 0,
  attrtype_name,
  attrtype_fields,
  attrtype_symbols,
  attrtype_items,
  attrtype_logicaltype,
};

/**
 * @brief AVRO JSON schema parser
 *
 * @param[out] schema parsed avro schema
 * @param[in] json_str avro schema (JSON string)
 *
 * @returns true if successful, false if error
 */
bool schema_parser::parse(std::vector<schema_entry>& schema, std::string_view json_str)
{
  // Empty schema
  if (json_str == "[]") return true;

  std::array<char, MAX_SCHEMA_DEPTH> depthbuf{};
  int depth = 0, parent_idx = -1, entry_idx = -1;
  json_state_e state = state_attrname;
  std::string str;
  std::unordered_map<std::string_view, type_kind_e> const typenames = {
    {"null", type_null},
    {"boolean", type_boolean},
    {"int", type_int},
    {"long", type_long},
    {"float", type_float},
    {"double", type_double},
    {"bytes", type_bytes},
    {"string", type_string},
    {"record", type_record},
    {"enum", type_enum},
    {"array", type_array},
    {"union", type_union},
    {"fixed", type_fixed},
    {"decimal", type_decimal},
    {"date", type_date},
    {"time-millis", type_time_millis},
    {"time-micros", type_time_micros},
    {"timestamp-millis", type_timestamp_millis},
    {"timestamp-micros", type_timestamp_micros},
    {"local-timestamp-millis", type_local_timestamp_millis},
    {"local-timestamp-micros", type_local_timestamp_micros},
    {"duration", type_duration}};
  std::unordered_map<std::string_view, attrtype_e> const attrnames = {
    {"type", attrtype_type},
    {"name", attrtype_name},
    {"fields", attrtype_fields},
    {"symbols", attrtype_symbols},
    {"items", attrtype_items},
    {"logicalType", attrtype_logicaltype}};
  attrtype_e cur_attr = attrtype_none;
  m_base              = json_str.begin();
  m_cur               = m_base;
  m_end               = json_str.end();
  while (more_data()) {
    int const c = *m_cur++;
    switch (c) {
      case '"':
        str = get_str();
        // printf("str: \"%s\" (cur_attr=%d, state=%d)\n", str.c_str(), cur_attr, state);
        if (state == state_attrname && cur_attr == attrtype_none &&
            typenames.find(str) != typenames.end()) {
          cur_attr = attrtype_type;
          state    = state_attrvalue_last;
        }
        if (state == state_attrname) {
          auto t   = attrnames.find(str);
          cur_attr = (t == attrnames.end()) ? attrtype_none : t->second;
          state    = state_attrcolon;
        } else if (state == state_attrvalue || state == state_attrvalue_last) {
          if (entry_idx < 0) {
            entry_idx = static_cast<int>(schema.size());
            schema.emplace_back(type_not_set, parent_idx);
            if (parent_idx >= 0) { schema[parent_idx].num_children++; }
          }
          if (cur_attr == attrtype_type) {
            auto t = typenames.find(str);
            if (t == typenames.end()) return false;
            schema[entry_idx].kind = t->second;
          } else if (cur_attr == attrtype_logicaltype) {
            auto t = typenames.find(str);
            if (t == typenames.end()) return false;
            schema[entry_idx].logical_kind = static_cast<logicaltype_kind_e>(t->second);
          } else if (cur_attr == attrtype_name) {
            if (entry_idx < 0) return false;
            schema[entry_idx].name = std::move(str);
          }
          if (state == state_attrvalue_last) { entry_idx = -1; }
          state    = state_nextattr;
          cur_attr = attrtype_none;
        } else if (state == state_nextsymbol) {
          if (entry_idx < 0) return false;
          schema[entry_idx].symbols.emplace_back(std::move(str));
        }
        break;
      case ':':
        if (state != state_attrcolon) return false;
        state = state_attrvalue;
        break;
      case ',':
        if (state != state_nextsymbol) {
          if (state != state_nextattr) return false;
          state = state_attrname;
        }
        break;
      case '{':
        if (state == state_attrvalue && cur_attr == attrtype_type) {
          if (entry_idx < 0) {
            entry_idx = static_cast<int>(schema.size());
            schema.emplace_back(type_record, parent_idx);
            if (parent_idx >= 0) { schema[parent_idx].num_children++; }
          }
          cur_attr = attrtype_none;
          state    = state_attrname;
        } else if (state == state_attrvalue && cur_attr == attrtype_items && entry_idx >= 0) {
          // Treat array as a one-field record
          parent_idx = entry_idx;
          entry_idx  = -1;
          cur_attr   = attrtype_none;
          state      = state_attrname;
        }
        if (depth >= MAX_SCHEMA_DEPTH || state != state_attrname) { return false; }
        depthbuf[depth++] = '{';
        break;
      case '}':
        if (depth == 0 || state != state_nextattr || depthbuf[depth - 1] != '{') return false;
        --depth;
        if (entry_idx < 0) {
          parent_idx = (parent_idx >= 0) ? schema[parent_idx].parent_idx : -1;
        } else {
          entry_idx = -1;
        }
        break;
      case '[':
        if (state == state_attrname && cur_attr == attrtype_none) {
          cur_attr = attrtype_type;
          state    = state_attrvalue;
        }
        if (depth >= MAX_SCHEMA_DEPTH || state != state_attrvalue) { return false; }
        depthbuf[depth++] = '[';
        if (cur_attr == attrtype_symbols) {
          state = state_nextsymbol;
          break;
        } else if (cur_attr == attrtype_type) {
          if (entry_idx < 0 || schema[entry_idx].kind != type_not_set) {
            entry_idx = static_cast<int>(schema.size());
            schema.emplace_back(type_union, parent_idx);
            if (parent_idx >= 0) { schema[parent_idx].num_children++; }
          } else {
            schema[entry_idx].kind = type_union;
          }
          parent_idx = entry_idx;
        } else if (cur_attr != attrtype_fields || entry_idx < 0 ||
                   schema[entry_idx].kind < type_record) {
          return false;
        } else {
          parent_idx = entry_idx;
        }
        entry_idx = -1;
        cur_attr  = attrtype_none;
        state     = state_attrname;
        break;
      case ']':
        if (depth == 0 || (state != state_nextattr && state != state_nextsymbol) ||
            depthbuf[depth - 1] != '[')
          return false;
        --depth;
        if (state == state_nextsymbol) {
          state = state_nextattr;
        } else if (parent_idx >= 0) {
          entry_idx  = parent_idx;
          parent_idx = schema[parent_idx].parent_idx;
        }
        break;
      case ' ':
      case '\x09':
      case '\x0d':
      case '\x0a':
        // Ignore spaces, tabs and CRLF
        break;
      default: return false;
    }
  }
  // printf("schema (%d entries) = %s\n", (int)schema.size(), m_base);
  return true;
}

/**
 * @brief Parse a string
 *
 * @returns parsed string, consuming the terminating quote
 */
std::string schema_parser::get_str()
{
  std::string s;
  char const* start = m_cur;
  char const* cur   = start;
  while (cur < m_end && *cur++ != '"')
    ;
  auto len = static_cast<int32_t>(cur - start - 1);
  m_cur    = cur;
  s.assign(start, std::max(len, 0));
  return s;
}

}  // namespace avro
}  // namespace io
}  // namespace cudf
