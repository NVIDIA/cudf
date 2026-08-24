/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/regex/flags.hpp>

#include <string>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Actions and Tokens (regex instruction types)
 *
 * ```
 *	02xx are operators, value == precedence
 *	03xx are tokens, i.e. operands for operators
 * ```
 */
enum InstType : int32_t {
  CHAR    = 0177,  // Literal character
  RBRA    = 0201,  // Right bracket, )
  LBRA    = 0202,  // Left bracket, (
  OR      = 0204,  // Alternation, |
  ANY     = 0300,  // Any character except newline, .
  ANYNL   = 0301,  // Any character including newline, .
  BOL     = 0303,  // Beginning of line, ^
  EOL     = 0304,  // End of line, $
  CCLASS  = 0305,  // Character class, []
  NCCLASS = 0306,  // Negated character class, [^ ]
  BOW     = 0307,  // Boundary of word, \b
  NBOW    = 0310,  // Not boundary of word, \B
  END     = 0377   // Terminate: match found
};

/**
 * @brief Range used for literals in reclass classes.
 */
struct reclass_range {
  char32_t first{};  /// first character in span
  char32_t last{};   /// last character in span (inclusive)
};

/**
 * @brief Class type for regex compiler instruction.
 */
struct reclass {
  int32_t builtins{0};  // bit mask identifying builtin classes
  std::vector<reclass_range> literals;
  reclass() {}
  reclass(int m) : builtins(m) {}
  reclass(int m, std::vector<reclass_range>&& l) : builtins(m), literals(std::move(l)) {}
};

constexpr int32_t CCLASS_W{1 << 0};   // [a-z], [A-Z], [0-9], and '_'
constexpr int32_t CCLASS_S{1 << 1};   // all spaces or ctrl characters
constexpr int32_t CCLASS_D{1 << 2};   // digits [0-9]
constexpr int32_t NCCLASS_W{1 << 3};  // not CCLASS_W or '\n'
constexpr int32_t NCCLASS_S{1 << 4};  // not CCLASS_S
constexpr int32_t NCCLASS_D{1 << 5};  // not CCLASS_D or '\n'

/**
 * @brief Structure of an encoded regex instruction
 */
struct reinst {
  int32_t type; /* operator type or instruction type */
  union {
    int32_t cls_id;   /* class pointer */
    char32_t c;       /* character */
    int32_t subid;    /* sub-expression id for RBRA and LBRA */
    int32_t right_id; /* right child of OR */
  } u1;
  union {            /* regexec relies on these two being in the same union */
    int32_t left_id; /* left child of OR */
    int32_t next_id; /* next instruction for CAT & LBRA */
  } u2;
  int32_t reserved4;
};

enum match_flags : int32_t {
  NONE        = 0,  // no special handling
  EMPTY_MATCH = 1,  // may match an empty string, e.g. a* or \b$
};

/**
 * @brief Regex program handles parsing a pattern into a vector
 * of chained instructions.
 */
class reprog {
 public:
  reprog(reprog const&)            = default;
  reprog(reprog&&)                 = default;
  ~reprog()                        = default;
  reprog& operator=(reprog const&) = default;
  reprog& operator=(reprog&&)      = default;

  /**
   * @brief Parses the given regex pattern and produces an instance
   * of this object
   *
   * @param pattern Regex pattern encoded as UTF-8
   * @param flags For interpreting certain `pattern` characters
   * @param capture For controlling how capture groups are processed
   * @return Instance of reprog
   */
  static reprog create_from(std::string_view pattern,
                            regex_flags const flags,
                            capture_groups const capture = capture_groups::EXTRACT);

  int32_t add_inst(int32_t type);
  int32_t add_inst(reinst const& inst);
  int32_t add_class(reclass const& cls);

  void set_groups_count(int32_t groups);
  [[nodiscard]] int32_t groups_count() const;

  [[nodiscard]] int32_t insts_count() const;
  [[nodiscard]] reinst& inst_at(int32_t id);
  [[nodiscard]] reinst const* insts_data() const;

  [[nodiscard]] int32_t classes_count() const;
  [[nodiscard]] reclass const& class_at(int32_t id) const;
  [[nodiscard]] reclass const* classes_data() const;

  [[nodiscard]] int32_t const* starts_data() const;
  [[nodiscard]] int32_t starts_count() const;

  void set_start_inst(int32_t id);
  [[nodiscard]] int32_t get_start_inst() const;

  void optimize();
  void finalize();
  void check_for_errors();

  [[nodiscard]] std::pair<literal_fast_path, std::string> check_for_literal_fast_path() const;
  [[nodiscard]] match_flags compute_match_flags() const;

#ifndef NDEBUG
  void print() const;
#endif

 private:
  std::vector<reinst> _insts;           // instructions
  std::vector<reclass> _classes;        // data for CCLASS instructions
  int32_t _startinst_id{};              // id of first instruction
  std::vector<int32_t> _startinst_ids;  // short-cut to speed-up ORs
  int32_t _num_capturing_groups{};
  regex_flags _flags{};

  reprog(regex_flags);
  void collapse_nops();
  void build_start_ids();
  void check_for_errors(int32_t id, int32_t next_id);
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf
