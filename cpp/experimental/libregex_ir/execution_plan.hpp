/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "regex_ir_detail.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace regex_ir::detail {

struct deterministic_nfa_node {
  character_predicate predicate           = character_predicate{};
  std::vector<std::size_t> targets        = std::vector<std::size_t>{};
  std::optional<write_capture> capture    = std::nullopt;
  std::optional<assertion_kind> assertion = std::nullopt;
  bool consumes : 1                       = false;
  bool accepts  : 1                       = false;
};

struct deterministic_nfa_graph {
  std::vector<deterministic_nfa_node> nodes = std::vector<deterministic_nfa_node>{};
  std::size_t entry                         = 0;
};

struct deterministic_capture_action {
  std::uint32_t slot = 0;
  bool reset_end : 1 = false;

  bool operator==(deterministic_capture_action const&) const = default;
};

struct deterministic_interval {
  std::uint32_t first    = 0;
  std::uint32_t last     = 0;
  std::uint16_t class_id = 0;
};

struct deterministic_machine {
  std::array<std::uint16_t, 256> byte_classes           = std::array<std::uint16_t, 256>{};
  std::array<std::uint64_t, 4> start_byte_bitmap        = std::array<std::uint64_t, 4>{};
  std::vector<deterministic_interval> unicode_intervals = std::vector<deterministic_interval>{};
  std::vector<std::uint16_t> transitions                = std::vector<std::uint16_t>{};
  std::vector<std::vector<deterministic_capture_action>> transition_capture_actions =
    std::vector<std::vector<deterministic_capture_action>>{};
  std::vector<std::vector<deterministic_capture_action>> accept_capture_actions =
    std::vector<std::vector<deterministic_capture_action>>{};
  std::vector<std::uint8_t> boundary_accepts     = std::vector<std::uint8_t>{};
  std::uint16_t initial_state                    = 0;
  std::uint16_t dead_state                       = 0;
  std::uint16_t class_count                      = 0;
  std::uint16_t state_count                      = 0;
  std::uint16_t state_mask                       = 16383;
  std::uint16_t restart_state                    = std::numeric_limits<std::uint16_t>::max();
  std::uint8_t transition_address_space          = 4;
  std::uint8_t boundary_class_count              = 0;
  std::uint8_t assertion_mask                    = 0;
  std::uint8_t start_byte_range_count            = 0;
  std::optional<assertion_kind> accept_assertion = std::nullopt;
  bool scan_input        : 1                     = false;
  bool accept_at_end     : 1                     = false;
  bool capture_one_pass  : 1                     = false;
  bool assertion_aware   : 1                     = false;
  bool start_byte_filter : 1                     = false;
};

struct glushkov_shift {
  std::uint64_t sources = 0;
  std::uint8_t amount   = 0;
};

struct glushkov_machine {
  deterministic_machine alphabet                        = deterministic_machine{};
  std::vector<std::uint64_t> reach_masks                = std::vector<std::uint64_t>{};
  std::vector<glushkov_shift> shifts                    = std::vector<glushkov_shift>{};
  std::array<std::uint64_t, 64> exception_successors    = std::array<std::uint64_t, 64>{};
  std::uint64_t first_set                               = 0;
  std::uint64_t accept_mask                             = 0;
  std::uint64_t exception_mask                          = 0;
  std::optional<std::uint8_t> start_byte                = std::nullopt;
  std::optional<std::uint32_t> fixed_match_bytes        = std::nullopt;
  std::optional<std::uint32_t> repeated_predicate_count = std::nullopt;
  predicate_class repeated_predicate_class              = predicate_class::NONE;
  std::uint8_t position_count                           = 0;
  bool scan_input    : 1                                = false;
  bool accept_at_end : 1                                = false;
};

class execution_plan {
 public:
  explicit execution_plan(instruction_ir ir);

  enum class string_operation_kind : std::uint8_t {
    BEGINS_WITH = 0,
    ENDS_WITH   = 1,
    ENDS_LINE   = 2,
    EQUALS      = 3,
    EQUALS_LINE = 4
  };

  struct string_operation {
    string_operation_kind kind;
    std::string literal;
  };

  struct fixed_ascii_suffix {
    std::vector<character_predicate> predicates;
    bool begins   : 1;
    bool line_end : 1;
  };

  struct utf8_literal {
    std::u32string codepoints;
    std::size_t byte_count;
  };

  instruction_ir ir_;
  std::optional<result_shape> anchored_boolean_result_     = std::nullopt;
  std::optional<std::string> exact_ascii_literal_metadata_ = std::nullopt;
  std::optional<std::string> exact_literal_bytes_          = std::nullopt;
  std::optional<builtin_character_class> repeated_builtin_ = std::nullopt;
  std::optional<deterministic_machine> deterministic_      = std::nullopt;
  std::optional<glushkov_machine> glushkov_                = std::nullopt;
  std::optional<string_operation> string_operations_       = std::nullopt;
  std::optional<fixed_ascii_suffix> fixed_ascii_suffix_    = std::nullopt;
  std::optional<std::string> line_tail_literal_            = std::nullopt;
  std::optional<std::uint32_t> word_run_minimum_           = std::nullopt;
  std::optional<std::string> ascii_literal_                = std::nullopt;
  std::optional<utf8_literal> utf8_literal_                = std::nullopt;
  std::optional<std::size_t> utf8_literal_pivot_           = std::nullopt;
  std::optional<std::string> mandatory_ascii_literal_      = std::nullopt;
  std::optional<std::uint8_t> prefix_seek_byte_            = std::nullopt;
  bool candidate_seeker_                                   = false;
  std::vector<std::uint32_t> whole_match_captures_         = std::vector<std::uint32_t>{};
  std::vector<std::size_t> capture_slots_                  = std::vector<std::size_t>{};
  std::optional<deterministic_nfa_graph> thompson_;
  std::size_t workspace_bytes_ = 0;
  executor_kind executor_      = executor_kind::ITERATIVE_THOMPSON;
  std::size_t frontier_words_  = 0;
  std::size_t closure_records_ = 0;
  std::size_t storage_words_   = 0;
  std::array<std::uint64_t, 4> candidate_bitmap_{};

  [[nodiscard]] std::optional<fixed_ascii_suffix> fixed_ascii_suffix_plan() const;

  [[nodiscard]] std::optional<string_operation> string_operation_plan() const;

  [[nodiscard]] std::optional<std::string> line_tail_literal() const;

  [[nodiscard]] std::optional<std::uint32_t> word_run_minimum() const;

  [[nodiscard]] static std::string encode_utf8_literal(std::u32string_view codepoints);

  [[nodiscard]] static std::optional<std::size_t> utf8_literal_pivot(utf8_literal const& literal);

  [[nodiscard]] std::optional<utf8_literal> exact_literal() const;

  [[nodiscard]] std::optional<std::string> exact_ascii_literal() const;

  [[nodiscard]] std::optional<utf8_literal> exact_utf8_literal() const;

  [[nodiscard]] std::optional<std::uint8_t> required_ascii_prefix() const;

  [[nodiscard]] std::optional<std::string> mandatory_ascii_literal() const;

  [[nodiscard]] bool begins_at_input_start() const;

  [[nodiscard]] std::vector<std::size_t> live_capture_slots() const;

  [[nodiscard]] std::vector<std::uint32_t> whole_match_captures() const;

  [[nodiscard]] bool uses_capture_buffer() const;

  [[nodiscard]] bool is_whole_match_capture(std::uint32_t capture_index) const;

  [[nodiscard]] bool uses_unicode_word_boundaries() const;

  void prepare_start_seeker(deterministic_nfa_graph const& graph);

  void prepare_thompson();

 private:
  void select();
};

}  // namespace regex_ir::detail
