/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/experimental/strings/regex.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class corpus_source : std::uint8_t {
  REBAR,
  RE2,
  PCRE2,
  HYPERSCAN,
  RIPGREP_EN,
  RIPGREP_RU,
  LINGUA_FRANCA,
  SNORT
};

struct regex_case {
  std::string_view name;
  std::string_view pattern;
  corpus_source source;
  cudf::strings::regex_flags flags{cudf::strings::regex_flags::DEFAULT};
};

// Patterns and workload shapes are adapted from the upstream sources documented in README.md.
constexpr std::array search_cases{
  regex_case{"rebar/long-words-ascii",
             R"(\b\w{25,}\b)",
             corpus_source::REBAR,
             cudf::strings::regex_flags::ASCII},
  regex_case{
    "rebar/contiguous-letters",
    R"((?:(a+)|(b+)|(c+)|(d+)|(e+)|(f+)|(g+)|(h+)|(i+)|(j+)|(k+)|(l+)|(m+)|(n+)|(o+)|(p+)|(q+)|(r+)|(s+)|(t+)|(u+)|(v+)|(w+)|(x+)|(y+)|(z+)))",
    corpus_source::REBAR},
  regex_case{"rebar/rust-functions", R"(fn (?:is|as)_(\w+))", corpus_source::REBAR},
  regex_case{"rebar/rare-literal", R"(ZQZQZQZQZQ)", corpus_source::REBAR},
  regex_case{"rebar/tricksy-literal", R"(fooYbarZquux)", corpus_source::REBAR},
  regex_case{"rebar/quadratic-haystack", R"(.*[^A-Z]|[A-Z])", corpus_source::REBAR},
  regex_case{"re2/easy-literal", R"(ABCDEFGHIJKLMNOPQRSTUVWXYZ$)", corpus_source::RE2},
  regex_case{
    "re2/easy-classes", R"(A[AB]B[BC]C[CD]D[DE]E[EF]F[FG]G[GH]H[HI]I[IJ]J$)", corpus_source::RE2},
  regex_case{"re2/easy-caseless",
             R"(ABCDEFGHIJKLMNOPQRSTUVWXYZ$)",
             corpus_source::RE2,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"re2/medium", R"([XYZ]ABCDEFGHIJKLMNOPQRSTUVWXYZ$)", corpus_source::RE2},
  regex_case{"re2/hard", R"([ -~]*ABCDEFGHIJKLMNOPQRSTUVWXYZ$)", corpus_source::RE2},
  regex_case{
    "re2/many-captures",
    R"(([ -~])*(A)(B)(C)(D)(E)(F)(G)(H)(I)(J)(K)(L)(M)(N)(O)(P)(Q)(R)(S)(T)(U)(V)(W)(X)(Y)(Z)$)",
    corpus_source::RE2},
  regex_case{"re2/phone", R"((\d{3}-|\(\d{3}\)\s+)(\d{3}-\d{4}))", corpus_source::RE2},
  regex_case{"pcre2/ff-char-absent", R"(Q)", corpus_source::PCRE2},
  regex_case{"pcre2/ff-lit-absent", R"(zqjx)", corpus_source::PCRE2},
  regex_case{"pcre2/ff-lit-found", R"(wombat)", corpus_source::PCRE2},
  regex_case{"pcre2/ff-range-absent", R"([QXZ])", corpus_source::PCRE2},
  regex_case{"pcre2/ff-range-digits", R"([0-9]{6})", corpus_source::PCRE2},
  regex_case{"pcre2/ff-class-dense", R"([aeiou]qzx)", corpus_source::PCRE2},
  regex_case{"pcre2/ff-off-range", R"(.{3}[QXZ])", corpus_source::PCRE2},
  regex_case{"pcre2/ff-off-digits", R"(.{2}[0-9]{6})", corpus_source::PCRE2},
  regex_case{"pcre2/ff-off-dense", R"(.{3}\w{12})", corpus_source::PCRE2},
  regex_case{"pcre2/ff-off-mixed", R"(a.{4}[QXZ])", corpus_source::PCRE2},
  regex_case{"pcre2/ff-caseless-absent",
             R"(zqjx)",
             corpus_source::PCRE2,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"pcre2/ff-caseless-found",
             R"(Hello)",
             corpus_source::PCRE2,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"pcre2/ff-pair-common", R"(th)", corpus_source::PCRE2},
  regex_case{"pcre2/email", R"([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})", corpus_source::PCRE2},
  regex_case{"pcre2/long-word", R"(\b\w{12,}\b)", corpus_source::PCRE2},
  regex_case{"pcre2/literal-alternation", R"((?:wombat|numbat|quokka))", corpus_source::PCRE2},
  regex_case{
    "pcre2/start-line", R"(^the\b)", corpus_source::PCRE2, cudf::strings::regex_flags::MULTILINE},
  regex_case{"hyperscan/http-request",
             R"((?:GET|POST|PUT|DELETE) /[A-Za-z0-9_./?=&%-]+ HTTP/1\.[01])",
             corpus_source::HYPERSCAN},
  regex_case{"hyperscan/host-header",
             R"(Host:[ \t]+[A-Za-z0-9.-]+)",
             corpus_source::HYPERSCAN,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"hyperscan/sql-union-select",
             R"(union[ \t]+select[ \t]+[A-Za-z0-9_, *]+)",
             corpus_source::HYPERSCAN,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"hyperscan/hex-token", R"([A-F0-9]{32})", corpus_source::HYPERSCAN},
  regex_case{"hyperscan/suspicious-command",
             R"((?:cmd\.exe|/bin/(?:ba)?sh|powershell))",
             corpus_source::HYPERSCAN,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"hyperscan/user-agent", R"(User-Agent:[^\r\n]{20,120})", corpus_source::HYPERSCAN},
  regex_case{"ripgrep/subtitles-literal", R"(Sherlock Holmes)", corpus_source::RIPGREP_EN},
  regex_case{"ripgrep/subtitles-literal-casei",
             R"(Sherlock Holmes)",
             corpus_source::RIPGREP_EN,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"ripgrep/subtitles-alternate",
             R"(Sherlock Holmes|John Watson|Irene Adler|Inspector Lestrade|Professor Moriarty)",
             corpus_source::RIPGREP_EN},
  regex_case{
    "ripgrep/subtitles-surrounding-words", R"(\w+\s+Holmes\s+\w+)", corpus_source::RIPGREP_EN},
  regex_case{"ripgrep/subtitles-no-literal",
             R"(\w{5}\s+\w{5}\s+\w{5}\s+\w{5}\s+\w{5}\s+\w{5}\s+\w{5})",
             corpus_source::RIPGREP_EN},
  regex_case{"ripgrep/subtitles-ru-literal", R"(Шерлок Холмс)", corpus_source::RIPGREP_RU},
  regex_case{
    "ripgrep/subtitles-ru-surrounding-words", R"(\w+\s+Холмс\s+\w+)", corpus_source::RIPGREP_RU},
  regex_case{"ripgrep/subtitles-ru-no-literal",
             R"(\w{5}\s+\w{5}\s+\w{5}\s+\w{5}\s+\w{5}\s+\w{5}\s+\w{5})",
             corpus_source::RIPGREP_RU},
  regex_case{"lingua-franca/node-modules", R"(node_modules)", corpus_source::LINGUA_FRANCA},
  regex_case{"lingua-franca/line-ending", R"(\r?\n)", corpus_source::LINGUA_FRANCA},
  regex_case{"lingua-franca/comment-tail", R"(#.*$)", corpus_source::LINGUA_FRANCA},
  regex_case{"lingua-franca/comma-whitespace", R"(\s*,\s*)", corpus_source::LINGUA_FRANCA},
  regex_case{"lingua-franca/html-tag", R"(<([\w:]+))", corpus_source::LINGUA_FRANCA},
  regex_case{"lingua-franca/camel-boundary", R"(([a-z\d])([A-Z]))", corpus_source::LINGUA_FRANCA},
  regex_case{"snort/path-traversal", R"((?:\.\./){2,}[A-Za-z0-9._/-]+)", corpus_source::SNORT},
  regex_case{"snort/php-transfer",
             R"(/[A-Za-z0-9_-]+/(?:upload|download)\.php)",
             corpus_source::SNORT,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"snort/sql-tautology",
             R"((?:or|and)[ \t]+["']?[0-9]+["']?[ \t]*=[ \t]*["']?[0-9]+)",
             corpus_source::SNORT,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"snort/shell-command",
             R"([;&|][ \t]*(?:cmd\.exe|powershell|/bin/(?:ba)?sh))",
             corpus_source::SNORT,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"snort/encoded-script",
             R"((?:%3[cC]|<)script(?:%3[eE]|>))",
             corpus_source::SNORT,
             cudf::strings::regex_flags::IGNORECASE},
  regex_case{"snort/executable-request",
             R"((?:GET|POST)[ \t]+/[^ \r\n]*\.(?:exe|dll|ps1)(?:[? ][^\r\n]*))",
             corpus_source::SNORT,
             cudf::strings::regex_flags::IGNORECASE},
};

template <std::size_t N>
std::vector<std::string> case_names(std::array<regex_case, N> const& cases)
{
  std::vector<std::string> result;
  result.reserve(cases.size());
  std::transform(cases.begin(), cases.end(), std::back_inserter(result), [](auto& item) {
    return std::string{item.name};
  });
  return result;
}

template <std::size_t N>
regex_case const* find_case(std::array<regex_case, N> const& cases, std::string_view name)
{
  auto found =
    std::find_if(cases.begin(), cases.end(), [name](auto& item) { return item.name == name; });
  return found == cases.end() ? nullptr : &*found;
}

std::uint64_t random_value(std::uint64_t& state)
{
  state ^= state >> 12U;
  state ^= state << 25U;
  state ^= state >> 27U;
  return state * 0x2545f4914f6cdd1dULL;
}

void inject(std::string& row, std::string_view needle, std::size_t position)
{
  if (needle.size() <= row.size()) {
    row.replace(std::min(position, row.size() - needle.size()), needle.size(), needle);
  }
}

std::string make_rebar_row(cudf::size_type width, std::size_t sample)
{
  static constexpr std::array fragments{
    std::string_view{"fn parse_expression benchmark_result "},
    std::string_view{"extraordinarily_long_identifier_name "},
    std::string_view{"aaaa bbbb cccc dddd eeee ffff "},
    std::string_view{"match regex haystack iterator captures "},
  };
  std::string row;
  for (std::size_t index = 0; row.size() < static_cast<std::size_t>(width); ++index) {
    row.append(fragments[(sample + index) % fragments.size()]);
  }
  row.resize(static_cast<std::size_t>(width));
  return row;
}

std::string make_re2_row(cudf::size_type width, std::size_t sample)
{
  auto random_state = std::uint64_t{0x9e3779b97f4a7c15ULL + sample};
  std::string row(static_cast<std::size_t>(width), ' ');
  for (auto& character : row) {
    character = static_cast<char>(0x20U + random_value(random_state) % 0x5fU);
  }
  if (sample % 4 == 0) { inject(row, "(650) 253-0001", row.size() / 3); }
  inject(row, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", row.size());
  return row;
}

std::string make_pcre2_row(cudf::size_type width, std::size_t sample)
{
  static constexpr std::array words{
    std::string_view{"the"},
    std::string_view{"regex"},
    std::string_view{"engine"},
    std::string_view{"search"},
    std::string_view{"through"},
    std::string_view{"ordinary"},
    std::string_view{"text"},
    std::string_view{"with"},
    std::string_view{"several"},
    std::string_view{"common"},
    std::string_view{"words"},
    std::string_view{"and"},
    std::string_view{"spaces"},
    std::string_view{"while"},
    std::string_view{"matching"},
    std::string_view{"quickly"},
    std::string_view{"across"},
    std::string_view{"rows"},
  };
  auto random_state = std::uint64_t{0xd1b54a32d192ed03ULL + sample};
  std::string row;
  while (row.size() < static_cast<std::size_t>(width)) {
    row.append(words[random_value(random_state) % words.size()]);
    row.push_back((random_value(random_state) & 15U) == 0 ? '\n' : ' ');
  }
  row.resize(static_cast<std::size_t>(width));
  switch (sample % 4) {
    case 0: inject(row, "wombat", row.size() / 4); break;
    case 1: inject(row, "user.name@example.com", row.size() / 3); break;
    case 2: inject(row, "1234567890", row.size() / 2); break;
    case 3: inject(row, "Hello", row.size() / 5); break;
  }
  inject(row, "\nthe ", row.size() * 3 / 4);
  return row;
}

std::string make_hyperscan_row(cudf::size_type width, std::size_t sample)
{
  static constexpr std::array packets{
    std::string_view{"GET /index.html?user=42 HTTP/1.1\r\nHost: example.com\r\nUser-Agent: "
                     "Mozilla/5.0 regex benchmark client\r\n\r\n"},
    std::string_view{"POST /login.php HTTP/1.1\r\nHost: accounts.example.net\r\n\r\nuser=admin "
                     "UNION SELECT username, password FROM users"},
    std::string_view{"alert payload powershell -enc QUJDREVGR0hJSktMTU5PUFFSU1RVVldY\r\n"},
    std::string_view{"PUT /api/item/17 HTTP/1.0\r\nHost:\tapi.example.org:8080\r\nUser-Agent: "
                     "automated security scanner 1.0\r\n\r\n"},
    std::string_view{
      "event token=0123456789ABCDEF0123456789ABCDEF command=/bin/bash status=blocked\r\n"},
    std::string_view{"ordinary packet payload without a selected signature\r\n"},
  };
  std::string row;
  while (row.size() < static_cast<std::size_t>(width)) {
    row.append(packets[(sample + row.size()) % packets.size()]);
  }
  row.resize(static_cast<std::size_t>(width));
  return row;
}

std::string make_ripgrep_english_row(cudf::size_type width, std::size_t sample)
{
  static constexpr std::array subtitles{
    std::string_view{
      "Sherlock Holmes studied the unusual letter while John Watson waited nearby.\n"},
    std::string_view{
      "Inspector Lestrade asked whether Professor Moriarty had returned to London.\n"},
    std::string_view{"Irene Adler knew that sherlock holmes would notice every ordinary detail.\n"},
    std::string_view{
      "These quiet words create realistic subtitle dialogue without a proper name.\n"},
    std::string_view{"Seven short words might align across this simple subtitle sentence today.\n"},
  };
  std::string row;
  for (std::size_t index = 0; row.size() < static_cast<std::size_t>(width); ++index) {
    row.append(subtitles[(sample + index) % subtitles.size()]);
  }
  row.resize(static_cast<std::size_t>(width));
  return row;
}

std::string make_lingua_franca_row(cudf::size_type width, std::size_t sample)
{
  static constexpr std::array fragments{
    std::string_view{"const nodePath = '/workspace/node_modules/pkg/index.js';\n"},
    std::string_view{"# generated configuration comment\r\nkey, value, nextValue\n"},
    std::string_view{R"(<html:section dataValue="camelCaseName">content</html:section>\n)"},
    std::string_view{"function parseHTTPResponse(inputValue) { return inputValue; }\n"},
  };
  std::string row;
  for (std::size_t index = 0; row.size() < static_cast<std::size_t>(width); ++index) {
    row.append(fragments[(sample + index) % fragments.size()]);
  }
  row.resize(static_cast<std::size_t>(width));
  return row;
}

std::string make_snort_row(cudf::size_type width, std::size_t sample)
{
  static constexpr std::array payloads{
    std::string_view{"GET /../../etc/passwd HTTP/1.1\r\nHost: example.test\r\n\r\n"},
    std::string_view{"POST /admin/upload.php HTTP/1.1\r\n\r\nfile=payload.exe"},
    std::string_view{"GET /login?id=1 OR 1=1 HTTP/1.1\r\n\r\n"},
    std::string_view{"command=ready;/bin/bash -c id\r\n"},
    std::string_view{"value=%3Cscript%3Ealert(document)\r\n"},
    std::string_view{"GET /downloads/update.ps1?arch=x64 HTTP/1.1\r\n\r\n"},
    std::string_view{"ordinary application traffic without a selected signature\r\n"},
  };
  std::string row;
  while (row.size() < static_cast<std::size_t>(width)) {
    row.append(payloads[(sample + row.size()) % payloads.size()]);
  }
  row.resize(static_cast<std::size_t>(width));
  return row;
}

std::string make_ripgrep_russian_row(cudf::size_type width, std::size_t sample)
{
  static constexpr std::array subtitles{
    std::string_view{"Шерлок Холмс внимательно изучал письмо, пока доктор Ватсон ждал ответа.\n"},
    std::string_view{"Инспектор спросил, почему Холмс сразу заметил эту важную деталь.\n"},
    std::string_view{"Обычный разговор продолжался вечером в тихой комнате старого дома.\n"},
    std::string_view{"Каждое слово звучало ясно, но разгадка оставалась совсем непростой.\n"},
  };
  std::string row;
  for (std::size_t index = 0; row.size() < static_cast<std::size_t>(width); ++index) {
    row.append(subtitles[(sample + index) % subtitles.size()]);
  }
  auto end = static_cast<std::size_t>(width);
  if (row.size() > end) {
    while (end > 0 && (static_cast<unsigned char>(row[end]) & 0xc0U) == 0x80U)
      --end;
    row.resize(end);
  }
  return row;
}

std::unique_ptr<cudf::table> make_input(regex_case const& definition,
                                        cudf::size_type num_rows,
                                        cudf::size_type row_width)
{
  constexpr std::size_t sample_count = 64;
  std::vector<std::string> samples;
  samples.reserve(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    switch (definition.source) {
      case corpus_source::REBAR: samples.push_back(make_rebar_row(row_width, sample)); break;
      case corpus_source::RE2: samples.push_back(make_re2_row(row_width, sample)); break;
      case corpus_source::PCRE2: samples.push_back(make_pcre2_row(row_width, sample)); break;
      case corpus_source::HYPERSCAN:
        samples.push_back(make_hyperscan_row(row_width, sample));
        break;
      case corpus_source::RIPGREP_EN:
        samples.push_back(make_ripgrep_english_row(row_width, sample));
        break;
      case corpus_source::RIPGREP_RU:
        samples.push_back(make_ripgrep_russian_row(row_width, sample));
        break;
      case corpus_source::LINGUA_FRANCA:
        samples.push_back(make_lingua_franca_row(row_width, sample));
        break;
      case corpus_source::SNORT: samples.push_back(make_snort_row(row_width, sample)); break;
    }
  }

  cudf::test::strings_column_wrapper samples_column(samples.begin(), samples.end());
  auto profile = data_profile_builder().no_validity().distribution(
    cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0ul, sample_count - 1);
  auto map =
    create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
  return cudf::gather(
    cudf::table_view{{samples_column}}, map->view(), cudf::out_of_bounds_policy::DONT_CHECK);
}

constexpr std::array validation_cases{
  regex_case{"regexeval/zip-plus-four", R"(^\d{5}-\d{4}$)", corpus_source::REBAR},
  regex_case{"regexeval/identifier", R"(^[a-zA-Z]\w{3,14}$)", corpus_source::REBAR},
  regex_case{
    "regexeval/ipv4",
    R"(^(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9])\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[0-9])$)",
    corpus_source::REBAR},
  regex_case{"regexeval/mac-address",
             R"(^([0-9a-fA-F][0-9a-fA-F]:){5}([0-9a-fA-F][0-9a-fA-F])$)",
             corpus_source::REBAR},
  regex_case{"regexeval/guid",
             R"(^\{?[a-fA-F\d]{8}-([a-fA-F\d]{4}-){3}[a-fA-F\d]{12}\}?$)",
             corpus_source::REBAR},
  regex_case{
    "regexeval/us-state",
    R"(^((AL)|(AK)|(AS)|(AZ)|(AR)|(CA)|(CO)|(CT)|(DE)|(DC)|(FM)|(FL)|(GA)|(GU)|(HI)|(ID)|(IL)|(IN)|(IA)|(KS)|(KY)|(LA)|(ME)|(MH)|(MD)|(MA)|(MI)|(MN)|(MS)|(MO)|(MT)|(NE)|(NV)|(NH)|(NJ)|(NM)|(NY)|(NC)|(ND)|(MP)|(OH)|(OK)|(OR)|(PW)|(PA)|(PR)|(RI)|(SC)|(SD)|(TN)|(TX)|(UT)|(VT)|(VI)|(VA)|(WA)|(WV)|(WI)|(WY))$)",
    corpus_source::REBAR},
};

std::vector<std::string> validation_samples(std::string_view name)
{
  if (name == "regexeval/zip-plus-four") {
    return {"22222-3333", "34545-2367", "123456789", "A3B 4C5"};
  }
  if (name == "regexeval/identifier") {
    return {"abcd", "aBc45DSD_sdf", "1234", "reallylongpassword"};
  }
  if (name == "regexeval/ipv4") {
    return {"127.0.0.1", "255.255.255.0", "1200.5.4.3", "abc.def.ghi.jkl"};
  }
  if (name == "regexeval/mac-address") {
    return {"01:23:45:67:89:ab", "fE:dC:bA:98:76:54", "01:23:45:67:89:ab:cd", "01:23:45:67:89:Az"};
  }
  if (name == "regexeval/guid") {
    return {"{e02ff0e4-00ad-090A-c030-0d00a0008ba0}",
            "e02ff0e4-00ad-090A-c030-0d00a0008ba0",
            "0xe02ff0e400ad090Ac0300d00a0008ba0",
            "f34fvfv"};
  }
  return {"NY", "PA", "Pennsylvania", "XX"};
}

std::unique_ptr<cudf::table> make_validation_input(regex_case const& definition,
                                                   cudf::size_type num_rows)
{
  auto base                          = validation_samples(definition.name);
  constexpr std::size_t sample_count = 64;
  std::vector<std::string> samples;
  samples.reserve(sample_count);
  for (std::size_t index = 0; index < sample_count; ++index) {
    samples.push_back(base[index % base.size()]);
  }

  cudf::test::strings_column_wrapper samples_column(samples.begin(), samples.end());
  auto profile = data_profile_builder().no_validity().distribution(
    cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0ul, sample_count - 1);
  auto map =
    create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
  return cudf::gather(
    cudf::table_view{{samples_column}}, map->view(), cudf::out_of_bounds_policy::DONT_CHECK);
}

void bench_corpus_search(nvbench::state& state)
{
  auto case_name  = state.get_string("case");
  auto definition = find_case(search_cases, case_name);
  if (definition == nullptr) {
    state.skip("unknown regex corpus case");
    return;
  }

  auto num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto api       = state.get_string("api");
  auto backend   = state.get_string("backend");
  auto input     = make_input(*definition, num_rows, row_width);
  cudf::strings_column_view strings(input->get_column(0).view());

  auto operation   = api == "contains" ? cudf::experimental::regex_operation::CONTAINS
                     : api == "count"  ? cudf::experimental::regex_operation::COUNT
                                       : cudf::experimental::regex_operation::FIND;
  auto interpreter = backend == "interpreter" ? cudf::strings::regex_program::create(
                                                  definition->pattern, definition->flags)
                                              : nullptr;
  auto jit         = backend == "jit" ? cudf::experimental::regex_jit_program::create(
                                  definition->pattern, operation, {}, definition->flags)
                                      : nullptr;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  auto data_size = input->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int32_t>(strings.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (api == "contains") {
      if (backend == "jit") {
        static_cast<void>(cudf::experimental::contains_re(strings, *jit));
      } else {
        static_cast<void>(cudf::strings::contains_re(strings, *interpreter));
      }
    } else if (api == "count") {
      if (backend == "jit") {
        static_cast<void>(cudf::experimental::count_re(strings, *jit));
      } else {
        static_cast<void>(cudf::strings::count_re(strings, *interpreter));
      }
    } else if (backend == "jit") {
      static_cast<void>(cudf::experimental::find_re(strings, *jit));
    } else {
      static_cast<void>(cudf::strings::find_re(strings, *interpreter));
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void bench_validation_corpus(nvbench::state& state)
{
  auto case_name  = state.get_string("case");
  auto definition = find_case(validation_cases, case_name);
  if (definition == nullptr) {
    state.skip("unknown regex validation case");
    return;
  }

  auto num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto api      = state.get_string("api");
  auto backend  = state.get_string("backend");
  auto input    = make_validation_input(*definition, num_rows);
  cudf::strings_column_view strings(input->get_column(0).view());

  auto operation   = api == "contains" ? cudf::experimental::regex_operation::CONTAINS
                     : api == "count"  ? cudf::experimental::regex_operation::COUNT
                                       : cudf::experimental::regex_operation::FIND;
  auto interpreter = backend == "interpreter" ? cudf::strings::regex_program::create(
                                                  definition->pattern, definition->flags)
                                              : nullptr;
  auto jit         = backend == "jit" ? cudf::experimental::regex_jit_program::create(
                                  definition->pattern, operation, {}, definition->flags)
                                      : nullptr;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(input->alloc_size());
  state.add_global_memory_writes<nvbench::int32_t>(strings.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (api == "contains") {
      if (backend == "jit") {
        static_cast<void>(cudf::experimental::contains_re(strings, *jit));
      } else {
        static_cast<void>(cudf::strings::contains_re(strings, *interpreter));
      }
    } else if (api == "count") {
      if (backend == "jit") {
        static_cast<void>(cudf::experimental::count_re(strings, *jit));
      } else {
        static_cast<void>(cudf::strings::count_re(strings, *interpreter));
      }
    } else if (backend == "jit") {
      static_cast<void>(cudf::experimental::find_re(strings, *jit));
    } else {
      static_cast<void>(cudf::strings::find_re(strings, *interpreter));
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

NVBENCH_BENCH(bench_corpus_search)
  .set_name("regex_corpus_search")
  .add_int64_axis("row_width", {128, 512})
  .add_int64_axis("num_rows", {262144})
  .add_string_axis("case", case_names(search_cases))
  .add_string_axis("api", {"contains", "count", "find"})
  .add_string_axis("backend", {"interpreter", "jit"});

NVBENCH_BENCH(bench_validation_corpus)
  .set_name("regex_validation_corpus")
  .add_int64_axis("num_rows", {262144})
  .add_string_axis("case", case_names(validation_cases))
  .add_string_axis("api", {"contains", "count", "find"})
  .add_string_axis("backend", {"interpreter", "jit"});
