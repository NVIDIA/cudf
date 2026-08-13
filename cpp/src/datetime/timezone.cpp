/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/timezone.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

namespace cudf {

namespace {

constexpr uint32_t tzif_magic           = ('T' << 0) | ('Z' << 8) | ('i' << 16) | ('f' << 24);
std::string const tzif_system_directory = "/usr/share/zoneinfo/";

/**
 * @brief Resolves a timezone alias via `Link` entries in `<tzif_dir>/tzdata.zi`.
 *
 * Returns the canonical name if it resolves to a TZif file on disk, else `std::nullopt`.
 */
std::optional<std::string> resolve_tz_alias(std::filesystem::path const& tzif_dir,
                                            std::string_view timezone_name)
{
  std::ifstream fin{tzif_dir / "tzdata.zi"};
  if (!fin) { return std::nullopt; }

  std::unordered_map<std::string, std::string> links;
  for (std::string line; std::getline(fin, line);) {
    // Handle CRLF line endings: `std::getline` keeps a trailing `\r`.
    if (!line.empty() && line.back() == '\r') { line.pop_back(); }

    std::string_view v{line};
    auto const take_token = [&v]() -> std::string_view {
      auto const ws_start = v.find_first_not_of(" \t");
      if (ws_start == std::string_view::npos) {
        v = {};
        return {};
      }
      v.remove_prefix(ws_start);
      auto const tok_end = v.find_first_of(" \t");
      auto const tok     = v.substr(0, tok_end);
      v.remove_prefix(tok_end == std::string_view::npos ? v.size() : tok_end);
      return tok;
    };

    auto const directive = take_token();
    if (directive.empty() || directive.front() == '#') { continue; }
    if (directive != "Link" && directive != "L") { continue; }

    auto const target = take_token();
    auto const alias  = take_token();
    if (target.empty() || alias.empty()) { continue; }
    links.emplace(std::string{alias}, std::string{target});
  }

  // Walk the alias chain, detecting cycles via the set of names already visited.
  std::string name{timezone_name};
  std::unordered_set<std::string> visited{name};
  while (true) {
    auto const it = links.find(name);
    if (it == links.end()) { break; }
    if (!visited.insert(it->second).second) {
      // Cycle in tzdata.zi; bail out instead of looping.
      return std::nullopt;
    }
    name = it->second;
  }

  // No link followed
  if (std::string_view{name} == timezone_name) { return std::nullopt; }

  std::error_code ec;
  if (!std::filesystem::exists(tzif_dir / name, ec)) { return std::nullopt; }
  return name;
}

#pragma pack(push, 1)
/**
 * @brief 32-bit TZif header
 */
struct timezone_file_header {
  uint32_t magic;          ///< "TZif"
  uint8_t version;         ///< 0:version1, '2':version2, '3':version3
  uint8_t reserved15[15];  ///< unused, reserved for future use // NOLINT
  uint32_t isutccnt;       ///< number of UTC/local indicators contained in the body
  uint32_t isstdcnt;       ///< number of standard/wall indicators contained in the body
  uint32_t leapcnt;        ///< number of leap second records contained in the body
  uint32_t timecnt;        ///< number of transition times contained in the body
  uint32_t typecnt;  ///< number of local time type Records contained in the body - MUST NOT be zero
  uint32_t charcnt;  ///< total number of octets used by the set of time zone designations contained
                     ///< in the body
};

struct localtime_type_record_s {
  int32_t utcoff;    // number of seconds to be added to UTC in order to determine local time
  uint8_t isdst;     // 0:standard time, 1:Daylight Savings Time (DST)
  uint8_t desigidx;  // index into the series of time zone designation characters
};

struct dst_transition_s {
  char type;  // Transition type ('J','M' or day)
  int month;  // Month of transition
  int week;   // Week of transition
  int day;    // Day of transition
  int time;   // Time of day (seconds)
};
#pragma pack(pop)

struct timezone_file {
  timezone_file_header header{};
  bool is_header_from_64bit = false;

  std::vector<int64_t> transition_times;
  std::vector<uint8_t> ttime_idx;
  std::vector<localtime_type_record_s> ttype;
  std::vector<char> posix_tz_string;

  [[nodiscard]] auto timecnt() const { return header.timecnt; }
  [[nodiscard]] auto typecnt() const { return header.typecnt; }

  // Based on https://datatracker.ietf.org/doc/html/draft-murchison-tzdist-tzif-16
  static constexpr auto leap_second_rec_size(bool is_64bit) noexcept
  {
    return (is_64bit ? sizeof(uint64_t) : sizeof(uint32_t)) + sizeof(uint32_t);
  }
  static constexpr auto file_content_size_32(timezone_file_header const& header) noexcept
  {
    return header.timecnt * sizeof(uint32_t) +                 // transition times
           header.timecnt * sizeof(uint8_t) +                  // transition time index
           header.typecnt * sizeof(localtime_type_record_s) +  // local time type records
           header.charcnt * sizeof(uint8_t) +                  // time zone designations
           header.leapcnt * leap_second_rec_size(false) +      // leap second records
           header.isstdcnt * sizeof(uint8_t) +                 // standard/wall indicators
           header.isutccnt * sizeof(uint8_t);                  // UTC/local indicators
  }

  /**
   * @brief Used because little-endian platform in assumed.
   */
  void header_to_little_endian()
  {
    header.isutccnt = __builtin_bswap32(header.isutccnt);
    header.isstdcnt = __builtin_bswap32(header.isstdcnt);
    header.leapcnt  = __builtin_bswap32(header.leapcnt);
    header.timecnt  = __builtin_bswap32(header.timecnt);
    header.typecnt  = __builtin_bswap32(header.typecnt);
    header.charcnt  = __builtin_bswap32(header.charcnt);
  }

  void read_header(std::ifstream& input_file, size_t file_size)
  {
    input_file.read(reinterpret_cast<char*>(&header), sizeof(header));
    CUDF_EXPECTS(!input_file.fail() && header.magic == tzif_magic,
                 "Error reading time zones file header.");
    header_to_little_endian();

    // Check for 64-bit header
    if (header.version != 0) {
      if (file_content_size_32(header) + sizeof(header) < file_size) {
        // skip the 32-bit content
        input_file.seekg(file_content_size_32(header), std::ios_base::cur);
        // read the 64-bit header
        input_file.read(reinterpret_cast<char*>(&header), sizeof(header));
        header_to_little_endian();
        is_header_from_64bit = true;
      }
    }
    CUDF_EXPECTS(
      header.typecnt > 0 && header.typecnt <= file_size / sizeof(localtime_type_record_s),
      "Invalid number of time types in timezone file.");
    CUDF_EXPECTS(header.timecnt <= file_size,
                 "Number of transition times is larger than the file size.");
  }

  timezone_file(std::optional<std::string_view> tzif_dir, std::string_view timezone_name)
  {
    using std::ios_base;

    auto const tz_dir      = std::filesystem::path{tzif_dir.value_or(tzif_system_directory)};
    auto const tz_filename = tz_dir / timezone_name;
    auto const open_flags  = ios_base::in | ios_base::binary | ios_base::ate;
    std::ifstream fin;
    fin.open(tz_filename, open_flags);

    // Fall back to resolving the alias through the tzdata.zi index file
    if (!fin) {
      if (auto const resolved = resolve_tz_alias(tz_dir, timezone_name)) {
        fin.clear();
        fin.open(tz_dir / *resolved, open_flags);
      }
    }
    if (!fin) {
      auto err = std::string{"Failed to open the timezone file '"} + tz_filename.string() + "'";
      std::error_code ec;
      if (!std::filesystem::exists(tz_dir / "tzdata.zi", ec)) {
        err += " (no tzdata.zi present in '" + tz_dir.string() + "' to resolve aliases)";
      } else {
        err += " (no matching Link entry in '" + (tz_dir / "tzdata.zi").string() +
               "', or its target is also missing)";
      }
      CUDF_FAIL(err);
    }
    auto const file_size = fin.tellg();
    fin.seekg(0);

    read_header(fin, file_size);

    // Read transition times (convert from 32-bit to 64-bit if necessary)
    transition_times.resize(timecnt());
    if (is_header_from_64bit) {
      fin.read(reinterpret_cast<char*>(transition_times.data()),
               transition_times.size() * sizeof(int64_t));
      for (auto& tt : transition_times) {
        tt = __builtin_bswap64(tt);
      }
    } else {
      std::vector<int32_t> tt32(timecnt());
      fin.read(reinterpret_cast<char*>(tt32.data()), tt32.size() * sizeof(int32_t));
      std::transform(
        tt32.cbegin(), tt32.cend(), std::back_inserter(transition_times), [](auto& tt) {
          return __builtin_bswap32(tt);
        });
    }
    ttime_idx.resize(timecnt());
    fin.read(reinterpret_cast<char*>(ttime_idx.data()), timecnt() * sizeof(uint8_t));

    // Read time types
    ttype.resize(typecnt());
    fin.read(reinterpret_cast<char*>(ttype.data()), typecnt() * sizeof(localtime_type_record_s));
    CUDF_EXPECTS(!fin.fail(), "Failed to read time types from the time zone file.");
    for (uint32_t i = 0; i < typecnt(); i++) {
      ttype[i].utcoff = __builtin_bswap32(ttype[i].utcoff);
    }

    // Read posix TZ string
    fin.seekg(header.charcnt + header.leapcnt * leap_second_rec_size(is_header_from_64bit) +
                header.isstdcnt + header.isutccnt,
              ios_base::cur);
    auto const file_pos = fin.tellg();
    if (file_size - file_pos > 1) {
      posix_tz_string.resize(file_size - file_pos);
      fin.read(posix_tz_string.data(), file_size - file_pos);
    }
  }
};

/**
 * @brief Posix TZ parser
 */
template <class Container>
class posix_parser {
 public:
  posix_parser(Container const& tz_string) : cur{tz_string.begin()}, end{tz_string.end()} {}

  /**
   * @brief Advances the parser past a name from the posix TZ string.
   */
  void skip_name();

  /**
   * @brief Parses a number from the posix TZ string.
   *
   * @return Parsed number
   */
  int64_t parse_number();

  /**
   * @brief Parses a UTC offset from the posix TZ string.
   *
   * @return Parsed offset
   */
  int32_t parse_offset();

  /**
   * @brief Parses a DST transition time from the posix TZ string.
   *
   * @return Parsed transition time
   */
  dst_transition_s parse_transition();

  /**
   * @brief Returns the remaining number of characters in the input.
   */
  [[nodiscard]] auto remaining_char_cnt() const { return end - cur; }

  /**
   * @brief Returns the next character in the input.
   */
  [[nodiscard]] char next_character() const { return *cur; }

 private:
  typename Container::const_iterator cur;
  typename Container::const_iterator const end;
};

/**
 * @brief Skips the next name token.
 *
 * Name can be a string of letters, such as EST, or an arbitrary string surrounded by angle
 * brackets, such as <UTC-05>
 */
template <class Container>
void posix_parser<Container>::skip_name()
{
  cur = std::find_if(cur, end, [](auto c) {
    return std::isdigit(c) || c == '-' || c == ',' || c == '+' || c == '<';
  });

  if (*cur == '<') cur = std::next(std::find(cur, end, '>'));
}

template <class Container>
int64_t posix_parser<Container>::parse_number()
{
  int64_t v = 0;
  while (cur < end) {
    auto const c = *cur - '0';
    if (c > 9 || c < 0) { break; }
    v = v * 10 + c;
    ++cur;
  }
  return v;
}

template <class Container>
int32_t posix_parser<Container>::parse_offset()
{
  CUDF_EXPECTS(cur < end, "Unexpected end of input stream");

  auto const sign = *cur;
  cur += (sign == '-' || sign == '+');

  auto const hours   = parse_number();
  auto scale         = 60 * 60;
  auto total_seconds = hours * scale;

  // Parse minutes and seconds, if present
  while (cur < end && scale > 1 && *cur == ':') {
    // Skip the ':' character
    ++cur;
    // Scale becomes 60, for minutes, and then 1, for seconds
    scale /= 60;
    total_seconds += parse_number() * scale;
  }

  return (sign == '-') ? -total_seconds : total_seconds;
}

template <class Container>
dst_transition_s posix_parser<Container>::parse_transition()
{
  CUDF_EXPECTS(cur < end, "Unexpected end of input stream");

  // Transition at 2AM by default
  int32_t time = 2 * 60 * 60;
  if (cur + 2 <= end && *cur == ',') {
    char const type = cur[1];
    int month       = 0;
    int week        = 0;
    int day         = 0;
    cur += (type == 'M' || type == 'J') ? 2 : 1;
    if (type == 'M') {
      month = parse_number();
      if (cur < end && *cur == '.') {
        ++cur;
        week = parse_number();
        if (cur < end && *cur == '.') {
          ++cur;
          day = parse_number();
        }
      }
    } else {
      day = parse_number();
    }
    if (cur < end && *cur == '/') {
      ++cur;
      time = parse_offset();
    }
    return {type, month, week, day, time};
  }
  return {0, 0, 0, 0, time};
}

/**
 * @brief Returns the number of days in a month.
 */
static int days_in_month(int month, bool is_leap_year)
{
  CUDF_EXPECTS(month > 0 && month <= 12, "Invalid month");

  if (month == 2) return 28 + is_leap_year;
  return 30 + ((0b1010110101010 >> month) & 1);
}

/**
 * @brief Converts a daylight saving transition time to a number of seconds.
 *
 * @param trans transition day information
 * @param year year of transition
 *
 * @return transition time in seconds from the beginning of the year
 */
static int64_t get_transition_time(dst_transition_s const& trans, int year)
{
  auto day = trans.day;

  auto const is_leap = cuda::std::chrono::year{year}.is_leap();

  if (trans.type == 'M') {
    auto const month = std::min(std::max(trans.month, 1), 12);
    auto week        = std::min(std::max(trans.week, 1), 52);

    // Year-to-year day adjustment
    auto const adjusted_month = (month + 9) % 12 + 1;
    auto const adjusted_year  = year - (month <= 2);
    auto day_of_week =
      ((26 * adjusted_month - 2) / 10 + 1 + (adjusted_year % 100) + (adjusted_year % 100) / 4 +
       (adjusted_year / 400) - 2 * (adjusted_year / 100)) %
      7;
    if (day_of_week < 0) { day_of_week += 7; }
    day = (day - day_of_week + 7) % 7;

    // Add weeks
    while (week > 1 && day + 7 < days_in_month(month, is_leap)) {
      week--;
      day += 7;
    }
    // Add months
    for (int m = 1; m < month; m++) {
      day += days_in_month(m, is_leap);
    }
  } else if (trans.type == 'J') {
    // Account for 29th of February on leap years
    day += (day > 31 + 29 && is_leap);
  }

  return trans.time + cuda::std::chrono::duration_cast<duration_s>(duration_D{day}).count();
}

}  // namespace

std::unique_ptr<table> make_timezone_transition_table(std::optional<std::string_view> tzif_dir,
                                                      std::string_view timezone_name,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::make_timezone_transition_table(tzif_dir, timezone_name, stream, mr);
}

namespace detail {

std::unique_ptr<table> make_timezone_transition_table(std::optional<std::string_view> tzif_dir,
                                                      std::string_view timezone_name,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  if (timezone_name == "UTC" || timezone_name.empty()) {
    // Return an empty table for UTC
    return std::make_unique<cudf::table>();
  }

  timezone_file const tzf(tzif_dir, timezone_name);

  std::vector<timestamp_s::rep> transition_times(1);
  std::vector<duration_s::rep> offsets(1);
  // One ancient rule entry, one per TZ file entry, 2 entries per year in the future cycle
  transition_times.reserve(1 + tzf.timecnt() + solar_cycle_entry_count);
  offsets.reserve(1 + tzf.timecnt() + solar_cycle_entry_count);
  size_t earliest_std_idx = 0;
  for (size_t t = 0; t < tzf.timecnt(); t++) {
    auto const ttime = tzf.transition_times[t];
    auto const idx   = tzf.ttime_idx[t];
    CUDF_EXPECTS(idx < tzf.typecnt(), "Out-of-range type index");
    auto const utcoff = tzf.ttype[idx].utcoff;
    transition_times.push_back(ttime);
    offsets.push_back(utcoff);
    if (!earliest_std_idx && !tzf.ttype[idx].isdst) {
      earliest_std_idx = transition_times.size() - 1;
    }
  }

  if (tzf.timecnt() != 0) {
    if (!earliest_std_idx) { earliest_std_idx = 1; }
    transition_times[0] = transition_times[earliest_std_idx];
    offsets[0]          = offsets[earliest_std_idx];
  } else {
    if (tzf.typecnt() == 0 || tzf.ttype[0].utcoff == 0) {
      // No transitions, offset is zero; Table would be a no-op.
      // Return an empty table to speed up parsing.
      return std::make_unique<cudf::table>();
    }
    // No transitions to use for the time/offset - use the first offset and apply to all timestamps
    transition_times[0] = std::numeric_limits<int64_t>::max();
    offsets[0]          = tzf.ttype[0].utcoff;
  }

  // Generate entries for times after the last transition
  auto future_std_offset = offsets[tzf.timecnt()];
  auto future_dst_offset = future_std_offset;
  dst_transition_s dst_start{};
  dst_transition_s dst_end{};
  if (!tzf.posix_tz_string.empty()) {
    posix_parser<decltype(tzf.posix_tz_string)> parser(tzf.posix_tz_string);
    parser.skip_name();
    future_std_offset = -parser.parse_offset();
    if (parser.remaining_char_cnt() > 1) {
      // Parse Daylight Saving Time information
      parser.skip_name();
      if (parser.remaining_char_cnt() > 0 && parser.next_character() != ',') {
        future_dst_offset = -parser.parse_offset();
      } else {
        future_dst_offset = future_std_offset + 60 * 60;
      }
      dst_start = parser.parse_transition();
      dst_end   = parser.parse_transition();
    } else {
      future_dst_offset = future_std_offset;
    }
  }

  // Add entries to fill the transition cycle
  int64_t year_timestamp = 0;
  for (int32_t year = 1970; year < 1970 + solar_cycle_years; ++year) {
    auto const dst_start_time = get_transition_time(dst_start, year);
    auto const dst_end_time   = get_transition_time(dst_end, year);

    // Two entries per year, since there are two transitions
    transition_times.push_back(year_timestamp + dst_start_time - future_std_offset);
    offsets.push_back(future_dst_offset);
    transition_times.push_back(year_timestamp + dst_end_time - future_dst_offset);
    offsets.push_back(future_std_offset);

    // Swap the newly added transitions if in descending order
    if (transition_times.rbegin()[1] > transition_times.rbegin()[0]) {
      std::swap(transition_times.rbegin()[0], transition_times.rbegin()[1]);
      std::swap(offsets.rbegin()[0], offsets.rbegin()[1]);
    }

    year_timestamp += cuda::std::chrono::duration_cast<duration_s>(
                        duration_D{365 + cuda::std::chrono::year{year}.is_leap()})
                        .count();
  }

  CUDF_EXPECTS(transition_times.size() == offsets.size(),
               "Error reading TZif file for timezone " + std::string{timezone_name});

  auto ttimes_typed = make_empty_host_vector<timestamp_s>(transition_times.size(), stream);
  std::transform(transition_times.cbegin(),
                 transition_times.cend(),
                 std::back_inserter(ttimes_typed),
                 [](auto ts) { return timestamp_s{duration_s{ts}}; });
  auto offsets_typed = make_empty_host_vector<duration_s>(offsets.size(), stream);
  std::transform(offsets.cbegin(), offsets.cend(), std::back_inserter(offsets_typed), [](auto ts) {
    return duration_s{ts};
  });

  auto d_ttimes  = cudf::detail::make_device_uvector_async(ttimes_typed, stream, mr);
  auto d_offsets = cudf::detail::make_device_uvector_async(offsets_typed, stream, mr);

  std::vector<std::unique_ptr<column>> tz_table_columns;
  tz_table_columns.emplace_back(
    std::make_unique<cudf::column>(std::move(d_ttimes), rmm::device_buffer{}, 0));
  tz_table_columns.emplace_back(
    std::make_unique<cudf::column>(std::move(d_offsets), rmm::device_buffer{}, 0));

  // Need to finish copies before transition_times and offsets go out of scope
  stream.synchronize();

  return std::make_unique<cudf::table>(std::move(tz_table_columns));
}

}  // namespace detail
}  // namespace cudf
