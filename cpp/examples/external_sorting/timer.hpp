/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <chrono>
#include <iostream>

/**
 * @brief Light-weight timer for measuring elapsed time.
 *
 * A timer object constructed from std::chrono, instrumenting at microseconds
 * precision. Can display elapsed durations at milli and micro second
 * scales. The timer starts at object construction.
 */
class timer {
 public:
  using micros = std::chrono::microseconds;
  using millis = std::chrono::milliseconds;

  timer() { reset(); }
  void reset() { start_time = std::chrono::high_resolution_clock::now(); }
  auto elapsed() const { return (std::chrono::high_resolution_clock::now() - start_time); }
  void print_elapsed_micros() const
  {
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<micros>(elapsed()).count()
              << "us\n\n";
  }
  void print_elapsed_millis() const
  {
    std::cout << "Elapsed Time: " << std::chrono::duration_cast<millis>(elapsed()).count()
              << "ms\n\n";
  }

 private:
  using time_point_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
  time_point_t start_time;
};
