/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "timer.hpp"

#include <cuda/iterator>

#include <chrono>
#include <iostream>

#pragma once

template <std::invocable F>
void benchmark(F&& f, std::size_t iterations)
{
  auto total_time = double{0.0};

  std::for_each(
    cuda::counting_iterator<std::size_t>{0}, cuda::counting_iterator{iterations}, [&](auto iter) {
      timer timer;

      f();

      auto elapsed_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(timer.elapsed()).count() /
        double{1000.0};

      std::cout << "Iteration: " << iter << ", time: " << elapsed_time_ms << " ms\n";

      if (iterations == 1 or (iter != 0)) { total_time += elapsed_time_ms; }
    });

  std::cout << "Average time (first iteration excluded if iterations > 1): "
            << total_time / std::max<std::size_t>(1, iterations - 1) << " ms\n\n";
}
