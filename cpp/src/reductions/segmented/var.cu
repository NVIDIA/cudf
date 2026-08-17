/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The segmented variance implementation is intentionally co-located with
// segmented_standard_deviation in std.cu. Both reductions use the same var_std intermediate and
// segmented CUB reduction shape; keeping them in one translation unit avoids emitting duplicate
// device kernel instantiations.
