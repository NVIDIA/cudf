// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod actor;
mod analyzer;
mod evaluate;
mod model;
mod resource;
#[cfg(all(feature = "filesystem", not(target_arch = "wasm32")))]
mod viewer;

pub(crate) use cudf_polars_quent_store as generated;

pub use analyzer::CudfPolarsUiAnalyzer;
#[cfg(all(feature = "filesystem", not(target_arch = "wasm32")))]
pub use viewer::Viewer;

#[cfg(test)]
mod tests;
