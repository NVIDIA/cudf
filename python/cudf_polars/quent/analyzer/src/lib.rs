// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod actor;
mod analyzer;
mod evaluate;
mod model;
mod resource;
mod viewer;

mod generated {
    #![allow(dead_code, unused_imports)]
    include!(concat!(env!("OUT_DIR"), "/cudfpolars.rs"));
}

pub use analyzer::CudfPolarsUiAnalyzer;
pub use viewer::Viewer;

#[cfg(test)]
mod tests;
