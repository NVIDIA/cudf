// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use quent_store_build::{EventOptions, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model.yaml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", model.display());

    let parsed = quent_yaml::parse_from_file(model)?;
    for warning in &parsed.warnings {
        println!("cargo:warning={warning}");
    }

    let generated = quent_store_build::generate(
        &parsed.schema,
        &Options {
            events: EventOptions {
                combined_event: true,
                serde: true,
                record_derives: &["Clone"],
                ..Default::default()
            },
            filesystem: cfg!(feature = "filesystem"),
            ..Default::default()
        },
    )?;
    for warning in generated.warnings {
        println!("cargo:warning={warning}");
    }
    Ok(())
}
