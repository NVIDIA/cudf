// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    quent_build_info::emit_source();

    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../model.yaml");
    println!("cargo:rerun-if-changed={}", model.display());

    let parsed = quent_yaml::parse_from_file(model)?;
    for warning in &parsed.warnings {
        println!("cargo:warning={warning}");
    }

    let generated = quent_instrumentation_build::generate(
        &parsed.schema,
        &quent_instrumentation_build::Options {
            serde: true,
            umbrella_event: true,
            analyzer_package: Some("cudf-polars-quent-analyzer".to_owned()),
            record_derives: &["Clone"],
            collector_sink: true,
            ..Default::default()
        },
    )?;
    for warning in generated.warnings {
        println!("cargo:warning={warning}");
    }

    let options = quent_schema_codegen_python::Options {
        module_name: "_quent_generated".to_owned(),
        instrumentation_path: "crate".to_owned(),
        exporters: quent_schema_codegen_python::Exporters {
            ndjson: true,
            collector: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let bindings = quent_schema_codegen_python::emit(&parsed.schema, &options)?;
    quent_schema_codegen_python::write_generated_files(&bindings, &out_dir)?;

    const PREAMBLE: &str = "\
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

";

    let mut stubs = quent_schema_codegen_python::emit_stubs(&parsed.schema, &options)?;
    for file in &mut stubs {
        if file.name == "_quent_generated/__init__.pyi" {
            file.name = "_quent/__init__.pyi".to_owned();
        }

        if file.name.ends_with(".pyi") {
            file.content.insert_str(0, PREAMBLE);
        }
    }
    quent_schema_codegen_python::write_generated_files(&stubs, &out_dir)?;
    Ok(())
}
