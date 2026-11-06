// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

const RUST_PREAMBLE: &str = "\
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

";
const PYTHON_PREAMBLE: &str = "\
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

";

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
            events: quent_instrumentation_build::EventOptions {
                serde: true,
                combined_event: true,
                record_derives: &["Clone"],
                ..Default::default()
            },
            analyzer_package: Some("cudf-polars-quent-analyzer".to_owned()),
            collector_sink: true,
            generate_timestamped_methods: true,
            ..Default::default()
        },
    )?;
    for warning in generated.warnings {
        println!("cargo:warning={warning}");
    }

    let options = quent_schema_codegen_python::Options {
        module_name: "cudf_polars_quent".to_owned(),
        instrumentation_path: "crate".to_owned(),
        exporters: quent_schema_codegen_python::Exporters {
            ndjson: true,
            collector: true,
            ..Default::default()
        },
        collector_server: true,
        timestamped_methods: true,
        ..Default::default()
    };
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let mut bindings = quent_schema_codegen_python::emit(&parsed.schema, &options)?;
    for file in &mut bindings {
        if file.name == "pyo3_bridge.rs" {
            file.name = "cudf_polars_quent.rs".to_owned();
            file.content.insert_str(0, RUST_PREAMBLE);
        }
    }
    quent_schema_codegen_python::write_generated_files(&bindings, &out_dir)?;
    let mut stubs = quent_schema_codegen_python::emit_stubs(&parsed.schema, &options)?;
    for file in &mut stubs {
        if file.name == "cudf_polars_quent/__init__.pyi" {
            file.name = "cudf_polars_quent.pyi".to_owned();
            file.content.insert_str(0, PYTHON_PREAMBLE);
        }
    }
    quent_schema_codegen_python::write_generated_files(&stubs, &out_dir)?;
    Ok(())
}
