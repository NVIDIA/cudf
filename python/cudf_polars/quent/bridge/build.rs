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
            events: quent_instrumentation_build::EventOptions {
                serde: true,
                combined_event: true,
                record_derives: &["Clone"],
                ..Default::default()
            },
            analyzer_package: Some("cudf-polars-quent-analyzer".to_owned()),
            collector_sink: true,
            ..Default::default()
        },
    )?;
    for warning in generated.warnings {
        println!("cargo:warning={warning}");
    }

    let options = quent_schema_codegen_python::Options {
        module_name: "cudf_polars_quent".to_owned(),
        bridge_file_name: "cudf_polars_quent.rs".to_owned(),
        stub_file_name: Some("cudf_polars_quent.pyi".to_owned()),
        bridge_preamble: "\
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"
        .to_owned(),
        stub_preamble: "\
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"
        .to_owned(),
        instrumentation_path: "crate".to_owned(),
        exporters: quent_schema_codegen_python::Exporters {
            ndjson: true,
            collector: true,
            ..Default::default()
        },
        collector_server: true,
        ..Default::default()
    };
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let bindings = quent_schema_codegen_python::emit(&parsed.schema, &options)?;
    quent_schema_codegen_python::write_generated_files(&bindings, &out_dir)?;
    let stubs = quent_schema_codegen_python::emit_stubs(&parsed.schema, &options)?;
    quent_schema_codegen_python::write_generated_files(&stubs, &out_dir)?;
    Ok(())
}
