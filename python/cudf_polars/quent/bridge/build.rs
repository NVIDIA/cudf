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
            combined_event: true,
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
        collector_server: true,
        ..Default::default()
    };
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let mut bindings = quent_schema_codegen_python::emit(&parsed.schema, &options)?;
    for file in &mut bindings {
        if file.name == "pyo3_bridge.rs" {
            let end = file
                .content
                .rfind('}')
                .ok_or("generated Python bridge has no module terminator")?;
            file.content.insert_str(
                end,
                r#"
    pub fn reservation_observers(
        context: &pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> pyo3::PyResult<(
        quent_instrumentation::Observer<crate::MemoryReservationRequest>,
        quent_instrumentation::Observer<crate::MemoryReservation>,
        quent_instrumentation::Observer<crate::Worker>,
        quent_instrumentation::Observer<crate::Actor>,
        quent_instrumentation::Observer<crate::ReservableMemory>,
    )> {
        let context = context.extract::<pyo3::PyRef<'_, PyContext>>()?;
        let context = context.inner.as_ref().ok_or_else(|| {
            ContextClosedError::new_err("cannot create a recorder from a closed context")
        })?;
        Ok((
            context.observer(),
            context.observer(),
            context.observer(),
            context.observer(),
            context.observer(),
        ))
    }
"#,
            );
        }
    }
    quent_schema_codegen_python::write_generated_files(&bindings, &out_dir)?;

    const PREAMBLE: &str = "\
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

";

    let mut stubs = quent_schema_codegen_python::emit_stubs(&parsed.schema, &options)?;
    for file in &mut stubs {
        if file.name == "_quent_generated/__init__.pyi" {
            file.name = "cudf_polars_quent.pyi".to_owned();
            file.content.push_str(
                r#"

def reservation_recorder_capsule(
    context: Context,
    worker_id: _uuid.UUID | str,
    reservable_memory_ids: Mapping[str, _uuid.UUID | str],
) -> object: ...
def reservation_recorder_diagnostics(capsule: object) -> list[str]: ...
def close_reservation_recorder(capsule: object) -> None: ...
"#,
            );
        }

        if file.name.ends_with(".pyi") {
            file.content.insert_str(0, PREAMBLE);
        }
    }
    quent_schema_codegen_python::write_generated_files(&stubs, &out_dir)?;
    Ok(())
}
