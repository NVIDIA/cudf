// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use cudf_polars_quent_store::{CudfPolars, CudfPolarsEvent, EngineEvent};
use quent_browser_bundle::Capabilities;
use quent_browser_bundle_build::{BuildOptions, GenerationSummary};

use crate::schema_digest;

/// Convert a completed collector directory or Quent ZIP to a browser bundle.
pub fn generate_bundle(
    input: &Path,
    output: &Path,
) -> Result<GenerationSummary, quent_browser_bundle_build::Error> {
    let schema_digest = schema_digest();
    quent_browser_bundle_build::generate_bundle::<CudfPolars>(
        input,
        output,
        &BuildOptions {
            model_name: "cudf-polars",
            schema_digest: &schema_digest,
            analyzer_revision: option_env!("QUENT_SOURCE_COMMIT")
                .unwrap_or(env!("CARGO_PKG_VERSION")),
            capabilities: Capabilities {
                data_flow: false,
                nvtx: false,
            },
        },
        |event| {
            matches!(
                event.data,
                CudfPolarsEvent::Engine(EngineEvent::Init { .. })
            )
            .then_some(event.id)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudf_polars_quent_store::Implementation;
    use quent_events::Event;
    use uuid::Uuid;

    const MODEL_INFO: &str = r#"{"model":{"name":"CudfPolars","package":"test","source":{"version":"test"},"analyzer_package":"cudf-polars-quent-analyzer"},"quent":{"version":"test"}}"#;

    #[test]
    fn converts_a_multi_event_context() {
        let temp = tempfile::tempdir().unwrap();
        let context_id = Uuid::now_v7();
        let engine_id = Uuid::now_v7();
        let context = temp.path().join(context_id.to_string());
        let engine_dir = context.join("Engine");
        std::fs::create_dir_all(&engine_dir).unwrap();
        let events = [
            Event::new(
                engine_id,
                1,
                EngineEvent::Init {
                    seq: 0,
                    instance_name: "test".into(),
                    implementation: Implementation {
                        name: "cudf-polars".into(),
                        version: "test".into(),
                        backend: "spmd".into(),
                        custom_attributes: Default::default(),
                    },
                },
            ),
            Event::new(engine_id, 2, EngineEvent::Exit { seq: 1 }),
        ];
        let ndjson = events
            .iter()
            .map(|event| serde_json::to_string(event).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(engine_dir.join(format!("{engine_id}.ndjson")), ndjson).unwrap();
        std::fs::write(context.join("model.qmi"), MODEL_INFO).unwrap();
        let output = temp.path().join("trace.quent-browser");

        let summary = generate_bundle(temp.path(), &output).unwrap();
        let bytes = std::fs::read(output).unwrap();
        let decoded = quent_browser_bundle::decode(&bytes).unwrap();
        let decoded_events: Vec<Event<CudfPolarsEvent>> =
            postcard::from_bytes(decoded.payload).unwrap();

        assert_eq!(summary.engine_id, engine_id);
        assert_eq!(summary.context_count, 1);
        assert_eq!(summary.event_count, 2);
        assert_eq!(decoded_events.len(), 2);
        assert_eq!(
            decoded.metadata.analyzer_revision,
            option_env!("QUENT_SOURCE_COMMIT").unwrap_or(env!("CARGO_PKG_VERSION"))
        );
        let quent = quent_events::build_info::quent();
        assert_eq!(
            decoded.metadata.quent_revision,
            quent.commit.unwrap_or(quent.version)
        );
        assert_eq!(
            decoded.metadata.model_provenance.unwrap()["model"]["name"],
            "CudfPolars"
        );
    }

    #[test]
    fn rejects_context_without_an_engine() {
        let temp = tempfile::tempdir().unwrap();
        let context = temp.path().join(Uuid::now_v7().to_string());
        std::fs::create_dir_all(&context).unwrap();
        std::fs::write(context.join("model.qmi"), MODEL_INFO).unwrap();

        let error =
            generate_bundle(temp.path(), &temp.path().join("trace.quent-browser")).unwrap_err();
        assert!(error.to_string().contains("exactly one engine"));
    }
}
