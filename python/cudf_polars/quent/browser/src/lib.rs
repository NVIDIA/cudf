// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Browser-hosted API facade for cudf-polars Quent traces.

#[cfg(feature = "generate")]
pub mod generate;

use std::collections::HashSet;

use cudf_polars_quent_analyzer::CudfPolarsUiAnalyzer;
use cudf_polars_quent_store::{CudfPolarsEvent, EngineEvent, SCHEMA_DIGEST};
use quent_events::Event;
use quent_query_engine_analyzer::ui::UiAnalyzer;
use quent_query_engine_browser::{BrowserError, BrowserService};
use uuid::Uuid;

/// Browser-bundle model name shared by the producer and this analyzer.
pub const MODEL_NAME: &str = "cudf-polars";

pub(crate) fn engine_id(event: &Event<CudfPolarsEvent>) -> Option<Uuid> {
    matches!(
        event.data,
        CudfPolarsEvent::Engine(EngineEvent::Init { .. })
    )
    .then_some(event.id)
}

fn load_browser_service(
    bytes: &[u8],
) -> Result<BrowserService<CudfPolarsUiAnalyzer>, BrowserError> {
    let bundle = quent_browser_bundle::decode(bytes).map_err(|error| {
        BrowserError::failure(format!("unable to decode browser bundle: {error}"))
    })?;
    if bundle.metadata.model_name != MODEL_NAME {
        return Err(BrowserError::failure(format!(
            "browser bundle model mismatch: expected {MODEL_NAME}, got {}",
            bundle.metadata.model_name
        )));
    }
    if bundle.metadata.schema_digest != SCHEMA_DIGEST {
        return Err(BrowserError::failure(format!(
            "browser bundle schema mismatch: expected {SCHEMA_DIGEST}, got {}",
            bundle.metadata.schema_digest
        )));
    }
    if bundle.metadata.codec != "postcard" {
        return Err(BrowserError::failure(format!(
            "unsupported cudf-polars event codec: {}",
            bundle.metadata.codec
        )));
    }

    let events: Vec<Event<CudfPolarsEvent>> =
        postcard::from_bytes(bundle.payload).map_err(|error| {
            BrowserError::failure(format!("unable to decode cudf-polars events: {error}"))
        })?;
    let engine_ids = events.iter().filter_map(engine_id).collect::<HashSet<_>>();
    if engine_ids.len() != 1 {
        return Err(BrowserError::failure(format!(
            "expected exactly one cudf-polars engine, found {}",
            engine_ids.len()
        )));
    }
    let engine_id = *engine_ids.iter().next().expect("one engine was validated");
    if engine_id != bundle.metadata.engine_id {
        return Err(BrowserError::failure(
            "browser bundle engine ID does not match its events",
        ));
    }
    let analyzer =
        CudfPolarsUiAnalyzer::try_new(engine_id, events.into_iter()).map_err(BrowserError::from)?;
    Ok(BrowserService::new(
        analyzer,
        engine_id,
        bundle.metadata.context_ids,
    ))
}

quent_query_engine_browser::export_wasm_browser_server!(CudfPolarsUiAnalyzer, load_browser_service);

#[cfg(test)]
mod tests {
    use quent_browser_bundle::{Capabilities, FORMAT_VERSION, Metadata};
    use uuid::Uuid;

    use super::*;

    fn bundle(model_name: &str, schema_digest: &str) -> Vec<u8> {
        quent_browser_bundle::encode(
            &Metadata {
                format_version: FORMAT_VERSION,
                model_name: model_name.into(),
                schema_digest: schema_digest.into(),
                model_provenance: None,
                analyzer_revision: "test".into(),
                quent_revision: "test".into(),
                codec: "postcard".into(),
                engine_id: Uuid::nil(),
                context_ids: vec![],
                capabilities: Capabilities::default(),
            },
            // Not a valid event list: identity mismatches must be reported first.
            b"\xff\xff\xff",
        )
        .unwrap()
    }

    #[test]
    fn rejects_schema_digest_mismatch_before_decoding_events() {
        let error = load_browser_service(&bundle(MODEL_NAME, "sha256:other"))
            .err()
            .unwrap();
        assert!(error.to_string().contains("schema mismatch"), "{error}");
    }

    #[test]
    fn rejects_model_mismatch_before_decoding_events() {
        let error = load_browser_service(&bundle("other-model", SCHEMA_DIGEST))
            .err()
            .unwrap();
        assert!(error.to_string().contains("model mismatch"), "{error}");
    }

    #[test]
    fn accepts_matching_schema_digest_and_proceeds_to_decoding() {
        let error = load_browser_service(&bundle(MODEL_NAME, SCHEMA_DIGEST))
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("unable to decode cudf-polars events"),
            "{error}"
        );
    }
}
