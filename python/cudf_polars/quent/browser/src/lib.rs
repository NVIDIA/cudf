// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Browser-hosted API facade for cudf-polars Quent traces.

#[cfg(feature = "generate")]
pub mod generate;

use std::collections::HashSet;

use cudf_polars_quent_analyzer::CudfPolarsUiAnalyzer;
use cudf_polars_quent_store::{CudfPolarsEvent, EngineEvent};
use quent_events::Event;
use quent_query_engine_analyzer::ui::UiAnalyzer;
use quent_query_engine_browser::{BrowserError, BrowserService};

const MODEL_NAME: &str = "cudf-polars";

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
    let expected_digest = schema_digest();
    if bundle.metadata.schema_digest != expected_digest {
        return Err(BrowserError::failure(format!(
            "browser bundle schema mismatch: expected {expected_digest}, got {}",
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
    let engine_ids = events
        .iter()
        .filter_map(|event| {
            matches!(
                event.data,
                CudfPolarsEvent::Engine(EngineEvent::Init { .. })
            )
            .then_some(event.id)
        })
        .collect::<HashSet<_>>();
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

/// Digest of the cudf-polars event schema accepted by this analyzer.
pub fn schema_digest() -> String {
    let digest = include_bytes!("../../model.yaml")
        .iter()
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
    format!("{digest:016x}")
}
