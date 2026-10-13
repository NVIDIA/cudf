// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::path::Path;

use quent_analyzer::context::ContextInventory;
use quent_events::Event;
use quent_query_engine_analyzer::ui::{QuentViewer, ViewerEventStream};
use quent_store::event::{EntityEventStore, ModelEventLoader};
use uuid::Uuid;

use crate::{
    CudfPolarsUiAnalyzer,
    generated::{self, CudfPolars, WorkerEvent},
};

/// Entry point discovered by `quent-open`.
pub struct Viewer;

impl QuentViewer for Viewer {
    type Analyzer = CudfPolarsUiAnalyzer;

    fn context_inventory(dir: &Path) -> quent_io::ImporterResult<ContextInventory> {
        let (context_id, root) = context_location(dir)?;
        let store = quent_store::event::filesystem::Store::<CudfPolars>::new(root);
        let engine_ids = store
            .entity_events::<generated::Engine>(context_id)
            .map_err(quent_io::ImporterError::other)?
            .map(|event| event.map(|event| event.id))
            .collect::<Result<HashSet<_>, _>>()
            .map_err(quent_io::ImporterError::other)?;
        let worker_engine_ids = store
            .entity_events::<generated::Worker>(context_id)
            .map_err(quent_io::ImporterError::other)?
            .filter_map(|event| match event {
                Ok(Event {
                    data: WorkerEvent::Init { engine, .. },
                    ..
                }) => Some(Ok(engine.target)),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<HashSet<_>, _>>()
            .map_err(quent_io::ImporterError::other)?;
        Ok(ContextInventory {
            analysis_target_ids: engine_ids.into_iter().chain(worker_engine_ids).collect(),
        })
    }

    fn import_events(dir: &Path) -> quent_io::ImporterResult<ViewerEventStream<Self::Analyzer>> {
        let (context_id, root) = context_location(dir)?;
        let events = quent_store::event::filesystem::Store::<CudfPolars>::new(root)
            .load_model_events(context_id)
            .map_err(quent_io::ImporterError::other)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(quent_io::ImporterError::other)?;
        Ok(Box::new(events.into_iter()))
    }
}

fn context_location(dir: &Path) -> quent_io::ImporterResult<(Uuid, &Path)> {
    let invalid_path = || {
        quent_io::ImporterError::other(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("context directory must end in a UUID: {}", dir.display()),
        ))
    };
    let context_id = dir
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| Uuid::parse_str(name).ok())
        .ok_or_else(invalid_path)?;
    let root = dir.parent().ok_or_else(invalid_path)?;
    Ok((context_id, root))
}
