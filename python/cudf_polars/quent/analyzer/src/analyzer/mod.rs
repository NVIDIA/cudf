// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod construction;
mod entities;
mod query_bundle;
mod timeline;

use std::collections::{HashMap, HashSet};

use quent_analyzer::{AnalyzerError, AnalyzerResult, Entity};
use quent_events::Event;
use quent_query_engine_analyzer::{QueryEngineModel, ui::UiAnalyzer};
use quent_query_engine_ui::{
    DataFlowTimelineBinned, EntityListResponse, OperatorFilter, QueryBundle, QueryFilter,
};
use quent_ui::{
    entities::request::EntityListRequest,
    timeline::{
        categorical::CategoricalTimelineRequest,
        request::{BulkTimelineRequest, SingleTimelineRequest},
        response::{BulkTimelinesResponse, SingleTimelineResponse},
    },
};
use uuid::Uuid;

use crate::{
    actor::ActorSpan,
    evaluate::EvaluateSpan,
    generated::CudfPolarsEvent,
    model::CudfPolarsModel,
    resource::{DeclaredResource, DeclaredResourceGroup},
};

/// Query-engine UI adapter for the schema-generated cudf-polars event model.
pub struct CudfPolarsUiAnalyzer {
    pub(super) model: CudfPolarsModel,
    pub(super) actors: HashMap<Uuid, ActorSpan>,
    pub(super) evaluates: Vec<EvaluateSpan>,
    pub(super) evaluate_indices: HashMap<Uuid, usize>,
    pub(super) resources: HashMap<Uuid, DeclaredResource>,
    pub(super) resource_groups: HashMap<Uuid, DeclaredResourceGroup>,
}

impl CudfPolarsUiAnalyzer {
    pub(super) fn query_operator_ids(&self, query_id: Uuid) -> AnalyzerResult<HashSet<Uuid>> {
        Ok(self
            .model
            .query_view(query_id)?
            .operators()
            .map(|operator| operator.id())
            .collect())
    }

    pub(super) fn evaluate_operator_id(&self, evaluate: &EvaluateSpan) -> Option<Uuid> {
        self.actors
            .get(&evaluate.actor_id)
            .map(|actor| actor.operator_id)
    }

    pub(super) fn evaluate(&self, id: Uuid) -> Option<&EvaluateSpan> {
        self.evaluate_indices
            .get(&id)
            .and_then(|&index| self.evaluates.get(index))
    }

    pub(super) fn is_known_resource_group(&self, group_id: Uuid, engine_id: Uuid) -> bool {
        group_id == engine_id
            || self.model.worker(group_id).is_ok()
            || self.resource_groups.contains_key(&group_id)
    }

    pub(super) fn resource_in_group(
        &self,
        resource: &DeclaredResource,
        group_id: Uuid,
        engine_id: Uuid,
    ) -> bool {
        group_id == engine_id
            || resource.parent_group_id == group_id
            || self
                .resource_groups
                .get(&resource.parent_group_id)
                .is_some_and(|group| group.parent_group_id == group_id)
    }
}

impl UiAnalyzer for CudfPolarsUiAnalyzer {
    type Event = CudfPolarsEvent;

    fn try_new(
        engine_id: Uuid,
        events: impl Iterator<Item = Event<Self::Event>>,
    ) -> AnalyzerResult<Self> {
        Self::from_events(engine_id, events)
    }

    fn extract_engine(
        engine_id: Uuid,
        events: impl Iterator<Item = Event<Self::Event>>,
    ) -> AnalyzerResult<quent_query_engine_ui::Engine> {
        Self::extract_engine_from_events(engine_id, events)
    }

    fn query_bundle(&self, query_id: Uuid) -> AnalyzerResult<QueryBundle> {
        self.build_query_bundle(query_id)
    }

    fn query_engine_model(&self) -> &impl QueryEngineModel {
        &self.model
    }

    fn single_resource_timeline(
        &self,
        request: SingleTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<SingleTimelineResponse> {
        self.build_single_resource_timeline(request)
    }

    fn list_entities(
        &self,
        request: EntityListRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<EntityListResponse> {
        self.build_entity_list(request)
    }

    fn bulk_resource_timeline(
        &self,
        request: BulkTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<BulkTimelinesResponse> {
        self.build_bulk_resource_timeline(request)
    }

    fn data_flow_timeline(
        &self,
        _request: CategoricalTimelineRequest<QueryFilter>,
    ) -> AnalyzerResult<DataFlowTimelineBinned> {
        Err(AnalyzerError::Unsupported)
    }
}
