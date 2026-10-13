// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity,
    timeline::binned::resource::{ResourceTimelineBuilder, ResourceTimelineByKeyBuilder},
};
use quent_query_engine_analyzer::QueryEngineModel;
use quent_query_engine_ui::{OperatorFilter, QueryFilter};
use quent_time::{TimeNanoSec, to_nanosecs};
use quent_ui::timeline::{
    request::{BulkTimelineRequest, SingleTimelineRequest, TimelineRequest},
    response::{
        BulkTimelinesResponse, BulkTimelinesResponseEntry, ResourceTimeline as UiResourceTimeline,
        ResourceTimelineBinned, ResourceTimelineBinnedByState, SingleTimelineResponse,
    },
};
use uuid::Uuid;

use super::CudfPolarsUiAnalyzer;
use crate::{
    evaluate::EvaluateUsage,
    resource::{
        DATA_CHANNEL_RESOURCE_TYPE, EVALUATE_ENTITY_TYPE, PROCESSOR_RESOURCE_TYPE,
        evaluate_resource_type,
    },
};

impl CudfPolarsUiAnalyzer {
    fn build_timeline(
        &self,
        query_id: Uuid,
        request: TimelineRequest<OperatorFilter>,
    ) -> AnalyzerResult<(quent_time::bin::BinnedSpanSec, UiResourceTimeline)> {
        let epoch = self.model.query_epoch(query_id)?;
        let config = request.config().try_into_binned_span(epoch)?;
        let query_operator_ids = self.query_operator_ids(query_id)?;
        let engine_id = self.model.query_view(query_id)?.engine()?.id();

        let (
            selected_resource_ids,
            entity_type_name,
            requested_operator_ids,
            resource_type_name,
            long_entities_threshold,
        ): (
            HashSet<Uuid>,
            Option<String>,
            HashSet<Uuid>,
            String,
            Option<TimeNanoSec>,
        ) = match request {
            TimelineRequest::Resource(request) => {
                let resource_type_name = self
                    .resources
                    .get(&request.resource_id)
                    .ok_or_else(|| {
                        AnalyzerError::InvalidArgument(format!(
                            "unknown resource {:?}",
                            request.resource_id
                        ))
                    })?
                    .type_name
                    .to_owned();
                (
                    [request.resource_id].into_iter().collect(),
                    request.entity_filter.entity_type_name,
                    request.application.operator_ids.into_iter().collect(),
                    resource_type_name,
                    request.long_entities_threshold_s.map(to_nanosecs),
                )
            }
            TimelineRequest::ResourceGroup(request) => {
                if !self.is_known_resource_group(request.resource_group_id, engine_id) {
                    return Err(AnalyzerError::InvalidArgument(format!(
                        "unknown resource group {:?}",
                        request.resource_group_id
                    )));
                }
                let resource_ids: HashSet<Uuid> = self
                    .resources
                    .values()
                    .filter(|resource| {
                        resource.type_name == request.resource_type_name
                            && self.resource_in_group(
                                resource,
                                request.resource_group_id,
                                engine_id,
                            )
                    })
                    .map(|resource| resource.id)
                    .collect();
                (
                    resource_ids,
                    request.entity_filter.entity_type_name,
                    request.app_params.operator_ids.into_iter().collect(),
                    request.resource_type_name,
                    request.long_entities_threshold_s.map(to_nanosecs),
                )
            }
        };

        if !matches!(
            resource_type_name.as_str(),
            PROCESSOR_RESOURCE_TYPE | DATA_CHANNEL_RESOURCE_TYPE
        ) {
            return Err(AnalyzerError::InvalidArgument(format!(
                "unknown resource type {resource_type_name:?}"
            )));
        }
        if entity_type_name
            .as_deref()
            .is_some_and(|name| name != EVALUATE_ENTITY_TYPE)
        {
            return Err(AnalyzerError::InvalidArgument(format!(
                "unknown timeline entity type {:?}; expected {EVALUATE_ENTITY_TYPE:?}",
                entity_type_name.unwrap()
            )));
        }

        let evaluates = self.evaluates.iter().filter(|evaluate| {
            self.evaluate_operator_id(evaluate)
                .is_some_and(|operator_id| {
                    query_operator_ids.contains(&operator_id)
                        && (requested_operator_ids.is_empty()
                            || requested_operator_ids.contains(&operator_id))
                })
        });
        let resource_type = evaluate_resource_type(&resource_type_name);
        let data = if entity_type_name.is_some() {
            let mut builder = ResourceTimelineByKeyBuilder::try_new(
                &resource_type,
                config,
                long_entities_threshold,
            )?;
            for evaluate in evaluates {
                if resource_type_name == PROCESSOR_RESOURCE_TYPE
                    && selected_resource_ids.contains(&evaluate.processor_id)
                {
                    builder.try_push(
                        "running",
                        &EvaluateUsage {
                            evaluate,
                            resource_id: evaluate.processor_id,
                            capacity: &evaluate.processor_unit,
                        },
                    )?;
                }
                if resource_type_name == DATA_CHANNEL_RESOURCE_TYPE
                    && let (Some((channel_id, _)), Some(channel_bytes)) =
                        (evaluate.channel, evaluate.channel_bytes.as_ref())
                    && selected_resource_ids.contains(&channel_id)
                {
                    builder.try_push(
                        "running",
                        &EvaluateUsage {
                            evaluate,
                            resource_id: channel_id,
                            capacity: channel_bytes,
                        },
                    )?;
                }
            }
            let result = builder.build();
            let long_fsms = result
                .long_entities
                .iter()
                .filter_map(|id| self.evaluate(*id))
                .map(|evaluate| evaluate.to_ui_fsm(epoch))
                .collect();
            let mut capacities_states_values = HashMap::new();
            for ((state_name, capacity_name), values) in result.data {
                capacities_states_values
                    .entry(capacity_name.to_owned())
                    .or_insert_with(HashMap::new)
                    .insert(state_name.to_owned(), values);
            }
            UiResourceTimeline::BinnedByState(ResourceTimelineBinnedByState {
                config: result.config.try_to_secs_relative(epoch)?,
                capacities_states_values,
                long_fsms,
            })
        } else {
            let mut builder =
                ResourceTimelineBuilder::try_new(&resource_type, config, long_entities_threshold)?;
            for evaluate in evaluates {
                if resource_type_name == PROCESSOR_RESOURCE_TYPE
                    && selected_resource_ids.contains(&evaluate.processor_id)
                {
                    builder.try_push(&EvaluateUsage {
                        evaluate,
                        resource_id: evaluate.processor_id,
                        capacity: &evaluate.processor_unit,
                    })?;
                }
                if resource_type_name == DATA_CHANNEL_RESOURCE_TYPE
                    && let (Some((channel_id, _)), Some(channel_bytes)) =
                        (evaluate.channel, evaluate.channel_bytes.as_ref())
                    && selected_resource_ids.contains(&channel_id)
                {
                    builder.try_push(&EvaluateUsage {
                        evaluate,
                        resource_id: channel_id,
                        capacity: channel_bytes,
                    })?;
                }
            }
            let result = builder.build();
            let long_fsms = result
                .long_entities
                .iter()
                .filter_map(|id| self.evaluate(*id))
                .map(|evaluate| evaluate.to_ui_fsm(epoch))
                .collect();
            UiResourceTimeline::Binned(ResourceTimelineBinned {
                config: result.config.try_to_secs_relative(epoch)?,
                capacities_values: result
                    .data
                    .into_iter()
                    .map(|(name, values)| (name.to_owned(), values))
                    .collect(),
                long_fsms,
            })
        };
        Ok((config.try_to_secs_relative(epoch)?, data))
    }

    pub(super) fn build_single_resource_timeline(
        &self,
        request: SingleTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<SingleTimelineResponse> {
        let (config, data) = self.build_timeline(request.app_params.query_id, request.entry)?;
        Ok(SingleTimelineResponse { config, data })
    }

    pub(super) fn build_bulk_resource_timeline(
        &self,
        request: BulkTimelineRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<BulkTimelinesResponse> {
        let entries = request
            .entries
            .into_iter()
            .map(|(entry_id, entry)| {
                let response = match self.build_timeline(request.app_params.query_id, entry) {
                    Ok((config, data)) => BulkTimelinesResponseEntry::Ok {
                        message: String::new(),
                        config,
                        data,
                    },
                    Err(error) => BulkTimelinesResponseEntry::Error {
                        message: error.to_string(),
                    },
                };
                (entry_id, response)
            })
            .collect();
        Ok(BulkTimelinesResponse { entries })
    }
}
