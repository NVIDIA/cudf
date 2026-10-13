// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use quent_analyzer::{AnalyzerError, AnalyzerResult, Entity};
use quent_query_engine_analyzer::QueryEngineModel;
use quent_query_engine_ui::{
    EntityListItem, EntityListResponse, OperatorFilter, QueryEngineFsm, QueryFilter,
};
use quent_time::{span::SpanUnixNanoSec, to_nanosecs};
use quent_ui::entities::request::{EntityListRequest, EntityScope, SortDir};

use super::CudfPolarsUiAnalyzer;
use crate::resource::EVALUATE_ENTITY_TYPE;

impl CudfPolarsUiAnalyzer {
    pub(super) fn build_entity_list(
        &self,
        request: EntityListRequest<QueryFilter, OperatorFilter>,
    ) -> AnalyzerResult<EntityListResponse> {
        let query_id = request.app_params.query_id;
        let epoch = self.model.query_epoch(query_id)?;
        let query_operator_ids = self.query_operator_ids(query_id)?;
        let entry = request.entry;
        let requested_operator_ids: HashSet<_> =
            entry.application.operator_ids.into_iter().collect();
        let window = entry.window.try_into_span(epoch)?;
        if entry
            .filter
            .entity_type_name
            .as_deref()
            .is_some_and(|name| name != EVALUATE_ENTITY_TYPE)
        {
            return Ok(EntityListResponse {
                items: vec![],
                total: 0,
            });
        }
        let engine_id = self.model.query_view(query_id)?.engine()?.id();
        let scope = match entry.filter.scope.as_ref() {
            Some(EntityScope::Resource { resource_id }) => {
                if !self.resources.contains_key(resource_id) {
                    return Err(AnalyzerError::InvalidArgument(format!(
                        "unknown resource {resource_id:?}"
                    )));
                }
                Some([*resource_id].into_iter().collect::<HashSet<_>>())
            }
            Some(EntityScope::ResourceGroup {
                resource_group_id,
                resource_type_name,
            }) => {
                if !self.is_known_resource_group(*resource_group_id, engine_id) {
                    return Err(AnalyzerError::InvalidArgument(format!(
                        "unknown resource group {resource_group_id:?}"
                    )));
                }
                Some(
                    self.resources
                        .values()
                        .filter(|resource| {
                            resource.type_name == resource_type_name
                                && self.resource_in_group(resource, *resource_group_id, engine_id)
                        })
                        .map(|resource| resource.id)
                        .collect(),
                )
            }
            None => None,
        };
        let min_usage = entry.filter.min_usage_s.map(to_nanosecs);
        let mut ranked: Vec<_> = self
            .evaluates
            .iter()
            .filter_map(|evaluate| {
                let operator_id = self.evaluate_operator_id(evaluate)?;
                if !query_operator_ids.contains(&operator_id)
                    || (!requested_operator_ids.is_empty()
                        && !requested_operator_ids.contains(&operator_id))
                {
                    return None;
                }
                let lifecycle =
                    SpanUnixNanoSec::try_new(evaluate.queued_at, evaluate.span.end()).ok()?;
                if !lifecycle.intersects(&window) {
                    return None;
                }
                let uses_scope = |resource_id| {
                    scope
                        .as_ref()
                        .is_none_or(|resource_ids| resource_ids.contains(&resource_id))
                };
                let metric = [
                    Some(evaluate.processor_id),
                    evaluate.channel.map(|(id, _)| id),
                ]
                .into_iter()
                .flatten()
                .filter(|&resource_id| uses_scope(resource_id))
                .filter_map(|_| evaluate.span.intersection(&window))
                .map(|span| span.duration())
                .max();
                let metric = match (&scope, metric) {
                    (Some(_), None) => return None,
                    (_, metric) => metric.unwrap_or(0),
                };
                if min_usage.is_some_and(|minimum| metric < minimum) {
                    return None;
                }
                Some((evaluate, operator_id, metric))
            })
            .collect();
        ranked.sort_by(|(left, _, left_metric), (right, _, right_metric)| {
            let order = left_metric.cmp(right_metric);
            let order = match entry.sort.dir {
                SortDir::Asc => order,
                SortDir::Desc => order.reverse(),
            };
            order.then_with(|| left.id.cmp(&right.id))
        });
        let total = ranked.len() as u32;
        let mut ranked = ranked.into_iter();
        let items: Vec<_> = if let Some(page) = entry.page {
            ranked
                .by_ref()
                .skip(page.page.saturating_mul(page.max) as usize)
                .take(page.max as usize)
                .collect()
        } else {
            ranked.collect()
        };
        Ok(EntityListResponse {
            items: items
                .into_iter()
                .map(|(evaluate, operator_id, usage_duration)| EntityListItem {
                    entity: QueryEngineFsm {
                        fsm: evaluate.to_ui_fsm(epoch),
                        operator_id: Some(operator_id),
                    },
                    usage_duration_s: quent_time::to_secs(usage_duration),
                })
                .collect(),
            total,
        })
    }
}
