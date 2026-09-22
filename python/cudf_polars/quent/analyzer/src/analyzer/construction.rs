// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use quent_analyzer::AnalyzerResult;
use quent_events::Event;
use quent_query_engine_analyzer::{OperatorEntityMut, QueryEngineModelMut};
use uuid::Uuid;

use super::CudfPolarsUiAnalyzer;
use crate::{
    actor::{ActorBuilder, ActorSpan},
    evaluate::{EvaluateBuilder, EvaluateSpan},
    generated::{CudfPolarsEvent, DataChannelEvent, EngineEvent, ProcessorEvent, ThreadPoolEvent},
    model::CudfPolarsModelBuilder,
    resource::{
        DATA_CHANNEL_RESOURCE_TYPE, DeclaredResource, DeclaredResourceGroup,
        PROCESSOR_RESOURCE_TYPE,
    },
};

impl CudfPolarsUiAnalyzer {
    pub(super) fn from_events(
        engine_id: Uuid,
        events: impl Iterator<Item = Event<CudfPolarsEvent>>,
    ) -> AnalyzerResult<Self> {
        let mut builder = CudfPolarsModelBuilder::try_new(engine_id)?;
        let mut actor_builders = HashMap::<Uuid, ActorBuilder>::new();
        let mut evaluate_builders = HashMap::<Uuid, EvaluateBuilder>::new();
        let mut resources = HashMap::new();
        let mut resource_groups = HashMap::new();
        for event in events {
            match &event.data {
                CudfPolarsEvent::Actor(actor_event) => {
                    actor_builders
                        .entry(event.id)
                        .or_default()
                        .push(event.timestamp, actor_event);
                }
                CudfPolarsEvent::Evaluate(evaluate_event) => {
                    evaluate_builders
                        .entry(event.id)
                        .or_default()
                        .push(event.timestamp, evaluate_event);
                }
                CudfPolarsEvent::ThreadPool(ThreadPoolEvent::Declared {
                    instance_name,
                    worker,
                }) => {
                    resource_groups.insert(
                        event.id,
                        DeclaredResourceGroup {
                            id: event.id,
                            instance_name: instance_name.clone(),
                            type_name: "thread_pool",
                            parent_group_id: worker.target,
                        },
                    );
                }
                CudfPolarsEvent::Processor(ProcessorEvent::Declared {
                    instance_name,
                    thread_pool,
                }) => {
                    resources.insert(
                        event.id,
                        DeclaredResource {
                            id: event.id,
                            instance_name: instance_name.clone(),
                            type_name: PROCESSOR_RESOURCE_TYPE,
                            parent_group_id: thread_pool.target,
                        },
                    );
                }
                CudfPolarsEvent::DataChannel(DataChannelEvent::Declared {
                    instance_name,
                    worker,
                    ..
                }) => {
                    resources.insert(
                        event.id,
                        DeclaredResource {
                            id: event.id,
                            instance_name: instance_name.clone(),
                            type_name: DATA_CHANNEL_RESOURCE_TYPE,
                            parent_group_id: worker.target,
                        },
                    );
                }
                _ => {}
            }
            builder.try_push(event)?;
        }
        let actors: Vec<ActorSpan> = actor_builders
            .into_iter()
            .map(|(id, builder)| builder.try_build(id))
            .collect::<AnalyzerResult<Vec<_>>>()?;
        let evaluates: Vec<EvaluateSpan> = evaluate_builders
            .into_iter()
            .map(|(id, builder)| builder.try_build(id))
            .collect::<AnalyzerResult<Vec<_>>>()?;
        let mut model = builder.try_build()?;
        for actor in &actors {
            if let Ok(operator) = model.operator_mut(actor.operator_id) {
                operator.extend_active_span(actor.span);
            }
        }
        Ok(Self {
            model,
            actors,
            evaluates,
            resources,
            resource_groups,
        })
    }

    pub(super) fn extract_engine_from_events(
        engine_id: Uuid,
        events: impl Iterator<Item = Event<CudfPolarsEvent>>,
    ) -> AnalyzerResult<quent_query_engine_ui::Engine> {
        for event in events {
            if event.id == engine_id
                && let CudfPolarsEvent::Engine(EngineEvent::Init {
                    instance_name,
                    implementation,
                    ..
                }) = event.data
            {
                return Ok(quent_query_engine_ui::Engine {
                    id: engine_id,
                    start_time_unix_ns: Some(event.timestamp),
                    duration_s: None,
                    instance_name: Some(instance_name),
                    implementation: Some(quent_query_engine_ui::EngineImplementationAttributes {
                        name: Some(implementation.name),
                        version: Some(implementation.version),
                        custom_attributes: implementation.custom_attributes.0,
                    }),
                });
            }
        }
        Ok(quent_query_engine_ui::Engine::new(engine_id))
    }
}
