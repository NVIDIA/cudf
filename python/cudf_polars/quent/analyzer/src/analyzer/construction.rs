// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use quent_analyzer::{AnalyzerError, AnalyzerResult};
use quent_events::Event;
use quent_query_engine_analyzer::{OperatorEntityMut, QueryEngineModelMut};
use uuid::Uuid;

use super::CudfPolarsUiAnalyzer;
use crate::{
    actor::{ActorBuilder, ActorSpan},
    evaluate::{EvaluateBuilder, EvaluateSpan},
    generated::{
        ActorEvent, CudfPolarsEvent, DataChannelEvent, DeviceMemoryEvent, EngineEvent,
        EvaluateEvent, OperatorEvent, PlanEvent, PortEvent, ProcessorEvent, QueryEvent,
        QueryGroupEvent, StorageEvent, ThreadPoolEvent, WorkerEvent,
    },
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
        let events = events_for_engine(engine_id, events);
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
            .filter_map(|(id, builder)| match builder.try_build(id) {
                Ok(actor) => Some(Ok(actor)),
                Err(AnalyzerError::IncompleteEntity(_)) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<AnalyzerResult<Vec<_>>>()?;
        let evaluates: Vec<EvaluateSpan> = evaluate_builders
            .into_iter()
            .filter_map(|(id, builder)| match builder.try_build(id) {
                Ok(evaluate) => Some(Ok(evaluate)),
                Err(AnalyzerError::IncompleteEntity(_)) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<AnalyzerResult<Vec<_>>>()?;
        let mut model = builder.try_build()?;
        for actor in &actors {
            if let Ok(operator) = model.operator_mut(actor.operator_id) {
                operator.extend_active_span(actor.span);
            }
        }
        let actors = actors.into_iter().map(|actor| (actor.id, actor)).collect();
        let evaluate_indices = evaluates
            .iter()
            .enumerate()
            .map(|(index, evaluate)| (evaluate.id, index))
            .collect();
        Ok(Self {
            model,
            actors,
            evaluates,
            evaluate_indices,
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

fn events_for_engine(
    engine_id: Uuid,
    events: impl Iterator<Item = Event<CudfPolarsEvent>>,
) -> Vec<Event<CudfPolarsEvent>> {
    let events = events.collect::<Vec<_>>();
    let child_to_parent = events
        .iter()
        .filter_map(|event| {
            let parent_id = match &event.data {
                CudfPolarsEvent::Worker(WorkerEvent::Init { engine, .. })
                | CudfPolarsEvent::QueryGroup(QueryGroupEvent::Declared { engine, .. }) => {
                    Some(engine.target)
                }
                CudfPolarsEvent::Query(QueryEvent::Initialized { query_group, .. }) => {
                    Some(query_group.target)
                }
                CudfPolarsEvent::Plan(PlanEvent::Declared { query, .. }) => Some(query.target),
                CudfPolarsEvent::Operator(OperatorEvent::Declared { plan, .. }) => {
                    Some(plan.target)
                }
                CudfPolarsEvent::Port(PortEvent::Declared { operator, .. })
                | CudfPolarsEvent::Actor(ActorEvent::Started { operator, .. }) => {
                    Some(operator.target)
                }
                CudfPolarsEvent::ThreadPool(ThreadPoolEvent::Declared { worker, .. })
                | CudfPolarsEvent::DeviceMemory(DeviceMemoryEvent::Declared { worker, .. })
                | CudfPolarsEvent::Storage(StorageEvent::Declared { worker, .. })
                | CudfPolarsEvent::DataChannel(DataChannelEvent::Declared { worker, .. }) => {
                    Some(worker.target)
                }
                CudfPolarsEvent::Processor(ProcessorEvent::Declared { thread_pool, .. }) => {
                    Some(thread_pool.target)
                }
                CudfPolarsEvent::Evaluate(EvaluateEvent::Queued { actor, .. }) => {
                    Some(actor.target)
                }
                _ => None,
            };
            parent_id.map(|parent_id| (event.id, parent_id))
        })
        .collect::<HashMap<_, _>>();

    let mut belongs_to_engine = HashMap::from([(engine_id, true)]);
    for event in &events {
        if belongs_to_engine.contains_key(&event.id) {
            continue;
        }
        let mut path = Vec::new();
        let mut visited = HashSet::new();
        let mut entity_id = event.id;
        let belongs = loop {
            if let Some(&belongs) = belongs_to_engine.get(&entity_id) {
                break belongs;
            }
            if !visited.insert(entity_id) {
                break false;
            }
            path.push(entity_id);
            let Some(&parent_id) = child_to_parent.get(&entity_id) else {
                break false;
            };
            entity_id = parent_id;
        };
        belongs_to_engine.extend(path.into_iter().map(|id| (id, belongs)));
    }

    events
        .into_iter()
        .filter(|event| belongs_to_engine.get(&event.id) == Some(&true))
        .collect()
}
