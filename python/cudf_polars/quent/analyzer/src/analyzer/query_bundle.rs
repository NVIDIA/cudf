// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use quent_analyzer::{AnalyzerResult, Entity, Span};
use quent_dynamic_attributes::DynamicAttribute;
use quent_query_engine_analyzer::{
    EngineEntity, OperatorEntity, PlanEntity, PortEntity, QueryEngineModel, QueryEntity,
    WorkerEntity,
};
use quent_query_engine_ui::{
    EntityRef as UiEntityRef, OperatorStatistic, OperatorStatistics, QueryBundle, QueryEntities,
};
use quent_ui::{
    FsmTypeSelection, Resource, ResourceGroup, ResourceGroupNode, ResourceGroupTypeDecl,
    ResourceTree, ResourceTypeDecl as UiResourceTypeDecl,
    fsm::{FsmStateTypeDecl, FsmTransitionDecl, FsmTypeDecl},
    quantity::{CapacityDecl as UiCapacityDecl, CapacityKind, PrefixSystem, QuantitySpec},
};
use uuid::Uuid;

use super::CudfPolarsUiAnalyzer;
use crate::reservation::{REQUEST_ENTITY_TYPE, RESERVATION_ENTITY_TYPE};
use crate::reservation_metrics::aggregate_operator_reservations;
use crate::resource::{
    DATA_CHANNEL_RESOURCE_TYPE, DEVICE_MEMORY_RESOURCE_TYPE, EVALUATE_ENTITY_TYPE,
    PROCESSOR_RESOURCE_TYPE, RESERVABLE_MEMORY_RESOURCE_TYPE,
};

impl CudfPolarsUiAnalyzer {
    pub(super) fn build_query_bundle(&self, query_id: Uuid) -> AnalyzerResult<QueryBundle> {
        let view = self.model.query_view(query_id)?;
        let query = view.query(query_id)?;
        let epoch = view.query_epoch(query_id)?;
        let worker_ids: HashSet<_> = view.workers().map(|worker| worker.id()).collect();
        let query_resource_group_ids: HashSet<_> = self
            .resource_groups
            .values()
            .filter(|group| worker_ids.contains(&group.parent_group_id))
            .map(|group| group.id)
            .collect();
        let resources: HashMap<_, _> = self
            .resources
            .values()
            .filter(|resource| {
                worker_ids.contains(&resource.parent_group_id)
                    || query_resource_group_ids.contains(&resource.parent_group_id)
            })
            .map(|resource| {
                (
                    resource.id,
                    Resource {
                        id: resource.id,
                        instance_name: resource.instance_name.clone(),
                        type_name: resource.type_name.to_owned(),
                        parent_group_id: resource.parent_group_id,
                    },
                )
            })
            .collect();

        let resource_types = [
            (
                PROCESSOR_RESOURCE_TYPE.to_owned(),
                UiResourceTypeDecl {
                    name: PROCESSOR_RESOURCE_TYPE.to_owned(),
                    capacities: vec![UiCapacityDecl {
                        name: "unit".to_owned(),
                        kind: CapacityKind::Occupancy,
                        quantity: "unit".to_owned(),
                    }],
                    display_order: vec![
                        FsmTypeSelection::All,
                        FsmTypeSelection::Type(EVALUATE_ENTITY_TYPE.to_owned()),
                    ],
                },
            ),
            (
                DEVICE_MEMORY_RESOURCE_TYPE.to_owned(),
                UiResourceTypeDecl {
                    name: DEVICE_MEMORY_RESOURCE_TYPE.to_owned(),
                    capacities: vec![UiCapacityDecl {
                        name: "bytes".to_owned(),
                        kind: CapacityKind::Occupancy,
                        quantity: "bytes".to_owned(),
                    }],
                    display_order: vec![FsmTypeSelection::All],
                },
            ),
            (
                DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
                UiResourceTypeDecl {
                    name: DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
                    capacities: vec![UiCapacityDecl {
                        name: "bytes".to_owned(),
                        kind: CapacityKind::Rate,
                        quantity: "bytes".to_owned(),
                    }],
                    display_order: vec![
                        FsmTypeSelection::All,
                        FsmTypeSelection::Type(EVALUATE_ENTITY_TYPE.to_owned()),
                    ],
                },
            ),
            (
                RESERVABLE_MEMORY_RESOURCE_TYPE.to_owned(),
                UiResourceTypeDecl {
                    name: RESERVABLE_MEMORY_RESOURCE_TYPE.to_owned(),
                    capacities: vec![UiCapacityDecl {
                        name: "bytes".to_owned(),
                        kind: CapacityKind::Occupancy,
                        quantity: "bytes".to_owned(),
                    }],
                    display_order: vec![
                        FsmTypeSelection::All,
                        FsmTypeSelection::Type(RESERVATION_ENTITY_TYPE.to_owned()),
                    ],
                },
            ),
        ]
        .into_iter()
        .collect();
        let group_type = |name: &str, contains_resource_types: Vec<String>| ResourceGroupTypeDecl {
            name: name.to_owned(),
            display_order: vec![
                FsmTypeSelection::All,
                FsmTypeSelection::Type(EVALUATE_ENTITY_TYPE.to_owned()),
            ],
            contains_resource_types,
        };
        let resource_group_types = [
            (
                "engine".to_owned(),
                group_type(
                    "engine",
                    vec![
                        PROCESSOR_RESOURCE_TYPE.to_owned(),
                        DEVICE_MEMORY_RESOURCE_TYPE.to_owned(),
                        DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
                        RESERVABLE_MEMORY_RESOURCE_TYPE.to_owned(),
                    ],
                ),
            ),
            (
                "worker".to_owned(),
                group_type(
                    "worker",
                    vec![
                        PROCESSOR_RESOURCE_TYPE.to_owned(),
                        DEVICE_MEMORY_RESOURCE_TYPE.to_owned(),
                        DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
                        RESERVABLE_MEMORY_RESOURCE_TYPE.to_owned(),
                    ],
                ),
            ),
            (
                "thread_pool".to_owned(),
                group_type("thread_pool", vec![PROCESSOR_RESOURCE_TYPE.to_owned()]),
            ),
        ]
        .into_iter()
        .collect();
        let resource_groups = self
            .resource_groups
            .values()
            .filter(|group| query_resource_group_ids.contains(&group.id))
            .map(|group| {
                (
                    group.id,
                    ResourceGroup {
                        id: group.id,
                        type_name: group.type_name.to_owned(),
                        instance_name: group.instance_name.clone(),
                        parent_group_id: Some(group.parent_group_id),
                    },
                )
            })
            .collect();
        let resource_tree = ResourceTree::ResourceGroup(ResourceGroupNode {
            id: UiEntityRef::Engine(view.engine()?.id()),
            children: worker_ids
                .iter()
                .map(|&worker_id| {
                    let mut children: Vec<_> = self
                        .resources
                        .values()
                        .filter(|resource| resource.parent_group_id == worker_id)
                        .map(|resource| ResourceTree::Resource(UiEntityRef::Resource(resource.id)))
                        .collect();
                    children.extend(
                        self.resource_groups
                            .values()
                            .filter(|group| group.parent_group_id == worker_id)
                            .map(|group| {
                                ResourceTree::ResourceGroup(ResourceGroupNode {
                                    id: UiEntityRef::ResourceGroup(group.id),
                                    children: self
                                        .resources
                                        .values()
                                        .filter(|resource| resource.parent_group_id == group.id)
                                        .map(|resource| {
                                            ResourceTree::Resource(UiEntityRef::Resource(
                                                resource.id,
                                            ))
                                        })
                                        .collect(),
                                })
                            }),
                    );
                    ResourceTree::ResourceGroup(ResourceGroupNode {
                        id: UiEntityRef::Worker(worker_id),
                        children,
                    })
                })
                .collect(),
        });
        let fsm_types = [
            (
                EVALUATE_ENTITY_TYPE.to_owned(),
                FsmTypeDecl {
                    name: EVALUATE_ENTITY_TYPE.to_owned(),
                    states: vec![
                        FsmStateTypeDecl {
                            name: "queued".to_owned(),
                            usages: vec![],
                        },
                        FsmStateTypeDecl {
                            name: "running".to_owned(),
                            usages: vec!["processor".to_owned(), "channel".to_owned()],
                        },
                        FsmStateTypeDecl {
                            name: "completed".to_owned(),
                            usages: vec![],
                        },
                        FsmStateTypeDecl {
                            name: "failed".to_owned(),
                            usages: vec![],
                        },
                    ],
                    transitions: vec![
                        FsmTransitionDecl::Entry("queued".to_owned()),
                        FsmTransitionDecl::Transition("queued".to_owned(), "running".to_owned()),
                        FsmTransitionDecl::Transition("running".to_owned(), "completed".to_owned()),
                        FsmTransitionDecl::Transition("running".to_owned(), "failed".to_owned()),
                        FsmTransitionDecl::Exit("completed".to_owned()),
                        FsmTransitionDecl::Exit("failed".to_owned()),
                    ],
                },
            ),
            (
                RESERVATION_ENTITY_TYPE.to_owned(),
                FsmTypeDecl {
                    name: RESERVATION_ENTITY_TYPE.to_owned(),
                    states: vec![
                        FsmStateTypeDecl {
                            name: "created".to_owned(),
                            usages: vec![],
                        },
                        FsmStateTypeDecl {
                            name: "held".to_owned(),
                            usages: vec!["memory".to_owned()],
                        },
                        FsmStateTypeDecl {
                            name: "released".to_owned(),
                            usages: vec![],
                        },
                    ],
                    transitions: vec![
                        FsmTransitionDecl::Entry("created".to_owned()),
                        FsmTransitionDecl::Transition("created".to_owned(), "held".to_owned()),
                        FsmTransitionDecl::Transition("held".to_owned(), "held".to_owned()),
                        FsmTransitionDecl::Transition("held".to_owned(), "released".to_owned()),
                        FsmTransitionDecl::Exit("released".to_owned()),
                    ],
                },
            ),
            (
                REQUEST_ENTITY_TYPE.to_owned(),
                FsmTypeDecl {
                    name: REQUEST_ENTITY_TYPE.to_owned(),
                    states: [
                        "submitted",
                        "waiting",
                        "granted",
                        "empty",
                        "failed",
                        "cancelled",
                    ]
                    .into_iter()
                    .map(|name| FsmStateTypeDecl {
                        name: name.to_owned(),
                        usages: vec![],
                    })
                    .collect(),
                    transitions: vec![
                        FsmTransitionDecl::Entry("submitted".to_owned()),
                        FsmTransitionDecl::Transition("submitted".to_owned(), "waiting".to_owned()),
                        FsmTransitionDecl::Transition("submitted".to_owned(), "granted".to_owned()),
                        FsmTransitionDecl::Transition("submitted".to_owned(), "empty".to_owned()),
                        FsmTransitionDecl::Transition("submitted".to_owned(), "failed".to_owned()),
                        FsmTransitionDecl::Transition(
                            "submitted".to_owned(),
                            "cancelled".to_owned(),
                        ),
                        FsmTransitionDecl::Transition("waiting".to_owned(), "granted".to_owned()),
                        FsmTransitionDecl::Transition("waiting".to_owned(), "empty".to_owned()),
                        FsmTransitionDecl::Transition("waiting".to_owned(), "failed".to_owned()),
                        FsmTransitionDecl::Transition("waiting".to_owned(), "cancelled".to_owned()),
                        FsmTransitionDecl::Exit("granted".to_owned()),
                        FsmTransitionDecl::Exit("empty".to_owned()),
                        FsmTransitionDecl::Exit("failed".to_owned()),
                        FsmTransitionDecl::Exit("cancelled".to_owned()),
                    ],
                },
            ),
        ]
        .into_iter()
        .collect();

        let query_operator_ids = view.operators().map(|operator| operator.id()).collect();
        let reservation_metrics =
            aggregate_operator_reservations(&self.reservations, &self.actors, &query_operator_ids);
        let mut operators: HashMap<_, _> = view
            .operators()
            .map(|operator| (operator.id(), operator.to_ui(epoch)))
            .collect();
        for (operator_id, metrics) in reservation_metrics {
            let operator = operators
                .get_mut(&operator_id)
                .expect("reservation metric operator came from this query");
            let statistics = operator
                .statistics
                .get_or_insert_with(|| OperatorStatistics {
                    custom_statistics: vec![],
                });
            statistics.custom_statistics.extend([
                OperatorStatistic {
                    value: DynamicAttribute::u64(
                        "reservation_peak_unconsumed_bytes",
                        metrics.peak_bytes,
                    ),
                    quantity: Some("bytes".to_owned()),
                },
                OperatorStatistic {
                    value: DynamicAttribute::f64(
                        "reservation_unconsumed_byte_seconds",
                        metrics.byte_seconds,
                    ),
                    quantity: Some("byte-seconds".to_owned()),
                },
                OperatorStatistic {
                    value: DynamicAttribute::u64(
                        "reservation_handle_count",
                        metrics.reservation_count,
                    ),
                    quantity: None,
                },
            ]);
        }

        let entities = QueryEntities {
            engine: view.engine()?.to_ui()?,
            query: query.to_ui()?,
            workers: view
                .workers()
                .map(|worker| (worker.id(), worker.to_ui(epoch)))
                .collect(),
            plans: view.plans().map(|plan| (plan.id(), plan.to_ui())).collect(),
            operators,
            ports: view
                .ports()
                .map(|port| (port.id(), port.to_ui(epoch)))
                .collect(),
            resource_types,
            resources,
            resource_groups,
            resource_group_types,
            fsm_types,
        };
        let unique_operator_names = view
            .operators()
            .filter_map(|operator| operator.operator_type_name().map(str::to_owned))
            .collect();
        let plan_tree = view.plan_tree(query_id)?.to_ui();
        Ok(QueryBundle {
            query_id,
            entities,
            plan_tree,
            resource_tree,
            unique_operator_names,
            quantity_specs: [
                ("bytes".to_owned(), QuantitySpec::bytes()),
                (
                    "byte-seconds".to_owned(),
                    QuantitySpec {
                        symbol: "B·s".to_owned(),
                        singular: "byte-second".to_owned(),
                        plural: "byte-seconds".to_owned(),
                        occupancy_prefix: PrefixSystem::Iec,
                        rate_prefix: PrefixSystem::Iec,
                    },
                ),
                ("unit".to_owned(), QuantitySpec::unit()),
            ]
            .into(),
            start_time_unix_ns: epoch,
            duration_s: quent_time::to_secs(query.span()?.duration()),
        })
    }
}
