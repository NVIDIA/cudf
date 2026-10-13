// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use quent_analyzer::{AnalyzerError, AnalyzerResult, Entity, Span};
use quent_query_engine_analyzer::{
    EngineEntity, OperatorEntity, PlanEntity, PortEntity, QueryEngineModel, QueryEntity,
    QueryGroupEntity, WorkerEntity,
};
use quent_query_engine_ui::{EntityRef as UiEntityRef, QueryBundle, QueryEntities};
use quent_ui::{
    Resource, ResourceGroup, ResourceGroupNode, ResourceGroupTypeDecl, ResourceTree,
    ResourceTypeDecl as UiResourceTypeDecl,
    fsm::{FsmStateTypeDecl, FsmTransitionDecl, FsmTypeDecl},
    quantity::{CapacityDecl as UiCapacityDecl, CapacityKind, QuantitySpec},
};
use uuid::Uuid;

use super::CudfPolarsUiAnalyzer;
use crate::resource::{DATA_CHANNEL_RESOURCE_TYPE, EVALUATE_ENTITY_TYPE, PROCESSOR_RESOURCE_TYPE};

impl CudfPolarsUiAnalyzer {
    pub(super) fn build_query_bundle(&self, query_id: Uuid) -> AnalyzerResult<QueryBundle> {
        let view = self.model.query_view(query_id)?;
        let query = view.query(query_id)?;
        let epoch = view.query_epoch(query_id)?;
        let query_group_id = query.query_group_id().ok_or_else(|| {
            AnalyzerError::IncompleteEntity(format!(
                "query {query_id} has no query-group reference"
            ))
        })?;
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
                    used_by: vec![EVALUATE_ENTITY_TYPE.to_owned()],
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
                    used_by: vec![EVALUATE_ENTITY_TYPE.to_owned()],
                },
            ),
        ]
        .into_iter()
        .collect();
        let group_type = |name: &str, contains_resource_types: Vec<String>| ResourceGroupTypeDecl {
            name: name.to_owned(),
            used_by_entity_types: vec![EVALUATE_ENTITY_TYPE.to_owned()],
            contains_resource_types,
        };
        let resource_group_types = [
            (
                "engine".to_owned(),
                group_type(
                    "engine",
                    vec![
                        PROCESSOR_RESOURCE_TYPE.to_owned(),
                        DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
                    ],
                ),
            ),
            (
                "worker".to_owned(),
                group_type(
                    "worker",
                    vec![
                        PROCESSOR_RESOURCE_TYPE.to_owned(),
                        DATA_CHANNEL_RESOURCE_TYPE.to_owned(),
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
        let fsm_types = [(
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
        )]
        .into_iter()
        .collect();

        let entities = QueryEntities {
            engine: view.engine()?.to_ui()?,
            query_group: view.query_group(query_group_id)?.to_ui(),
            query: query.to_ui()?,
            workers: view
                .workers()
                .map(|worker| (worker.id(), worker.to_ui(epoch)))
                .collect(),
            plans: view.plans().map(|plan| (plan.id(), plan.to_ui())).collect(),
            operators: view
                .operators()
                .map(|operator| (operator.id(), operator.to_ui(epoch)))
                .collect(),
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
                ("unit".to_owned(), QuantitySpec::unit()),
            ]
            .into(),
            start_time_unix_ns: epoch,
            duration_s: quent_time::to_secs(query.span()?.duration()),
        })
    }
}
