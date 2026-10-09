// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::AnalyzerError;
use quent_dynamic_attributes::DynamicAttributes;
use quent_events::{EntityRef, Event, EventModel};
use quent_query_engine_analyzer::ui::{QuentViewer, UiAnalyzer};
use quent_query_engine_ui::{OperatorFilter, QueryFilter};
use quent_ui::{
    entities::request::{EntityListRequest, EntityScope, SortDir},
    timeline::{
        request::{BulkTimelineRequest, TimelineRequest},
        response::{BulkTimelinesResponseEntry, ResourceTimeline as UiResourceTimeline},
    },
};
use uuid::Uuid;

use crate::{
    CudfPolarsUiAnalyzer, Viewer,
    generated::{
        ActorEvent, CudfPolars, CudfPolarsEvent, EngineEvent, EvaluateEvent, Implementation,
        OperatorEvent, OperatorStatistics, PlanEvent, ProcessorEvent, QueryEvent, QueryGroupEvent,
        ThreadPoolEvent, WorkerEvent,
    },
    resource::{EVALUATE_ENTITY_TYPE, PROCESSOR_RESOURCE_TYPE},
};

fn entity_list_request(
    query_id: Uuid,
    scope: Option<EntityScope>,
    operator_ids: Vec<Uuid>,
    min_usage_s: Option<f64>,
) -> EntityListRequest<QueryFilter, OperatorFilter> {
    EntityListRequest {
        entry: quent_ui::entities::request::EntityListEntry {
            window: quent_ui::entities::request::TimeWindow {
                start: 0.0,
                end: 20e-9,
            },
            filter: quent_ui::entities::request::EntityListFilter {
                scope,
                entity_type_name: Some(EVALUATE_ENTITY_TYPE.to_owned()),
                min_usage_s,
            },
            sort: quent_ui::entities::request::Sort {
                key: quent_ui::entities::request::EntitySortKey::UsageDuration,
                dir: SortDir::Desc,
            },
            page: None,
            application: OperatorFilter { operator_ids },
        },
        app_params: QueryFilter { query_id },
    }
}

#[test]
fn builds_query_bundle_from_generated_events() {
    let engine_id = Uuid::now_v7();
    let foreign_engine_id = Uuid::now_v7();
    let worker_id = Uuid::now_v7();
    let foreign_worker_id = Uuid::now_v7();
    let idle_worker_id = Uuid::now_v7();
    let query_group_id = Uuid::now_v7();
    let query_id = Uuid::now_v7();
    let plan_id = Uuid::now_v7();
    let idle_plan_id = Uuid::now_v7();
    let operator_id = Uuid::now_v7();
    let actor_id = Uuid::now_v7();
    let incomplete_actor_id = Uuid::now_v7();
    let thread_pool_id = Uuid::now_v7();
    let processor_id = Uuid::now_v7();
    let evaluate_id = Uuid::now_v7();
    let incomplete_evaluate_id = Uuid::now_v7();
    let events = vec![
        Event::new(
            engine_id,
            1,
            CudfPolarsEvent::Engine(EngineEvent::Init {
                seq: 0,
                instance_name: "cudf-polars".to_owned(),
                implementation: Implementation {
                    name: "cudf-polars".to_owned(),
                    version: "test".to_owned(),
                    backend: "spmd".to_owned(),
                    custom_attributes: DynamicAttributes::new(),
                },
            }),
        ),
        Event::new(
            foreign_engine_id,
            1,
            CudfPolarsEvent::Engine(EngineEvent::Init {
                seq: 0,
                instance_name: "foreign-cudf-polars".to_owned(),
                implementation: Implementation {
                    name: "cudf-polars".to_owned(),
                    version: "test".to_owned(),
                    backend: "spmd".to_owned(),
                    custom_attributes: DynamicAttributes::new(),
                },
            }),
        ),
        Event::new(
            foreign_worker_id,
            2,
            CudfPolarsEvent::Worker(WorkerEvent::Init {
                seq: 0,
                instance_name: "foreign-rank-0".to_owned(),
                engine: EntityRef::new(foreign_engine_id, ()),
            }),
        ),
        Event::new(
            worker_id,
            2,
            CudfPolarsEvent::Worker(WorkerEvent::Init {
                seq: 0,
                instance_name: "rank-0".to_owned(),
                engine: EntityRef::new(engine_id, ()),
            }),
        ),
        Event::new(
            idle_worker_id,
            2,
            CudfPolarsEvent::Worker(WorkerEvent::Init {
                seq: 0,
                instance_name: "rank-1".to_owned(),
                engine: EntityRef::new(engine_id, ()),
            }),
        ),
        Event::new(
            query_group_id,
            3,
            CudfPolarsEvent::QueryGroup(QueryGroupEvent::Declared {
                instance_name: Some("group".to_owned()),
                engine: EntityRef::new(engine_id, ()),
            }),
        ),
        Event::new(
            query_id,
            4,
            CudfPolarsEvent::Query(QueryEvent::Initialized {
                seq: 0,
                instance_name: "iteration-1".to_owned(),
                query_group: EntityRef::new(query_group_id, ()),
            }),
        ),
        Event::new(
            query_id,
            5,
            CudfPolarsEvent::Query(QueryEvent::Planning { seq: 1 }),
        ),
        Event::new(
            query_id,
            6,
            CudfPolarsEvent::Query(QueryEvent::Executing { seq: 2 }),
        ),
        Event::new(
            plan_id,
            7,
            CudfPolarsEvent::Plan(PlanEvent::Declared {
                instance_name: "logical".to_owned(),
                query: EntityRef::new(query_id, ()),
                parent_plan: None,
                worker: Some(EntityRef::new(worker_id, ())),
                edges: vec![],
            }),
        ),
        Event::new(
            idle_plan_id,
            7,
            CudfPolarsEvent::Plan(PlanEvent::Declared {
                instance_name: "idle-physical".to_owned(),
                query: EntityRef::new(query_id, ()),
                parent_plan: Some(EntityRef::new(plan_id, ())),
                worker: Some(EntityRef::new(idle_worker_id, ())),
                edges: vec![],
            }),
        ),
        Event::new(
            operator_id,
            8,
            CudfPolarsEvent::Operator(OperatorEvent::Declared {
                plan: EntityRef::new(plan_id, ()),
                parent_operators: vec![],
                instance_name: "scan".to_owned(),
                type_name: "Scan".to_owned(),
                node_id: "0".to_owned(),
            }),
        ),
        Event::new(
            thread_pool_id,
            8,
            CudfPolarsEvent::ThreadPool(ThreadPoolEvent::Declared {
                instance_name: "host threads".to_owned(),
                worker: EntityRef::new(worker_id, ()),
            }),
        ),
        Event::new(
            processor_id,
            8,
            CudfPolarsEvent::Processor(ProcessorEvent::Declared {
                instance_name: "thread-0".to_owned(),
                thread_pool: EntityRef::new(thread_pool_id, ()),
            }),
        ),
        Event::new(
            actor_id,
            9,
            CudfPolarsEvent::Actor(ActorEvent::Started {
                seq: 0,
                operator: EntityRef::new(operator_id, ()),
                worker: EntityRef::new(worker_id, ()),
            }),
        ),
        Event::new(
            actor_id,
            10,
            CudfPolarsEvent::Actor(ActorEvent::Running { seq: 1 }),
        ),
        Event::new(
            evaluate_id,
            11,
            CudfPolarsEvent::Evaluate(EvaluateEvent::Queued {
                seq: 0,
                instance_name: "Scan-evaluate".to_owned(),
                actor: EntityRef::new(actor_id, ()),
            }),
        ),
        Event::new(
            evaluate_id,
            12,
            CudfPolarsEvent::Evaluate(EvaluateEvent::Running {
                seq: 1,
                io: false,
                input_bytes: 10,
                processor: EntityRef::new(processor_id, crate::generated::ProcessorUsage {}),
                channel: None,
            }),
        ),
        Event::new(
            evaluate_id,
            18,
            CudfPolarsEvent::Evaluate(EvaluateEvent::Completed {
                seq: 2,
                output_bytes: 20,
            }),
        ),
        Event::new(
            actor_id,
            20,
            CudfPolarsEvent::Actor(ActorEvent::Completed {
                seq: 2,
                values: OperatorStatistics {
                    input_bytes: 10,
                    output_bytes: 20,
                    output_rows: Some(1),
                    chunk_count: 1,
                    duplicated: false,
                    decision: None,
                },
            }),
        ),
        Event::new(
            operator_id,
            20,
            CudfPolarsEvent::Operator(OperatorEvent::Statistics {
                values: OperatorStatistics {
                    input_bytes: 10,
                    output_bytes: 20,
                    output_rows: Some(1),
                    chunk_count: 1,
                    duplicated: false,
                    decision: None,
                },
            }),
        ),
        Event::new(
            incomplete_actor_id,
            20,
            CudfPolarsEvent::Actor(ActorEvent::Started {
                seq: 0,
                operator: EntityRef::new(operator_id, ()),
                worker: EntityRef::new(worker_id, ()),
            }),
        ),
        Event::new(
            incomplete_evaluate_id,
            20,
            CudfPolarsEvent::Evaluate(EvaluateEvent::Queued {
                seq: 0,
                instance_name: "incomplete-evaluate".to_owned(),
                actor: EntityRef::new(incomplete_actor_id, ()),
            }),
        ),
        Event::new(
            query_id,
            21,
            CudfPolarsEvent::Query(QueryEvent::Completed { seq: 3 }),
        ),
        Event::new(
            worker_id,
            22,
            CudfPolarsEvent::Worker(WorkerEvent::Exit { seq: 1 }),
        ),
        Event::new(
            idle_worker_id,
            22,
            CudfPolarsEvent::Worker(WorkerEvent::Exit { seq: 1 }),
        ),
        Event::new(
            engine_id,
            23,
            CudfPolarsEvent::Engine(EngineEvent::Exit { seq: 1 }),
        ),
    ];

    let analyzer = CudfPolarsUiAnalyzer::try_new(engine_id, events.into_iter()).unwrap();
    let bundle = analyzer.query_bundle(query_id).unwrap();

    assert_eq!(
        bundle.entities.query.instance_name.as_deref(),
        Some("iteration-1")
    );
    assert_eq!(
        bundle.entities.query_group.instance_name.as_deref(),
        Some("group")
    );
    assert_eq!(bundle.entities.plans.len(), 2);
    assert!(bundle.entities.workers.contains_key(&idle_worker_id));
    assert!(!bundle.entities.workers.contains_key(&foreign_worker_id));
    let quent_ui::ResourceTree::ResourceGroup(resource_root) = &bundle.resource_tree else {
        panic!("expected the resource tree root to be the engine")
    };
    assert!(resource_root.children.iter().any(|child| {
        matches!(
            child,
            quent_ui::ResourceTree::ResourceGroup(node)
                if node.id == quent_query_engine_ui::EntityRef::Worker(idle_worker_id)
        )
    }));
    assert!(
        bundle.entities.operators[&operator_id]
            .active_span
            .is_some()
    );
    let statistics = bundle.entities.operators[&operator_id]
        .statistics
        .as_ref()
        .expect("operator statistics should be present");
    assert_eq!(
        statistics.custom_statistics["input_bytes"]
            .quantity
            .as_deref(),
        Some("bytes")
    );
    assert_eq!(
        statistics.custom_statistics["output_bytes"]
            .quantity
            .as_deref(),
        Some("bytes")
    );
    assert!(
        statistics.custom_statistics["chunk_count"]
            .quantity
            .is_none()
    );
    assert!(bundle.entities.resources.contains_key(&processor_id));
    assert!(
        bundle
            .entities
            .resource_groups
            .contains_key(&thread_pool_id)
    );
    assert!(
        bundle
            .entities
            .resource_types
            .contains_key(PROCESSOR_RESOURCE_TYPE)
    );
    assert!(bundle.entities.fsm_types.contains_key(EVALUATE_ENTITY_TYPE));

    let response = analyzer
        .bulk_resource_timeline(BulkTimelineRequest {
            entries: [(
                "processor:evaluate".to_owned(),
                TimelineRequest::Resource(quent_ui::timeline::request::ResourceTimelineRequest {
                    resource_id: processor_id,
                    long_entities_threshold_s: Some(1e-9),
                    entity_filter: quent_ui::timeline::request::EntityFilter {
                        entity_type_name: Some(EVALUATE_ENTITY_TYPE.to_owned()),
                    },
                    application: OperatorFilter {
                        operator_ids: vec![],
                    },
                    config: quent_ui::timeline::request::TimelineConfig {
                        num_bins: 4,
                        start: 0.0,
                        end: 20e-9,
                    },
                }),
            )]
            .into_iter()
            .collect(),
            app_params: QueryFilter { query_id },
        })
        .unwrap();
    let BulkTimelinesResponseEntry::Ok { data, .. } = &response.entries["processor:evaluate"]
    else {
        panic!("evaluate timeline entry failed")
    };
    let UiResourceTimeline::BinnedByState(timeline) = data else {
        panic!("expected a state-keyed evaluate timeline")
    };
    assert_eq!(
        timeline.capacities_states_values["unit"]["running"].len(),
        4
    );
    assert!(
        timeline.capacities_states_values["unit"]["running"]
            .iter()
            .any(|&value| value > 0.0)
    );
    assert_eq!(timeline.long_fsms.len(), 1);
    assert_eq!(timeline.long_fsms[0].id, evaluate_id);
    assert_eq!(timeline.long_fsms[0].instance_name, "Scan-evaluate");

    let entities = analyzer
        .list_entities(entity_list_request(
            query_id,
            Some(EntityScope::Resource {
                resource_id: processor_id,
            }),
            vec![],
            Some(1e-9),
        ))
        .unwrap();
    assert_eq!(entities.total, 1);
    assert_eq!(entities.items[0].entity.fsm.id, evaluate_id);
    assert_eq!(entities.items[0].entity.operator_id, Some(operator_id));

    let entities = analyzer
        .list_entities(entity_list_request(
            query_id,
            None,
            vec![Uuid::now_v7()],
            None,
        ))
        .unwrap();
    assert_eq!(entities.total, 0);
    assert!(entities.items.is_empty());

    let error = analyzer
        .list_entities(entity_list_request(
            query_id,
            Some(EntityScope::Resource {
                resource_id: Uuid::now_v7(),
            }),
            vec![],
            None,
        ))
        .expect_err("unknown resource should be rejected");
    assert!(matches!(error, AnalyzerError::InvalidArgument(_)));

    let error = analyzer
        .list_entities(entity_list_request(
            query_id,
            Some(EntityScope::ResourceGroup {
                resource_group_id: Uuid::now_v7(),
                resource_type_name: PROCESSOR_RESOURCE_TYPE.to_owned(),
            }),
            vec![],
            None,
        ))
        .expect_err("unknown resource group should be rejected");
    assert!(matches!(error, AnalyzerError::InvalidArgument(_)));
}

#[test]
fn imports_generated_filesystem_layout() {
    let root = tempfile::tempdir().unwrap();
    let context_id = Uuid::now_v7();
    let context = root.path().join(context_id.to_string());
    let engine_dir = context.join("Engine");
    std::fs::create_dir_all(&engine_dir).unwrap();
    quent_events::build_info::ArtifactInfo::new(CudfPolars::model_info())
        .write_sidecar(&context)
        .unwrap();

    let event = Event::new(
        Uuid::now_v7(),
        1,
        EngineEvent::Init {
            seq: 0,
            instance_name: "cudf-polars".to_owned(),
            implementation: Implementation {
                name: "cudf-polars".to_owned(),
                version: "test".to_owned(),
                backend: "spmd".to_owned(),
                custom_attributes: DynamicAttributes::new(),
            },
        },
    );
    std::fs::write(
        engine_dir.join("events.ndjson"),
        format!("{}\n", serde_json::to_string(&event).unwrap()),
    )
    .unwrap();

    let events = Viewer::import_events(&context).unwrap().collect::<Vec<_>>();
    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0].data,
        CudfPolarsEvent::Engine(EngineEvent::Init { .. })
    ));
}

#[test]
fn context_inventory_discovers_engine_and_worker_scope() {
    let root = tempfile::tempdir().unwrap();
    let engine_id = Uuid::now_v7();
    let worker_id = Uuid::now_v7();
    let context_id = Uuid::now_v7();
    let context = root.path().join(context_id.to_string());
    std::fs::create_dir_all(&context).unwrap();
    quent_events::build_info::ArtifactInfo::new(CudfPolars::model_info())
        .write_sidecar(&context)
        .unwrap();
    let engine_dir = context.join("Engine");
    let worker_dir = context.join("Worker");
    std::fs::create_dir_all(&engine_dir).unwrap();
    std::fs::create_dir_all(&worker_dir).unwrap();
    let engine = Event::new(
        engine_id,
        1,
        EngineEvent::Init {
            seq: 0,
            instance_name: "cudf-polars".to_owned(),
            implementation: Implementation {
                name: "cudf-polars".to_owned(),
                version: "test".to_owned(),
                backend: "spmd".to_owned(),
                custom_attributes: DynamicAttributes::new(),
            },
        },
    );
    let worker = Event::new(
        worker_id,
        2,
        WorkerEvent::Init {
            seq: 0,
            instance_name: "rank-0".to_owned(),
            engine: EntityRef::new(engine_id, ()),
        },
    );
    std::fs::write(
        engine_dir.join("events.ndjson"),
        format!("{}\n", serde_json::to_string(&engine).unwrap()),
    )
    .unwrap();
    std::fs::write(
        worker_dir.join("events.ndjson"),
        format!("{}\n", serde_json::to_string(&worker).unwrap()),
    )
    .unwrap();

    let inventory = Viewer::context_inventory(&context).unwrap();
    assert_eq!(inventory.analysis_target_ids, [engine_id].into());
}
