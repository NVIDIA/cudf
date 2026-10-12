// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, hash_map::Entry};

use quent_analyzer::{
    AnalyzerError, AnalyzerResult, Entity, Model,
    entity::native::{AnalyzedEntity, EntityEventAccumulator},
    fsm::{
        Fsm, FsmUsages,
        native::{AnalyzedFsm, AnalyzedFsmBuilder, AnalyzedTransition, TransitionEvent},
    },
    resource::{Usage, Using},
};
use quent_dynamic_attributes::{DynamicAttributes, DynamicValue};
use quent_events::Event;
use quent_query_engine_analyzer::{
    EngineEntity, OperatorEntity, OperatorEntityMut, PlanEntity, PortEntity, QueryEngineModel,
    QueryEngineModelMut, QueryEntity, QueryGroupEntity, WorkerEntity, plan_tree::PlanTree,
};
use quent_query_engine_ui::{self as ui, EntityRef};
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec, try_to_secs_relative};
use uuid::Uuid;

use crate::generated::{
    EngineEvent, OperatorEvent, OperatorStatistics, PlanEvent, PortEvent, QueryEvent,
    QueryGroupEvent, WorkerEvent,
};

macro_rules! entity_impl {
    ($name:ident) => {
        impl Entity for $name {
            fn id(&self) -> Uuid {
                self.0.id()
            }
            fn type_name(&self) -> &str {
                self.0.type_name()
            }
            fn earliest_timestamp(&self) -> TimeUnixNanoSec {
                self.0.earliest_timestamp()
            }
            fn latest_timestamp(&self) -> TimeUnixNanoSec {
                self.0.latest_timestamp()
            }
        }
    };
}

#[derive(Default)]
struct EngineData {
    instance_name: Option<String>,
    implementation: Option<crate::generated::Implementation>,
    exited: bool,
}

impl EntityEventAccumulator for EngineData {
    type Payload = EngineEvent;

    fn push(&mut self, event: EngineEvent) {
        match event {
            EngineEvent::Init {
                instance_name,
                implementation,
                ..
            } => {
                self.instance_name = Some(instance_name);
                self.implementation = Some(implementation);
            }
            EngineEvent::Exit { .. } => self.exited = true,
        }
    }
}

#[derive(Debug)]
pub struct Engine(AnalyzedEntity<EngineData>);
entity_impl!(Engine);

impl EngineEntity for Engine {
    fn to_ui(&self) -> AnalyzerResult<ui::Engine> {
        let data = self.0.accumulator();
        let start = self.earliest_timestamp();
        Ok(ui::Engine {
            id: self.id(),
            start_time_unix_ns: Some(start),
            duration_s: data
                .exited
                .then(|| try_to_secs_relative(self.latest_timestamp(), start))
                .transpose()?,
            instance_name: data.instance_name.clone(),
            implementation: data.implementation.as_ref().map(|implementation| {
                ui::EngineImplementationAttributes {
                    name: Some(implementation.name.clone()),
                    version: Some(implementation.version.clone()),
                    custom_attributes: implementation.custom_attributes.0.clone(),
                }
            }),
        })
    }
}

#[derive(Default)]
struct WorkerData {
    engine_id: Option<Uuid>,
    instance_name: Option<String>,
    exited: bool,
}

impl EntityEventAccumulator for WorkerData {
    type Payload = WorkerEvent;

    fn push(&mut self, event: WorkerEvent) {
        match event {
            WorkerEvent::Init {
                instance_name,
                engine,
                ..
            } => {
                self.engine_id = Some(engine.target);
                self.instance_name = Some(instance_name);
            }
            WorkerEvent::Exit { .. } => self.exited = true,
        }
    }
}

#[derive(Debug)]
pub struct Worker(AnalyzedEntity<WorkerData>);
entity_impl!(Worker);

impl WorkerEntity for Worker {
    fn to_ui(&self, _epoch: TimeUnixNanoSec) -> ui::Worker {
        let data = self.0.accumulator();
        ui::Worker {
            id: self.id(),
            parent_engine_id: data.engine_id,
            instance_name: data.instance_name.clone(),
            start_unix_ns: Some(self.earliest_timestamp()),
            end_unix_ns: data.exited.then(|| self.latest_timestamp()),
        }
    }
}

#[derive(Default)]
struct QueryGroupData {
    engine_id: Option<Uuid>,
    instance_name: Option<String>,
}

impl EntityEventAccumulator for QueryGroupData {
    type Payload = QueryGroupEvent;

    fn push(&mut self, event: QueryGroupEvent) {
        let QueryGroupEvent::Declared {
            instance_name,
            engine,
        } = event;
        self.engine_id = Some(engine.target);
        self.instance_name = instance_name;
    }
}

#[derive(Debug)]
pub struct QueryGroup(AnalyzedEntity<QueryGroupData>);
entity_impl!(QueryGroup);

impl QueryGroupEntity for QueryGroup {
    fn to_ui(&self) -> ui::QueryGroup {
        let data = self.0.accumulator();
        ui::QueryGroup {
            id: self.id(),
            instance_name: data.instance_name.clone(),
            engine_id: data.engine_id,
        }
    }
}

impl TransitionEvent for QueryEvent {
    fn name(&self) -> &'static str {
        match self {
            Self::Initialized { .. } => "initialized",
            Self::Planning { .. } => "planning",
            Self::Executing { .. } => "executing",
            Self::Completed { .. } => "completed",
            Self::Failed { .. } => "failed",
        }
    }

    fn sequence(&self) -> u16 {
        match self {
            Self::Initialized { seq, .. }
            | Self::Planning { seq }
            | Self::Executing { seq }
            | Self::Completed { seq }
            | Self::Failed { seq, .. } => *seq,
        }
    }

    fn is_initial(&self) -> bool {
        matches!(self, Self::Initialized { .. })
    }

    fn is_final(&self) -> bool {
        matches!(self, Self::Completed { .. } | Self::Failed { .. })
    }

    fn is_valid_next(&self, next: &Self) -> bool {
        match self {
            Self::Initialized { .. } => matches!(next, Self::Planning { .. }),
            Self::Planning { .. } => matches!(next, Self::Executing { .. }),
            Self::Executing { .. } => {
                matches!(next, Self::Completed { .. } | Self::Failed { .. })
            }
            Self::Completed { .. } | Self::Failed { .. } => false,
        }
    }
}

type QueryBuilder = AnalyzedFsmBuilder<QueryEvent>;

#[derive(Debug)]
pub struct Query(AnalyzedFsm<QueryEvent>);

impl Query {
    fn query_group_id_inner(&self) -> Option<Uuid> {
        match &self.0.transition(0)?.data {
            QueryEvent::Initialized { query_group, .. } => Some(query_group.target),
            _ => None,
        }
    }
}

impl Entity for Query {
    fn id(&self) -> Uuid {
        self.0.id()
    }
    fn type_name(&self) -> &str {
        self.0.type_name()
    }
    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }
    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl Fsm for Query {
    type TransitionType = AnalyzedTransition<QueryEvent>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.0.transition(index)
    }
}

impl<'a> FsmUsages<'a> for Query {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.0.usages_with_state_names()
    }
}

impl Using for Query {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.0.usages()
    }
}

impl QueryEntity for Query {
    fn query_group_id(&self) -> Option<Uuid> {
        self.query_group_id_inner()
    }

    fn to_ui(&self) -> AnalyzerResult<ui::Query> {
        let transitions = self.0.transitions();
        let epoch = transitions.first().map(Timestamp::timestamp);
        let mut planning_s = None;
        let mut executing_s = None;
        let mut completed_s = None;
        if let Some(epoch) = epoch {
            for (index, transition) in transitions.iter().enumerate() {
                match transition.data {
                    QueryEvent::Planning { .. } => {
                        planning_s = Some(try_to_secs_relative(transition.timestamp(), epoch)?);
                    }
                    QueryEvent::Executing { .. } => {
                        executing_s = Some(try_to_secs_relative(transition.timestamp(), epoch)?);
                        if let Some(next) = transitions.get(index + 1) {
                            completed_s = Some(try_to_secs_relative(next.timestamp(), epoch)?);
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(ui::Query {
            id: self.id(),
            query_group_id: self.query_group_id_inner().unwrap_or_default(),
            instance_name: transitions
                .first()
                .and_then(|transition| match &transition.data {
                    QueryEvent::Initialized { instance_name, .. } => Some(instance_name.clone()),
                    _ => None,
                }),
            start_unix_ns: epoch,
            planning_s,
            executing_s,
            completed_s,
        })
    }
}

#[derive(Default)]
struct PlanData {
    instance_name: Option<String>,
    query_id: Option<Uuid>,
    parent_plan_id: Option<Uuid>,
    worker_id: Option<Uuid>,
    edges: Vec<(Uuid, Uuid)>,
}

impl EntityEventAccumulator for PlanData {
    type Payload = PlanEvent;

    fn push(&mut self, event: PlanEvent) {
        let PlanEvent::Declared {
            instance_name,
            query,
            parent_plan,
            worker,
            edges,
        } = event;
        self.instance_name = Some(instance_name);
        self.query_id = Some(query.target);
        self.parent_plan_id = parent_plan.map(|value| value.target);
        self.worker_id = worker.map(|value| value.target);
        self.edges = edges
            .into_iter()
            .map(|edge| (edge.source.target, edge.target.target))
            .collect();
    }
}

#[derive(Debug)]
pub struct Plan(AnalyzedEntity<PlanData>);
entity_impl!(Plan);

impl PlanEntity for Plan {
    fn parent_query_id(&self) -> Option<Uuid> {
        self.0
            .accumulator()
            .parent_plan_id
            .is_none()
            .then_some(self.0.accumulator().query_id)
            .flatten()
    }
    fn parent_plan_id(&self) -> Option<Uuid> {
        self.0.accumulator().parent_plan_id
    }
    fn worker_id(&self) -> Option<Uuid> {
        self.0.accumulator().worker_id
    }
    fn edges(&self) -> impl Iterator<Item = (Uuid, Uuid)> + '_ {
        self.0.accumulator().edges.iter().copied()
    }
    fn to_ui(&self) -> ui::Plan {
        let data = self.0.accumulator();
        ui::Plan {
            id: self.id(),
            instance_name: data.instance_name.clone(),
            parent: data.parent_plan_id.or(data.query_id),
            worker_id: data.worker_id,
            edges: data
                .edges
                .iter()
                .map(|&(source, target)| ui::Edge { source, target })
                .collect(),
        }
    }
}

#[derive(Default)]
struct OperatorData {
    plan_id: Option<Uuid>,
    parent_operator_ids: Vec<Uuid>,
    instance_name: Option<String>,
    type_name: Option<String>,
    custom_attributes: DynamicAttributes,
    statistics: Option<OperatorStatistics>,
}

impl OperatorData {
    fn add_serialized_attribute(&mut self, name: &str, value: &impl serde::Serialize) {
        if let Ok(value) = serde_json::to_string(value) {
            self.custom_attributes.add(name, value);
        }
    }
}

impl EntityEventAccumulator for OperatorData {
    type Payload = OperatorEvent;

    fn push(&mut self, event: OperatorEvent) {
        match event {
            OperatorEvent::Declared {
                plan,
                parent_operators,
                instance_name,
                type_name,
                node_id,
            } => {
                self.plan_id = Some(plan.target);
                self.parent_operator_ids = parent_operators
                    .into_iter()
                    .map(|value| value.target)
                    .collect();
                self.instance_name = Some(instance_name);
                self.type_name = Some(type_name);
                self.custom_attributes.add("node_id", node_id);
            }
            OperatorEvent::Statistics { values } => self.statistics = Some(values),
            OperatorEvent::ScanDetails { values } => {
                self.add_serialized_attribute("scan_details", &values);
            }
            OperatorEvent::StreamingScanDetails { values } => {
                self.add_serialized_attribute("streaming_scan_details", &values);
            }
            OperatorEvent::JoinDetails { values } => {
                self.add_serialized_attribute("join_details", &values);
            }
            OperatorEvent::JoinWithPrefilterDetails { values } => {
                self.add_serialized_attribute("join_with_prefilter_details", &values);
            }
            OperatorEvent::PushdownFilterHintDetails { values } => {
                self.add_serialized_attribute("pushdown_filter_hint_details", &values);
            }
            OperatorEvent::GroupByDetails { values } => {
                self.add_serialized_attribute("group_by_details", &values);
            }
            OperatorEvent::ShuffleDetails { values } => {
                self.add_serialized_attribute("shuffle_details", &values);
            }
            OperatorEvent::SortDetails { values } => {
                self.add_serialized_attribute("sort_details", &values);
            }
            OperatorEvent::FilterDetails { values } => {
                self.add_serialized_attribute("filter_details", &values);
            }
            OperatorEvent::SelectDetails { values } => {
                self.add_serialized_attribute("select_details", &values);
            }
            OperatorEvent::HstackDetails { values } => {
                self.add_serialized_attribute("hstack_details", &values);
            }
        }
    }
}

fn to_ui_operator_statistics(statistics: &OperatorStatistics) -> ui::OperatorStatistics {
    let statistic = |value: DynamicValue, quantity: Option<&str>| ui::OperatorStatistic {
        value: Some(value),
        quantity: quantity.map(str::to_owned),
    };
    let mut custom_statistics = HashMap::from([
        (
            "input_bytes".to_owned(),
            statistic(statistics.input_bytes.into(), Some("bytes")),
        ),
        (
            "output_bytes".to_owned(),
            statistic(statistics.output_bytes.into(), Some("bytes")),
        ),
        (
            "chunk_count".to_owned(),
            statistic(statistics.chunk_count.into(), None),
        ),
        (
            "duplicated".to_owned(),
            statistic(statistics.duplicated.into(), None),
        ),
    ]);
    if let Some(value) = statistics.output_rows {
        custom_statistics.insert("output_rows".to_owned(), statistic(value.into(), None));
    }
    if let Some(value) = &statistics.decision {
        custom_statistics.insert(
            "decision".to_owned(),
            statistic(value.as_str().into(), None),
        );
    }
    ui::OperatorStatistics { custom_statistics }
}

#[derive(Debug)]
pub struct Operator {
    entity: AnalyzedEntity<OperatorData>,
    active_span: Option<SpanUnixNanoSec>,
}

impl Entity for Operator {
    fn id(&self) -> Uuid {
        self.entity.id()
    }
    fn type_name(&self) -> &str {
        self.entity.type_name()
    }
    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.entity.earliest_timestamp()
    }
    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.entity.latest_timestamp()
    }
}

impl OperatorEntity for Operator {
    fn plan_id(&self) -> Option<Uuid> {
        self.entity.accumulator().plan_id
    }
    fn parent_operator_ids(&self) -> impl ExactSizeIterator<Item = Uuid> + '_ {
        self.entity
            .accumulator()
            .parent_operator_ids
            .iter()
            .copied()
    }
    fn active_span(&self) -> Option<SpanUnixNanoSec> {
        self.active_span
    }
    fn operator_type_name(&self) -> Option<&str> {
        self.entity.accumulator().type_name.as_deref()
    }
    fn to_ui(&self, epoch: TimeUnixNanoSec) -> ui::Operator {
        let data = self.entity.accumulator();
        ui::Operator {
            id: self.id(),
            plan_id: data.plan_id,
            parent_operator_ids: data.parent_operator_ids.clone(),
            instance_name: data.instance_name.clone(),
            operator_type_name: data.type_name.clone(),
            custom_attributes: data
                .custom_attributes
                .iter()
                .map(|attribute| (attribute.key.clone(), attribute.value.clone()))
                .collect(),
            statistics: data.statistics.as_ref().map(to_ui_operator_statistics),
            active_span: self
                .active_span
                .and_then(|span| span.try_to_secs_relative(epoch).ok()),
        }
    }
}

impl OperatorEntityMut for Operator {
    fn extend_active_span(&mut self, span: SpanUnixNanoSec) {
        self.active_span = Some(match self.active_span {
            Some(existing) => existing.extend(&span),
            None => span,
        });
    }
}

#[derive(Default)]
struct PortData {
    operator_id: Option<Uuid>,
    instance_name: Option<String>,
}

impl EntityEventAccumulator for PortData {
    type Payload = PortEvent;

    fn push(&mut self, event: PortEvent) {
        let PortEvent::Declared {
            operator,
            instance_name,
        } = event;
        self.operator_id = Some(operator.target);
        self.instance_name = Some(instance_name);
    }
}

#[derive(Debug)]
pub struct Port(AnalyzedEntity<PortData>);
entity_impl!(Port);

impl PortEntity for Port {
    fn operator_id(&self) -> Option<Uuid> {
        self.0.accumulator().operator_id
    }
    fn to_ui(&self, _epoch: TimeUnixNanoSec) -> ui::Port {
        ui::Port {
            id: self.id(),
            operator_id: self.0.accumulator().operator_id,
            instance_name: self.0.accumulator().instance_name.clone(),
            statistics: None,
        }
    }
}

pub struct CudfPolarsModel {
    engine: Engine,
    workers: HashMap<Uuid, Worker>,
    query_groups: HashMap<Uuid, QueryGroup>,
    queries: HashMap<Uuid, Query>,
    plans: HashMap<Uuid, Plan>,
    operators: HashMap<Uuid, Operator>,
    ports: HashMap<Uuid, Port>,
}

impl Model for CudfPolarsModel {
    type EntityIdType = EntityRef;

    fn try_entity_ref(&self, id: Uuid) -> AnalyzerResult<EntityRef> {
        if self.engine.id() == id {
            Ok(EntityRef::Engine(id))
        } else if self.workers.contains_key(&id) {
            Ok(EntityRef::Worker(id))
        } else if self.query_groups.contains_key(&id) {
            Ok(EntityRef::QueryGroup(id))
        } else if self.queries.contains_key(&id) {
            Ok(EntityRef::Query(id))
        } else if self.plans.contains_key(&id) {
            Ok(EntityRef::Plan(id))
        } else if self.operators.contains_key(&id) {
            Ok(EntityRef::Operator(id))
        } else if self.ports.contains_key(&id) {
            Ok(EntityRef::Port(id))
        } else {
            Err(AnalyzerError::InvalidId(id))
        }
    }
}

impl QueryEngineModel for CudfPolarsModel {
    type Engine = Engine;
    type Query = Query;
    type QueryGroup = QueryGroup;
    type Worker = Worker;
    type Plan = Plan;
    type Operator = Operator;
    type Port = Port;

    fn engine(&self) -> AnalyzerResult<&Engine> {
        Ok(&self.engine)
    }
    fn query(&self, id: Uuid) -> AnalyzerResult<&Query> {
        self.queries.get(&id).ok_or(AnalyzerError::InvalidId(id))
    }
    fn query_group(&self, id: Uuid) -> AnalyzerResult<&QueryGroup> {
        self.query_groups
            .get(&id)
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn worker(&self, id: Uuid) -> AnalyzerResult<&Worker> {
        self.workers.get(&id).ok_or(AnalyzerError::InvalidId(id))
    }
    fn plan(&self, id: Uuid) -> AnalyzerResult<&Plan> {
        self.plans.get(&id).ok_or(AnalyzerError::InvalidId(id))
    }
    fn operator(&self, id: Uuid) -> AnalyzerResult<&Operator> {
        self.operators.get(&id).ok_or(AnalyzerError::InvalidId(id))
    }
    fn port(&self, id: Uuid) -> AnalyzerResult<&Port> {
        self.ports.get(&id).ok_or(AnalyzerError::InvalidId(id))
    }
    fn queries(&self) -> impl Iterator<Item = &Query> {
        self.queries.values()
    }
    fn query_groups(&self) -> impl Iterator<Item = &QueryGroup> {
        self.query_groups.values()
    }
    fn workers(&self) -> impl Iterator<Item = &Worker> {
        self.workers.values()
    }
    fn plans(&self) -> impl Iterator<Item = &Plan> {
        self.plans.values()
    }
    fn operators(&self) -> impl Iterator<Item = &Operator> {
        self.operators.values()
    }
    fn ports(&self) -> impl Iterator<Item = &Port> {
        self.ports.values()
    }
    fn plan_tree(&self, query_id: Uuid) -> AnalyzerResult<PlanTree> {
        PlanTree::try_new(self.plans.values(), query_id)
    }
}

impl QueryEngineModelMut for CudfPolarsModel {
    fn operator_mut(&mut self, id: Uuid) -> AnalyzerResult<&mut Operator> {
        self.operators
            .get_mut(&id)
            .ok_or(AnalyzerError::InvalidId(id))
    }
}

pub struct QueryView<'a> {
    engine: &'a Engine,
    query: &'a Query,
    group: &'a QueryGroup,
    workers: HashMap<Uuid, &'a Worker>,
    plans: HashMap<Uuid, &'a Plan>,
    operators: HashMap<Uuid, &'a Operator>,
    ports: HashMap<Uuid, &'a Port>,
}

impl CudfPolarsModel {
    pub fn query_view(&self, query_id: Uuid) -> AnalyzerResult<QueryView<'_>> {
        let query = self.query(query_id)?;
        let group = self.query_group(query.query_group_id().unwrap_or_default())?;
        let plans: HashMap<_, _> = self
            .query_plans(query_id)?
            .map(|plan| (plan.id(), plan))
            .collect();
        let workers = plans
            .values()
            .filter_map(|plan| plan.worker_id().and_then(|id| self.worker(id).ok()))
            .map(|worker| (worker.id(), worker))
            .collect();
        let operators: HashMap<_, _> = self
            .plans_operators(plans.values().copied())?
            .map(|operator| (operator.id(), operator))
            .collect();
        let ports = self
            .operators_ports(operators.values().copied())?
            .map(|port| (port.id(), port))
            .collect();
        Ok(QueryView {
            engine: &self.engine,
            query,
            group,
            workers,
            plans,
            operators,
            ports,
        })
    }
}

impl Model for QueryView<'_> {
    type EntityIdType = EntityRef;
    fn try_entity_ref(&self, id: Uuid) -> AnalyzerResult<EntityRef> {
        if self.engine.id() == id {
            Ok(EntityRef::Engine(id))
        } else if self.query.id() == id {
            Ok(EntityRef::Query(id))
        } else if self.group.id() == id {
            Ok(EntityRef::QueryGroup(id))
        } else if self.workers.contains_key(&id) {
            Ok(EntityRef::Worker(id))
        } else if self.plans.contains_key(&id) {
            Ok(EntityRef::Plan(id))
        } else if self.operators.contains_key(&id) {
            Ok(EntityRef::Operator(id))
        } else if self.ports.contains_key(&id) {
            Ok(EntityRef::Port(id))
        } else {
            Err(AnalyzerError::InvalidId(id))
        }
    }
}

impl QueryEngineModel for QueryView<'_> {
    type Engine = Engine;
    type Query = Query;
    type QueryGroup = QueryGroup;
    type Worker = Worker;
    type Plan = Plan;
    type Operator = Operator;
    type Port = Port;

    fn engine(&self) -> AnalyzerResult<&Engine> {
        Ok(self.engine)
    }
    fn query(&self, id: Uuid) -> AnalyzerResult<&Query> {
        (self.query.id() == id)
            .then_some(self.query)
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn query_group(&self, id: Uuid) -> AnalyzerResult<&QueryGroup> {
        (self.group.id() == id)
            .then_some(self.group)
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn worker(&self, id: Uuid) -> AnalyzerResult<&Worker> {
        self.workers
            .get(&id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn plan(&self, id: Uuid) -> AnalyzerResult<&Plan> {
        self.plans
            .get(&id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn operator(&self, id: Uuid) -> AnalyzerResult<&Operator> {
        self.operators
            .get(&id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn port(&self, id: Uuid) -> AnalyzerResult<&Port> {
        self.ports
            .get(&id)
            .copied()
            .ok_or(AnalyzerError::InvalidId(id))
    }
    fn queries(&self) -> impl Iterator<Item = &Query> {
        std::iter::once(self.query)
    }
    fn query_groups(&self) -> impl Iterator<Item = &QueryGroup> {
        std::iter::once(self.group)
    }
    fn workers(&self) -> impl Iterator<Item = &Worker> {
        self.workers.values().copied()
    }
    fn plans(&self) -> impl Iterator<Item = &Plan> {
        self.plans.values().copied()
    }
    fn operators(&self) -> impl Iterator<Item = &Operator> {
        self.operators.values().copied()
    }
    fn ports(&self) -> impl Iterator<Item = &Port> {
        self.ports.values().copied()
    }
    fn plan_tree(&self, query_id: Uuid) -> AnalyzerResult<PlanTree> {
        PlanTree::try_new(self.plans.values().copied(), query_id)
    }
}

pub struct CudfPolarsModelBuilder {
    engine_id: Uuid,
    engine: Option<Engine>,
    workers: HashMap<Uuid, Worker>,
    groups: HashMap<Uuid, QueryGroup>,
    queries: HashMap<Uuid, QueryBuilder>,
    plans: HashMap<Uuid, Plan>,
    operators: HashMap<Uuid, Operator>,
    ports: HashMap<Uuid, Port>,
}

impl CudfPolarsModelBuilder {
    pub fn try_new(engine_id: Uuid) -> AnalyzerResult<Self> {
        if engine_id.is_nil() {
            return Err(AnalyzerError::Validation(
                "engine id cannot be nil".to_owned(),
            ));
        }
        Ok(Self {
            engine_id,
            engine: None,
            workers: HashMap::new(),
            groups: HashMap::new(),
            queries: HashMap::new(),
            plans: HashMap::new(),
            operators: HashMap::new(),
            ports: HashMap::new(),
        })
    }

    pub fn try_push(
        &mut self,
        event: Event<crate::generated::CudfPolarsEvent>,
    ) -> AnalyzerResult<()> {
        let Event {
            id,
            timestamp,
            data,
        } = event;
        match data {
            crate::generated::CudfPolarsEvent::Engine(data) => {
                if id != self.engine_id {
                    return Ok(());
                }
                let event = Event::new(id, timestamp, data);
                if let Some(engine) = &mut self.engine {
                    engine.0.push(event)
                } else {
                    self.engine = Some(Engine(AnalyzedEntity::try_from_event(event)?));
                    Ok(())
                }
            }
            crate::generated::CudfPolarsEvent::Worker(data) => push_entity(
                &mut self.workers,
                id,
                timestamp,
                data,
                |entity| &mut entity.0,
                Worker,
            ),
            crate::generated::CudfPolarsEvent::QueryGroup(data) => push_entity(
                &mut self.groups,
                id,
                timestamp,
                data,
                |entity| &mut entity.0,
                QueryGroup,
            ),
            crate::generated::CudfPolarsEvent::Plan(data) => push_entity(
                &mut self.plans,
                id,
                timestamp,
                data,
                |entity| &mut entity.0,
                Plan,
            ),
            crate::generated::CudfPolarsEvent::Port(data) => push_entity(
                &mut self.ports,
                id,
                timestamp,
                data,
                |entity| &mut entity.0,
                Port,
            ),
            crate::generated::CudfPolarsEvent::Operator(data) => {
                let event = Event::new(id, timestamp, data);
                match self.operators.entry(id) {
                    Entry::Occupied(entry) => entry.into_mut().entity.push(event),
                    Entry::Vacant(entry) => {
                        entry.insert(Operator {
                            entity: AnalyzedEntity::try_from_event(event)?,
                            active_span: None,
                        });
                        Ok(())
                    }
                }
            }
            crate::generated::CudfPolarsEvent::Query(data) => {
                let builder = match self.queries.entry(id) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => entry.insert(QueryBuilder::try_new(id)?),
                };
                builder.push_transition(Event::new(id, timestamp, data));
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn try_build(self) -> AnalyzerResult<CudfPolarsModel> {
        Ok(CudfPolarsModel {
            engine: self.engine.ok_or_else(|| {
                AnalyzerError::IncompleteEntity(format!("engine {} has no events", self.engine_id))
            })?,
            workers: self.workers,
            query_groups: self.groups,
            queries: self
                .queries
                .into_iter()
                .map(|(id, builder)| builder.try_build().map(Query).map(|query| (id, query)))
                .collect::<AnalyzerResult<_>>()?,
            plans: self.plans,
            operators: self.operators,
            ports: self.ports,
        })
    }
}

fn push_entity<E, A, T>(
    entities: &mut HashMap<Uuid, T>,
    id: Uuid,
    timestamp: TimeUnixNanoSec,
    data: E,
    inner: impl Fn(&mut T) -> &mut AnalyzedEntity<A>,
    wrap: impl Fn(AnalyzedEntity<A>) -> T,
) -> AnalyzerResult<()>
where
    E: quent_events::EventPayload,
    A: EntityEventAccumulator<Payload = E>,
{
    let event = Event::new(id, timestamp, data);
    match entities.entry(id) {
        Entry::Occupied(entry) => inner(entry.into_mut()).push(event),
        Entry::Vacant(entry) => {
            entry.insert(wrap(AnalyzedEntity::try_from_event(event)?));
            Ok(())
        }
    }
}
