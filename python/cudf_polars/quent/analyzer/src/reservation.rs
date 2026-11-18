// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::{
    AnalyzerError, AnalyzerResult,
    resource::{CapacityValue, Usage},
};
use quent_dynamic_attributes::DynamicAttribute;
use quent_store::entity::{
    EntityHandle, EntityStore,
    native::{NativeEntity, NativeStore},
};
use quent_time::{TimeUnixNanoSec, span::SpanUnixNanoSec, to_secs_relative};
use quent_ui::{FiniteStateMachine, FsmTransition, FsmUsage};
use uuid::Uuid;

use crate::generated::{
    ReservationOccupancy, ReservationOccupancyEvent,
    reservation_occupancy::ReservationOccupancyEvents,
};

pub(crate) const OCCUPANCY_ENTITY_TYPE: &str = "ReservationOccupancy";

struct OccupancyTransition {
    timestamp: TimeUnixNanoSec,
    name: String,
    usage: Option<(Uuid, u64)>,
    attributes: Vec<DynamicAttribute>,
}

pub(crate) struct OccupancySpan {
    pub(crate) id: Uuid,
    pub(crate) actor_id: Option<Uuid>,
    pub(crate) segments: Vec<OccupancyUsageSegment>,
    transitions: Vec<OccupancyTransition>,
}

pub(crate) struct OccupancyUsageSegment {
    pub(crate) occupancy_id: Uuid,
    pub(crate) resource_id: Uuid,
    pub(crate) bytes: CapacityValue,
    pub(crate) span: SpanUnixNanoSec,
}

impl<'a> Usage<'a> for &'a OccupancyUsageSegment {
    fn entity_id(&self) -> Uuid {
        self.occupancy_id
    }

    fn resource_id(&self) -> Uuid {
        self.resource_id
    }

    fn capacities(&self) -> impl Iterator<Item = &'a CapacityValue> {
        std::iter::once(&self.bytes)
    }

    fn span(&self) -> SpanUnixNanoSec {
        self.span
    }
}

impl OccupancySpan {
    pub(crate) fn lifecycle(&self) -> AnalyzerResult<SpanUnixNanoSec> {
        Ok(SpanUnixNanoSec::try_new(
            self.transitions
                .first()
                .expect("occupancy has a created transition")
                .timestamp,
            self.transitions
                .last()
                .expect("occupancy has a released transition")
                .timestamp,
        )?)
    }

    pub(crate) fn to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> FiniteStateMachine {
        FiniteStateMachine {
            id: self.id,
            type_name: OCCUPANCY_ENTITY_TYPE.to_owned(),
            instance_name: format!("reservation occupancy {}", self.id),
            transitions: self
                .transitions
                .iter()
                .map(|transition| FsmTransition {
                    name: transition.name.clone(),
                    usages: transition
                        .usage
                        .map(|(resource, bytes)| FsmUsage {
                            resource,
                            capacities: vec![("bytes".to_owned(), Some(bytes))],
                        })
                        .into_iter()
                        .collect(),
                    timestamp: to_secs_relative(transition.timestamp, epoch),
                    attributes: transition.attributes.clone(),
                    derived_attributes: vec![],
                })
                .collect(),
        }
    }
}

pub(crate) fn build_occupancies(
    events: Vec<quent_events::Event<ReservationOccupancyEvent>>,
    trace_end: TimeUnixNanoSec,
) -> AnalyzerResult<Vec<OccupancySpan>> {
    let store = NativeStore::<ReservationOccupancy>::try_from_events(events)
        .map_err(|error| AnalyzerError::Validation(error.to_string()))?;
    store
        .entities()
        .map_err(|error| AnalyzerError::Validation(error.to_string()))?
        .map(|id| {
            let entity = store
                .entity(id)
                .map_err(|error| AnalyzerError::Validation(error.to_string()))?
                .expect("ID came from this store");
            build_occupancy(entity, trace_end)
        })
        .collect()
}

fn build_occupancy(
    entity: &NativeEntity<ReservationOccupancy>,
    trace_end: TimeUnixNanoSec,
) -> AnalyzerResult<OccupancySpan> {
    let id = entity.id().uuid();
    let created = entity.created().ok_or_else(|| {
        AnalyzerError::IncompleteEntity(format!("reservation occupancy {id} has no created event"))
    })?;
    let released = entity.released();
    let actor_id = created.actor().as_ref().map(|actor| actor.target);
    let held = entity.held().collect::<Vec<_>>();
    let mut segments = Vec::with_capacity(held.len());
    let mut created_attributes = vec![DynamicAttribute::string(
        "worker_id",
        created.worker().target.to_string(),
    )];
    created_attributes.extend(
        actor_id.map(|actor_id| DynamicAttribute::string("actor_id", actor_id.to_string())),
    );
    created_attributes.extend(
        created
            .site()
            .as_ref()
            .map(|site| DynamicAttribute::string("site", site.clone())),
    );
    let mut transitions = vec![OccupancyTransition {
        timestamp: created.timestamp(),
        name: "created".to_owned(),
        usage: None,
        attributes: created_attributes,
    }];
    for (index, current) in held.iter().enumerate() {
        let end = held.get(index + 1).map_or_else(
            || released.map_or(trace_end, |released| released.timestamp()),
            |next| next.timestamp(),
        );
        let memory = current.memory();
        if end > current.timestamp() {
            segments.push(OccupancyUsageSegment {
                occupancy_id: id,
                resource_id: memory.target,
                bytes: CapacityValue::new("bytes", memory.data.bytes),
                span: SpanUnixNanoSec::try_new(current.timestamp(), end)?,
            });
        }
        transitions.push(OccupancyTransition {
            timestamp: current.timestamp(),
            name: "held".to_owned(),
            usage: Some((memory.target, memory.data.bytes)),
            attributes: vec![],
        });
    }
    transitions.push(OccupancyTransition {
        timestamp: released.map_or(trace_end, |released| released.timestamp()),
        name: "released".to_owned(),
        usage: None,
        attributes: vec![],
    });
    Ok(OccupancySpan {
        id,
        actor_id,
        segments,
        transitions,
    })
}
