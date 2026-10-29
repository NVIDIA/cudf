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
    MemoryReservation, MemoryReservationEvent, MemoryReservationRequest,
    MemoryReservationRequestEvent, memory_reservation::MemoryReservationEvents,
    memory_reservation_request::MemoryReservationRequestEvents,
};

pub(crate) const RESERVATION_ENTITY_TYPE: &str = "MemoryReservation";
pub(crate) const REQUEST_ENTITY_TYPE: &str = "MemoryReservationRequest";

struct ReservationTransition {
    timestamp: TimeUnixNanoSec,
    name: String,
    usage: Option<(Uuid, u64)>,
    attributes: Vec<DynamicAttribute>,
}

pub(crate) struct ReservationSpan {
    pub(crate) id: Uuid,
    pub(crate) actor_id: Option<Uuid>,
    pub(crate) segments: Vec<ReservationUsageSegment>,
    transitions: Vec<ReservationTransition>,
}

pub(crate) struct ReservationUsageSegment {
    pub(crate) reservation_id: Uuid,
    pub(crate) resource_id: Uuid,
    pub(crate) bytes: CapacityValue,
    pub(crate) span: SpanUnixNanoSec,
}

impl<'a> Usage<'a> for &'a ReservationUsageSegment {
    fn entity_id(&self) -> Uuid {
        self.reservation_id
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

impl ReservationSpan {
    pub(crate) fn to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> FiniteStateMachine {
        FiniteStateMachine {
            id: self.id,
            type_name: RESERVATION_ENTITY_TYPE.to_owned(),
            instance_name: format!("reservation {}", self.id),
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

pub(crate) struct RequestSpan {
    pub(crate) id: Uuid,
    pub(crate) actor_id: Option<Uuid>,
    pub(crate) site: String,
    pub(crate) lifecycle: SpanUnixNanoSec,
    pub(crate) wait: Option<SpanUnixNanoSec>,
    transitions: Vec<(TimeUnixNanoSec, String, Vec<DynamicAttribute>)>,
}

impl RequestSpan {
    pub(crate) fn to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> FiniteStateMachine {
        FiniteStateMachine {
            id: self.id,
            type_name: REQUEST_ENTITY_TYPE.to_owned(),
            instance_name: self.site.clone(),
            transitions: self
                .transitions
                .iter()
                .map(|(timestamp, name, attributes)| FsmTransition {
                    name: name.clone(),
                    usages: vec![],
                    timestamp: to_secs_relative(*timestamp, epoch),
                    attributes: attributes.clone(),
                    derived_attributes: vec![],
                })
                .collect(),
        }
    }
}

pub(crate) fn build_reservations(
    events: Vec<quent_events::Event<MemoryReservationEvent>>,
    trace_end: TimeUnixNanoSec,
) -> AnalyzerResult<Vec<ReservationSpan>> {
    let store = NativeStore::<MemoryReservation>::try_from_events(events)
        .map_err(|error| AnalyzerError::Validation(error.to_string()))?;
    store
        .entities()
        .map_err(|error| AnalyzerError::Validation(error.to_string()))?
        .map(|id| {
            let entity = store
                .entity(id)
                .map_err(|error| AnalyzerError::Validation(error.to_string()))?
                .expect("ID came from this store");
            build_reservation(entity, trace_end)
        })
        .collect()
}

fn build_reservation(
    entity: &NativeEntity<MemoryReservation>,
    trace_end: TimeUnixNanoSec,
) -> AnalyzerResult<ReservationSpan> {
    let id = entity.id().uuid();
    let created = entity.created().ok_or_else(|| {
        AnalyzerError::IncompleteEntity(format!("reservation {id} has no created event"))
    })?;
    let released = entity.released();
    let actor_id = created.actor().as_ref().map(|actor| actor.target);
    let held = entity.held().collect::<Vec<_>>();
    let mut segments = Vec::with_capacity(held.len());
    let mut created_attributes = created.attributes().iter().cloned().collect::<Vec<_>>();
    created_attributes.extend([
        DynamicAttribute::string("worker_id", created.worker().target.to_string()),
        DynamicAttribute::string("site", created.site().clone()),
        DynamicAttribute::u64("sequence", *created.sequence()),
    ]);
    created_attributes.extend(
        actor_id.map(|actor_id| DynamicAttribute::string("actor_id", actor_id.to_string())),
    );
    created_attributes.extend(
        created
            .request()
            .as_ref()
            .map(|request| DynamicAttribute::string("request_id", request.target.to_string())),
    );
    created_attributes.extend(
        created
            .parent()
            .as_ref()
            .map(|parent| DynamicAttribute::string("parent_id", parent.target.to_string())),
    );
    let mut transitions = vec![ReservationTransition {
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
            segments.push(ReservationUsageSegment {
                reservation_id: id,
                resource_id: memory.target,
                bytes: CapacityValue::new("bytes", memory.data.bytes),
                span: SpanUnixNanoSec::try_new(current.timestamp(), end)?,
            });
        }
        transitions.push(ReservationTransition {
            timestamp: current.timestamp(),
            name: "held".to_owned(),
            usage: Some((memory.target, memory.data.bytes)),
            attributes: vec![
                DynamicAttribute::string("change", current.change().clone()),
                DynamicAttribute::u64("sequence", *current.sequence()),
            ],
        });
    }
    let release_attributes = released.map_or_else(
        || vec![DynamicAttribute::string("reason", "truncated".to_owned())],
        |released| {
            vec![
                DynamicAttribute::string("reason", released.reason().clone()),
                DynamicAttribute::u64("sequence", *released.sequence()),
            ]
        },
    );
    transitions.push(ReservationTransition {
        timestamp: released.map_or(trace_end, |released| released.timestamp()),
        name: "released".to_owned(),
        usage: None,
        attributes: release_attributes,
    });
    Ok(ReservationSpan {
        id,
        actor_id,
        segments,
        transitions,
    })
}

pub(crate) fn build_requests(
    events: Vec<quent_events::Event<MemoryReservationRequestEvent>>,
    trace_end: TimeUnixNanoSec,
) -> AnalyzerResult<Vec<RequestSpan>> {
    let store = NativeStore::<MemoryReservationRequest>::try_from_events(events)
        .map_err(|error| AnalyzerError::Validation(error.to_string()))?;
    store
        .entities()
        .map_err(|error| AnalyzerError::Validation(error.to_string()))?
        .map(|id| {
            let entity = store
                .entity(id)
                .map_err(|error| AnalyzerError::Validation(error.to_string()))?
                .expect("ID came from this store");
            build_request(entity, trace_end)
        })
        .collect()
}

fn build_request(
    entity: &NativeEntity<MemoryReservationRequest>,
    trace_end: TimeUnixNanoSec,
) -> AnalyzerResult<RequestSpan> {
    let id = entity.id().uuid();
    let submitted = entity.submitted().ok_or_else(|| {
        AnalyzerError::IncompleteEntity(format!("reservation request {id} has no submitted event"))
    })?;
    let terminal = entity
        .granted()
        .map(|event| {
            (
                event.timestamp(),
                "granted",
                vec![
                    DynamicAttribute::u64("size", *event.size()),
                    DynamicAttribute::u64("overbooked_bytes", *event.overbooked_bytes()),
                    DynamicAttribute::u8("timed_out", u8::from(*event.timed_out())),
                ],
            )
        })
        .or_else(|| {
            entity.empty().map(|event| {
                (
                    event.timestamp(),
                    "empty",
                    vec![DynamicAttribute::u8(
                        "timed_out",
                        u8::from(*event.timed_out()),
                    )],
                )
            })
        })
        .or_else(|| {
            entity.failed().map(|event| {
                (
                    event.timestamp(),
                    "failed",
                    vec![DynamicAttribute::string("error", event.error().clone())],
                )
            })
        })
        .or_else(|| {
            entity
                .cancelled()
                .map(|event| (event.timestamp(), "cancelled", vec![]))
        })
        .unwrap_or_else(|| {
            (
                trace_end,
                "cancelled",
                vec![DynamicAttribute::u8("truncated", 1)],
            )
        });
    let mut transitions = vec![(
        submitted.timestamp(),
        "submitted".to_owned(),
        [
            DynamicAttribute::u64("size", *submitted.size()),
            DynamicAttribute::string("policy", submitted.policy().clone()),
        ]
        .into_iter()
        .chain(
            submitted
                .actor()
                .as_ref()
                .map(|actor| DynamicAttribute::string("actor_id", actor.target.to_string())),
        )
        .collect(),
    )];
    let wait = entity
        .waiting()
        .map(|waiting| {
            transitions.push((waiting.timestamp(), "waiting".to_owned(), vec![]));
            SpanUnixNanoSec::try_new(waiting.timestamp(), terminal.0)
        })
        .transpose()?;
    transitions.push((terminal.0, terminal.1.to_owned(), terminal.2));
    Ok(RequestSpan {
        id,
        actor_id: submitted.actor().as_ref().map(|actor| actor.target),
        site: submitted.site().clone(),
        lifecycle: SpanUnixNanoSec::try_new(submitted.timestamp(), terminal.0)?,
        wait,
        transitions,
    })
}

#[cfg(test)]
mod tests {
    use quent_events::{DynamicAttributes, EntityRef, Event};

    use super::*;
    use crate::generated::{Actor, ReservableMemory, ReservableMemoryUsage, Worker};

    #[test]
    fn builds_actor_attributed_accounting_segments() {
        let reservation_id = Uuid::from_u128(1);
        let actor_id = Uuid::from_u128(2);
        let memory_id = Uuid::from_u128(3);
        let reservations = build_reservations(
            vec![
                Event::new(
                    reservation_id,
                    10,
                    MemoryReservationEvent::Created {
                        seq: 0,
                        worker: EntityRef::<Worker>::new(Uuid::from_u128(4), ()),
                        actor: Some(EntityRef::<Actor>::new(actor_id, ())),
                        site: "test.reserve".to_owned(),
                        request: None,
                        parent: None,
                        attributes: DynamicAttributes::new(),
                        sequence: 0,
                    },
                ),
                Event::new(
                    reservation_id,
                    10,
                    MemoryReservationEvent::Held {
                        seq: 1,
                        memory: EntityRef::<ReservableMemory, _>::new(
                            memory_id,
                            ReservableMemoryUsage { bytes: 64 },
                        ),
                        change: "granted".to_owned(),
                        sequence: 1,
                    },
                ),
                Event::new(
                    reservation_id,
                    20,
                    MemoryReservationEvent::Held {
                        seq: 2,
                        memory: EntityRef::<ReservableMemory, _>::new(
                            memory_id,
                            ReservableMemoryUsage { bytes: 16 },
                        ),
                        change: "consumed".to_owned(),
                        sequence: 2,
                    },
                ),
                Event::new(
                    reservation_id,
                    30,
                    MemoryReservationEvent::Released {
                        seq: 3,
                        reason: "consumed".to_owned(),
                        sequence: 3,
                    },
                ),
            ],
            30,
        )
        .unwrap();

        assert_eq!(reservations[0].actor_id, Some(actor_id));
        assert_eq!(reservations[0].segments.len(), 2);
        assert_eq!(reservations[0].segments[0].span.duration(), 10);
        assert_eq!(reservations[0].segments[1].span.duration(), 10);

        let fsm = reservations[0].to_ui_fsm(0);
        assert!(
            fsm.transitions[0]
                .attributes
                .contains(&DynamicAttribute::string(
                    "worker_id",
                    Uuid::from_u128(4).to_string()
                ))
        );
        assert!(
            fsm.transitions[0]
                .attributes
                .contains(&DynamicAttribute::string("actor_id", actor_id.to_string()))
        );
        assert!(
            fsm.transitions[0]
                .attributes
                .contains(&DynamicAttribute::string("site", "test.reserve"))
        );
        assert!(
            fsm.transitions[1]
                .attributes
                .contains(&DynamicAttribute::string("change", "granted"))
        );
        assert!(
            fsm.transitions[2]
                .attributes
                .contains(&DynamicAttribute::string("change", "consumed"))
        );
        assert!(
            fsm.transitions[3]
                .attributes
                .contains(&DynamicAttribute::string("reason", "consumed"))
        );
    }

    #[test]
    fn builds_wait_spans_for_every_terminal_state() {
        let worker = EntityRef::<Worker>::new(Uuid::from_u128(1), ());
        let actor = EntityRef::<Actor>::new(Uuid::from_u128(2), ());
        let reservation = EntityRef::<MemoryReservation>::new(Uuid::from_u128(3), ());
        let mut events = Vec::new();
        for (offset, terminal) in [
            MemoryReservationRequestEvent::Granted {
                seq: 2,
                reservation: Some(reservation.clone()),
                memory_type: "device".to_owned(),
                size: 64,
                overbooked_bytes: 0,
                timed_out: false,
                sequence: 2,
            },
            MemoryReservationRequestEvent::Empty {
                seq: 2,
                timed_out: true,
                sequence: 5,
            },
            MemoryReservationRequestEvent::Failed {
                seq: 2,
                error: "failed".to_owned(),
                sequence: 8,
            },
            MemoryReservationRequestEvent::Cancelled {
                seq: 2,
                sequence: 11,
            },
        ]
        .into_iter()
        .enumerate()
        {
            let id = Uuid::from_u128(10 + offset as u128);
            events.push(Event::new(
                id,
                10,
                MemoryReservationRequestEvent::Submitted {
                    seq: 0,
                    worker: worker.clone(),
                    actor: Some(actor.clone()),
                    site: "test.reserve".to_owned(),
                    size: 64,
                    memory_types: vec!["device".to_owned()],
                    policy: "wait".to_owned(),
                    net_memory_delta: None,
                    attributes: DynamicAttributes::new(),
                    sequence: (offset * 3) as u64,
                },
            ));
            events.push(Event::new(
                id,
                20,
                MemoryReservationRequestEvent::Waiting {
                    seq: 1,
                    sequence: (offset * 3 + 1) as u64,
                },
            ));
            events.push(Event::new(id, 30, terminal));
        }

        let requests = build_requests(events, 30).unwrap();
        assert_eq!(requests.len(), 4);
        assert!(
            requests
                .iter()
                .all(|request| request.wait.unwrap().duration() == 10)
        );
    }

    #[test]
    fn repairs_incomplete_lifecycles_at_trace_end() {
        let worker = EntityRef::<Worker>::new(Uuid::from_u128(1), ());
        let memory_id = Uuid::from_u128(2);
        let reservation_id = Uuid::from_u128(3);
        let request_id = Uuid::from_u128(4);
        let reservations = build_reservations(
            vec![
                Event::new(
                    reservation_id,
                    10,
                    MemoryReservationEvent::Created {
                        seq: 0,
                        worker: worker.clone(),
                        actor: None,
                        site: "test.reserve".to_owned(),
                        request: None,
                        parent: None,
                        attributes: DynamicAttributes::new(),
                        sequence: 0,
                    },
                ),
                Event::new(
                    reservation_id,
                    20,
                    MemoryReservationEvent::Held {
                        seq: 1,
                        memory: EntityRef::<ReservableMemory, _>::new(
                            memory_id,
                            ReservableMemoryUsage { bytes: 64 },
                        ),
                        change: "granted".to_owned(),
                        sequence: 1,
                    },
                ),
            ],
            40,
        )
        .unwrap();
        assert_eq!(reservations[0].segments[0].span.duration(), 20);
        assert_eq!(reservations[0].transitions.last().unwrap().name, "released");
        assert!(
            reservations[0]
                .transitions
                .last()
                .unwrap()
                .attributes
                .contains(&DynamicAttribute::string("reason", "truncated".to_owned()))
        );

        let requests = build_requests(
            vec![
                Event::new(
                    request_id,
                    10,
                    MemoryReservationRequestEvent::Submitted {
                        seq: 0,
                        worker,
                        actor: None,
                        site: "test.reserve".to_owned(),
                        size: 64,
                        memory_types: vec!["device".to_owned()],
                        policy: "wait".to_owned(),
                        net_memory_delta: None,
                        attributes: DynamicAttributes::new(),
                        sequence: 0,
                    },
                ),
                Event::new(
                    request_id,
                    20,
                    MemoryReservationRequestEvent::Waiting {
                        seq: 1,
                        sequence: 1,
                    },
                ),
            ],
            40,
        )
        .unwrap();
        assert_eq!(requests[0].wait.unwrap().duration(), 20);
        assert_eq!(requests[0].transitions.last().unwrap().1, "cancelled");
        assert!(
            requests[0]
                .transitions
                .last()
                .unwrap()
                .2
                .contains(&DynamicAttribute::u8("truncated", 1))
        );
    }

    #[test]
    fn rejects_lifecycles_without_initial_events() {
        let reservation = build_reservations(
            vec![Event::new(
                Uuid::from_u128(1),
                10,
                MemoryReservationEvent::Held {
                    seq: 0,
                    memory: EntityRef::<ReservableMemory, _>::new(
                        Uuid::from_u128(2),
                        ReservableMemoryUsage { bytes: 64 },
                    ),
                    change: "granted".to_owned(),
                    sequence: 0,
                },
            )],
            20,
        );
        assert!(reservation.is_err());

        let request = build_requests(
            vec![Event::new(
                Uuid::from_u128(3),
                10,
                MemoryReservationRequestEvent::Waiting {
                    seq: 0,
                    sequence: 0,
                },
            )],
            20,
        );
        assert!(request.is_err());
    }
}
