// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};

use uuid::Uuid;

use crate::{actor::ActorSpan, reservation::ReservationSpan};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct OperatorReservationMetrics {
    pub(crate) peak_bytes: u64,
    pub(crate) byte_seconds: f64,
    pub(crate) reservation_count: u64,
}

#[derive(Default)]
struct OperatorReservationAccumulator {
    byte_deltas: BTreeMap<u64, i128>,
    byte_seconds: f64,
    reservation_count: u64,
}

pub(crate) fn aggregate_operator_reservations(
    reservations: &[ReservationSpan],
    actors: &HashMap<Uuid, ActorSpan>,
    query_operator_ids: &HashSet<Uuid>,
) -> HashMap<Uuid, OperatorReservationMetrics> {
    let mut accumulators = HashMap::<Uuid, OperatorReservationAccumulator>::new();

    for reservation in reservations {
        let Some(operator_id) = reservation
            .actor_id
            .and_then(|actor_id| actors.get(&actor_id))
            .map(|actor| actor.operator_id)
            .filter(|operator_id| query_operator_ids.contains(operator_id))
        else {
            continue;
        };
        let accumulator = accumulators.entry(operator_id).or_default();
        accumulator.reservation_count += 1;
        for segment in &reservation.segments {
            let Some(bytes) = segment.bytes.value else {
                continue;
            };
            *accumulator
                .byte_deltas
                .entry(segment.span.start())
                .or_default() += i128::from(bytes);
            *accumulator
                .byte_deltas
                .entry(segment.span.end())
                .or_default() -= i128::from(bytes);
            accumulator.byte_seconds +=
                bytes as f64 * segment.span.duration() as f64 / 1_000_000_000.0;
        }
    }

    accumulators
        .into_iter()
        .map(|(operator_id, accumulator)| {
            let mut held_bytes = 0_i128;
            let mut peak_bytes = 0_i128;
            for delta in accumulator.byte_deltas.values() {
                held_bytes += delta;
                peak_bytes = peak_bytes.max(held_bytes);
            }
            debug_assert_eq!(held_bytes, 0);
            (
                operator_id,
                OperatorReservationMetrics {
                    peak_bytes: peak_bytes
                        .try_into()
                        .expect("reservation occupancy cannot be negative"),
                    byte_seconds: accumulator.byte_seconds,
                    reservation_count: accumulator.reservation_count,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use quent_events::{DynamicAttributes, EntityRef, Event};
    use quent_time::span::SpanUnixNanoSec;

    use super::*;
    use crate::{
        generated::{
            Actor, MemoryReservation, MemoryReservationEvent, ReservableMemory,
            ReservableMemoryUsage, Worker,
        },
        reservation::build_reservations,
    };

    fn reservation_events(
        id: Uuid,
        actor_id: Option<Uuid>,
        parent_id: Option<Uuid>,
        held: &[(u64, u64, &str)],
        released_at: u64,
    ) -> Vec<Event<MemoryReservationEvent>> {
        let worker = EntityRef::<Worker>::new(Uuid::from_u128(100), ());
        let memory_id = Uuid::from_u128(101);
        let mut events = vec![Event::new(
            id,
            held[0].0.saturating_sub(1),
            MemoryReservationEvent::Created {
                seq: 0,
                worker,
                actor: actor_id.map(|id| EntityRef::<Actor>::new(id, ())),
                site: "test.reserve".to_owned(),
                request: None,
                parent: parent_id.map(|id| EntityRef::<MemoryReservation>::new(id, ())),
                attributes: DynamicAttributes::new(),
                sequence: 0,
            },
        )];
        events.extend(
            held.iter()
                .enumerate()
                .map(|(index, &(timestamp, bytes, change))| {
                    Event::new(
                        id,
                        timestamp,
                        MemoryReservationEvent::Held {
                            seq: (index + 1) as u16,
                            memory: EntityRef::<ReservableMemory, _>::new(
                                memory_id,
                                ReservableMemoryUsage { bytes },
                            ),
                            change: change.to_owned(),
                            sequence: (index + 1) as u64,
                        },
                    )
                }),
        );
        events.push(Event::new(
            id,
            released_at,
            MemoryReservationEvent::Released {
                seq: (held.len() + 1) as u16,
                reason: "consumed".to_owned(),
                sequence: (held.len() + 1) as u64,
            },
        ));
        events
    }

    #[test]
    fn aggregates_concurrent_resizes_splits_and_non_overlapping_reservations() {
        let actor_id = Uuid::from_u128(1);
        let operator_id = Uuid::from_u128(2);
        let first_id = Uuid::from_u128(10);
        let split_id = Uuid::from_u128(11);
        let later_id = Uuid::from_u128(12);
        let unattributed_id = Uuid::from_u128(13);
        let mut events = reservation_events(
            first_id,
            Some(actor_id),
            None,
            &[(10, 10, "granted"), (20, 4, "consumed")],
            30,
        );
        events.extend(reservation_events(
            split_id,
            Some(actor_id),
            Some(first_id),
            &[(15, 6, "split_off")],
            25,
        ));
        events.extend(reservation_events(
            later_id,
            Some(actor_id),
            None,
            &[(40, 20, "granted")],
            45,
        ));
        events.extend(reservation_events(
            unattributed_id,
            None,
            None,
            &[(10, 1_000, "granted")],
            50,
        ));
        let reservations = build_reservations(events, 50).unwrap();
        let actors = [(
            actor_id,
            ActorSpan {
                id: actor_id,
                operator_id,
                span: SpanUnixNanoSec::try_new(0, 100).unwrap(),
            },
        )]
        .into();

        let metrics =
            aggregate_operator_reservations(&reservations, &actors, &[operator_id].into());

        assert_eq!(
            metrics[&operator_id],
            OperatorReservationMetrics {
                peak_bytes: 20,
                byte_seconds: 300.0 / 1_000_000_000.0,
                reservation_count: 3,
            }
        );
    }

    #[test]
    fn excludes_actors_outside_the_query() {
        let actor_id = Uuid::from_u128(1);
        let operator_id = Uuid::from_u128(2);
        let reservations = build_reservations(
            reservation_events(
                Uuid::from_u128(10),
                Some(actor_id),
                None,
                &[(10, 10, "granted")],
                20,
            ),
            20,
        )
        .unwrap();
        let actors = [(
            actor_id,
            ActorSpan {
                id: actor_id,
                operator_id,
                span: SpanUnixNanoSec::try_new(0, 100).unwrap(),
            },
        )]
        .into();

        assert!(
            aggregate_operator_reservations(&reservations, &actors, &HashSet::new()).is_empty()
        );
    }
}
