// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};

use uuid::Uuid;

use crate::{actor::ActorSpan, reservation::OccupancySpan};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct OperatorReservationMetrics {
    pub(crate) peak_bytes: u64,
    pub(crate) byte_seconds: f64,
}

#[derive(Default)]
struct OperatorReservationAccumulator {
    byte_deltas: BTreeMap<u64, i128>,
    byte_seconds: f64,
}

pub(crate) fn aggregate_operator_occupancy(
    occupancies: &[OccupancySpan],
    actors: &HashMap<Uuid, ActorSpan>,
    query_operator_ids: &HashSet<Uuid>,
) -> HashMap<Uuid, OperatorReservationMetrics> {
    let mut accumulators = HashMap::<Uuid, OperatorReservationAccumulator>::new();

    for occupancy in occupancies {
        let Some(operator_id) = occupancy
            .actor_id
            .and_then(|actor_id| actors.get(&actor_id))
            .map(|actor| actor.operator_id)
            .filter(|operator_id| query_operator_ids.contains(operator_id))
        else {
            continue;
        };
        let accumulator = accumulators.entry(operator_id).or_default();
        for segment in &occupancy.segments {
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
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use quent_events::{EntityRef, Event};
    use quent_time::span::SpanUnixNanoSec;

    use super::*;
    use crate::{
        generated::{
            Actor, ReservableMemory, ReservableMemoryUsage, ReservationOccupancyEvent, Worker,
        },
        reservation::build_occupancies,
    };

    fn occupancy_events(
        id: Uuid,
        actor_id: Option<Uuid>,
        memory_id: Uuid,
        held: &[(u64, u64)],
        released_at: Option<u64>,
    ) -> Vec<Event<ReservationOccupancyEvent>> {
        let mut events = vec![Event::new(
            id,
            held[0].0.saturating_sub(1),
            ReservationOccupancyEvent::Created {
                seq: 0,
                worker: EntityRef::<Worker>::new(Uuid::from_u128(100), ()),
                actor: actor_id.map(|actor| EntityRef::<Actor>::new(actor, ())),
                site: Some("test.reserve".to_owned()),
            },
        )];
        events.extend(held.iter().enumerate().map(|(index, &(timestamp, bytes))| {
            Event::new(
                id,
                timestamp,
                ReservationOccupancyEvent::Held {
                    seq: (index + 1) as u16,
                    memory: EntityRef::<ReservableMemory, _>::new(
                        memory_id,
                        ReservableMemoryUsage { bytes },
                    ),
                },
            )
        }));
        if let Some(timestamp) = released_at {
            events.push(Event::new(
                id,
                timestamp,
                ReservationOccupancyEvent::Released {
                    seq: (held.len() + 1) as u16,
                },
            ));
        }
        events
    }

    #[test]
    fn aggregates_concurrent_memory_types_zero_crossings_and_truncated_occupancy() {
        let actor_id = Uuid::from_u128(1);
        let operator_id = Uuid::from_u128(2);
        let mut events = occupancy_events(
            Uuid::from_u128(10),
            Some(actor_id),
            Uuid::from_u128(20),
            &[(10, 10), (20, 4), (20, 6), (30, 0)],
            Some(40),
        );
        events.extend(occupancy_events(
            Uuid::from_u128(11),
            Some(actor_id),
            Uuid::from_u128(21),
            &[(15, 5), (20, 7)],
            None,
        ));
        events.extend(occupancy_events(
            Uuid::from_u128(12),
            None,
            Uuid::from_u128(20),
            &[(10, 1_000)],
            Some(40),
        ));
        let occupancies = build_occupancies(events, 40).unwrap();
        let actors = [(
            actor_id,
            ActorSpan {
                id: actor_id,
                operator_id,
                span: SpanUnixNanoSec::try_new(0, 100).unwrap(),
            },
        )]
        .into();

        let metrics = aggregate_operator_occupancy(&occupancies, &actors, &[operator_id].into());

        assert_eq!(
            metrics[&operator_id],
            OperatorReservationMetrics {
                peak_bytes: 15,
                byte_seconds: 325.0 / 1_000_000_000.0,
            }
        );
    }

    #[test]
    fn excludes_occupancy_outside_the_query() {
        let actor_id = Uuid::from_u128(1);
        let operator_id = Uuid::from_u128(2);
        let occupancies = build_occupancies(
            occupancy_events(
                Uuid::from_u128(10),
                Some(actor_id),
                Uuid::from_u128(20),
                &[(10, 10)],
                Some(20),
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

        assert!(aggregate_operator_occupancy(&occupancies, &actors, &HashSet::new()).is_empty());
    }
}
