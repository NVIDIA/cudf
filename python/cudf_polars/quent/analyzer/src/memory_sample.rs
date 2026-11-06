// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sampled memory resource occupancy readings.
//!
//! One `MemoryResourceSampler` FSM covers one query on one worker. Each of its
//! `sampling` transitions is a reading of memory resource occupancy that holds a
//! `SampledMemoryResource` usage until the next transition (the next reading, or
//! `stopped`). Summed over the resource, consecutive readings form a step
//! function.
//!
//! The FSM sequence number is a `u16` that wraps on long sampling runs, so
//! readings are ordered by timestamp rather than by sequence number.

use quent_analyzer::{
    AnalyzerError, AnalyzerResult,
    resource::{CapacityValue, Usage},
};
use quent_time::{TimeUnixNanoSec, span::SpanUnixNanoSec};
use uuid::Uuid;

use crate::generated::MemoryResourceSamplerEvent;

struct Reading {
    timestamp: TimeUnixNanoSec,
    bytes: u64,
}

#[derive(Default)]
pub(crate) struct MemorySampleBuilder {
    query_id: Option<Uuid>,
    memory_id: Option<Uuid>,
    readings: Vec<Reading>,
    stopped_at: Option<TimeUnixNanoSec>,
}

impl MemorySampleBuilder {
    pub(crate) fn push(&mut self, timestamp: TimeUnixNanoSec, event: &MemoryResourceSamplerEvent) {
        match event {
            MemoryResourceSamplerEvent::Sampling { query, memory, .. } => {
                self.query_id = Some(query.target);
                self.memory_id = Some(memory.target);
                self.readings.push(Reading {
                    timestamp,
                    bytes: memory.data.bytes,
                });
            }
            MemoryResourceSamplerEvent::Stopped { .. } => self.stopped_at = Some(timestamp),
        }
    }

    /// Build one span per reading whose end is known.
    ///
    /// A reading ends at the next reading, or when the sampler stops. If the
    /// sampler never stopped (e.g. the process died), its last reading has no
    /// known end and is dropped; the earlier readings are kept.
    pub(crate) fn try_build(mut self, id: Uuid) -> AnalyzerResult<Vec<MemorySampleSpan>> {
        let incomplete = |field| {
            AnalyzerError::IncompleteEntity(format!("memory sampler {id} is missing {field}"))
        };
        let query_id = self.query_id.ok_or_else(|| incomplete("query"))?;
        let memory_id = self.memory_id.ok_or_else(|| incomplete("memory"))?;
        self.readings.sort_by_key(|reading| reading.timestamp);
        let ends = self
            .readings
            .iter()
            .skip(1)
            .map(|reading| Some(reading.timestamp))
            .chain(std::iter::once(self.stopped_at));
        self.readings
            .iter()
            .zip(ends)
            .filter_map(|(reading, end)| end.map(|end| (reading, end)))
            .map(|(reading, end)| {
                Ok(MemorySampleSpan {
                    id,
                    query_id,
                    memory_id,
                    span: SpanUnixNanoSec::try_new(reading.timestamp, end)?,
                    bytes: CapacityValue::new("bytes", reading.bytes),
                })
            })
            .collect()
    }
}

pub(crate) struct MemorySampleSpan {
    pub(crate) id: Uuid,
    pub(crate) query_id: Uuid,
    pub(crate) memory_id: Uuid,
    pub(crate) span: SpanUnixNanoSec,
    pub(crate) bytes: CapacityValue,
}

pub(crate) struct MemorySampleUsage<'a>(pub(crate) &'a MemorySampleSpan);

impl<'a> Usage<'a> for MemorySampleUsage<'a> {
    fn entity_id(&self) -> Uuid {
        self.0.id
    }

    fn resource_id(&self) -> Uuid {
        self.0.memory_id
    }

    fn capacities(&self) -> impl Iterator<Item = &'a CapacityValue> {
        std::iter::once(&self.0.bytes)
    }

    fn span(&self) -> SpanUnixNanoSec {
        self.0.span
    }
}

#[cfg(test)]
mod tests {
    use quent_events::EntityRef;

    use super::*;
    use crate::generated::SampledMemoryResourceUsage;

    fn sampling(bytes: u64, query: Uuid, memory: Uuid) -> MemoryResourceSamplerEvent {
        MemoryResourceSamplerEvent::Sampling {
            seq: 0,
            query: EntityRef::new(query, ()),
            memory: EntityRef::new(memory, SampledMemoryResourceUsage { bytes }),
        }
    }

    #[test]
    fn rejects_sampler_without_readings() {
        let id = Uuid::now_v7();
        let mut builder = MemorySampleBuilder::default();
        builder.push(1, &MemoryResourceSamplerEvent::Stopped { seq: 1 });

        let error = builder
            .try_build(id)
            .err()
            .expect("sampler should be incomplete");
        assert!(
            matches!(error, AnalyzerError::IncompleteEntity(message) if message.contains(&id.to_string()))
        );
    }

    #[test]
    fn each_reading_ends_at_the_next_transition() {
        let (query, memory) = (Uuid::now_v7(), Uuid::now_v7());
        let mut builder = MemorySampleBuilder::default();
        builder.push(10, &sampling(1, query, memory));
        builder.push(20, &sampling(2, query, memory));
        builder.push(35, &sampling(3, query, memory));
        builder.push(40, &MemoryResourceSamplerEvent::Stopped { seq: 3 });

        let spans = builder.try_build(Uuid::now_v7()).unwrap();
        let bounds: Vec<_> = spans
            .iter()
            .map(|sample| (sample.span.start(), sample.span.end()))
            .collect();
        assert_eq!(bounds, [(10, 20), (20, 35), (35, 40)]);
        assert!(spans.iter().all(|sample| sample.query_id == query));
        assert!(spans.iter().all(|sample| sample.memory_id == memory));
    }

    #[test]
    fn drops_only_the_unterminated_last_reading() {
        let (query, memory) = (Uuid::now_v7(), Uuid::now_v7());
        let mut builder = MemorySampleBuilder::default();
        builder.push(10, &sampling(1, query, memory));
        builder.push(20, &sampling(2, query, memory));

        let spans = builder.try_build(Uuid::now_v7()).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            (spans[0].span.start(), spans[0].span.end()),
            (10, 20),
            "the reading that has a successor is kept"
        );
    }
}
