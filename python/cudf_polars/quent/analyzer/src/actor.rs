// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::{AnalyzerError, AnalyzerResult};
use quent_time::{TimeUnixNanoSec, span::SpanUnixNanoSec};
use uuid::Uuid;

use crate::generated::ActorEvent;

#[derive(Default)]
pub(crate) struct ActorBuilder {
    operator_id: Option<Uuid>,
    worker_id: Option<Uuid>,
    running_at: Option<TimeUnixNanoSec>,
    finished_at: Option<TimeUnixNanoSec>,
}

impl ActorBuilder {
    pub(crate) fn push(&mut self, timestamp: TimeUnixNanoSec, event: &ActorEvent) {
        match event {
            ActorEvent::Started {
                operator, worker, ..
            } => {
                self.operator_id = Some(operator.target);
                self.worker_id = Some(worker.target);
            }
            ActorEvent::Running { .. } => self.running_at = Some(timestamp),
            ActorEvent::Completed { .. } | ActorEvent::Failed { .. } => {
                self.finished_at = Some(timestamp);
            }
        }
    }

    pub(crate) fn try_build(self, id: Uuid) -> AnalyzerResult<ActorSpan> {
        let incomplete =
            |field| AnalyzerError::IncompleteEntity(format!("actor {id} is missing {field}"));
        let operator_id = self.operator_id.ok_or_else(|| incomplete("operator"))?;
        self.worker_id.ok_or_else(|| incomplete("worker"))?;
        let start = self.running_at.ok_or_else(|| incomplete("running event"))?;
        let end = self
            .finished_at
            .ok_or_else(|| incomplete("completed or failed event"))?;
        Ok(ActorSpan {
            id,
            operator_id,
            span: SpanUnixNanoSec::try_new(start, end)?,
        })
    }
}

pub(crate) struct ActorSpan {
    pub(crate) id: Uuid,
    pub(crate) operator_id: Uuid,
    pub(crate) span: SpanUnixNanoSec,
}

#[cfg(test)]
mod tests {
    use quent_analyzer::AnalyzerError;
    use quent_events::EntityRef;

    use super::*;

    #[test]
    fn rejects_incomplete_lifecycle() {
        let id = Uuid::now_v7();
        let mut builder = ActorBuilder::default();
        builder.push(
            1,
            &ActorEvent::Started {
                seq: 0,
                operator: EntityRef::new(Uuid::now_v7(), ()),
                worker: EntityRef::new(Uuid::now_v7(), ()),
            },
        );

        let error = builder
            .try_build(id)
            .err()
            .expect("actor should be incomplete");
        assert!(
            matches!(error, AnalyzerError::IncompleteEntity(message) if message.contains(&id.to_string()))
        );
    }
}
