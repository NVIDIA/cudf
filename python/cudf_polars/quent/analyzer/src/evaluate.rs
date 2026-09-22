// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::{
    AnalyzerError, AnalyzerResult,
    resource::{CapacityValue, Usage},
};
use quent_time::{TimeUnixNanoSec, span::SpanUnixNanoSec, to_secs_relative};
use quent_ui::{FiniteStateMachine, FsmTransition, FsmUsage};
use uuid::Uuid;

use crate::{generated::EvaluateEvent, resource::EVALUATE_ENTITY_TYPE};

#[derive(Default)]
pub(crate) struct EvaluateBuilder {
    instance_name: Option<String>,
    actor_id: Option<Uuid>,
    processor_id: Option<Uuid>,
    channel: Option<(Uuid, u64)>,
    queued_at: Option<TimeUnixNanoSec>,
    running_at: Option<TimeUnixNanoSec>,
    finished_at: Option<TimeUnixNanoSec>,
    finished_state: Option<&'static str>,
}

impl EvaluateBuilder {
    pub(crate) fn push(&mut self, timestamp: TimeUnixNanoSec, event: &EvaluateEvent) {
        match event {
            EvaluateEvent::Queued {
                instance_name,
                actor,
                ..
            } => {
                self.instance_name = Some(instance_name.clone());
                self.actor_id = Some(actor.target);
                self.queued_at = Some(timestamp);
            }
            EvaluateEvent::Running {
                processor, channel, ..
            } => {
                self.processor_id = Some(processor.target);
                self.channel = channel
                    .as_ref()
                    .map(|channel| (channel.target, channel.data.bytes));
                self.running_at = Some(timestamp);
            }
            EvaluateEvent::Completed { .. } => {
                self.finished_at = Some(timestamp);
                self.finished_state = Some("completed");
            }
            EvaluateEvent::Failed { .. } => {
                self.finished_at = Some(timestamp);
                self.finished_state = Some("failed");
            }
        }
    }

    pub(crate) fn try_build(self, id: Uuid) -> AnalyzerResult<EvaluateSpan> {
        let incomplete =
            |field| AnalyzerError::IncompleteEntity(format!("evaluate {id} is missing {field}"));
        let instance_name = self
            .instance_name
            .ok_or_else(|| incomplete("instance name"))?;
        let actor_id = self.actor_id.ok_or_else(|| incomplete("actor"))?;
        let processor_id = self.processor_id.ok_or_else(|| incomplete("processor"))?;
        let queued_at = self
            .queued_at
            .ok_or_else(|| incomplete("queued event timestamp"))?;
        let start = self.running_at.ok_or_else(|| incomplete("running event"))?;
        let end = self
            .finished_at
            .ok_or_else(|| incomplete("completed or failed event"))?;
        let finished_state = self
            .finished_state
            .ok_or_else(|| incomplete("finished state"))?;
        let channel_bytes = self
            .channel
            .map(|(_, bytes)| CapacityValue::new("bytes", bytes));
        Ok(EvaluateSpan {
            id,
            instance_name,
            actor_id,
            processor_id,
            channel: self.channel,
            queued_at,
            span: SpanUnixNanoSec::try_new(start, end)?,
            finished_state,
            processor_unit: CapacityValue::new("unit", 1),
            channel_bytes,
        })
    }
}

pub(crate) struct EvaluateSpan {
    pub(crate) id: Uuid,
    pub(crate) instance_name: String,
    pub(crate) actor_id: Uuid,
    pub(crate) processor_id: Uuid,
    pub(crate) channel: Option<(Uuid, u64)>,
    pub(crate) queued_at: TimeUnixNanoSec,
    pub(crate) span: SpanUnixNanoSec,
    finished_state: &'static str,
    pub(crate) processor_unit: CapacityValue,
    pub(crate) channel_bytes: Option<CapacityValue>,
}

pub(crate) struct EvaluateUsage<'a> {
    pub(crate) evaluate: &'a EvaluateSpan,
    pub(crate) resource_id: Uuid,
    pub(crate) capacity: &'a CapacityValue,
}

impl<'a> Usage<'a> for EvaluateUsage<'a> {
    fn entity_id(&self) -> Uuid {
        self.evaluate.id
    }

    fn resource_id(&self) -> Uuid {
        self.resource_id
    }

    fn capacities(&self) -> impl Iterator<Item = &'a CapacityValue> {
        std::iter::once(self.capacity)
    }

    fn span(&self) -> SpanUnixNanoSec {
        self.evaluate.span
    }
}

impl EvaluateSpan {
    pub(crate) fn to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> FiniteStateMachine {
        let mut running_usages = vec![FsmUsage {
            resource: self.processor_id,
            capacities: vec![("unit".to_owned(), Some(1))],
        }];
        if let Some((channel_id, bytes)) = self.channel {
            running_usages.push(FsmUsage {
                resource: channel_id,
                capacities: vec![("bytes".to_owned(), Some(bytes))],
            });
        }
        let transition = |name: &str, timestamp, usages| FsmTransition {
            name: name.to_owned(),
            usages,
            timestamp: to_secs_relative(timestamp, epoch),
            attributes: vec![],
            derived_attributes: vec![],
        };
        FiniteStateMachine {
            id: self.id,
            type_name: EVALUATE_ENTITY_TYPE.to_owned(),
            instance_name: self.instance_name.clone(),
            transitions: vec![
                transition("queued", self.queued_at, vec![]),
                transition("running", self.span.start(), running_usages),
                transition(self.finished_state, self.span.end(), vec![]),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use quent_analyzer::AnalyzerError;
    use quent_events::EntityRef;

    use super::*;

    #[test]
    fn rejects_incomplete_lifecycle() {
        let id = Uuid::now_v7();
        let mut builder = EvaluateBuilder::default();
        builder.push(
            1,
            &EvaluateEvent::Queued {
                seq: 0,
                instance_name: "evaluate".to_owned(),
                actor: EntityRef::new(Uuid::now_v7(), ()),
            },
        );

        let error = builder
            .try_build(id)
            .err()
            .expect("evaluate should be incomplete");
        assert!(
            matches!(error, AnalyzerError::IncompleteEntity(message) if message.contains(&id.to_string()))
        );
    }
}
