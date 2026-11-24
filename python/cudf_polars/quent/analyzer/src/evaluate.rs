// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::{
    AnalyzerError, AnalyzerResult,
    resource::{CapacityValue, Usage},
};
use quent_dynamic_attributes::{DynamicAttribute, DynamicList, DynamicStruct};
use quent_time::{TimeUnixNanoSec, span::SpanUnixNanoSec, to_secs_relative};
use quent_ui::{FiniteStateMachine, FsmTransition, FsmUsage};
use uuid::Uuid;

use crate::{
    generated::{DataFrameStatistics, EvaluateEvent},
    resource::EVALUATE_ENTITY_TYPE,
};

fn dataframe_statistics(value: &DataFrameStatistics) -> DynamicStruct {
    DynamicStruct(vec![DynamicAttribute::list(
        "shape",
        DynamicList::U64(value.shape.clone()),
    )])
}

#[derive(Default)]
pub(crate) struct EvaluateBuilder {
    instance_name: Option<String>,
    actor_id: Option<Uuid>,
    processor_id: Option<Uuid>,
    channel: Option<(Uuid, u64)>,
    queued_at: Option<TimeUnixNanoSec>,
    queued_attributes: Vec<DynamicAttribute>,
    running_at: Option<TimeUnixNanoSec>,
    running_attributes: Vec<DynamicAttribute>,
    finished_at: Option<TimeUnixNanoSec>,
    finished_state: Option<&'static str>,
    finished_attributes: Vec<DynamicAttribute>,
}

impl EvaluateBuilder {
    pub(crate) fn push(&mut self, timestamp: TimeUnixNanoSec, event: &EvaluateEvent) {
        match event {
            EvaluateEvent::Queued {
                instance_name,
                actor,
                task,
                ..
            } => {
                self.instance_name = Some(instance_name.clone());
                self.actor_id = Some(actor.target);
                self.queued_at = Some(timestamp);
                if let Some(task) = task {
                    self.queued_attributes = vec![
                        DynamicAttribute::string("task_node_id", task.node_id.clone()),
                        DynamicAttribute::string("task_node_type", task.node_type.clone()),
                    ];
                }
            }
            EvaluateEvent::Running {
                io,
                input_bytes,
                input,
                processor,
                channel,
                ..
            } => {
                self.processor_id = Some(processor.target);
                self.channel = channel
                    .as_ref()
                    .map(|channel| (channel.target, channel.data.bytes));
                self.running_attributes = vec![
                    DynamicAttribute::u8("io", u8::from(*io)),
                    DynamicAttribute::u64("input_bytes", *input_bytes),
                ];
                self.running_attributes.push(DynamicAttribute::list(
                    "input_dataframes",
                    DynamicList::Struct(
                        input.dataframes.iter().map(dataframe_statistics).collect(),
                    ),
                ));
                if let Some(sequence_number) = input.sequence_number {
                    self.running_attributes
                        .push(DynamicAttribute::u64("sequence_number", sequence_number));
                }
                if let Some(content_sizes) = &input.content_sizes {
                    self.running_attributes.push(DynamicAttribute::list(
                        "content_sizes",
                        DynamicList::U64(content_sizes.clone()),
                    ));
                }
                if let Some(spillable) = input.spillable {
                    self.running_attributes
                        .push(DynamicAttribute::u8("spillable", u8::from(spillable)));
                }
                self.running_at = Some(timestamp);
            }
            EvaluateEvent::Completed {
                output_bytes,
                output_dataframe,
                ..
            } => {
                self.finished_at = Some(timestamp);
                self.finished_state = Some("completed");
                self.finished_attributes = vec![
                    DynamicAttribute::u64("output_bytes", *output_bytes),
                    DynamicAttribute::structure(
                        "output_dataframe",
                        dataframe_statistics(output_dataframe),
                    ),
                ];
            }
            EvaluateEvent::Failed { error, .. } => {
                self.finished_at = Some(timestamp);
                self.finished_state = Some("failed");
                self.finished_attributes = vec![DynamicAttribute::string("error", error.clone())];
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
            queued_attributes: self.queued_attributes,
            running_attributes: self.running_attributes,
            finished_state,
            finished_attributes: self.finished_attributes,
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
    queued_attributes: Vec<DynamicAttribute>,
    running_attributes: Vec<DynamicAttribute>,
    finished_state: &'static str,
    finished_attributes: Vec<DynamicAttribute>,
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
        let transition = |name: &str, timestamp, usages, attributes| FsmTransition {
            name: name.to_owned(),
            usages,
            timestamp: to_secs_relative(timestamp, epoch),
            attributes,
            derived_attributes: vec![],
        };
        FiniteStateMachine {
            id: self.id,
            type_name: EVALUATE_ENTITY_TYPE.to_owned(),
            instance_name: self.instance_name.clone(),
            transitions: vec![
                transition(
                    "queued",
                    self.queued_at,
                    vec![],
                    self.queued_attributes.clone(),
                ),
                transition(
                    "running",
                    self.span.start(),
                    running_usages,
                    self.running_attributes.clone(),
                ),
                transition(
                    self.finished_state,
                    self.span.end(),
                    vec![],
                    self.finished_attributes.clone(),
                ),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use quent_analyzer::AnalyzerError;
    use quent_events::EntityRef;

    use super::*;

    fn running_builder() -> EvaluateBuilder {
        let mut builder = EvaluateBuilder::default();
        builder.push(
            1,
            &EvaluateEvent::Queued {
                seq: 0,
                instance_name: "evaluate".to_owned(),
                actor: EntityRef::new(Uuid::now_v7(), ()),
                task: None,
            },
        );
        builder.push(
            2,
            &EvaluateEvent::Running {
                seq: 1,
                io: false,
                input_bytes: 10,
                input: crate::generated::EvaluateInput {
                    dataframes: vec![DataFrameStatistics { shape: vec![2, 3] }],
                    sequence_number: Some(4),
                    content_sizes: Some(vec![6, 4]),
                    spillable: Some(true),
                },
                processor: EntityRef::new(Uuid::now_v7(), crate::generated::ProcessorUsage {}),
                channel: None,
            },
        );
        builder
    }

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
                task: None,
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

    #[test]
    fn includes_running_attributes_in_ui_fsm() {
        let mut builder = running_builder();
        builder.push(
            3,
            &EvaluateEvent::Completed {
                seq: 2,
                output_bytes: 20,
                output_dataframe: DataFrameStatistics { shape: vec![4, 5] },
            },
        );

        let fsm = builder.try_build(Uuid::now_v7()).unwrap().to_ui_fsm(0);

        assert_eq!(
            fsm.transitions[1].attributes,
            vec![
                DynamicAttribute::u8("io", 0),
                DynamicAttribute::u64("input_bytes", 10),
                DynamicAttribute::list(
                    "input_dataframes",
                    DynamicList::Struct(vec![DynamicStruct(vec![DynamicAttribute::list(
                        "shape",
                        DynamicList::U64(vec![2, 3])
                    ),])]),
                ),
                DynamicAttribute::u64("sequence_number", 4),
                DynamicAttribute::list("content_sizes", DynamicList::U64(vec![6, 4])),
                DynamicAttribute::u8("spillable", 1),
            ]
        );
    }

    #[test]
    fn includes_completed_attributes_in_ui_fsm() {
        let mut builder = running_builder();
        builder.push(
            3,
            &EvaluateEvent::Completed {
                seq: 2,
                output_bytes: 20,
                output_dataframe: DataFrameStatistics { shape: vec![4, 5] },
            },
        );

        let fsm = builder.try_build(Uuid::now_v7()).unwrap().to_ui_fsm(0);

        assert_eq!(
            fsm.transitions[2].attributes,
            vec![
                DynamicAttribute::u64("output_bytes", 20),
                DynamicAttribute::structure(
                    "output_dataframe",
                    DynamicStruct(vec![DynamicAttribute::list(
                        "shape",
                        DynamicList::U64(vec![4, 5]),
                    )]),
                ),
            ]
        );
    }

    #[test]
    fn includes_failed_attributes_in_ui_fsm() {
        let mut builder = running_builder();
        builder.push(
            3,
            &EvaluateEvent::Failed {
                seq: 2,
                error: "evaluation failed".to_owned(),
            },
        );

        let fsm = builder.try_build(Uuid::now_v7()).unwrap().to_ui_fsm(0);

        assert_eq!(
            fsm.transitions[2].attributes,
            vec![DynamicAttribute::string("error", "evaluation failed")]
        );
    }
}
