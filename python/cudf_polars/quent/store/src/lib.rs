// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema-generated cudf-polars stored-event types.

#![allow(dead_code, unused_imports)]

include!(concat!(env!("OUT_DIR"), "/cudfpolars.rs"));

impl quent_analyzer::fsm::native::TransitionEvent for QueryEvent {
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
