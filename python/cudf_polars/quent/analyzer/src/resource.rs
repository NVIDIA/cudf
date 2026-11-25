// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use quent_analyzer::resource::{CapacityDecl, ResourceTypeDecl};
use uuid::Uuid;

pub(crate) const PROCESSOR_RESOURCE_TYPE: &str = "processor";
pub(crate) const DEVICE_MEMORY_RESOURCE_TYPE: &str = "device_memory";
pub(crate) const DATA_CHANNEL_RESOURCE_TYPE: &str = "data_channel";
pub(crate) const EVALUATE_ENTITY_TYPE: &str = "Evaluate";

#[derive(Clone)]
pub(crate) struct DeclaredResource {
    pub(crate) id: Uuid,
    pub(crate) instance_name: String,
    pub(crate) type_name: String,
    pub(crate) parent_group_id: Uuid,
    pub(crate) memory_type: Option<String>,
}

#[derive(Clone)]
pub(crate) struct DeclaredResourceGroup {
    pub(crate) id: Uuid,
    pub(crate) instance_name: String,
    pub(crate) type_name: &'static str,
    pub(crate) parent_group_id: Uuid,
}

pub(crate) fn resource_type(name: &str) -> ResourceTypeDecl {
    let mut resource_type = match name {
        PROCESSOR_RESOURCE_TYPE => ResourceTypeDecl::unit(name),
        DEVICE_MEMORY_RESOURCE_TYPE => {
            ResourceTypeDecl::new(name, [CapacityDecl::new_occupancy("bytes")])
        }
        DATA_CHANNEL_RESOURCE_TYPE => {
            ResourceTypeDecl::new(name, [CapacityDecl::new_rate("bytes")])
        }
        _ if is_reservable_memory_resource_type(name) => {
            ResourceTypeDecl::new(name, [CapacityDecl::new_occupancy("bytes")])
        }
        _ => unreachable!("resource type validated by caller"),
    };
    if name != DEVICE_MEMORY_RESOURCE_TYPE && !is_reservable_memory_resource_type(name) {
        resource_type
            .used_by
            .push(EVALUATE_ENTITY_TYPE.parse().expect("valid schema path"));
    }
    resource_type
}

pub(crate) fn reservable_memory_resource_type(memory_type: &str) -> String {
    format!("reservable_{memory_type}_memory")
}

pub(crate) fn is_reservable_memory_resource_type(name: &str) -> bool {
    name.starts_with("reservable_") && name.ends_with("_memory")
}
