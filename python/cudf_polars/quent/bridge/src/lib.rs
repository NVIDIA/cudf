// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyInt, PyMapping, PyMappingMethods, PyString, PyTuple};

include!(concat!(env!("OUT_DIR"), "/cudfpolars.rs"));

mod generated_python {
    include!(concat!(env!("OUT_DIR"), "/pyo3_bridge.rs"));

    pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        __quent_pyo3_bridge::_quent_generated(module)
    }

    pub use __quent_pyo3_bridge::reservation_event_observers;
}

const MEMORY_DEVICE: u32 = 0;
const MEMORY_HOST: u32 = 1;
const MEMORY_PINNED_HOST: u32 = 2;
const MEMORY_DISK: u32 = 3;

#[derive(Clone, Debug)]
enum OwnedValue {
    String(String),
    I64(i64),
    Bool(bool),
    F64(f64),
}

#[derive(Clone, Debug)]
struct OwnedAttribute {
    key: String,
    value: OwnedValue,
}

#[derive(Clone, Debug)]
enum ReservationEventData {
    RequestSubmitted {
        id: u64,
        size: u64,
        memory_types: Vec<u32>,
        policy: u32,
        delta: Option<i64>,
        attributes: Vec<OwnedAttribute>,
    },
    RequestWaiting {
        id: u64,
    },
    RequestResolved {
        id: u64,
        outcome: u32,
        reservation_id: u64,
        memory_type: u32,
        size: u64,
        overbooked_bytes: u64,
        timed_out: bool,
        error: String,
    },
    ReservationCreated {
        id: u64,
        memory_type: u32,
        size: u64,
        attributes: Vec<OwnedAttribute>,
        request_id: u64,
        parent_id: u64,
    },
    ReservationResized {
        id: u64,
        size: u64,
        reason: u32,
    },
    ReservationReleased {
        id: u64,
        reason: u32,
    },
    MemoryLimitChanged {
        memory_type: u32,
        configured_limit: i64,
    },
}

#[derive(Clone, Debug)]
struct ReservationEvent {
    sequence: u64,
    timestamp_ns: u64,
    data: ReservationEventData,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestStatus {
    Submitted,
    Waiting,
}

#[derive(Clone, Copy, Debug)]
struct RequestState {
    status: RequestStatus,
    next_fsm_sequence: u16,
}

#[derive(Clone, Copy, Debug)]
struct ReservationState {
    memory_type: u32,
    next_fsm_sequence: u16,
}

#[derive(Clone, Debug, Default)]
struct Lifecycle {
    next_sequence: u64,
    requests: HashMap<u64, RequestState>,
    known_requests: HashSet<u64>,
    reservations: HashMap<u64, ReservationState>,
    known_reservations: HashSet<u64>,
}

impl Lifecycle {
    fn advance_fsm(sequence: &mut u16, entity: &str, id: u64) -> Result<u16, String> {
        let current = *sequence;
        *sequence = sequence
            .checked_add(1)
            .ok_or_else(|| format!("{entity} {id} exceeded the FSM sequence limit"))?;
        Ok(current)
    }

    fn validate(
        &mut self,
        event: &ReservationEvent,
        memories: &HashSet<u32>,
    ) -> Result<(), String> {
        if event.sequence != self.next_sequence {
            return Err(format!(
                "reservation event sequence discontinuity: expected {}, got {}",
                self.next_sequence, event.sequence
            ));
        }
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| "reservation event sequence overflow".to_owned())?;

        match &event.data {
            ReservationEventData::RequestSubmitted {
                id,
                memory_types,
                policy,
                ..
            } => {
                if !self.known_requests.insert(*id) {
                    return Err(format!("duplicate request id {id}"));
                }
                for memory_type in memory_types {
                    memory_type_name(*memory_type)?;
                }
                policy_name(*policy)?;
                self.requests.insert(
                    *id,
                    RequestState {
                        status: RequestStatus::Submitted,
                        next_fsm_sequence: 1,
                    },
                );
            }
            ReservationEventData::RequestWaiting { id } => {
                let state = self
                    .requests
                    .get_mut(id)
                    .ok_or_else(|| format!("waiting for unknown request {id}"))?;
                if state.status != RequestStatus::Submitted {
                    return Err(format!("request {id} cannot wait more than once"));
                }
                Self::advance_fsm(&mut state.next_fsm_sequence, "request", *id)?;
                state.status = RequestStatus::Waiting;
            }
            ReservationEventData::RequestResolved {
                id,
                outcome,
                reservation_id,
                memory_type,
                ..
            } => {
                let mut state = self
                    .requests
                    .remove(id)
                    .ok_or_else(|| format!("resolution for unknown request {id}"))?;
                Self::advance_fsm(&mut state.next_fsm_sequence, "request", *id)?;
                match outcome {
                    0 => {
                        memory_type_name(*memory_type)?;
                        if !self.known_reservations.contains(reservation_id) {
                            return Err(format!(
                                "request {id} granted unknown reservation {reservation_id}"
                            ));
                        }
                    }
                    1..=3 => {}
                    value => return Err(format!("invalid request outcome {value}")),
                }
            }
            ReservationEventData::ReservationCreated {
                id,
                memory_type,
                size,
                request_id,
                parent_id,
                ..
            } => {
                if *size == 0 {
                    return Err(format!("reservation {id} was created with zero bytes"));
                }
                if !memories.contains(memory_type) {
                    return Err(format!(
                        "reservation uses undeclared {}",
                        memory_type_description(*memory_type)
                    ));
                }
                if !self.known_reservations.insert(*id) {
                    return Err(format!("duplicate reservation id {id}"));
                }
                if *request_id != 0 && !self.requests.contains_key(request_id) {
                    return Err(format!(
                        "reservation {id} references unknown request {request_id}"
                    ));
                }
                if *parent_id != 0 && !self.known_reservations.contains(parent_id) {
                    return Err(format!(
                        "reservation {id} references unknown parent {parent_id}"
                    ));
                }
                self.reservations.insert(
                    *id,
                    ReservationState {
                        memory_type: *memory_type,
                        next_fsm_sequence: 2,
                    },
                );
            }
            ReservationEventData::ReservationResized { id, size, reason } => {
                if *size == 0 {
                    return Err(format!("reservation {id} resized to zero bytes"));
                }
                resize_reason(*reason)?;
                let state = self
                    .reservations
                    .get_mut(id)
                    .ok_or_else(|| format!("resize for unknown reservation {id}"))?;
                Self::advance_fsm(&mut state.next_fsm_sequence, "reservation", *id)?;
            }
            ReservationEventData::ReservationReleased { id, reason } => {
                release_reason(*reason)?;
                let mut state = self
                    .reservations
                    .remove(id)
                    .ok_or_else(|| format!("release for unknown reservation {id}"))?;
                Self::advance_fsm(&mut state.next_fsm_sequence, "reservation", *id)?;
            }
            ReservationEventData::MemoryLimitChanged { memory_type, .. } => {
                if !memories.contains(memory_type) {
                    return Err(format!(
                        "memory limit changed for undeclared {}",
                        memory_type_description(*memory_type)
                    ));
                }
            }
        }
        Ok(())
    }

    fn fsm_sequence_before(&self, event: &ReservationEvent) -> u16 {
        match &event.data {
            ReservationEventData::RequestSubmitted { .. }
            | ReservationEventData::ReservationCreated { .. } => 0,
            ReservationEventData::RequestWaiting { id }
            | ReservationEventData::RequestResolved { id, .. } => self
                .requests
                .get(id)
                .map_or(0, |state| state.next_fsm_sequence),
            ReservationEventData::ReservationResized { id, .. }
            | ReservationEventData::ReservationReleased { id, .. } => self
                .reservations
                .get(id)
                .map_or(0, |state| state.next_fsm_sequence),
            ReservationEventData::MemoryLimitChanged { .. } => 0,
        }
    }
}

#[pyclass]
struct ReservationEventIngestor {
    worker: EntityRef<Worker>,
    memory_ids: HashMap<u32, Uuid>,
    request_observer: Observer<MemoryReservationRequest>,
    reservation_observer: Observer<MemoryReservation>,
    actor_observer: Observer<Actor>,
    memory_observer: Observer<ReservableMemory>,
    lifecycle: Lifecycle,
    diagnostics: Vec<String>,
    closed: bool,
}

impl ReservationEventIngestor {
    fn ingest_owned(&mut self, events: Vec<ReservationEvent>) -> Result<(), String> {
        if self.closed {
            return Err("reservation event ingestor is closed".to_owned());
        }

        let memory_types = self.memory_ids.keys().copied().collect();
        let mut next = self.lifecycle.clone();
        for event in &events {
            next.validate(event, &memory_types)?;
        }

        let mut current = self.lifecycle.clone();
        for event in events {
            let fsm_sequence = current.fsm_sequence_before(&event);
            self.forward_event(&event, fsm_sequence, &current)?;
            current.validate(&event, &memory_types)?;
        }
        self.lifecycle = next;
        Ok(())
    }

    fn forward_event(
        &mut self,
        event: &ReservationEvent,
        seq: u16,
        lifecycle: &Lifecycle,
    ) -> Result<(), String> {
        let timestamp = event.timestamp_ns;
        let sequence = event.sequence;
        match &event.data {
            ReservationEventData::RequestSubmitted {
                id,
                size,
                memory_types,
                policy,
                delta,
                attributes,
            } => {
                let translated = self.translate_attributes(attributes);
                let data = MemoryReservationRequestEvent::Submitted {
                    seq,
                    worker: self.worker.clone(),
                    actor: translated.actor,
                    site: translated.site,
                    size: *size,
                    memory_types: memory_types
                        .iter()
                        .map(|value| memory_type_name(*value))
                        .collect::<Result<_, _>>()?,
                    policy: policy_name(*policy)?.to_owned(),
                    net_memory_delta: *delta,
                    attributes: translated.dynamic,
                    sequence,
                };
                self.request_observer.forward(raw_event(
                    stable_id(b"request", *id),
                    timestamp,
                    data,
                ));
            }
            ReservationEventData::RequestWaiting { id } => {
                self.request_observer.forward(raw_event(
                    stable_id(b"request", *id),
                    timestamp,
                    MemoryReservationRequestEvent::Waiting { seq, sequence },
                ));
            }
            ReservationEventData::RequestResolved {
                id,
                outcome,
                reservation_id,
                memory_type,
                size,
                overbooked_bytes,
                timed_out,
                error,
            } => {
                let data = match outcome {
                    0 => MemoryReservationRequestEvent::Granted {
                        seq,
                        reservation: self.reservation_ref(*reservation_id),
                        memory_type: memory_type_name(*memory_type)?,
                        size: *size,
                        overbooked_bytes: *overbooked_bytes,
                        timed_out: *timed_out,
                        sequence,
                    },
                    1 => MemoryReservationRequestEvent::Empty {
                        seq,
                        timed_out: *timed_out,
                        sequence,
                    },
                    2 => MemoryReservationRequestEvent::Failed {
                        seq,
                        error: error.clone(),
                        sequence,
                    },
                    3 => MemoryReservationRequestEvent::Cancelled { seq, sequence },
                    value => return Err(format!("invalid request outcome {value}")),
                };
                self.request_observer.forward(raw_event(
                    stable_id(b"request", *id),
                    timestamp,
                    data,
                ));
            }
            ReservationEventData::ReservationCreated {
                id,
                memory_type,
                size,
                attributes,
                request_id,
                parent_id,
            } => {
                let translated = self.translate_attributes(attributes);
                let entity_id = stable_id(b"reservation", *id);
                self.reservation_observer.forward(raw_event(
                    entity_id,
                    timestamp,
                    MemoryReservationEvent::Created {
                        seq,
                        worker: self.worker.clone(),
                        actor: translated.actor,
                        site: translated.site,
                        request: (*request_id != 0).then(|| self.request_ref(*request_id)),
                        parent: (*parent_id != 0).then(|| self.reservation_ref(*parent_id)),
                        attributes: translated.dynamic,
                        sequence,
                    },
                ));
                self.reservation_observer.forward(raw_event(
                    entity_id,
                    timestamp,
                    MemoryReservationEvent::Held {
                        seq: seq + 1,
                        memory: self.memory_usage(*memory_type, *size)?,
                        change: if *parent_id == 0 {
                            "granted"
                        } else {
                            "split_child"
                        }
                        .to_owned(),
                        sequence,
                    },
                ));
            }
            ReservationEventData::ReservationResized { id, size, reason } => {
                let memory_type = lifecycle
                    .reservations
                    .get(id)
                    .ok_or_else(|| format!("resize for unknown reservation {id}"))?
                    .memory_type;
                self.reservation_observer.forward(raw_event(
                    stable_id(b"reservation", *id),
                    timestamp,
                    MemoryReservationEvent::Held {
                        seq,
                        memory: self.memory_usage(memory_type, *size)?,
                        change: resize_reason(*reason)?.to_owned(),
                        sequence,
                    },
                ));
            }
            ReservationEventData::ReservationReleased { id, reason } => {
                self.reservation_observer.forward(raw_event(
                    stable_id(b"reservation", *id),
                    timestamp,
                    MemoryReservationEvent::Released {
                        seq,
                        reason: release_reason(*reason)?.to_owned(),
                        sequence,
                    },
                ));
            }
            ReservationEventData::MemoryLimitChanged {
                memory_type,
                configured_limit,
            } => {
                let id = *self.memory_ids.get(memory_type).ok_or_else(|| {
                    format!(
                        "memory limit changed for undeclared {}",
                        memory_type_description(*memory_type)
                    )
                })?;
                self.memory_observer.forward(raw_event(
                    id,
                    timestamp,
                    ReservableMemoryEvent::Resized {
                        limits: ReservableMemoryBounds {
                            bytes: (*configured_limit).max(0) as u64,
                        },
                        configured_limit: *configured_limit,
                        sequence,
                    },
                ));
            }
        }
        Ok(())
    }

    fn request_ref(&self, id: u64) -> EntityRef<MemoryReservationRequest> {
        self.request_observer
            .handle_with_id(stable_id(b"request", id))
            .as_entity_ref()
    }

    fn reservation_ref(&self, id: u64) -> EntityRef<MemoryReservation> {
        self.reservation_observer
            .handle_with_id(stable_id(b"reservation", id))
            .as_entity_ref()
    }

    fn memory_usage(
        &self,
        memory_type: u32,
        bytes: u64,
    ) -> Result<EntityRef<ReservableMemory, ReservableMemoryUsage>, String> {
        let id = self
            .memory_ids
            .get(&memory_type)
            .ok_or_else(|| {
                format!(
                    "reservation uses undeclared {}",
                    memory_type_description(memory_type)
                )
            })?
            .to_owned();
        Ok(self
            .memory_observer
            .handle_with_id(id)
            .as_entity_ref_with(ReservableMemoryUsage { bytes }))
    }

    fn translate_attributes(&mut self, attributes: &[OwnedAttribute]) -> TranslatedAttributes {
        let mut site = None;
        let mut actor = None;
        let mut dynamic = DynamicAttributes::new();
        for attribute in attributes {
            match attribute.key.as_str() {
                "rapidsmpf.site" => {
                    if let OwnedValue::String(value) = &attribute.value {
                        site = Some(value.clone());
                    } else {
                        self.diagnostics
                            .push("rapidsmpf.site is not a string".to_owned());
                    }
                }
                "quent.actor_id" => match &attribute.value {
                    OwnedValue::String(value) => match Uuid::parse_str(value) {
                        Ok(id) => {
                            actor = Some(self.actor_observer.handle_with_id(id).as_entity_ref())
                        }
                        Err(_) => {
                            self.diagnostics
                                .push(format!("invalid actor UUID: {value}"));
                            dynamic.add("invalid_actor_id", value.clone());
                        }
                    },
                    value => {
                        self.diagnostics
                            .push("quent.actor_id is not a string".to_owned());
                        add_dynamic(&mut dynamic, "invalid_actor_id".to_owned(), value);
                    }
                },
                _ => add_dynamic(&mut dynamic, attribute.key.clone(), &attribute.value),
            }
        }
        TranslatedAttributes {
            site: site.unwrap_or_else(|| "unknown".to_owned()),
            actor,
            dynamic,
        }
    }
}

#[pymethods]
impl ReservationEventIngestor {
    #[new]
    fn new(
        context: &Bound<'_, PyAny>,
        worker_id: &Bound<'_, PyAny>,
        reservable_memory_ids: &Bound<'_, PyMapping>,
    ) -> PyResult<Self> {
        let (
            request_observer,
            reservation_observer,
            worker_observer,
            actor_observer,
            memory_observer,
        ) = generated_python::reservation_event_observers(context)?;
        let worker_id = parse_uuid(worker_id, "worker_id")?;
        Ok(Self {
            worker: worker_observer.handle_with_id(worker_id).as_entity_ref(),
            memory_ids: parse_memory_ids(reservable_memory_ids)?,
            request_observer,
            reservation_observer,
            actor_observer,
            memory_observer,
            lifecycle: Lifecycle::default(),
            diagnostics: Vec::new(),
            closed: false,
        })
    }

    fn ingest(&mut self, events: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut parsed = Vec::new();
        for (index, event) in events.try_iter()?.enumerate() {
            let event = event?
                .cast_into::<PyMapping>()
                .map_err(|_| PyValueError::new_err(format!("event {index} is not a mapping")))?;
            parsed.push(
                parse_event(&event)
                    .map_err(|error| PyValueError::new_err(format!("event {index}: {error}")))?,
            );
        }
        self.ingest_owned(parsed).map_err(PyValueError::new_err)
    }

    fn diagnostics(&self) -> Vec<String> {
        self.diagnostics.clone()
    }

    fn close(&mut self) -> PyResult<()> {
        if self.closed {
            return Ok(());
        }
        let mut errors = Vec::new();
        if !self.lifecycle.requests.is_empty() {
            errors.push(format!(
                "{} live reservation request(s)",
                self.lifecycle.requests.len()
            ));
        }
        if !self.lifecycle.reservations.is_empty() {
            errors.push(format!(
                "{} live reservation(s)",
                self.lifecycle.reservations.len()
            ));
        }
        if errors.is_empty() {
            self.closed = true;
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(errors.join("; ")))
        }
    }
}

struct TranslatedAttributes {
    site: String,
    actor: Option<EntityRef<Actor>>,
    dynamic: DynamicAttributes,
}

fn raw_event<P>(id: Uuid, timestamp_ns: u64, data: P) -> Event<P> {
    Event::new(id, timestamp_ns, data)
}

fn add_dynamic(attributes: &mut DynamicAttributes, key: String, value: &OwnedValue) {
    match value {
        OwnedValue::String(value) => attributes.add(key, value.clone()),
        OwnedValue::I64(value) => attributes.add(key, *value),
        OwnedValue::Bool(value) => attributes.add(key, *value),
        OwnedValue::F64(value) => attributes.add(key, *value),
    }
}

fn parse_event(mapping: &Bound<'_, PyMapping>) -> PyResult<ReservationEvent> {
    let sequence = mapping.get_item("sequence")?.extract()?;
    let timestamp_ns = mapping.get_item("timestamp_ns")?.extract()?;
    let kind: String = mapping.get_item("kind")?.extract()?;
    let data = match kind.as_str() {
        "request_submitted" => ReservationEventData::RequestSubmitted {
            id: mapping.get_item("id")?.extract()?,
            size: mapping.get_item("size")?.extract()?,
            memory_types: mapping.get_item("memory_types")?.extract()?,
            policy: mapping.get_item("policy")?.extract()?,
            delta: mapping.get_item("net_memory_delta")?.extract()?,
            attributes: parse_attributes(&mapping.get_item("attributes")?)?,
        },
        "request_waiting" => ReservationEventData::RequestWaiting {
            id: mapping.get_item("id")?.extract()?,
        },
        "request_resolved" => ReservationEventData::RequestResolved {
            id: mapping.get_item("id")?.extract()?,
            outcome: mapping.get_item("outcome")?.extract()?,
            reservation_id: mapping.get_item("reservation_id")?.extract()?,
            memory_type: mapping.get_item("memory_type")?.extract()?,
            size: mapping.get_item("size")?.extract()?,
            overbooked_bytes: mapping.get_item("overbooked_bytes")?.extract()?,
            timed_out: mapping.get_item("timed_out")?.extract()?,
            error: mapping.get_item("error")?.extract()?,
        },
        "reservation_created" => ReservationEventData::ReservationCreated {
            id: mapping.get_item("id")?.extract()?,
            memory_type: mapping.get_item("memory_type")?.extract()?,
            size: mapping.get_item("size")?.extract()?,
            attributes: parse_attributes(&mapping.get_item("attributes")?)?,
            request_id: mapping.get_item("request_id")?.extract()?,
            parent_id: mapping.get_item("parent_id")?.extract()?,
        },
        "reservation_resized" => ReservationEventData::ReservationResized {
            id: mapping.get_item("id")?.extract()?,
            size: mapping.get_item("size")?.extract()?,
            reason: mapping.get_item("reason")?.extract()?,
        },
        "reservation_released" => ReservationEventData::ReservationReleased {
            id: mapping.get_item("id")?.extract()?,
            reason: mapping.get_item("reason")?.extract()?,
        },
        "memory_limit_changed" => ReservationEventData::MemoryLimitChanged {
            memory_type: mapping.get_item("memory_type")?.extract()?,
            configured_limit: mapping.get_item("configured_limit")?.extract()?,
        },
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown event kind {kind:?}"
            )));
        }
    };
    Ok(ReservationEvent {
        sequence,
        timestamp_ns,
        data,
    })
}

fn parse_attributes(value: &Bound<'_, PyAny>) -> PyResult<Vec<OwnedAttribute>> {
    let mapping = value
        .clone()
        .cast_into::<PyMapping>()
        .map_err(|_| PyValueError::new_err("attributes is not a mapping"))?;
    let mut result = Vec::new();
    for item in mapping.items()?.try_iter()? {
        let item = item?.cast_into::<PyTuple>()?;
        let key: String = item.get_item(0)?.extract()?;
        let value = item.get_item(1)?;
        let value = if value.is_instance_of::<PyBool>() {
            OwnedValue::Bool(value.extract()?)
        } else if value.is_instance_of::<PyString>() {
            OwnedValue::String(value.extract()?)
        } else if value.is_instance_of::<PyInt>() {
            OwnedValue::I64(value.extract().map_err(|_| {
                PyValueError::new_err(format!("integer attribute {key:?} does not fit in i64"))
            })?)
        } else if value.is_instance_of::<PyFloat>() {
            OwnedValue::F64(value.extract()?)
        } else {
            return Err(PyValueError::new_err(format!(
                "attribute {key:?} must be str, int, bool, or float"
            )));
        };
        result.push(OwnedAttribute { key, value });
    }
    Ok(result)
}

fn parse_uuid(value: &Bound<'_, PyAny>, field: &str) -> PyResult<Uuid> {
    let text = value.str()?.to_str()?.to_owned();
    Uuid::parse_str(&text).map_err(|error| PyValueError::new_err(format!("{field}: {error}")))
}

fn parse_memory_ids(mapping: &Bound<'_, PyMapping>) -> PyResult<HashMap<u32, Uuid>> {
    let mut result = HashMap::new();
    for item in mapping.items()?.try_iter()? {
        let item = item?.cast_into::<PyTuple>()?;
        let key: String = item.get_item(0)?.extract()?;
        let id = parse_uuid(
            &item.get_item(1)?,
            &format!("reservable_memory_ids[{key:?}]"),
        )?;
        let kind = memory_type_value(&key)?;
        if result.insert(kind, id).is_some() {
            return Err(PyValueError::new_err(format!(
                "duplicate reservable memory type {key:?}"
            )));
        }
    }
    Ok(result)
}

fn stable_id(namespace: &[u8], id: u64) -> Uuid {
    let mut bytes = [0_u8; 16];
    let namespace_len = namespace.len().min(8);
    bytes[..namespace_len].copy_from_slice(&namespace[..namespace_len]);
    bytes[8..].copy_from_slice(&id.to_be_bytes());
    Uuid::from_bytes(bytes)
}

fn memory_type_value(value: &str) -> PyResult<u32> {
    match value {
        "device" => Ok(MEMORY_DEVICE),
        "host" => Ok(MEMORY_HOST),
        "pinned_host" => Ok(MEMORY_PINNED_HOST),
        "disk" => Ok(MEMORY_DISK),
        _ => Err(PyValueError::new_err(format!(
            "invalid reservable memory type {value:?}"
        ))),
    }
}

fn memory_type_name(value: u32) -> Result<String, String> {
    match value {
        MEMORY_DEVICE => Ok("device".to_owned()),
        MEMORY_HOST => Ok("host".to_owned()),
        MEMORY_PINNED_HOST => Ok("pinned_host".to_owned()),
        MEMORY_DISK => Ok("disk".to_owned()),
        _ => Err(format!("invalid memory type {value}")),
    }
}

fn memory_type_description(value: u32) -> String {
    memory_type_name(value).unwrap_or_else(|_| format!("memory type {value}"))
}

fn policy_name(value: u32) -> Result<&'static str, String> {
    match value {
        0 => Ok("immediate"),
        1 => Ok("overbook"),
        2 => Ok("spill"),
        3 => Ok("wait"),
        4 => Ok("wait_or_overbook"),
        5 => Ok("wait_or_fail"),
        _ => Err(format!("invalid reservation policy {value}")),
    }
}

fn resize_reason(value: u32) -> Result<&'static str, String> {
    match value {
        0 => Ok("consumed"),
        1 => Ok("split_off"),
        _ => Err(format!("invalid reservation resize reason {value}")),
    }
}

fn release_reason(value: u32) -> Result<&'static str, String> {
    match value {
        0 => Ok("cleared"),
        1 => Ok("consumed"),
        2 => Ok("split"),
        _ => Err(format!("invalid reservation release reason {value}")),
    }
}

#[pymodule]
fn cudf_polars_quent(module: &Bound<'_, PyModule>) -> PyResult<()> {
    generated_python::register(module)?;
    module.add_class::<ReservationEventIngestor>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ingestor() -> ReservationEventIngestor {
        let context = Context::<CudfPolars>::try_new(Noop).expect("no-op context");
        ReservationEventIngestor {
            worker: context
                .observer::<Worker>()
                .handle_with_id(Uuid::from_u128(1))
                .as_entity_ref(),
            memory_ids: HashMap::from([(MEMORY_DEVICE, Uuid::from_u128(2))]),
            request_observer: context.observer(),
            reservation_observer: context.observer(),
            actor_observer: context.observer(),
            memory_observer: context.observer(),
            lifecycle: Lifecycle::default(),
            diagnostics: Vec::new(),
            closed: false,
        }
    }

    fn event(sequence: u64, data: ReservationEventData) -> ReservationEvent {
        ReservationEvent {
            sequence,
            timestamp_ns: 1_234_567_890 + sequence,
            data,
        }
    }

    fn submitted(id: u64) -> ReservationEventData {
        ReservationEventData::RequestSubmitted {
            id,
            size: 64,
            memory_types: vec![MEMORY_DEVICE],
            policy: 3,
            delta: None,
            attributes: Vec::new(),
        }
    }

    #[test]
    fn requires_strict_sequence_continuity_across_batches() {
        let mut ingestor = ingestor();
        ingestor.ingest_owned(vec![event(0, submitted(1))]).unwrap();
        for invalid in [0, 2] {
            let error = ingestor
                .ingest_owned(vec![event(
                    invalid,
                    ReservationEventData::RequestWaiting { id: 1 },
                )])
                .unwrap_err();
            assert!(error.contains(&format!("expected 1, got {invalid}")));
            assert_eq!(ingestor.lifecycle.next_sequence, 1);
        }
    }

    #[test]
    fn supports_multibatch_request_and_all_terminal_outcomes() {
        for outcome in 1..=3 {
            let mut ingestor = ingestor();
            ingestor.ingest_owned(vec![event(0, submitted(1))]).unwrap();
            ingestor
                .ingest_owned(vec![
                    event(1, ReservationEventData::RequestWaiting { id: 1 }),
                    event(
                        2,
                        ReservationEventData::RequestResolved {
                            id: 1,
                            outcome,
                            reservation_id: 0,
                            memory_type: MEMORY_DEVICE,
                            size: 0,
                            overbooked_bytes: 0,
                            timed_out: outcome == 1,
                            error: "failure".to_owned(),
                        },
                    ),
                ])
                .unwrap();
            assert!(ingestor.lifecycle.requests.is_empty());
        }

        let mut ingestor = ingestor();
        ingestor
            .ingest_owned(vec![
                event(0, submitted(1)),
                event(
                    1,
                    ReservationEventData::ReservationCreated {
                        id: 10,
                        memory_type: MEMORY_DEVICE,
                        size: 64,
                        attributes: Vec::new(),
                        request_id: 1,
                        parent_id: 0,
                    },
                ),
                event(
                    2,
                    ReservationEventData::RequestResolved {
                        id: 1,
                        outcome: 0,
                        reservation_id: 10,
                        memory_type: MEMORY_DEVICE,
                        size: 64,
                        overbooked_bytes: 4,
                        timed_out: false,
                        error: String::new(),
                    },
                ),
                event(
                    3,
                    ReservationEventData::ReservationReleased { id: 10, reason: 0 },
                ),
            ])
            .unwrap();
        assert!(ingestor.lifecycle.requests.is_empty());
        assert!(ingestor.lifecycle.reservations.is_empty());
    }

    #[test]
    fn retains_released_reservations_for_split_references() {
        let mut ingestor = ingestor();
        ingestor
            .ingest_owned(vec![
                event(
                    0,
                    ReservationEventData::ReservationCreated {
                        id: 5,
                        memory_type: MEMORY_DEVICE,
                        size: 64,
                        attributes: Vec::new(),
                        request_id: 0,
                        parent_id: 0,
                    },
                ),
                event(
                    1,
                    ReservationEventData::ReservationReleased { id: 5, reason: 2 },
                ),
                event(
                    2,
                    ReservationEventData::ReservationCreated {
                        id: 6,
                        memory_type: MEMORY_DEVICE,
                        size: 32,
                        attributes: Vec::new(),
                        request_id: 0,
                        parent_id: 5,
                    },
                ),
            ])
            .unwrap();
        assert!(ingestor.lifecycle.known_reservations.contains(&5));
        assert!(ingestor.lifecycle.reservations.contains_key(&6));
        assert_eq!(ingestor.lifecycle.reservations[&6].next_fsm_sequence, 2);
    }

    #[test]
    fn maps_attributes_resizes_and_limit_changes() {
        let mut ingestor = ingestor();
        ingestor
            .ingest_owned(vec![
                event(
                    0,
                    ReservationEventData::RequestSubmitted {
                        id: 1,
                        size: 64,
                        memory_types: vec![MEMORY_DEVICE],
                        policy: 0,
                        delta: Some(-64),
                        attributes: vec![
                            OwnedAttribute {
                                key: "rapidsmpf.site".to_owned(),
                                value: OwnedValue::String("test.site".to_owned()),
                            },
                            OwnedAttribute {
                                key: "string".to_owned(),
                                value: OwnedValue::String("value".to_owned()),
                            },
                            OwnedAttribute {
                                key: "integer".to_owned(),
                                value: OwnedValue::I64(-7),
                            },
                            OwnedAttribute {
                                key: "boolean".to_owned(),
                                value: OwnedValue::Bool(true),
                            },
                            OwnedAttribute {
                                key: "float".to_owned(),
                                value: OwnedValue::F64(1.5),
                            },
                        ],
                    },
                ),
                event(
                    1,
                    ReservationEventData::ReservationCreated {
                        id: 10,
                        memory_type: MEMORY_DEVICE,
                        size: 64,
                        attributes: Vec::new(),
                        request_id: 1,
                        parent_id: 0,
                    },
                ),
                event(
                    2,
                    ReservationEventData::ReservationResized {
                        id: 10,
                        size: 32,
                        reason: 0,
                    },
                ),
                event(
                    3,
                    ReservationEventData::MemoryLimitChanged {
                        memory_type: MEMORY_DEVICE,
                        configured_limit: -1,
                    },
                ),
            ])
            .unwrap();
        assert_eq!(ingestor.lifecycle.reservations[&10].next_fsm_sequence, 3);
        assert!(ingestor.diagnostics.is_empty());
    }

    #[test]
    fn raw_event_preserves_explicit_timestamp() {
        let event = raw_event(Uuid::from_u128(1), 987_654_321, ());
        assert_eq!(event.timestamp, 987_654_321);
    }

    #[test]
    fn malformed_batch_is_atomic() {
        let mut ingestor = ingestor();
        let error = ingestor
            .ingest_owned(vec![
                event(0, submitted(1)),
                event(1, ReservationEventData::RequestWaiting { id: 99 }),
            ])
            .unwrap_err();
        assert!(error.contains("unknown request 99"));
        assert_eq!(ingestor.lifecycle.next_sequence, 0);
        assert!(ingestor.lifecycle.requests.is_empty());
    }

    #[test]
    fn incomplete_close_is_rejected() {
        Python::initialize();
        let mut ingestor = ingestor();
        ingestor.ingest_owned(vec![event(0, submitted(1))]).unwrap();
        let error = ingestor.close().unwrap_err();
        assert!(error.to_string().contains("1 live reservation request"));
        assert!(!ingestor.closed);
    }
}
