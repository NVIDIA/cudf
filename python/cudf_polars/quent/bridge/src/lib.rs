// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyCapsule, PyCapsuleMethods, PyMapping, PyMappingMethods, PyTuple};

include!(concat!(env!("OUT_DIR"), "/cudfpolars.rs"));

mod generated_python {
    include!(concat!(env!("OUT_DIR"), "/pyo3_bridge.rs"));

    pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        __quent_pyo3_bridge::_quent_generated(module)
    }

    pub use __quent_pyo3_bridge::reservation_observers;
}

const ABI_VERSION: u32 = 1;
const CAPSULE_NAME: &CStr = c"rapidsmpf.reservation_recorder";

const MEMORY_DEVICE: u32 = 0;
const MEMORY_HOST: u32 = 1;
const MEMORY_PINNED_HOST: u32 = 2;
const MEMORY_DISK: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
struct StringView {
    data: *const u8,
    size: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
union AttributeValue {
    string_value: StringView,
    i64_value: i64,
    bool_value: bool,
    f64_value: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReservationAttribute {
    key: StringView,
    type_: u32,
    value: AttributeValue,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttributesView {
    data: *const ReservationAttribute,
    size: usize,
}

#[repr(C)]
struct ReservationRecorder {
    abi_version: u32,
    struct_size: usize,
    user: *mut c_void,
    retain_user: Option<unsafe extern "C" fn(*mut c_void)>,
    release_user: Option<unsafe extern "C" fn(*mut c_void)>,
    request_submitted: Option<
        unsafe extern "C" fn(
            *mut c_void,
            u64,
            u64,
            usize,
            *const u32,
            usize,
            u32,
            bool,
            i64,
            AttributesView,
        ),
    >,
    request_waiting: Option<unsafe extern "C" fn(*mut c_void, u64, u64)>,
    request_resolved: Option<
        unsafe extern "C" fn(*mut c_void, u64, u64, u32, u64, u32, usize, usize, bool, StringView),
    >,
    reservation_created:
        Option<unsafe extern "C" fn(*mut c_void, u64, u64, u32, usize, AttributesView, u64, u64)>,
    reservation_resized: Option<unsafe extern "C" fn(*mut c_void, u64, u64, usize, u32)>,
    reservation_released: Option<unsafe extern "C" fn(*mut c_void, u64, u64, u32)>,
    memory_limit_changed: Option<unsafe extern "C" fn(*mut c_void, u64, u32, i64)>,
}

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

#[derive(Debug)]
enum PendingEvent {
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
    Invalid(String),
}

struct Core {
    next_sequence: u64,
    pending: BTreeMap<u64, PendingEvent>,
    requests: HashMap<u64, DynamicFsmHandle<MemoryReservationRequest>>,
    reservations: HashMap<u64, DynamicFsmHandle<MemoryReservation>>,
    reservation_types: HashMap<u64, u32>,
    reservation_refs: HashMap<u64, EntityRef<MemoryReservation>>,
    first_error: Option<String>,
    diagnostics: Vec<String>,
}

struct RecorderState {
    worker: EntityRef<Worker>,
    memories: HashMap<u32, Handle<ReservableMemory>>,
    request_observer: Observer<MemoryReservationRequest>,
    reservation_observer: Observer<MemoryReservation>,
    actor_observer: Observer<Actor>,
    core: Mutex<Core>,
}

impl RecorderState {
    fn new(
        request_observer: Observer<MemoryReservationRequest>,
        reservation_observer: Observer<MemoryReservation>,
        worker_observer: Observer<Worker>,
        actor_observer: Observer<Actor>,
        memory_observer: Observer<ReservableMemory>,
        worker_id: Uuid,
        memory_ids: HashMap<u32, Uuid>,
    ) -> Self {
        Self {
            worker: worker_observer.handle_with_id(worker_id).as_entity_ref(),
            memories: memory_ids
                .into_iter()
                .map(|(kind, id)| (kind, memory_observer.handle_with_id(id)))
                .collect(),
            request_observer,
            reservation_observer,
            actor_observer,
            core: Mutex::new(Core {
                next_sequence: 0,
                pending: BTreeMap::new(),
                requests: HashMap::new(),
                reservations: HashMap::new(),
                reservation_types: HashMap::new(),
                reservation_refs: HashMap::new(),
                first_error: None,
                diagnostics: Vec::new(),
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Core> {
        self.core
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn enqueue(&self, sequence: u64, event: PendingEvent) {
        let mut core = self.lock();
        if sequence < core.next_sequence {
            let next_sequence = core.next_sequence;
            core.latch(format!(
                "duplicate or stale recorder sequence {sequence}; next is {}",
                next_sequence
            ));
            return;
        }
        if core.pending.insert(sequence, event).is_some() {
            core.latch(format!("duplicate recorder sequence {sequence}"));
            return;
        }
        loop {
            let next_sequence = core.next_sequence;
            let Some(event) = core.pending.remove(&next_sequence) else {
                break;
            };
            let current = core.next_sequence;
            core.next_sequence += 1;
            if core.first_error.is_none()
                && let Err(error) = self.apply(&mut core, current, event)
            {
                core.latch(error);
            }
        }
    }

    fn apply(&self, core: &mut Core, sequence: u64, event: PendingEvent) -> Result<(), String> {
        match event {
            PendingEvent::RequestSubmitted {
                id,
                size,
                memory_types,
                policy,
                delta,
                attributes,
            } => {
                if core.requests.contains_key(&id) {
                    return Err(format!("duplicate request id {id}"));
                }
                let translated = self.translate_attributes(core, attributes);
                let memory_types = memory_types
                    .into_iter()
                    .map(memory_type_name)
                    .collect::<Result<Vec<_>, _>>()?;
                let mut handle = self
                    .request_observer
                    .handle_with_id(stable_id(b"request", id))
                    .into_dynamic();
                handle
                    .submitted(
                        self.worker.clone(),
                        translated.actor,
                        translated.site,
                        size,
                        memory_types,
                        policy_name(policy)?.to_owned(),
                        delta,
                        translated.dynamic,
                        sequence,
                    )
                    .map_err(|error| error.to_string())?;
                core.requests.insert(id, handle);
            }
            PendingEvent::RequestWaiting { id } => {
                core.requests
                    .get_mut(&id)
                    .ok_or_else(|| format!("waiting for unknown request {id}"))?
                    .waiting(sequence)
                    .map_err(|error| error.to_string())?;
            }
            PendingEvent::RequestResolved {
                id,
                outcome,
                reservation_id,
                memory_type,
                size,
                overbooked_bytes,
                timed_out,
                error,
            } => {
                let mut handle = core
                    .requests
                    .remove(&id)
                    .ok_or_else(|| format!("resolution for unknown request {id}"))?;
                match outcome {
                    0 => {
                        let reservation =
                            core.reservation_refs.get(&reservation_id).ok_or_else(|| {
                                format!("request {id} granted unknown reservation {reservation_id}")
                            })?;
                        handle
                            .granted(
                                reservation.clone(),
                                memory_type_name(memory_type)?,
                                size,
                                overbooked_bytes,
                                timed_out,
                                sequence,
                            )
                            .map_err(|error| error.to_string())?;
                    }
                    1 => handle
                        .empty(timed_out, sequence)
                        .map_err(|error| error.to_string())?,
                    2 => handle
                        .failed(error, sequence)
                        .map_err(|error| error.to_string())?,
                    3 => handle
                        .cancelled(sequence)
                        .map_err(|error| error.to_string())?,
                    value => return Err(format!("invalid request outcome {value}")),
                }
            }
            PendingEvent::ReservationCreated {
                id,
                memory_type,
                size,
                attributes,
                request_id,
                parent_id,
            } => {
                if size == 0 {
                    return Err(format!("reservation {id} was created with zero bytes"));
                }
                if core.reservation_refs.contains_key(&id) {
                    return Err(format!("duplicate reservation id {id}"));
                }
                let translated = self.translate_attributes(core, attributes);
                let request = (request_id != 0)
                    .then(|| {
                        core.requests
                            .get(&request_id)
                            .map(DynamicFsmHandle::as_entity_ref)
                            .ok_or_else(|| {
                                format!("reservation {id} references unknown request {request_id}")
                            })
                    })
                    .transpose()?;
                let parent = (parent_id != 0)
                    .then(|| {
                        core.reservation_refs
                            .get(&parent_id)
                            .cloned()
                            .ok_or_else(|| {
                                format!("reservation {id} references unknown parent {parent_id}")
                            })
                    })
                    .transpose()?;
                let memory = self.memory_usage(memory_type, size)?;
                let mut handle = self
                    .reservation_observer
                    .handle_with_id(stable_id(b"reservation", id))
                    .into_dynamic();
                handle
                    .created(
                        self.worker.clone(),
                        translated.actor,
                        translated.site,
                        request,
                        parent,
                        translated.dynamic,
                        sequence,
                    )
                    .map_err(|error| error.to_string())?;
                handle
                    .held(
                        memory,
                        if parent_id == 0 {
                            "granted"
                        } else {
                            "split_child"
                        }
                        .to_owned(),
                        sequence,
                    )
                    .map_err(|error| error.to_string())?;
                core.reservation_refs.insert(id, handle.as_entity_ref());
                core.reservation_types.insert(id, memory_type);
                core.reservations.insert(id, handle);
            }
            PendingEvent::ReservationResized { id, size, reason } => {
                if size == 0 {
                    return Err(format!("reservation {id} resized to zero bytes"));
                }
                let memory_type = self.reservation_memory_type(core, id)?;
                let memory = self.memory_usage(memory_type, size)?;
                core.reservations
                    .get_mut(&id)
                    .ok_or_else(|| format!("resize for unknown reservation {id}"))?
                    .held(memory, resize_reason(reason)?.to_owned(), sequence)
                    .map_err(|error| error.to_string())?;
            }
            PendingEvent::ReservationReleased { id, reason } => {
                let mut handle = core
                    .reservations
                    .remove(&id)
                    .ok_or_else(|| format!("release for unknown reservation {id}"))?;
                core.reservation_types.remove(&id);
                handle
                    .released(release_reason(reason)?.to_owned(), sequence)
                    .map_err(|error| error.to_string())?;
            }
            PendingEvent::MemoryLimitChanged {
                memory_type,
                configured_limit,
            } => {
                let handle = self.memories.get(&memory_type).ok_or_else(|| {
                    format!(
                        "memory limit changed for undeclared {}",
                        memory_type_name(memory_type)
                            .unwrap_or_else(|_| { format!("memory type {memory_type}") })
                    )
                })?;
                handle
                    .resized(
                        ReservableMemoryBounds {
                            bytes: configured_limit.max(0) as u64,
                        },
                        configured_limit,
                        sequence,
                    )
                    .map_err(|error| error.to_string())?;
            }
            PendingEvent::Invalid(error) => return Err(error),
        }
        Ok(())
    }

    fn translate_attributes(
        &self,
        core: &mut Core,
        attributes: Vec<OwnedAttribute>,
    ) -> TranslatedAttributes {
        let mut site = None;
        let mut actor = None;
        let mut dynamic = DynamicAttributes::new();
        for attribute in attributes {
            match attribute.key.as_str() {
                "rapidsmpf.site" => {
                    if let OwnedValue::String(value) = attribute.value {
                        site = Some(value);
                    } else {
                        core.diagnostic("rapidsmpf.site is not a string".to_owned());
                    }
                }
                "quent.actor_id" => match attribute.value {
                    OwnedValue::String(value) => match Uuid::parse_str(&value) {
                        Ok(id) => {
                            actor = Some(self.actor_observer.handle_with_id(id).as_entity_ref())
                        }
                        Err(_) => {
                            core.diagnostic(format!("invalid actor UUID: {value}"));
                            dynamic.add("invalid_actor_id", value);
                        }
                    },
                    value => {
                        core.diagnostic("quent.actor_id is not a string".to_owned());
                        add_dynamic(&mut dynamic, "invalid_actor_id".to_owned(), value);
                    }
                },
                _ => add_dynamic(&mut dynamic, attribute.key, attribute.value),
            }
        }
        TranslatedAttributes {
            site: site.unwrap_or_else(|| "unknown".to_owned()),
            actor,
            dynamic,
        }
    }

    fn memory_usage(
        &self,
        memory_type: u32,
        bytes: u64,
    ) -> Result<EntityRef<ReservableMemory, ReservableMemoryUsage>, String> {
        self.memories
            .get(&memory_type)
            .ok_or_else(|| {
                format!(
                    "reservation uses undeclared {}",
                    memory_type_name(memory_type)
                        .unwrap_or_else(|_| format!("memory type {memory_type}"))
                )
            })
            .map(|memory| memory.as_entity_ref_with(ReservableMemoryUsage { bytes }))
    }

    fn reservation_memory_type(&self, core: &Core, id: u64) -> Result<u32, String> {
        core.reservation_types
            .get(&id)
            .copied()
            .ok_or_else(|| format!("resize for unknown reservation {id}"))
    }
}

struct TranslatedAttributes {
    site: String,
    actor: Option<EntityRef<Actor>>,
    dynamic: DynamicAttributes,
}

impl Core {
    fn latch(&mut self, error: String) {
        self.diagnostics.push(error.clone());
        self.first_error.get_or_insert(error);
    }

    fn diagnostic(&mut self, diagnostic: String) {
        self.diagnostics.push(diagnostic);
    }
}

fn add_dynamic(attributes: &mut DynamicAttributes, key: String, value: OwnedValue) {
    match value {
        OwnedValue::String(value) => attributes.add(key, value),
        OwnedValue::I64(value) => attributes.add(key, value),
        OwnedValue::Bool(value) => attributes.add(key, value),
        OwnedValue::F64(value) => attributes.add(key, value),
    }
}

fn stable_id(namespace: &[u8], id: u64) -> Uuid {
    let mut bytes = [0_u8; 16];
    let namespace_len = namespace.len().min(8);
    bytes[..namespace_len].copy_from_slice(&namespace[..namespace_len]);
    bytes[8..].copy_from_slice(&id.to_be_bytes());
    Uuid::from_bytes(bytes)
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

unsafe fn copy_string(view: StringView, field: &str) -> Result<String, String> {
    if view.size == 0 {
        return Ok(String::new());
    }
    if view.data.is_null() {
        return Err(format!("{field} has a null pointer with non-zero length"));
    }
    // SAFETY: rapidsmpf guarantees callback views are valid for the callback duration.
    let bytes = unsafe { std::slice::from_raw_parts(view.data, view.size) };
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| format!("{field} is not UTF-8: {error}"))
}

unsafe fn copy_attributes(view: AttributesView) -> Result<Vec<OwnedAttribute>, String> {
    if view.size == 0 {
        return Ok(Vec::new());
    }
    if view.data.is_null() {
        return Err("attributes have a null pointer with non-zero length".to_owned());
    }
    // SAFETY: rapidsmpf guarantees callback views are valid for the callback duration.
    let attributes = unsafe { std::slice::from_raw_parts(view.data, view.size) };
    attributes
        .iter()
        .map(|attribute| {
            // SAFETY: each key and the selected union member are valid for this callback.
            let key = unsafe { copy_string(attribute.key, "attribute key") }?;
            let value = match attribute.type_ {
                0 => {
                    // SAFETY: the ABI tag selects string_value.
                    OwnedValue::String(unsafe {
                        copy_string(attribute.value.string_value, "string attribute")?
                    })
                }
                // SAFETY: each ABI tag selects the corresponding union member.
                1 => OwnedValue::I64(unsafe { attribute.value.i64_value }),
                2 => OwnedValue::Bool(unsafe { attribute.value.bool_value }),
                3 => OwnedValue::F64(unsafe { attribute.value.f64_value }),
                value => return Err(format!("invalid attribute type {value} for {key}")),
            };
            Ok(OwnedAttribute { key, value })
        })
        .collect()
}

unsafe fn copy_memory_types(data: *const u32, size: usize) -> Result<Vec<u32>, String> {
    if size == 0 {
        return Ok(Vec::new());
    }
    if data.is_null() {
        return Err("memory types have a null pointer with non-zero length".to_owned());
    }
    // SAFETY: rapidsmpf guarantees callback arrays are valid for the callback duration.
    Ok(unsafe { std::slice::from_raw_parts(data, size) }.to_vec())
}

unsafe fn state_from_user<'a>(user: *mut c_void) -> Option<&'a RecorderState> {
    // SAFETY: rapidsmpf retains user for every installed registration and in-flight callback.
    unsafe { (user as *const RecorderState).as_ref() }
}

fn callback(
    user: *mut c_void,
    sequence: u64,
    build: impl FnOnce() -> Result<PendingEvent, String>,
) {
    // SAFETY: callback callers pass the recorder's retained Arc pointer.
    let Some(state) = (unsafe { state_from_user(user) }) else {
        return;
    };
    let event = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(build)) {
        Ok(Ok(event)) => event,
        Ok(Err(error)) => PendingEvent::Invalid(error),
        Err(_) => PendingEvent::Invalid("panic caught in reservation recorder callback".to_owned()),
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.enqueue(sequence, event);
    }));
    if result.is_err() {
        let mut core = state.lock();
        core.latch("panic caught while applying reservation recorder event".to_owned());
    }
}

unsafe extern "C" fn retain_user(user: *mut c_void) {
    if !user.is_null() {
        // SAFETY: user was created with Arc::into_raw and remains a valid Arc allocation.
        unsafe { Arc::increment_strong_count(user as *const RecorderState) };
    }
}

unsafe extern "C" fn release_user(user: *mut c_void) {
    if !user.is_null() {
        // SAFETY: balances one Arc strong count owned by rapidsmpf.
        unsafe { Arc::decrement_strong_count(user as *const RecorderState) };
    }
}

unsafe extern "C" fn request_submitted(
    user: *mut c_void,
    sequence: u64,
    id: u64,
    size: usize,
    memory_types: *const u32,
    memory_types_size: usize,
    policy: u32,
    has_delta: bool,
    delta: i64,
    attributes: AttributesView,
) {
    callback(user, sequence, || {
        Ok(PendingEvent::RequestSubmitted {
            id,
            size: u64::try_from(size).map_err(|_| "request size exceeds u64".to_owned())?,
            // SAFETY: copied before this callback returns.
            memory_types: unsafe { copy_memory_types(memory_types, memory_types_size) }?,
            policy,
            delta: has_delta.then_some(delta),
            // SAFETY: copied before this callback returns.
            attributes: unsafe { copy_attributes(attributes) }?,
        })
    });
}

unsafe extern "C" fn request_waiting(user: *mut c_void, sequence: u64, id: u64) {
    callback(user, sequence, || Ok(PendingEvent::RequestWaiting { id }));
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn request_resolved(
    user: *mut c_void,
    sequence: u64,
    id: u64,
    outcome: u32,
    reservation_id: u64,
    memory_type: u32,
    size: usize,
    overbooked_bytes: usize,
    timed_out: bool,
    error: StringView,
) {
    callback(user, sequence, || {
        Ok(PendingEvent::RequestResolved {
            id,
            outcome,
            reservation_id,
            memory_type,
            size: u64::try_from(size).map_err(|_| "granted size exceeds u64".to_owned())?,
            overbooked_bytes: u64::try_from(overbooked_bytes)
                .map_err(|_| "overbooked bytes exceed u64".to_owned())?,
            timed_out,
            // SAFETY: copied before this callback returns.
            error: unsafe { copy_string(error, "request error") }?,
        })
    });
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn reservation_created(
    user: *mut c_void,
    sequence: u64,
    id: u64,
    memory_type: u32,
    size: usize,
    attributes: AttributesView,
    request_id: u64,
    parent_id: u64,
) {
    callback(user, sequence, || {
        Ok(PendingEvent::ReservationCreated {
            id,
            memory_type,
            size: u64::try_from(size).map_err(|_| "reservation size exceeds u64".to_owned())?,
            // SAFETY: copied before this callback returns.
            attributes: unsafe { copy_attributes(attributes) }?,
            request_id,
            parent_id,
        })
    });
}

unsafe extern "C" fn reservation_resized(
    user: *mut c_void,
    sequence: u64,
    id: u64,
    size: usize,
    reason: u32,
) {
    callback(user, sequence, || {
        Ok(PendingEvent::ReservationResized {
            id,
            size: u64::try_from(size).map_err(|_| "reservation size exceeds u64".to_owned())?,
            reason,
        })
    });
}

unsafe extern "C" fn reservation_released(user: *mut c_void, sequence: u64, id: u64, reason: u32) {
    callback(user, sequence, || {
        Ok(PendingEvent::ReservationReleased { id, reason })
    });
}

unsafe extern "C" fn memory_limit_changed(
    user: *mut c_void,
    sequence: u64,
    memory_type: u32,
    configured_limit: i64,
) {
    callback(user, sequence, || {
        Ok(PendingEvent::MemoryLimitChanged {
            memory_type,
            configured_limit,
        })
    });
}

#[repr(C)]
struct CapsuleRegistration {
    recorder: ReservationRecorder,
    state: *const RecorderState,
}

impl Drop for CapsuleRegistration {
    fn drop(&mut self) {
        // SAFETY: this is the original strong count transferred with Arc::into_raw.
        unsafe { drop(Arc::from_raw(self.state)) };
    }
}

unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    // SAFETY: CPython calls this with the capsule created below.
    let pointer = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, CAPSULE_NAME.as_ptr()) };
    if !pointer.is_null() {
        // SAFETY: the pointer was allocated with Box::into_raw exactly once.
        unsafe { drop(Box::from_raw(pointer as *mut CapsuleRegistration)) };
    }
}

fn parse_uuid(value: &Bound<'_, PyAny>, field: &str) -> PyResult<Uuid> {
    let text = value.str()?.to_str()?.to_owned();
    Uuid::parse_str(&text).map_err(|error| PyValueError::new_err(format!("{field}: {error}")))
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

fn registration<'a>(capsule: &'a Bound<'a, PyCapsule>) -> PyResult<&'a CapsuleRegistration> {
    let pointer = capsule.pointer_checked(Some(CAPSULE_NAME))?;
    // SAFETY: the capsule owns a live CapsuleRegistration until its destructor runs.
    Ok(unsafe { pointer.cast::<CapsuleRegistration>().as_ref() })
}

#[pyfunction]
fn reservation_recorder_capsule<'py>(
    py: Python<'py>,
    context: &Bound<'py, PyAny>,
    worker_id: &Bound<'py, PyAny>,
    reservable_memory_ids: &Bound<'py, PyMapping>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let (request_observer, reservation_observer, worker_observer, actor_observer, memory_observer) =
        generated_python::reservation_observers(context)?;
    let state = Arc::new(RecorderState::new(
        request_observer,
        reservation_observer,
        worker_observer,
        actor_observer,
        memory_observer,
        parse_uuid(worker_id, "worker_id")?,
        parse_memory_ids(reservable_memory_ids)?,
    ));
    let state = Arc::into_raw(state);
    let registration = Box::new(CapsuleRegistration {
        recorder: ReservationRecorder {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<ReservationRecorder>(),
            user: state.cast_mut().cast(),
            retain_user: Some(retain_user),
            release_user: Some(release_user),
            request_submitted: Some(request_submitted),
            request_waiting: Some(request_waiting),
            request_resolved: Some(request_resolved),
            reservation_created: Some(reservation_created),
            reservation_resized: Some(reservation_resized),
            reservation_released: Some(reservation_released),
            memory_limit_changed: Some(memory_limit_changed),
        },
        state,
    });
    let pointer = NonNull::new(Box::into_raw(registration).cast::<c_void>())
        .expect("Box::into_raw never returns null");
    // SAFETY: the boxed registration remains alive until capsule_destructor.
    unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            pointer,
            CAPSULE_NAME,
            Some(capsule_destructor),
        )
    }
}

#[pyfunction]
fn reservation_recorder_diagnostics(capsule: &Bound<'_, PyCapsule>) -> PyResult<Vec<String>> {
    let registration = registration(capsule)?;
    // SAFETY: the capsule retains the original Arc strong count.
    let state = unsafe { &*registration.state };
    Ok(state.lock().diagnostics.clone())
}

#[pyfunction]
fn close_reservation_recorder(capsule: &Bound<'_, PyCapsule>) -> PyResult<()> {
    let registration = registration(capsule)?;
    // SAFETY: the capsule retains the original Arc strong count.
    let state = unsafe { &*registration.state };
    let core = state.lock();
    let mut errors = core.diagnostics.clone();
    if !core.requests.is_empty() {
        errors.push(format!(
            "{} live reservation request(s)",
            core.requests.len()
        ));
    }
    if !core.reservations.is_empty() {
        errors.push(format!("{} live reservation(s)", core.reservations.len()));
    }
    if !core.pending.is_empty() {
        errors.push(format!(
            "{} pending event(s), next sequence {}",
            core.pending.len(),
            core.next_sequence
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(errors.join("; ")))
    }
}

#[pymodule]
fn cudf_polars_quent(module: &Bound<'_, PyModule>) -> PyResult<()> {
    generated_python::register(module)?;
    module.add_function(wrap_pyfunction!(reservation_recorder_capsule, module)?)?;
    module.add_function(wrap_pyfunction!(reservation_recorder_diagnostics, module)?)?;
    module.add_function(wrap_pyfunction!(close_reservation_recorder, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> RecorderState {
        let context = Context::<CudfPolars>::try_new(Noop).expect("no-op context");
        RecorderState::new(
            context.observer(),
            context.observer(),
            context.observer(),
            context.observer(),
            context.observer(),
            Uuid::from_u128(1),
            HashMap::from([(MEMORY_DEVICE, Uuid::from_u128(2))]),
        )
    }

    fn submitted(id: u64, attributes: Vec<OwnedAttribute>) -> PendingEvent {
        PendingEvent::RequestSubmitted {
            id,
            size: 64,
            memory_types: vec![MEMORY_DEVICE],
            policy: 3,
            delta: None,
            attributes,
        }
    }

    #[test]
    fn reorders_callbacks_before_applying_transitions() {
        let state = state();
        state.enqueue(1, PendingEvent::RequestWaiting { id: 7 });
        assert_eq!(state.lock().next_sequence, 0);

        state.enqueue(0, submitted(7, Vec::new()));
        let core = state.lock();
        assert_eq!(core.next_sequence, 2);
        assert_eq!(
            core.requests.get(&7).expect("request").state(),
            MemoryReservationRequestDynamicState::Waiting
        );
        assert!(core.first_error.is_none());
    }

    #[test]
    fn first_error_is_latched_while_sequence_keeps_advancing() {
        let state = state();
        state.enqueue(0, PendingEvent::Invalid("first".to_owned()));
        state.enqueue(2, PendingEvent::RequestWaiting { id: 99 });
        state.enqueue(1, PendingEvent::Invalid("second".to_owned()));

        let core = state.lock();
        assert_eq!(core.next_sequence, 3);
        assert_eq!(core.first_error.as_deref(), Some("first"));
        assert!(core.pending.is_empty());
    }

    #[test]
    fn callback_panic_is_latched_without_unwinding() {
        let state = state();
        callback(
            (&raw const state).cast_mut().cast(),
            0,
            || -> Result<PendingEvent, String> { panic!("deliberate callback panic") },
        );

        let core = state.lock();
        assert_eq!(core.next_sequence, 1);
        assert_eq!(
            core.first_error.as_deref(),
            Some("panic caught in reservation recorder callback")
        );
    }

    #[test]
    fn negative_memory_limit_is_accepted_and_zero_clamped() {
        let state = state();
        state.enqueue(
            0,
            PendingEvent::MemoryLimitChanged {
                memory_type: MEMORY_DEVICE,
                configured_limit: -7,
            },
        );

        let core = state.lock();
        assert_eq!(core.next_sequence, 1);
        assert!(core.first_error.is_none());
    }

    #[test]
    fn malformed_actor_is_unattributed_and_diagnosed() {
        let state = state();
        state.enqueue(
            0,
            submitted(
                3,
                vec![
                    OwnedAttribute {
                        key: "rapidsmpf.site".to_owned(),
                        value: OwnedValue::String("shuffle.extract".to_owned()),
                    },
                    OwnedAttribute {
                        key: "quent.actor_id".to_owned(),
                        value: OwnedValue::String("not-a-uuid".to_owned()),
                    },
                ],
            ),
        );

        let core = state.lock();
        assert!(core.first_error.is_none());
        assert_eq!(core.diagnostics, ["invalid actor UUID: not-a-uuid"]);
        assert!(core.requests.contains_key(&3));
    }

    #[test]
    fn released_reservations_retain_identity_for_split_children() {
        let state = state();
        state.enqueue(
            0,
            PendingEvent::ReservationCreated {
                id: 5,
                memory_type: MEMORY_DEVICE,
                size: 64,
                attributes: Vec::new(),
                request_id: 0,
                parent_id: 0,
            },
        );
        state.enqueue(1, PendingEvent::ReservationReleased { id: 5, reason: 0 });
        state.enqueue(
            2,
            PendingEvent::ReservationCreated {
                id: 6,
                memory_type: MEMORY_DEVICE,
                size: 32,
                attributes: Vec::new(),
                request_id: 0,
                parent_id: 5,
            },
        );

        let core = state.lock();
        assert!(!core.reservations.contains_key(&5));
        assert!(!core.reservation_types.contains_key(&5));
        assert!(core.reservation_refs.contains_key(&5));
        assert!(core.reservations.contains_key(&6));
        assert!(core.first_error.is_none());
    }
}
