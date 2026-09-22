// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use quent_collector::server::CollectorService;
use quent_collector_proto::collector_server::CollectorServer;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

include!(concat!(env!("OUT_DIR"), "/cudfpolars.rs"));

mod generated_python {
    include!(concat!(env!("OUT_DIR"), "/pyo3_bridge.rs"));

    pub fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        __quent_pyo3_bridge::_quent_generated(module)
    }
}

struct SharedCollectorSink(Arc<crate::Context<crate::CudfPolars>>);

impl quent_collector::CollectorSink for SharedCollectorSink {
    fn ingest(&self, entity: &str, event: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        quent_collector::CollectorSink::ingest(self.0.as_ref(), entity, event)
    }
}

/*
TODO: delete this Collector once the bindings generator provides an API.

https://github.com/rapidsai/quent/issues/286#issuecomment-5778598679 has more.

For now, we can either hack this in, or limit ourselves to the local filesystem exporter
plus some manual merging on the client / rank 0.
*/

#[pyclass(name = "Collector")]
struct PyCollector {
    address: String,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<Result<(), String>>>,
}

impl PyCollector {
    fn stop(&mut self) -> Result<(), String> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| "collector thread panicked".to_owned())??;
        }
        Ok(())
    }
}

impl Drop for PyCollector {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[pymethods]
impl PyCollector {
    #[new]
    #[pyo3(signature = (output_root, advertise_host))]
    fn new(output_root: PathBuf, advertise_host: String) -> PyResult<Self> {
        std::fs::create_dir_all(&output_root)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let listener = TcpListener::bind(("0.0.0.0", 0))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let port = listener
            .local_addr()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .port();
        listener
            .set_nonblocking(true)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let address = format!("http://{advertise_host}:{port}");
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|error| error.to_string())?;
            let contexts = Arc::new(Mutex::new(Vec::new()));
            let server_contexts = Arc::clone(&contexts);
            let result = runtime.block_on(async move {
                let listener =
                    tokio::net::TcpListener::from_std(listener).map_err(|e| e.to_string())?;
                let incoming = TcpListenerStream::new(listener);
                let service = CollectorService::new(move |context_id| {
                    let options = quent_io::ExporterOptions::FileSystem(
                        quent_io::FileSystemExporterOptions::new(
                            quent_io::FileSystemFormat::Ndjson,
                            output_root.clone(),
                        ),
                    );
                    let context = Arc::new(
                        crate::Context::<crate::CudfPolars>::try_with_id(context_id, options)
                            .map_err(|error| error.to_string())?,
                    );
                    server_contexts.lock().unwrap().push(Arc::clone(&context));
                    Ok(SharedCollectorSink(context))
                });
                Server::builder()
                    .add_service(CollectorServer::new(service))
                    .serve_with_incoming_shutdown(incoming, async {
                        let _ = shutdown_rx.await;
                    })
                    .await
                    .map_err(|error| error.to_string())
            });
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                let all_released = contexts
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|context| Arc::strong_count(context) == 1);
                if all_released {
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    return Err("collector contexts did not close within 10 seconds".to_owned());
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            contexts.lock().unwrap().clear();
            drop(runtime);
            result
        });
        Ok(Self {
            address,
            shutdown: Some(shutdown_tx),
            thread: Some(thread),
        })
    }

    #[getter]
    fn address(&self) -> &str {
        &self.address
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.stop()).map_err(PyRuntimeError::new_err)
    }

    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.close(py)
    }
}

#[pymodule(name = "_quent")]
fn _quent(module: &Bound<'_, PyModule>) -> PyResult<()> {
    generated_python::register(module)?;
    module.add_class::<PyCollector>()?;
    Ok(())
}
