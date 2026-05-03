//! PyO3 export: `v_vae_core`.

use crate::frame_bus;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(frame_bus_init_pool, m)?)?;
    m.add_function(wrap_pyfunction!(frame_bus_push_frame, m)?)?;
    m.add_function(wrap_pyfunction!(frame_bus_schedule_save_dct_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(frame_bus_schedule_export, m)?)?;
    m.add_function(wrap_pyfunction!(frame_bus_pushed_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_pending_tasks, m)?)?;
    m.add_function(wrap_pyfunction!(await_pending_writes, m)?)?;
    Ok(())
}

fn str_to_py(s: String) -> PyErr {
    PyRuntimeError::new_err(s)
}

#[pyfunction]
#[pyo3(signature = (batch_size, height, width, channels))]
fn frame_bus_init_pool(
    batch_size: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> PyResult<()> {
    frame_bus::init_pool(batch_size, height, width, channels).map_err(str_to_py)
}

#[pyfunction]
#[pyo3(signature = (index, ptr, num_elements))]
fn frame_bus_push_frame(index: usize, ptr: usize, num_elements: usize) -> PyResult<()> {
    frame_bus::push_frame(index, ptr, num_elements).map_err(str_to_py)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (paths, fps, header_mode = 0u8, workflow_json = None, zstd_level = 3i32, require_full = true))]
fn frame_bus_schedule_save_dct_parallel(
    paths: Vec<String>,
    fps: f32,
    header_mode: u8,
    workflow_json: Option<String>,
    zstd_level: i32,
    require_full: bool,
) -> PyResult<()> {
    frame_bus::schedule_save_dct_parallel(
        paths,
        fps,
        header_mode,
        workflow_json,
        zstd_level,
        require_full,
    )
    .map_err(str_to_py)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    paths,
    format,
    package_as_zip = false,
    quality = 92u8,
    fps = 24.0,
    header_mode = 0u8,
    workflow_json = None,
    zstd_level = 3i32,
    require_full = true,
    zip_inner_prefix = ""
))]
fn frame_bus_schedule_export(
    paths: Vec<String>,
    format: String,
    package_as_zip: bool,
    quality: u8,
    fps: f32,
    header_mode: u8,
    workflow_json: Option<String>,
    zstd_level: i32,
    require_full: bool,
    zip_inner_prefix: &str,
) -> PyResult<()> {
    frame_bus::schedule_export(
        paths,
        format,
        package_as_zip,
        quality,
        fps,
        header_mode,
        workflow_json,
        zstd_level,
        require_full,
        zip_inner_prefix.to_string(),
    )
    .map_err(str_to_py)
}

#[pyfunction]
fn frame_bus_pushed_count() -> usize {
    frame_bus::pushed_frame_count()
}

#[pyfunction]
fn get_pending_tasks() -> usize {
    frame_bus::pending_async_count()
}

#[pyfunction]
fn await_pending_writes(py: Python<'_>) -> PyResult<()> {
    py.allow_threads(|| loop {
        if frame_bus::pending_async_count() == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(8));
    });
    Ok(())
}
