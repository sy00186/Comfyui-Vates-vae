//! **v_vae_core**：`FrameBus` 像素总线 + 异步导出（DCT / PNG / JPEG / WebP / ZIP）。

mod export_codec;
mod frame_bus;
mod python_binding;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModule;

#[cfg(feature = "python")]
#[pymodule]
fn v_vae_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_binding::register(m)
}
