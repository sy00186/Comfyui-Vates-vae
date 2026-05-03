//! Singleton FrameBus：预分配 NHWC f32、`push_frame` 原生指针拷贝、异步导出（DCT / PNG / JPEG / WebP / ZIP）。

use crate::export_codec::{self, ExportFmt};
use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use vates_core::Encoder;
use zip::write::{FileOptions, ZipWriter};
use zip::CompressionMethod;

static BUS: Lazy<Mutex<Option<FrameBusInner>>> = Lazy::new(|| Mutex::new(None));
static ASYNC_PENDING: AtomicUsize = AtomicUsize::new(0);
static PUSHED_FRAMES: AtomicUsize = AtomicUsize::new(0);

struct FrameBusInner {
    pixels: Vec<f32>,
    batch: usize,
    h: usize,
    w: usize,
    c: usize,
}

#[inline]
pub(crate) fn frame_stride(h: usize, w: usize, c: usize) -> Result<usize, String> {
    h.checked_mul(w)
        .and_then(|x| x.checked_mul(c))
        .ok_or_else(|| "H*W*C overflow".to_string())
}

#[inline]
fn total_len(batch: usize, stride: usize) -> Result<usize, String> {
    batch
        .checked_mul(stride)
        .ok_or_else(|| "batch*stride overflow".to_string())
}

pub(crate) fn init_pool(
    batch_size: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> Result<(), String> {
    if batch_size < 1 || height < 1 || width < 1 || channels < 1 {
        return Err("batch_size, height, width, channels must be positive".to_string());
    }
    let stride = frame_stride(height, width, channels)?;
    let cap = total_len(batch_size, stride)?;
    let mut g = BUS.lock().map_err(|_| "FrameBus mutex poisoned".to_string())?;

    let mut pix = Vec::new();
    pix.try_reserve_exact(cap)
        .map_err(|_| "try_reserve_exact failed".to_string())?;
    pix.resize(cap, 0.0_f32);

    PUSHED_FRAMES.store(0, Ordering::SeqCst);

    *g = Some(FrameBusInner {
        pixels: pix,
        batch: batch_size,
        h: height,
        w: width,
        c: channels,
    });

    Ok(())
}

pub(crate) fn push_frame(index: usize, ptr: usize, num_elements: usize) -> Result<(), String> {
    let mut guard = BUS.lock().map_err(|_| "FrameBus mutex poisoned".to_string())?;
    let Some(inner) = guard.as_mut() else {
        return Err("FrameBus not initialized".to_string());
    };
    if index >= inner.batch {
        return Err(format!("index {index} >= batch {}", inner.batch));
    }
    let stride = frame_stride(inner.h, inner.w, inner.c)?;
    if num_elements != stride {
        return Err(format!(
            "num_elements {num_elements} != stride {stride} (h,w,c)=({},{},{})",
            inner.h, inner.w, inner.c
        ));
    }
    if ptr == 0 {
        return Err("null ptr".to_string());
    }

    let off = index
        .checked_mul(stride)
        .ok_or_else(|| "index*stride overflow".to_string())?;

    let dst = unsafe { inner.pixels.as_mut_ptr().add(off) };
    unsafe {
        std::ptr::copy_nonoverlapping(ptr as *const f32, dst, stride);
    }

    PUSHED_FRAMES.fetch_add(1, Ordering::SeqCst);
    Ok(())
}

pub(crate) fn pushed_frame_count() -> usize {
    PUSHED_FRAMES.load(Ordering::SeqCst)
}

fn take_bus_inner(require_full: bool, paths_len: usize, package_zip: bool) -> Result<FrameBusInner, String> {
    let mut g = BUS.lock().map_err(|_| "FrameBus mutex poisoned".to_string())?;
    let Some(inner_ref) = g.as_ref() else {
        return Err("FrameBus empty or not initialized".to_string());
    };
    let pushed = PUSHED_FRAMES.load(Ordering::SeqCst);
    if require_full && pushed != inner_ref.batch {
        return Err(format!(
            "incomplete frames: pushed {pushed} / batch {}",
            inner_ref.batch
        ));
    }
    if package_zip {
        if paths_len != 1 {
            return Err(format!(
                "zip export expects paths.len()==1, got {paths_len}"
            ));
        }
    } else if paths_len != inner_ref.batch {
        return Err(format!(
            "paths {paths_len} != batch {}",
            inner_ref.batch
        ));
    }
    g.take().ok_or_else(|| "FrameBus take failed".to_string())
}

fn write_zip_file(
    zip_path: &str,
    stem: &str,
    fmt: ExportFmt,
    frames: Vec<Vec<u8>>,
) -> Result<(), String> {
    let file = File::create(zip_path).map_err(|e| e.to_string())?;
    let mut zip = ZipWriter::new(file);
    let opts = FileOptions::<()>::default().compression_method(CompressionMethod::Deflated);
    let ext = export_codec::ext_for(fmt);
    for (i, bytes) in frames.into_iter().enumerate() {
        let name = format!("{stem}_{i:04}.{ext}");
        zip.start_file(name, opts).map_err(|e| e.to_string())?;
        zip.write_all(&bytes).map_err(|e| e.to_string())?;
    }
    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

fn encode_all_frames_ordered_for_zip(
    pixels: &[f32],
    batch: usize,
    stride: usize,
    h: usize,
    w: usize,
    c: usize,
    fmt: ExportFmt,
    quality: u8,
    fps: f32,
    header_mode: u8,
    wf: Option<&str>,
    zstd_level: i32,
) -> Result<Vec<Vec<u8>>, String> {
    let parts: Vec<Result<(usize, Vec<u8>), String>> = (0..batch)
        .into_par_iter()
        .map(|frame_idx| {
            let begin = frame_idx * stride;
            let chunk = pixels
                .get(begin..begin + stride)
                .ok_or_else(|| format!("zip: frame slice oob {}", frame_idx))?;
            let bytes = export_codec::encode_frame_bytes(
                chunk,
                h,
                w,
                c,
                fmt,
                quality,
                fps,
                header_mode,
                wf,
                zstd_level,
            )?;
            Ok((frame_idx, bytes))
        })
        .collect();

    let mut tuples = Vec::with_capacity(batch);
    for r in parts {
        tuples.push(r?);
    }
    tuples.sort_by_key(|(i, _)| *i);
    Ok(tuples.into_iter().map(|(_, b)| b).collect())
}

pub(crate) fn schedule_export(
    paths: Vec<String>,
    format: String,
    package_as_zip: bool,
    quality: u8,
    fps: f32,
    header_mode: u8,
    workflow_json: Option<String>,
    zstd_level: i32,
    require_full: bool,
    zip_inner_prefix: String,
) -> Result<(), String> {
    let fmt = export_codec::parse_format(&format)?;
    let inner = take_bus_inner(require_full, paths.len(), package_as_zip)?;

    let FrameBusInner {
        pixels,
        batch,
        h,
        w,
        c,
    } = inner;

    let stride = frame_stride(h, w, c)?;
    if stride * batch != pixels.len() {
        return Err("pixel buffer length mismatch".to_string());
    }

    PUSHED_FRAMES.store(0, Ordering::SeqCst);
    ASYNC_PENDING.fetch_add(1, Ordering::SeqCst);

    let wf = workflow_json.clone();
    let stem = if zip_inner_prefix.trim().is_empty() {
        "frame".to_string()
    } else {
        zip_inner_prefix
    };

    std::thread::spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
            if package_as_zip {
                let zip_path = paths
                    .first()
                    .ok_or_else(|| "missing zip path".to_string())?
                    .clone();
                let frames = encode_all_frames_ordered_for_zip(
                    &pixels,
                    batch,
                    stride,
                    h,
                    w,
                    c,
                    fmt,
                    quality,
                    fps,
                    header_mode,
                    wf.as_deref(),
                    zstd_level,
                )?;
                write_zip_file(&zip_path, &stem, fmt, frames)?;
                return Ok(());
            }

            match fmt {
                ExportFmt::Dct => {
                    paths
                        .par_iter()
                        .enumerate()
                        .try_for_each(|(frame_idx, p)| -> Result<(), String> {
                            let begin = frame_idx * stride;
                            let chunk = pixels
                                .get(begin..begin + stride)
                                .ok_or_else(|| format!("dct oob {}", frame_idx))?;
                            Encoder::encode_batch_bhwc_file(
                                chunk,
                                1u32,
                                h as u32,
                                w as u32,
                                c as u32,
                                fps,
                                PathBuf::from(p),
                                zstd_level,
                                false,
                                header_mode,
                                wf.as_deref(),
                            )
                            .map_err(|e| format!("dct: {e}"))
                        })?;
                }
                ExportFmt::Png | ExportFmt::Jpg | ExportFmt::Webp => {
                    paths
                        .par_iter()
                        .enumerate()
                        .try_for_each(|(frame_idx, p)| -> Result<(), String> {
                            let begin = frame_idx * stride;
                            let chunk = pixels
                                .get(begin..begin + stride)
                                .ok_or_else(|| format!("raster oob {}", frame_idx))?;
                            let bytes = export_codec::encode_frame_bytes(
                                chunk,
                                h,
                                w,
                                c,
                                fmt,
                                quality,
                                fps,
                                header_mode,
                                wf.as_deref(),
                                zstd_level,
                            )?;
                            std::fs::write(p, bytes).map_err(|e| e.to_string())
                        })?;
                }
            }
            Ok(())
        }));

        match r {
            Ok(Ok(())) => {}
            Ok(Err(e)) => eprintln!("[v_vae_core] export failed: {e}"),
            Err(payload) => eprintln!("[v_vae_core] export panic: {payload:?}"),
        }
        ASYNC_PENDING.fetch_sub(1, Ordering::SeqCst);
    });

    Ok(())
}

pub(crate) fn schedule_save_dct_parallel(
    paths: Vec<String>,
    fps: f32,
    header_mode: u8,
    workflow_json: Option<String>,
    zstd_level: i32,
    require_full: bool,
) -> Result<(), String> {
    schedule_export(
        paths,
        "dct".to_string(),
        false,
        95,
        fps,
        header_mode,
        workflow_json,
        zstd_level,
        require_full,
        String::new(),
    )
}

pub(crate) fn pending_async_count() -> usize {
    ASYNC_PENDING.load(Ordering::SeqCst)
}
