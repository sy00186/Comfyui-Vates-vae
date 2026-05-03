//! Singleton FrameBus: preallocated NHWC f32 buffer, push_frame memcpy from ptr,
//! schedule_save_dct_parallel: background Rayon parallel .dct via Encoder::encode_batch_bhwc_file.

use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use vates_core::{DctError, Encoder};

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
fn frame_stride(h: usize, w: usize, c: usize) -> Result<usize, String> {
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

/// ptr points to contiguous CPU f32 NHWC [H,W,C]; num_elements must equal H*W*C.
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

pub(crate) fn schedule_save_dct_parallel(
    paths: Vec<String>,
    fps: f32,
    header_mode: u8,
    workflow_json: Option<String>,
    zstd_level: i32,
    require_full: bool,
) -> Result<(), String> {
    let inner = {
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
        if paths.len() != inner_ref.batch {
            return Err(format!(
                "paths {} != batch {}",
                paths.len(),
                inner_ref.batch
            ));
        }
        g.take().expect("just peeked Some")
    };

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

    std::thread::spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            paths
                .par_iter()
                .enumerate()
                .try_for_each(|(frame_idx, p)| -> Result<(), DctError> {
                    let begin = frame_idx * stride;
                    let chunk = pixels
                        .get(begin..begin + stride)
                        .ok_or_else(|| DctError::ShapeOverflow)?;

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
                        workflow_json.as_deref(),
                    )
                })
        }));

        match r {
            Ok(Ok(())) => {}
            Ok(Err(e)) => eprintln!("[v_vae_core] parallel DCT save failed: {e}"),
            Err(payload) => eprintln!("[v_vae_core] encode thread panic: {payload:?}"),
        }
        ASYNC_PENDING.fetch_sub(1, Ordering::SeqCst);
    });

    Ok(())
}

pub(crate) fn pending_async_count() -> usize {
    ASYNC_PENDING.load(Ordering::SeqCst)
}
