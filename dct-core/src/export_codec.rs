//! 自 FrameBus **NHWC f32** 切片编码为 PNG/JPEG/WebP 内存字节，或 DCT（经临时文件读回字节）。

use image::codecs::jpeg::JpegEncoder;
use image::ExtendedColorType;
use std::io::Cursor;
use tempfile::NamedTempFile;
use vates_core::Encoder;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExportFmt {
    Png,
    Jpg,
    Webp,
    Dct,
}

pub(crate) fn parse_format(s: &str) -> Result<ExportFmt, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "png" => Ok(ExportFmt::Png),
        "jpg" | "jpeg" => Ok(ExportFmt::Jpg),
        "webp" => Ok(ExportFmt::Webp),
        "dct" => Ok(ExportFmt::Dct),
        other => Err(format!("unknown output_format: {other}")),
    }
}

pub(crate) fn ext_for(fmt: ExportFmt) -> &'static str {
    match fmt {
        ExportFmt::Png => "png",
        ExportFmt::Jpg => "jpg",
        ExportFmt::Webp => "webp",
        ExportFmt::Dct => "dct",
    }
}

pub(crate) fn nhwc_f32_to_rgb8(chunk: &[f32], h: usize, w: usize, c: usize) -> Result<Vec<u8>, String> {
    if c < 3 {
        return Err("channels must be >= 3 for RGB export".to_string());
    }
    let expect = h
        .checked_mul(w)
        .and_then(|x| x.checked_mul(c))
        .ok_or_else(|| "H*W*C overflow".to_string())?;
    if chunk.len() != expect {
        return Err(format!(
            "chunk len {} != H*W*C {}",
            chunk.len(),
            expect
        ));
    }
    let mut rgb = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * c;
            let dst = (y * w + x) * 3;
            for k in 0..3 {
                let v = chunk[src + k].clamp(0.0_f32, 1.0_f32);
                rgb[dst + k] = (v * 255.0_f32).round() as u8;
            }
        }
    }
    Ok(rgb)
}

pub(crate) fn encode_png_bytes(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let img =
        image::RgbImage::from_raw(width, height, rgb.to_vec()).ok_or_else(|| {
            "RgbImage::from_raw failed (dimension mismatch)".to_string()
        })?;
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .map_err(|e| format!("png encode: {e}"))?;
    Ok(buf)
}

pub(crate) fn encode_jpeg_bytes(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    let q = quality.clamp(1, 100);
    let mut buf = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut buf, q);
    enc.encode(rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|e| format!("jpeg encode: {e}"))?;
    Ok(buf)
}

pub(crate) fn encode_webp_bytes(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    let enc = webp::Encoder::from_rgb(rgb, width, height);
    let q = quality.clamp(1, 100) as f32;
    let mem = enc.encode(q);
    Ok(mem.to_vec())
}

pub(crate) fn encode_dct_bytes(
    chunk_nhwc: &[f32],
    height: u32,
    width: u32,
    channels: u32,
    fps: f32,
    header_mode: u8,
    workflow_json: Option<&str>,
    zstd_level: i32,
) -> Result<Vec<u8>, String> {
    let tmp = NamedTempFile::new().map_err(|e| format!("tempfile: {e}"))?;
    Encoder::encode_batch_bhwc_file(
        chunk_nhwc,
        1,
        height,
        width,
        channels,
        fps,
        tmp.path(),
        zstd_level,
        false,
        header_mode,
        workflow_json,
    )
    .map_err(|e| format!("dct encode: {e}"))?;
    std::fs::read(tmp.path()).map_err(|e| format!("read dct bytes: {e}"))
}

pub(crate) fn encode_frame_bytes(
    chunk_nhwc: &[f32],
    h: usize,
    w: usize,
    c: usize,
    fmt: ExportFmt,
    quality: u8,
    fps: f32,
    header_mode: u8,
    workflow_json: Option<&str>,
    zstd_level: i32,
) -> Result<Vec<u8>, String> {
    let wu = u32::try_from(w).map_err(|_| "width".to_string())?;
    let hu = u32::try_from(h).map_err(|_| "height".to_string())?;
    let ch_u32 = u32::try_from(c).map_err(|_| "channels".to_string())?;
    match fmt {
        ExportFmt::Dct => encode_dct_bytes(
            chunk_nhwc,
            hu,
            wu,
            ch_u32,
            fps,
            header_mode,
            workflow_json,
            zstd_level,
        ),
        ExportFmt::Png => {
            let rgb = nhwc_f32_to_rgb8(chunk_nhwc, h, w, c)?;
            encode_png_bytes(&rgb, wu, hu)
        }
        ExportFmt::Jpg => {
            let rgb = nhwc_f32_to_rgb8(chunk_nhwc, h, w, c)?;
            encode_jpeg_bytes(&rgb, wu, hu, quality)
        }
        ExportFmt::Webp => {
            let rgb = nhwc_f32_to_rgb8(chunk_nhwc, h, w, c)?;
            encode_webp_bytes(&rgb, wu, hu, quality)
        }
    }
}
