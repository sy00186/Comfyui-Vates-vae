"""
Vates VAE 分布式三节点：Splitter 初始化 Rust FrameBus；Worker VAE 解码并 ``push_frame``；
Collector 触发 Rayon 异步 DCT 落盘。
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import folder_paths
import torch

import comfy.model_management as model_management

logger = logging.getLogger(__name__)

_PLUGIN_ROOT = Path(__file__).resolve().parent
_DCT_CORE = _PLUGIN_ROOT / "dct-core"
if _DCT_CORE.is_dir() and str(_DCT_CORE) not in sys.path:
    sys.path.insert(0, str(_DCT_CORE))

VATES_BATCH_SIGNAL = "VATES_BATCH_SIGNAL"


def _pick_v_vae_core():  # noqa: ANN401
    for cand in ("v_vae_core.pyd", "v_vae_core.so"):
        p = _DCT_CORE / cand
        if not p.is_file():
            continue
        spec = importlib.util.spec_from_file_location("v_vae_core_ext_shim", p)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    try:
        import v_vae_core as vv  # type: ignore
    except ImportError as e:
        raise ImportError(
            "缺少 `v_vae_core`。请在 ComfyUI-Vates-VAE/dct-core 目录执行: python install.py"
        ) from e
    return vv


def _sanitize_prefix(name: str) -> str:
    s = re.sub(r'[/\\\\]', "_", str(name).strip())
    return s if s else "vates_vae"


def _decode_rgb(vae: Any, latent_bchw: torch.Tensor) -> torch.Tensor:
    dec = getattr(vae, "decode", None)
    if callable(dec):
        return dec(latent_bchw)
    dec_t = getattr(vae, "decode_tiled", None)
    if callable(dec_t):
        return dec_t(latent_bchw)
    raise RuntimeError("VAE 对象不支持 decode / decode_tiled")


class VatesVAESplitter:
    """LATENT -> VATES_BATCH_SIGNAL，并调用 Rust ``frame_bus_init_pool``。"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "spatial_scale": ("INT", {"default": 8, "min": 1, "max": 128}),
            },
        }

    RETURN_TYPES = (VATES_BATCH_SIGNAL,)
    RETURN_NAMES = ("signal",)
    FUNCTION = "split"
    CATEGORY = "Vates/VAE"

    def split(
        self,
        latent: dict,
        spatial_scale: int = 8,
    ) -> tuple[dict]:
        vv = _pick_v_vae_core()
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            raise RuntimeError("FrameBus 不支持 NestedTensor latent")
        b = int(samples.shape[0])
        lh = int(samples.shape[2])
        lw = int(samples.shape[3])
        ph = lh * int(spatial_scale)
        pw = lw * int(spatial_scale)
        ch = 3

        vv.frame_bus_init_pool(b, ph, pw, ch)

        signal = {
            "v": 1,
            "batch": b,
            "pixel_h": ph,
            "pixel_w": pw,
            "pixel_c": ch,
            "latent": latent,
            "spatial_scale": int(spatial_scale),
        }
        return (signal,)


class VatesVAEWorker:
    """解码 latent 切片，经原生指针 ``push_frame``；每帧后 ``soft_empty_cache``。"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {
                "signal": (VATES_BATCH_SIGNAL,),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = (VATES_BATCH_SIGNAL,)
    RETURN_NAMES = ("signal",)
    FUNCTION = "decode_push"
    CATEGORY = "Vates/VAE"

    def decode_push(self, signal: dict, vae: Any) -> tuple[dict]:
        vv = _pick_v_vae_core()
        latent = signal["latent"]
        samples = latent["samples"]
        batch = int(signal["batch"])
        ph = int(signal["pixel_h"])
        pw = int(signal["pixel_w"])
        pc = int(signal.get("pixel_c", 3))
        lh = int(samples.shape[2])
        lw = int(samples.shape[3])

        if getattr(samples, "is_nested", False):
            raise RuntimeError("不支持 NestedTensor")

        for i in range(batch):
            slab = samples[i : i + 1]
            with torch.inference_mode():
                decoded = _decode_rgb(vae, slab)

            if decoded.ndim != 4:
                raise RuntimeError(f"解码输出维度异常: shape={tuple(decoded.shape)}")
            # [B,H,W,C]
            if decoded.shape[-1] >= 3:
                plane = decoded[:, :, :, :3]
            else:
                plane = decoded

            im = plane.float()
            if im.device.type != "cpu":
                im = im.cpu()
            im = im.contiguous()
            # 单帧 [1,H,W,3] — 使用首帧平面，元素数须为 H*W*C
            frame = im[0]
            if tuple(frame.shape) != (ph, pw, pc):
                raise RuntimeError(
                    f"VAE 解码尺寸 {tuple(frame.shape)} 与 Splitter 预留池 ({ph},{pw},{pc}) 不符；"
                    f"请调整 spatial_scale（当前 latent {lh}x{lw}，scale={signal.get('spatial_scale', '?')}）。"
                )

            nelem = int(frame.numel())
            ptr = int(frame.data_ptr())
            vv.frame_bus_push_frame(i, ptr, nelem)

            del decoded, im, frame, slab
            model_management.soft_empty_cache()

        pushed = int(vv.frame_bus_pushed_count())
        if pushed != batch:
            logger.warning("push 计数=%s 期望 batch=%s", pushed, batch)

        return (signal,)


class VatesVAECollector:
    """触发 ``schedule_save_dct_parallel``；可选阻塞等待磁盘写入。"""

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {
                "signal": (VATES_BATCH_SIGNAL,),
                "save_prefix": ("STRING", {"default": "vates_vae_batch"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0}),
            },
            "optional": {
                "header_mode": ("INT", {"default": 0, "min": 0, "max": 255}),
                "zstd_level": ("INT", {"default": 3, "min": 1, "max": 22}),
                "workflow_json": ("STRING", {"default": "", "multiline": True}),
                "wait_for_disk": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "collect_save"
    CATEGORY = "Vates/VAE"

    def collect_save(
        self,
        signal: dict,
        save_prefix: str,
        fps: float,
        header_mode: int = 0,
        zstd_level: int = 3,
        workflow_json: str = "",
        wait_for_disk: bool = False,
    ) -> dict:
        vv = _pick_v_vae_core()
        batch = int(signal["batch"])
        safe = _sanitize_prefix(save_prefix)
        out_dir = folder_paths.get_output_directory()
        paths = [os.path.join(out_dir, f"{safe}_{i:04d}.dct") for i in range(batch)]

        wf = workflow_json.strip() or None
        vv.frame_bus_schedule_save_dct_parallel(
            paths,
            float(fps),
            int(header_mode) & 0xFF,
            wf,
            int(zstd_level),
            True,
        )

        pending = int(vv.get_pending_tasks())
        msg = (
            f"Vates VAE：已提交异步 DCT，batch={batch}，后台任务={pending}。\n"
            f"输出目录: {out_dir}\n前缀: {safe}_*.dct"
        )

        if wait_for_disk:
            vv.await_pending_writes()
            msg += "\n已完成磁盘写入（wait_for_disk=true）。"

        return {"ui": {"text": (msg,)}, "result": (msg,)}
