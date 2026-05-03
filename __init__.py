"""ComfyUI-Vates-VAE：分布式 VAE / FrameBus / DCT Collector。"""

from __future__ import annotations

from .vae_nodes import (
    VATES_BATCH_SIGNAL,
    VatesVAECollector,
    VatesVAESplitter,
    VatesVAEWorker,
)

NODE_CLASS_MAPPINGS = {
    "VatesVAESplitter": VatesVAESplitter,
    "VatesVAEWorker": VatesVAEWorker,
    "VatesVAECollector": VatesVAECollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VatesVAESplitter": "Vates · VAE Splitter (FrameBus)",
    "VatesVAEWorker": "Vates · VAE Worker (Decode + Push)",
    "VatesVAECollector": "Vates · VAE Collector (Async DCT)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "VATES_BATCH_SIGNAL",
]
