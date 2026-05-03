# ComfyUI-Vates-VAE（V-VAE）

**Vates** 品牌分布式 VAE 解码与导出流水线：**Splitter → Worker → Collector**，底层由 Rust **`v_vae_core`** 的 **`FrameBus`**（连续像素池 + `copy_nonoverlapping`）与后台线程内 **Rayon** 并行编码驱动；支持 **`.dct` / PNG / JPEG / WebP** 及可选 **ZIP**（均在 Rust 内从像素池读取，不经 Python 回传像素）。

## 架构概览

| 节点 | 作用 |
|------|------|
| **Vates · VAE Splitter (FrameBus)** | 读 `LATENT`，按 latent 空间 × `spatial_scale`（默认 8）推算像素尺寸，调用 **`frame_bus_init_pool`** 预分配缓冲；输出 **`VATES_BATCH_SIGNAL`**。 |
| **Vates · VAE Worker (Decode + Push)** | 接收 **SIGNAL + VAE**，逐帧 `decode`，CPU `float` 连续张量经 **`data_ptr()`** 推送 **`push_frame`**；每帧后 **`comfy.model_management.soft_empty_cache()`**。 |
| **Vates · VAE Collector (Async Export)** | 接收 **SIGNAL + `save_prefix`**，在 **`folder_paths.get_output_directory()`** 下按 **`output_format`** 生成 **`prefix_XXXX`** + 对应扩展名（`dct` / `png` / `jpg` / `webp`），或 **`package_as_zip=true`** 时生成单个 **`prefix.zip`**（内含按序命名的帧）；调用 **`frame_bus_schedule_export`**（Rust 内存编码 + 可选 Deflate ZIP）；**`quality`**（1–100）作用于 JPEG/WebP，PNG 忽略；可选 **`wait_for_disk`**。 |

Rust 编码路径复用 **`ComfyUI-Vates-BatchLoader/dct-core`** 中的 **`vates_core::Encoder`**（与既有 `.dct` 生态一致）。

## 依赖与路径

- **ComfyUI** 完整安装（含 `folder_paths`、`model_management`）。
- **本仓库旁需存在**：`ComfyUI-Vates-BatchLoader/dct-core`（`v_vae_core` 的 Cargo **`path`** 依赖）。
- **Rust**：编译扩展需要 [Rust toolchain](https://rustup.rs/)。

## 安装

```bash
cd ComfyUI-Vates-VAE/dct-core
python install.py
```

成功后在 `dct-core/` 目录可见对齐产物 **`v_vae_core.pyd`**（Windows）或 **`v_vae_core.so`**（Linux/macOS）。随后重启 ComfyUI。

## 工作流连线

```
LATENT ──► [Splitter] ──► VATES_BATCH_SIGNAL ──► [Worker] ──► SIGNAL ──► [Collector]
                              │                      │
                              └──────────────────────┘
                                      VAE ────────────┘
```

1. **Splitter** 仅需 **LATENT**；可按模型调整 **`spatial_scale`**，使预留像素尺寸与 VAE 解码结果 **H×W×3** 一致。  
2. **Worker** 将 SIGNAL 与 **VAE** 相连；若解码尺寸与预留不一致会 **报错**，避免静默损坏 DCT。  
3. **Collector** 填写 **`save_prefix`**；必选 **`output_format`**（`png` / `jpg` / `dct` / `webp`）；可选 **`package_as_zip`**、**`quality`**（1–100）、**`wait_for_disk`**、`fps`、`zstd_level`、`header_mode`、`workflow_json`。

## 性能要点

- **像素池**：启动阶段一次性分配，降低长时间 batch 下的分配碎片。  
- **推送**：Python 侧不做额外整块拷贝；Rust 侧单次 **`memcpy`** 入池（GPU→CPU 仍由解码路径决定）。  
- **落盘**：异步线程内 **Rayon** 并行逐帧编码（栅格或 DCT），ZIP 模式下先在内存中收集各帧字节再 **`ZipWriter`** 落盘；主图不阻塞等待磁盘。

## 自定义类型

连线类型名：**`VATES_BATCH_SIGNAL`**（Splitter / Worker / Collector 三者端口一致）。

## 许可证

与依赖的 `vates_core`（MIT OR Apache-2.0）保持一致；详见各 `Cargo.toml` / `pyproject.toml`。
