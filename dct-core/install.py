#!/usr/bin/env python3
"""编译并对齐 `v_vae_core`（FrameBus）。在 `dct-core/` 执行：python install.py"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "target" / "release"

from vae_repo_meta import expected_v_vae_version


def _copy_native_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32" and dst.is_file():
        fd, tmp_name = tempfile.mkstemp(suffix=dst.suffix, dir=str(dst.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return
    shutil.copy2(src, dst)


def _pick_windows_native_artifact(release_dir: Path) -> Path | None:
    if not release_dir.is_dir():
        return None
    tagged = list(release_dir.glob("v_vae_core.cp*.pyd"))
    if tagged:
        return max(tagged, key=lambda p: p.stat().st_mtime)
    for name in ("v_vae_core.dll", "libv_vae_core.dll"):
        p = release_dir / name
        if p.is_file():
            return p
    plain = release_dir / "v_vae_core.pyd"
    if plain.is_file():
        return plain
    return None


def _align_native_artifacts() -> list[str]:
    out: list[str] = []
    if not RELEASE.is_dir():
        return out
    if sys.platform == "win32":
        src = _pick_windows_native_artifact(RELEASE)
        if src is not None:
            dst = ROOT / "v_vae_core.pyd"
            _copy_native_artifact(src, dst)
            out.append(str(dst))
        return out
    if sys.platform == "darwin":
        for name in ("libv_vae_core.dylib", "libv_vae_core.so"):
            p = RELEASE / name
            if p.is_file():
                dst = ROOT / "v_vae_core.so"
                _copy_native_artifact(p, dst)
                _chmod_exe(dst)
                out.append(str(dst))
                break
        return out
    for name in ("libv_vae_core.so", "v_vae_core.so"):
        p = RELEASE / name
        if p.is_file():
            dst = ROOT / "v_vae_core.so"
            _copy_native_artifact(p, dst)
            _chmod_exe(dst)
            out.append(str(dst))
            break
    return out


def _chmod_exe(dst: Path) -> None:
    try:
        os.chmod(
            dst,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH,
        )
    except OSError:
        pass


def _verify_import() -> bool:
    exp = expected_v_vae_version()
    script = (
        "import v_vae_core as v; "
        f"exp={exp!r}; "
        "assert getattr(v,'__version__',None)==exp; "
        "assert callable(v.frame_bus_init_pool); "
        "assert callable(v.frame_bus_push_frame); "
        "assert callable(getattr(v, 'frame_bus_schedule_export', None))"
    )
    try:
        subprocess.check_call(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _cargo_build() -> bool:
    if shutil.which("cargo") is None:
        print("[V-VAE install] 需要 Rust：https://rustup.rs/", flush=True)
        return False
    cmd = ["cargo", "build", "--release", "--no-default-features", "--features", "python"]
    print(f"[V-VAE install] {' '.join(cmd)} cwd={ROOT}", flush=True)
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    if _verify_import():
        print("[V-VAE install] v_vae_core 已可用。", flush=True)
        return 0
    if _cargo_build():
        copied = _align_native_artifacts()
        if copied:
            print("[V-VAE install] aligned:", *copied, sep="\n  ", flush=True)
        if _verify_import():
            print("[V-VAE install] OK。", flush=True)
            return 0
    print("[V-VAE install] 失败：请在本目录 cargo build --release --features python", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
