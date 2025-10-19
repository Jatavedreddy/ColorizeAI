#!/usr/bin/env python3
"""
Generic TorchScript exporter for external PyTorch models (e.g., DDColor).

Usage example (DDColor-like):

  python tools/export_torchscript.py \
    --module ddcolor.model_builder \
    --builder build_model \
    --checkpoint /path/to/ddcolor_checkpoint.pth \
    --output weights/ddcolor_scripted.pt \
    --input-shape 1 1 256 256 \
    --method trace

Notes:
- You must know the Python module and the builder function that returns an initialized model.
- The builder function should return a torch.nn.Module matching the checkpoint.
- The model must accept a single input tensor shaped as provided by --input-shape.
- For DDColor, this is typically [N,1,256,256] L-channel (Lab) input with output [N,2,256,256] ab.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import List

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a PyTorch model to TorchScript")
    p.add_argument("--module", required=True, help="Python module path containing the builder, e.g., ddcolor.model")
    p.add_argument("--builder", required=True, help="Builder function name within the module, e.g., build_model")
    p.add_argument("--checkpoint", required=True, help="Path to .pth/.pt state dict or full model")
    p.add_argument("--output", required=True, help="Output TorchScript file path (e.g., weights/ddcolor.pt)")
    p.add_argument("--input-shape", nargs=4, type=int, default=[1, 1, 256, 256], metavar=("N", "C", "H", "W"),
                   help="Input tensor shape for example input (default: 1 1 256 256)")
    p.add_argument("--method", choices=["trace", "script"], default="trace",
                   help="TorchScript method to use (default: trace)")
    p.add_argument("--device", default="cpu", help="Device for export (cpu/cuda)")
    return p.parse_args()


def main():
    args = parse_args()

    mod = importlib.import_module(args.module)
    builder = getattr(mod, args.builder, None)
    if builder is None:
        raise RuntimeError(f"Builder function {args.builder} not found in module {args.module}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = builder()
    model.to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Try to load as state_dict first; fallback to full model
    try:
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            # Remove DistributedDataParallel prefixes if present
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    except Exception:
        # Maybe it's a whole scripted/saved model; try to load and copy weights if possible
        loaded = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(loaded, torch.nn.Module):
            model.load_state_dict(loaded.state_dict(), strict=False)
        else:
            raise

    model.eval()

    n, c, h, w = args.input_shape
    example = torch.randn(n, c, h, w, device=device)

    if args.method == "script":
        scripted = torch.jit.script(model)
    else:
        scripted = torch.jit.trace(model, example)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    print(f"✅ Saved TorchScript model to: {out_path}")


if __name__ == "__main__":
    main()
