"""Tropicalize & Export TSRNGist checkpoint for FPGA inference.

Replaces softmax/sigmoid with hard equivalents, quantizes to INT8,
exports hex weight files for Verilog $readmemh.

Usage:
  python tropicalize_export.py checkpoints/tsrn_gist_enwik8_best_small_scale.pt
"""
import argparse, json, math, os, sys
from typing import Dict, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from tsrn_gist import TSRNGist, load_enwik8, evaluate
from tsrn_dml import CharDataset


def quantize_int8(t: Tensor) -> Tuple[Tensor, float]:
    """Symmetric INT8: scale = max_abs / 127."""
    mx = t.float().abs().max().item()
    if mx < 1e-10:
        return torch.zeros_like(t, dtype=torch.int8), 1.0
    scale = mx / 127.0
    return torch.clamp(torch.round(t.float() / scale), -128, 127).to(torch.int8), scale


def quantize_int16(t: Tensor, frac: int = 8) -> Tuple[Tensor, float]:
    """Q8.8 fixed-point for embeddings."""
    s = 2.0 ** frac
    return torch.clamp(torch.round(t.float() * s), -32768, 32767).to(torch.int16), 1.0/s


def export_hex_rows(t_q: Tensor, path: str, row_width: int = 32):
    """Export quantized tensor as hex, packing INT8 into 32-bit words."""
    flat = t_q.flatten().tolist()
    with open(path, 'w') as f:
        for i in range(0, len(flat), row_width):
            row = flat[i:i+row_width]
            words = []
            for j in range(0, len(row), 4):
                c = row[j:j+4]
                while len(c) < 4: c.append(0)
                w = sum((int(b) & 0xFF) << (k*8) for k, b in enumerate(c))
                words.append(f"{w:08x}")
            f.write(' '.join(words) + '\n')


def export_int16_flat(t_q: Tensor, path: str):
    """Export INT16 tensor, one hex value per line."""
    with open(path, 'w') as f:
        for v in t_q.flatten().tolist():
            f.write(f"{int(v) & 0xFFFF:04x}\n")


def export_rotor_lut(theta: Tensor, path: str, frac: int = 14):
    """Export cos/sin LUT for Clifford rotors."""
    s = 2.0 ** frac
    c = torch.cos(theta.detach()).float()
    sn = torch.sin(theta.detach()).float()
    with open(path, 'w') as f:
        for i in range(len(theta)):
            cv = int(torch.clamp(torch.round(c[i]*s), -32768, 32767).item())
            sv = int(torch.clamp(torch.round(sn[i]*s), -32768, 32767).item())
            f.write(f"{cv & 0xFFFF:04x} {sv & 0xFFFF:04x}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to TSRNGist checkpoint")
    parser.add_argument("--output", default="../Tropical-ISA/tsrn_inference/firmware")
    args = parser.parse_args()

    out = args.output
    hex_dir = os.path.join(out, "hex")
    os.makedirs(hex_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"  Config: {cfg}")
    print(f"  Best val BPC: {ckpt.get('best_val_bpc', '?')}")

    model = TSRNGist(
        vocab=cfg["vocab"], d_model=cfg["d_model"],
        context_len=cfg["context_len"], n_blocks=cfg["n_blocks"],
        top_k=cfg["top_k"], n_heads=cfg["n_heads"],
        mem_depth=cfg["mem_depth"],
        max_gists=cfg.get("max_gists", 64),
        gist_top_k=cfg.get("gist_top_k", 4), dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Params: {model.count_params():,}")

    # Validate original
    ds = load_enwik8(context_len=cfg["context_len"])
    model.gist_buffer.reset()
    _, _, orig_bpc = evaluate(model, ds, torch.device("cpu"),
                              n_batches=50, batch_size=8, split="val")
    print(f"\n  Original val BPC: {orig_bpc:.4f}")

    # Quantize and export all parameters
    manifest = {"config": cfg, "orig_bpc": round(orig_bpc, 4), "layers": {}}
    total_bytes = 0
    weight_addr = 0  # running address in weight ROM

    print(f"\n  Exporting weights to {hex_dir}/")
    for name, param in model.named_parameters():
        p = param.detach().float()
        safe_name = name.replace(".", "_")

        if 'embed' in name or 'pos_' in name:
            t_q, scale = quantize_int16(p)
            fpath = os.path.join(hex_dir, f"{safe_name}.hex")
            export_int16_flat(t_q, fpath)
            nbytes = t_q.numel() * 2
            manifest["layers"][name] = {
                "dtype": "int16", "scale": scale, "shape": list(p.shape),
                "bytes": nbytes, "file": f"hex/{safe_name}.hex"
            }
        elif 'theta' in name and p.dim() == 1:
            # Clifford rotor angles — export as cos/sin LUT
            fpath = os.path.join(hex_dir, f"{safe_name}_lut.hex")
            export_rotor_lut(p, fpath)
            nbytes = len(p) * 4
            manifest["layers"][name] = {
                "dtype": "rotor_lut", "shape": list(p.shape),
                "bytes": nbytes, "file": f"hex/{safe_name}_lut.hex"
            }
        else:
            t_q, scale = quantize_int8(p)
            fpath = os.path.join(hex_dir, f"{safe_name}.hex")
            export_hex_rows(t_q, fpath)
            nbytes = t_q.numel()
            manifest["layers"][name] = {
                "dtype": "int8", "scale": scale, "shape": list(p.shape),
                "bytes": nbytes, "file": f"hex/{safe_name}.hex",
                "rom_addr": weight_addr
            }
            weight_addr += (nbytes + 3) // 4  # word-aligned

        total_bytes += nbytes
        print(f"    {name:50s} {str(list(p.shape)):20s} -> {nbytes:>6} B")

    # Export manifest
    manifest["total_bytes"] = total_bytes
    manifest["weight_rom_words"] = weight_addr
    mpath = os.path.join(out, "export_manifest.json")
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Total: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print(f"  Weight ROM: {weight_addr:,} x 32-bit words")
    print(f"  Manifest: {mpath}")

    # Export hardware config header
    hdr_path = os.path.join(out, "tsrn_config.svh")
    with open(hdr_path, 'w') as f:
        f.write(f"// Auto-generated from {args.checkpoint}\n")
        f.write(f"parameter D_MODEL    = {cfg['d_model']};\n")
        f.write(f"parameter N_HEADS    = {cfg['n_heads']};\n")
        f.write(f"parameter DH         = {cfg['d_model']//cfg['n_heads']};\n")
        f.write(f"parameter CTX_LEN    = {cfg['context_len']};\n")
        f.write(f"parameter VOCAB_SIZE = {cfg['vocab']};\n")
        f.write(f"parameter TOP_K      = {cfg['top_k']};\n")
        f.write(f"parameter N_BLOCKS   = {cfg['n_blocks']};\n")
        f.write(f"parameter MEM_DEPTH  = {cfg['mem_depth']};\n")
        f.write(f"parameter GIST_BUF   = {cfg.get('max_gists',64)};\n")
        f.write(f"parameter GIST_TOP_K = {cfg.get('gist_top_k',4)};\n")
        f.write(f"parameter WEIGHT_ROM_DEPTH = {weight_addr};\n")
    print(f"  SV header: {hdr_path}")

    # Generate consolidated weight_rom.hex for Verilog $readmemh
    # Packs all INT8 weight matrices sequentially into 32-bit words
    rom_path = os.path.join(hex_dir, "weight_rom.hex")
    rom_words = []
    for name, info in manifest["layers"].items():
        if info["dtype"] == "int8" and "rom_addr" in info:
            fpath = os.path.join(out, info["file"])
            with open(fpath, 'r') as f:
                for line in f:
                    for word in line.strip().split():
                        if word:
                            rom_words.append(word)
    # Pad to power-of-2 depth (16384)
    while len(rom_words) < 16384:
        rom_words.append("00000000")
    with open(rom_path, 'w') as f:
        for w in rom_words:
            f.write(w + '\n')
    print(f"  Weight ROM hex: {rom_path} ({len(rom_words)} words)")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
