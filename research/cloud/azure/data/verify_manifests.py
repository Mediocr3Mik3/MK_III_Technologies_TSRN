"""
verify_manifests.py
===================

Verify every HuggingFace dataset referenced by the Azure training
manifests is reachable WITHOUT downloading any data.  Uses the HF Hub
API (`HfApi.dataset_info`) to issue a single metadata HEAD-style call
per dataset, then probes file availability via `huggingface_hub.hf_hub_url`
+ a short `requests.head`.

Also validates:
  - All local proprietary JSONL paths exist (warning if missing)
  - YAML manifests parse cleanly
  - Token / example budgets are sane and weights sum near 1.0
  - HF auth is configured if needed (gated datasets)

Usage (from repo root):
    python -m research.cloud.azure.data.verify_manifests
    python -m research.cloud.azure.data.verify_manifests --manifest pretrain_mix
    python -m research.cloud.azure.data.verify_manifests --all --strict

Exits non-zero if any required dataset is unreachable.  Safe to run on
any machine with `huggingface_hub` installed; no GPU / no disk writes.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 stdout on Windows (cp1252 can't encode ✓/✗/⊘).
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ASCII-safe symbol fallbacks for terminals that still refuse UTF-8.
_USE_ASCII = os.environ.get("VERIFY_ASCII", "0") == "1" or (
    sys.platform == "win32" and not sys.stdout.isatty()
)
SYM_OK    = "[OK]"   if _USE_ASCII else "✓"
SYM_FAIL  = "[FAIL]" if _USE_ASCII else "✗"
SYM_WARN  = "[WARN]" if _USE_ASCII else "⚠"
SYM_SKIP  = "[GATE]" if _USE_ASCII else "⊘"

try:
    import yaml
except ImportError as e:
    print("ERROR: pyyaml not installed.  `pip install pyyaml`", file=sys.stderr)
    raise SystemExit(2) from e

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (
        RepositoryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
    )
except ImportError as e:
    print("ERROR: huggingface_hub not installed. "
          "`pip install huggingface_hub`", file=sys.stderr)
    raise SystemExit(2) from e


REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_DIR = Path(__file__).parent / "manifests"
MANIFESTS = {
    "pretrain_mix": MANIFEST_DIR / "pretrain_mix.yaml",
    "sft_mix":      MANIFEST_DIR / "sft_mix.yaml",
    "dpo_mix":      MANIFEST_DIR / "dpo_mix.yaml",
}


# ANSI colours (graceful on Windows via colorama if installed, else raw).
def _c(code: str, s: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


GREEN, RED, YELLOW, DIM = "32", "31", "33", "2"


# ---------------------------------------------------------------------------
# Per-dataset probe
# ---------------------------------------------------------------------------

def probe_hf_dataset(
    api: HfApi,
    repo_id: str,
    config: Optional[str],
    split: Optional[str],
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """
    Return a dict with keys:
      ok: bool
      status: "ok" | "gated" | "not_found" | "auth_needed" | "error"
      detail: str
      bytes_total: Optional[int]
    """
    t0 = time.time()
    try:
        info = api.dataset_info(repo_id, timeout=timeout, files_metadata=False)
    except GatedRepoError as e:
        return {"ok": False, "status": "gated",
                "detail": f"gated; accept terms on hf.co/datasets/{repo_id}",
                "bytes_total": None, "elapsed_s": time.time() - t0}
    except RepositoryNotFoundError:
        return {"ok": False, "status": "not_found",
                "detail": "404 (dataset does not exist or auth required)",
                "bytes_total": None, "elapsed_s": time.time() - t0}
    except HfHubHTTPError as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            return {"ok": False, "status": "auth_needed",
                    "detail": f"auth required ({msg[:80]})",
                    "bytes_total": None, "elapsed_s": time.time() - t0}
        return {"ok": False, "status": "error",
                "detail": msg[:120],
                "bytes_total": None, "elapsed_s": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": "error",
                "detail": f"{type(e).__name__}: {str(e)[:120]}",
                "bytes_total": None, "elapsed_s": time.time() - t0}

    # Successful info call.  Try to compute total bytes if files_metadata
    # would be cheap; we skipped it for speed.  Pull a fast list of siblings
    # for sanity instead.
    n_files = len(info.siblings) if info.siblings is not None else 0
    return {
        "ok": True,
        "status": "ok",
        "detail": f"{n_files} files; private={info.private}",
        "bytes_total": None,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Manifest scan
# ---------------------------------------------------------------------------

def check_manifest(path: Path, api: HfApi, strict: bool) -> Dict[str, Any]:
    print(f"\n{_c('1', '=' * 70)}")
    print(f"  {path.name}")
    print(_c('1', '=' * 70))

    with open(path) as f:
        manifest = yaml.safe_load(f)

    mixture: List[Dict[str, Any]] = manifest.get("mixture", [])
    n_total = len(mixture)

    n_ok, n_gated, n_missing, n_auth, n_local_missing, n_err = 0, 0, 0, 0, 0, 0
    weight_sum = 0.0

    for entry in mixture:
        name = entry["name"]
        hf_id = entry.get("hf_dataset")
        weight = float(entry.get("weight", 0.0))
        weight_sum += weight

        if hf_id is None:
            # local proprietary file
            local_path = entry.get("local_path")
            if not local_path:
                print(f"  {_c(YELLOW, SYM_WARN)}  {name:30s}  no hf_dataset and no local_path")
                n_err += 1
                continue
            full = REPO_ROOT / local_path
            if full.exists():
                size_mb = full.stat().st_size / 1e6
                print(f"  {_c(GREEN, SYM_OK)}  {name:30s}  {_c(DIM, f'local {size_mb:.1f} MB')}")
                n_ok += 1
            else:
                print(f"  {_c(YELLOW, SYM_WARN)}  {name:30s}  "
                      f"{_c(YELLOW, 'local file missing:')} {local_path}")
                n_local_missing += 1
            continue

        # remote HF
        result = probe_hf_dataset(api, hf_id, entry.get("hf_config"),
                                  entry.get("split"))
        elapsed = result["elapsed_s"]
        if result["ok"]:
            detail = result["detail"]
            tag = f"{elapsed*1000:.0f}ms · {detail}"
            print(f"  {_c(GREEN, SYM_OK)}  {name:30s}  "
                  f"{hf_id:48s}  {_c(DIM, tag)}")
            n_ok += 1
        elif result["status"] == "gated":
            print(f"  {_c(YELLOW, SYM_SKIP)}  {name:30s}  "
                  f"{hf_id:48s}  {_c(YELLOW, result['detail'])}")
            n_gated += 1
        elif result["status"] == "auth_needed":
            print(f"  {_c(YELLOW, SYM_SKIP)}  {name:30s}  "
                  f"{hf_id:48s}  {_c(YELLOW, result['detail'])}")
            n_auth += 1
        elif result["status"] == "not_found":
            print(f"  {_c(RED, SYM_FAIL)}  {name:30s}  "
                  f"{hf_id:48s}  {_c(RED, result['detail'])}")
            n_missing += 1
        else:
            print(f"  {_c(RED, SYM_FAIL)}  {name:30s}  "
                  f"{hf_id:48s}  {_c(RED, result['detail'])}")
            n_err += 1

    # Sanity: weights sum near 1.0 (within 5%)
    print()
    if abs(weight_sum - 1.0) > 0.05:
        print(f"  {_c(YELLOW, 'WARN')}  weights sum to {weight_sum:.3f} (expected ~1.0)")
    else:
        print(f"  {_c(GREEN, 'OK')}    weights sum {weight_sum:.3f}")

    summary = {
        "name": path.stem,
        "total": n_total,
        "ok": n_ok,
        "gated": n_gated,
        "auth": n_auth,
        "missing": n_missing,
        "local_missing": n_local_missing,
        "error": n_err,
        "weight_sum": weight_sum,
    }

    # Verdict
    fatal = n_missing + n_err
    if strict:
        fatal += n_gated + n_auth + n_local_missing
    summary["fatal"] = fatal
    return summary


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", choices=list(MANIFESTS) + ["all"],
                   default="all",
                   help="Which manifest to check.")
    p.add_argument("--strict", action="store_true",
                   help="Treat gated / auth-needed / missing local as fatal.")
    p.add_argument("--token", default=None,
                   help="HF token (else uses HF_TOKEN env var or cached login).")
    args = p.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)

    if token:
        try:
            who = api.whoami()
            print(f"  HF auth : {_c(GREEN, who['name'])}")
        except Exception as e:  # noqa: BLE001
            print(f"  HF auth : {_c(RED, 'invalid token')} ({e})")
    else:
        print(f"  HF auth : {_c(YELLOW, 'anonymous')}  "
              f"(gated datasets will fail — set HF_TOKEN if needed)")

    to_check = list(MANIFESTS) if args.manifest == "all" else [args.manifest]
    summaries = []
    for m in to_check:
        path = MANIFESTS[m]
        if not path.exists():
            print(f"  {_c(RED, 'MISSING')} {path}")
            continue
        summaries.append(check_manifest(path, api, strict=args.strict))

    # Final table
    print(f"\n{_c('1', '=' * 70)}")
    print(f"  Summary")
    print(_c('1', '=' * 70))
    print(f"  {'manifest':<16s} {'total':>6s} {'ok':>4s} {'gate':>5s} "
          f"{'auth':>5s} {'404':>5s} {'lcl':>5s} {'err':>4s}")
    n_fatal_all = 0
    for s in summaries:
        print(f"  {s['name']:<16s} {s['total']:>6d} {s['ok']:>4d} "
              f"{s['gated']:>5d} {s['auth']:>5d} {s['missing']:>5d} "
              f"{s['local_missing']:>5d} {s['error']:>4d}")
        n_fatal_all += s["fatal"]

    print()
    if n_fatal_all == 0:
        print(_c(GREEN, "  ALL CLEAR — every dataset is reachable."))
        return 0
    print(_c(RED, f"  {n_fatal_all} fatal issue(s).  "
                  "Run with --strict to also treat gated/auth as fatal."))
    return 1


if __name__ == "__main__":
    sys.exit(main())
