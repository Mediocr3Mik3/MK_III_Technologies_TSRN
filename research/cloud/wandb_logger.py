"""
WandB logging helper for the TropFormer cloud trainer.
=======================================================

Thin wrapper around the wandb SDK that:
  * Is fully optional — if wandb is not installed or no project is given,
    every call is a no-op.
  * Is rank-aware — only rank 0 actually talks to the WandB server.
  * Mirrors the existing `log_entry` dict used by `train_cloud.py`, so the
    integration is one line per call site.

Usage in train_cloud.py:
    from research.cloud.wandb_logger import WandBLogger

    wb = WandBLogger.maybe_init(args, cfg, run_tag, is_main=is_main,
                                model=model, world_size=world)
    ...
    wb.log(log_entry, step=step)              # eval-step metrics
    wb.log_text("sample", sample_text, step)  # generation samples
    wb.log_summary(test_bpc=..., wall_time_hours=...)  # final
    wb.finish()
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


class WandBLogger:
    """No-op friendly WandB wrapper. Safe on non-rank-0 and when wandb is off."""

    def __init__(self, run: Any = None, enabled: bool = False):
        self._run = run
        self._enabled = enabled

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    @classmethod
    def maybe_init(
        cls,
        args: Any,
        cfg: Dict[str, Any],
        run_tag: str,
        is_main: bool,
        model: Any = None,
        world_size: int = 1,
    ) -> "WandBLogger":
        """Initialise WandB iff a project name was supplied AND we're rank 0.

        Returns a no-op logger in every other case (multi-rank workers,
        wandb-not-installed, or user did not pass --wandb-project).
        """
        project = getattr(args, "wandb_project", None)
        if not project or not is_main:
            return cls(run=None, enabled=False)

        try:
            import wandb  # type: ignore
        except ImportError:
            print("[wandb] wandb not installed; skipping logging. "
                  "pip install wandb to enable.")
            return cls(run=None, enabled=False)

        # Allow WANDB_MODE=offline / disabled via env without crashing.
        mode = os.environ.get("WANDB_MODE", "online")

        run_name = getattr(args, "wandb_run_name", None) or run_tag
        entity = getattr(args, "wandb_entity", None)
        tags = []
        if cfg.get("use_kleene_ssm"):
            tags.append("kleene_ssm")
        if cfg.get("tier"):
            tags.append(f"tier_{cfg['tier']}")
        if cfg.get("use_hyperbolic"):
            tags.append("hyperbolic")
        if cfg.get("gist_chaining"):
            tags.append("gist_chaining")
        tags.append(f"world_{world_size}")

        # Sanitised config payload (no tensors, no callables).
        wb_config = {k: v for k, v in cfg.items()
                     if isinstance(v, (int, float, str, bool, list, tuple, type(None)))}
        wb_config.update({
            "preset": getattr(args, "preset", None),
            "tag": getattr(args, "tag", None),
            "world_size": world_size,
        })

        try:
            run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags,
                config=wb_config,
                resume="allow",
                mode=mode,
            )
        except Exception as e:  # pragma: no cover
            print(f"[wandb] init failed: {e!r}; continuing without logging.")
            return cls(run=None, enabled=False)

        # Watch params / grad norms occasionally (not too noisy).
        if model is not None:
            try:
                wandb.watch(model, log="gradients", log_freq=2000, log_graph=False)
            except Exception:
                pass

        print(f"[wandb] live at: {run.url}")
        return cls(run=run, enabled=True)

    # ---------------------------------------------------------------------
    # Logging surface
    # ---------------------------------------------------------------------
    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a flat dict of scalars. Safe to call when disabled."""
        if not self._enabled:
            return
        try:
            import wandb  # type: ignore
            # wandb chokes on non-numeric keys mixed in; coerce safely.
            clean = {k: v for k, v in payload.items()
                     if isinstance(v, (int, float))}
            wandb.log(clean, step=step)
        except Exception as e:  # pragma: no cover
            print(f"[wandb] log failed: {e!r}")

    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """Log a free-text artefact (e.g. a generation sample)."""
        if not self._enabled:
            return
        try:
            import wandb  # type: ignore
            wandb.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)
        except Exception as e:  # pragma: no cover
            print(f"[wandb] log_text failed: {e!r}")

    def log_summary(self, **kwargs: Any) -> None:
        """Update the run's summary panel with final metrics."""
        if not self._enabled:
            return
        try:
            for k, v in kwargs.items():
                self._run.summary[k] = v
        except Exception as e:  # pragma: no cover
            print(f"[wandb] summary failed: {e!r}")

    def save_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        """Optional: upload a checkpoint as a WandB artifact."""
        if not self._enabled or not os.path.exists(path):
            return
        try:
            import wandb  # type: ignore
            art = wandb.Artifact(name=name, type=artifact_type)
            art.add_file(path)
            self._run.log_artifact(art)
        except Exception as e:  # pragma: no cover
            print(f"[wandb] artifact failed: {e!r}")

    def finish(self) -> None:
        if not self._enabled:
            return
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:  # pragma: no cover
            pass
