"""GPU residency and pre-flight for the models of a profile.

The owner's serving hardware is one or more 12 GB GPUs with 16 GB system RAM. Not every model can
be resident at once (reranker + verifier + 4-bit 8B generator ≈ 8–9 GB, a 14B at 4-bit does not
fit next to them). ``ModelPool`` applies the residency policy from configs/resources.yaml:
``resident`` roles are built once and kept; ``on_demand`` roles are built when acquired and freed
when released (``torch.cuda.empty_cache`` when torch is present). ``preflight`` sums the VRAM
estimates and refuses a profile that cannot fit, instead of failing on the first request.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Any

from app.core.admission import Admission
from app.core.budget import ResourceConfig
from models.registry.loader import ModelRegistry

log = logging.getLogger("litbiorag.pool")


class PreflightError(RuntimeError):
    pass


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # noqa: BLE001
        return None


class ModelPool:
    def __init__(self, registry: ModelRegistry, resources: ResourceConfig | None = None):
        self.registry = registry
        self.res = resources or ResourceConfig.from_yaml()
        gpu = self.res.gpu or {}
        self.residency: dict[str, str] = dict(gpu.get("residency") or {})
        self.estimates: dict[str, float] = {
            k: float(v) for k, v in (gpu.get("vram_estimate_gb") or {}).items()
        }
        self.vram_gb = float(gpu.get("vram_gb") or 0.0)
        self._resident: dict[str, Any] = {}
        self._loads: dict[str, int] = {}
        # admission control: how many requests may hold each role at once (H1)
        self.admission = Admission.from_config(self.res.concurrency)

    # ---- policy ---------------------------------------------------------------------------
    def policy(self, role: str) -> str:
        return self.residency.get(role, "resident")

    def estimate_gb(self, role: str) -> float:
        spec = self.registry.spec(role)
        if spec.is_test_double or spec.kind in ("openai_compatible", "anthropic"):
            return 0.0
        v = spec.params.get("vram_gb")
        return float(v) if v is not None else self.estimates.get(role, 0.0)

    def preflight(self) -> dict[str, Any]:
        """Peak VRAM = all resident roles + the largest on_demand role. Raises when > vram_gb."""
        roles = self.registry.roles()
        resident = {r: self.estimate_gb(r) for r in roles if self.policy(r) == "resident"}
        on_demand = {r: self.estimate_gb(r) for r in roles if self.policy(r) != "resident"}
        peak = sum(resident.values()) + (max(on_demand.values()) if on_demand else 0.0)
        report = {
            "resident_gb": resident, "on_demand_gb": on_demand, "peak_gb": round(peak, 2),
            "vram_gb": self.vram_gb, "fits": not self.vram_gb or peak <= self.vram_gb,
        }  # fmt: skip
        if not report["fits"]:
            raise PreflightError(
                f"profile needs {peak:.1f} GB peak VRAM but the device has {self.vram_gb:.1f} GB; "
                f"move a role to on_demand, quantize, or use a hosted generator: {report}"
            )
        return report

    # ---- lifecycle ------------------------------------------------------------------------
    def acquire(self, role: str) -> Any:
        if role in self._resident:
            return self._resident[role]
        obj = self.registry.build(role, cache=False)
        self._loads[role] = self._loads.get(role, 0) + 1
        if self.policy(role) == "resident":
            self._resident[role] = obj
        return obj

    def release(self, role: str, obj: Any) -> None:
        if self.policy(role) == "resident":
            return
        del obj
        gc.collect()
        t = _torch()
        if t is not None and t.cuda.is_available():
            t.cuda.empty_cache()

    @contextmanager
    def use(self, role: str, *, on_queued=None):
        """Admit (bounded concurrency, FIFO queue), then acquire; raises ``Overloaded`` when the
        role's queue is full or the wait times out, before any model is loaded."""
        with self.admission.admit(role, on_queued=on_queued):
            obj = self.acquire(role)
            try:
                yield obj
            finally:
                self.release(role, obj)

    @property
    def load_counts(self) -> dict[str, int]:
        return dict(self._loads)

    def vram_report(self) -> dict[str, float] | None:
        t = _torch()
        if t is None or not t.cuda.is_available():
            return None
        free, total = t.cuda.mem_get_info()
        return {"free_gb": free / 2**30, "total_gb": total / 2**30}
