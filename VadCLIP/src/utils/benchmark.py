import time
from dataclasses import dataclass
from typing import List, Sequence, Union, Dict, Any


try:  # Optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


DeviceLike = Union[str, object, None]


def _iter_cuda_devices(devices: Sequence[DeviceLike]) -> List[Any]:
    if torch is None:
        return []

    cuda_devices: List[Any] = []
    for d in devices:
        if d is None:
            continue
        try:
            td = d if isinstance(d, torch.device) else torch.device(str(d))
        except Exception:
            continue
        if td.type == "cuda":
            cuda_devices.append(td)
    return cuda_devices


def synchronize_if_cuda(devices: Sequence[DeviceLike]) -> None:
    """Synchronize CUDA streams for accurate wall-clock timing.

    Safe no-op on CPU-only, missing torch, or non-cuda devices.
    """

    if torch is None or (not torch.cuda.is_available()):
        return

    for td in _iter_cuda_devices(devices):
        try:
            torch.cuda.synchronize(td)
        except Exception:
            # Be permissive: timing should never crash evaluation.
            pass


@dataclass
class TimingResult:
    durations_s: List[float]

    @property
    def mean_s(self) -> float:
        return float(sum(self.durations_s) / max(1, len(self.durations_s)))

    def percentile_s(self, p: float) -> float:
        if not self.durations_s:
            return 0.0
        if p <= 0:
            return float(min(self.durations_s))
        if p >= 100:
            return float(max(self.durations_s))

        xs = sorted(self.durations_s)
        k = (len(xs) - 1) * (p / 100.0)
        f = int(k)
        c = min(len(xs) - 1, f + 1)
        if f == c:
            return float(xs[f])
        return float(xs[f] + (xs[c] - xs[f]) * (k - f))

    def to_json(self) -> Dict[str, Any]:
        return {
            "count": len(self.durations_s),
            "mean_s": self.mean_s,
            "p50_s": self.percentile_s(50),
            "p90_s": self.percentile_s(90),
            "p95_s": self.percentile_s(95),
            "min_s": float(min(self.durations_s)) if self.durations_s else 0.0,
            "max_s": float(max(self.durations_s)) if self.durations_s else 0.0,
        }


def time_callable(
    fn,
    *,
    devices: Sequence[DeviceLike] = (),
    warmup: int = 0,
    repeat: int = 1,
) -> TimingResult:
    """Time `fn()` with optional warmup and CUDA synchronization.

    Returns:
      TimingResult with `repeat` durations (warmup excluded).
    """

    warmup = max(0, int(warmup))
    repeat = max(1, int(repeat))

    for _ in range(warmup):
        synchronize_if_cuda(devices)
        _ = fn()
        synchronize_if_cuda(devices)

    durations: List[float] = []
    for _ in range(repeat):
        synchronize_if_cuda(devices)
        t0 = time.perf_counter()
        _ = fn()
        synchronize_if_cuda(devices)
        t1 = time.perf_counter()
        durations.append(float(t1 - t0))

    return TimingResult(durations_s=durations)
