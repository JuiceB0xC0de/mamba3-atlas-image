"""B0-2: online per-(layer, head, class, block) statistics for Stage B.

Everything here is O(1) in token count. Nothing stores per-token tensors.

WHY BLOCK-KEYED (decision D1 in DESIGN-stage-b-c-buildplan.md):
  Label-permutation nulls and document-block bootstrap are ANALYSIS-time
  operations that need CAPTURE-time support. Pooling per class destroys the
  block structure they require, and every null would then cost another GPU run.
  So per-block scalars are recorded during capture; the nulls are free after.

MEMORY BUDGET, the reason for the split below:
  Stream B has ~11.3k blocks (one per prompt). With ~10 captured layers and 64
  heads that is 7.2M cells per scalar quantity: fine at ~29 MB float32 each.
  A per-block 4x4 Gram would be 116M floats (~464 MB) and is NOT worth it.

  Therefore:
    per BLOCK  -> scalar summaries only (means over the block's tokens).
                  Enough for bootstrap and permutation of those scalars.
    per CLASS  -> the rich objects: Grams, histograms, reservoirs.

Quantiles come from fixed-edge histograms rather than exact sorting, because
exact per-head quantiles would require keeping every token. Edges are DECLARED
UP FRONT per quantity (see DEFAULT_EDGES) and recorded in the artifact; a
quantity that saturates its outermost bin is reported as censored, never
silently clipped.
"""

from __future__ import annotations

import numpy as np
import torch

# --------------------------------------------------------------------------
# edge definitions
# --------------------------------------------------------------------------


def linear_edges(lo: float, hi: float, bins: int = 128) -> torch.Tensor:
    return torch.linspace(lo, hi, bins + 1)


def log_edges(lo: float, hi: float, bins: int = 128) -> torch.Tensor:
    """Log-spaced edges, for heavy-tailed quantities.

    dt and the retention horizons are heavy-tailed across heads: on siso-187m L0
    the mean half-life is 145 tokens while the median is 2.3. Linear bins would
    put almost every observation in the first bin.
    """
    return torch.logspace(np.log10(lo), np.log10(hi), bins + 1)


# Declared ranges per captured quantity. Recorded in the artifact so a reader
# can tell a real value from a binning artifact.
DEFAULT_EDGES = {
    "lambda": ("linear", 0.0, 1.0),          # sigmoid output, bounded
    "alpha": ("linear", 0.0, 1.0),           # exp(A*dt), bounded above by 1
    "Delta": ("log", 1e-4, 1e2),             # softplus, heavy tailed
    "trap_scale": ("log", 1e-5, 1e3),        # sum of two dt-scaled terms
    "local_halflife": ("log", 1e-2, 1e5),    # ln2 / -ADT, very heavy tailed
    "retention_horizon": ("log", 1e0, 1e5),  # positions, censored at window end
    "diff_energy": ("linear", 0.0, 1.0),     # a ratio in [0,1]
    "resid_absmean": ("log", 1e-4, 1e3),
    "pathway_norm": ("log", 1e-6, 1e4),
}


def make_edges(name: str, bins: int = 128) -> torch.Tensor:
    kind, lo, hi = DEFAULT_EDGES[name]
    return linear_edges(lo, hi, bins) if kind == "linear" else log_edges(lo, hi, bins)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


class StreamingMoments:
    """Vectorized Welford over a leading (L, H) grid.

    update() takes (L, H, T) and folds T observations per cell.
    """

    def __init__(self, shape, device="cpu", dtype=torch.float64):
        self.count = torch.zeros(shape, device=device, dtype=dtype)
        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        self.m2 = torch.zeros(shape, device=device, dtype=dtype)
        self.absum = torch.zeros(shape, device=device, dtype=dtype)
        self.vmin = torch.full(shape, float("inf"), device=device, dtype=dtype)
        self.vmax = torch.full(shape, float("-inf"), device=device, dtype=dtype)

    def update(self, x: torch.Tensor) -> None:
        """x: (L, H, T). Chunked Welford, exact for any T."""
        x = x.to(self.mean.dtype)
        n_b = x.shape[-1]
        if n_b == 0:
            return
        mean_b = x.mean(dim=-1)
        m2_b = ((x - mean_b.unsqueeze(-1)) ** 2).sum(dim=-1)

        n_a = self.count
        delta = mean_b - self.mean
        n_tot = n_a + n_b
        # guard the first update, where n_a == 0
        self.mean = torch.where(n_tot > 0, self.mean + delta * (n_b / n_tot.clamp_min(1)), self.mean)
        self.m2 = self.m2 + m2_b + delta**2 * n_a * n_b / n_tot.clamp_min(1)
        self.count = n_tot
        self.absum += x.abs().sum(dim=-1)
        self.vmin = torch.minimum(self.vmin, x.amin(dim=-1))
        self.vmax = torch.maximum(self.vmax, x.amax(dim=-1))

    def finish(self) -> dict:
        var = self.m2 / self.count.clamp_min(1)
        return {
            "count": self.count.cpu().numpy(),
            "mean": self.mean.cpu().numpy(),
            "std": var.sqrt().cpu().numpy(),
            "absmean": (self.absum / self.count.clamp_min(1)).cpu().numpy(),
            "min": self.vmin.cpu().numpy(),
            "max": self.vmax.cpu().numpy(),
        }


class Histogram:
    """Fixed-edge histogram per (L, H) cell, for quantiles without storing tokens.

    Tracks under/overflow explicitly so saturation is visible as censoring
    rather than silently piling into the end bins.
    """

    def __init__(self, shape, edges: torch.Tensor, device="cpu"):
        self.edges = edges.to(device)
        self.bins = len(edges) - 1
        self.counts = torch.zeros((*shape, self.bins), device=device, dtype=torch.float64)
        self.under = torch.zeros(shape, device=device, dtype=torch.float64)
        self.over = torch.zeros(shape, device=device, dtype=torch.float64)

    def update(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """x: (L, H, T); optional boolean mask with the identical shape.

        Masking is vectorized so censored observations can be excluded without
        per-head Python branches or GPU-to-CPU synchronization.
        """
        idx = torch.bucketize(x.to(self.edges.dtype).contiguous(), self.edges) - 1
        observed = (torch.ones_like(idx, dtype=torch.bool) if mask is None
                    else mask.to(device=idx.device, dtype=torch.bool))
        if observed.shape != idx.shape:
            raise ValueError(f"histogram mask shape {tuple(observed.shape)} != "
                             f"values shape {tuple(idx.shape)}")
        self.under += ((idx < 0) & observed).sum(dim=-1)
        self.over += ((idx >= self.bins) & observed).sum(dim=-1)
        valid = (idx >= 0) & (idx < self.bins) & observed
        clamped = idx.clamp(0, self.bins - 1)
        L, H, _ = x.shape
        flat = self.counts.view(L * H, self.bins)
        off = (
            torch.arange(L * H, device=x.device).unsqueeze(-1) * self.bins
        ) + clamped.view(L * H, -1)
        # Fixed-size weighted scatter: boolean indexing would materialize a
        # dynamic-length CUDA tensor and can force a host-visible size query.
        # Invalid/censored observations use zero weight at their clamped bin.
        selected = off.reshape(-1)
        weights = valid.reshape(-1).to(torch.float64)
        flat.view(-1).index_add_(
            0, selected, weights)

    def quantile(self, q: float) -> torch.Tensor:
        """Linear interpolation inside the containing bin. Returns (L, H)."""
        total = self.counts.sum(-1) + self.under + self.over
        target = q * total
        cum = torch.cumsum(self.counts, dim=-1) + self.under.unsqueeze(-1)
        idx = (cum < target.unsqueeze(-1)).sum(-1).clamp(max=self.bins - 1)
        lo = self.edges[:-1][idx]
        hi = self.edges[1:][idx]
        below = torch.where(
            idx > 0, cum.gather(-1, (idx - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1),
            self.under,
        )
        inbin = self.counts.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        frac = ((target - below) / inbin.clamp_min(1e-12)).clamp(0, 1)
        return lo + frac * (hi - lo)

    def censored_fraction(self) -> torch.Tensor:
        total = self.counts.sum(-1) + self.under + self.over
        return (self.under + self.over) / total.clamp_min(1)


class GramAccum:
    """Per-head Gram across the rank axis, accumulated over tokens.

    Pooled per class, NOT per block: a per-block 4x4 Gram would cost ~464 MB
    on stream B for no analysis we actually run.
    """

    def __init__(self, nheads, rank, device="cpu"):
        self.g = torch.zeros((nheads, rank, rank), device=device, dtype=torch.float64)
        self.n = 0

    def update(self, v: torch.Tensor) -> None:
        """v: (T, nheads, rank, d_state). Normalized along d_state first."""
        vn = torch.nn.functional.normalize(v.float(), dim=-1)
        self.g += torch.einsum("thrd,thsd->hrs", vn, vn).double()
        self.n += v.shape[0]

    def finish(self) -> np.ndarray:
        return (self.g / max(self.n, 1)).cpu().numpy()


class Reservoir:
    """Algorithm R uniform sample, bounded, for spot checks and sanity plots."""

    def __init__(self, k: int, seed: int = 0):
        self.k = k
        self.buf: list = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        for v in np.asarray(values).reshape(-1):
            self.seen += 1
            if len(self.buf) < self.k:
                self.buf.append(v)
            else:
                j = self.rng.integers(0, self.seen)
                if j < self.k:
                    self.buf[int(j)] = v

    def finish(self) -> np.ndarray:
        return np.asarray(self.buf)


class BlockRecorder:
    """Per-block scalar summaries: the substrate for every null.

    One row per block, each row (L, H) per registered quantity. Blocks are
    prompts (stream B) or documents (stream A). Resampling and label permutation
    operate on these rows afterward, on CPU, at no GPU cost.

    Denominators are PER CELL, not one shared scalar. A caller may contribute
    one layer's row at a time (add(..., lp=slot)) and only that row's count
    advances. The previous design zero-filled a full (L, H, T) tensor per layer
    and advanced a shared count on every call, dividing every layer mean by the
    number of selected layers. That is exactly why counts are now per cell.

    Device-side staging: accumulation happens on `device` and end() performs
    ONE batched D2H copy per block (sums and counts packed together) instead of
    one per add() call.
    """

    def __init__(self, quantities, n_layers, n_heads, device="cpu",
                 n_blocks=None):
        self.quantities = list(quantities)
        self.shape = (n_layers, n_heads)
        self.device = device
        self.n_blocks = n_blocks
        self.rows = ({q: np.zeros((n_blocks, *self.shape), np.float32)
                      for q in self.quantities}
                     if n_blocks is not None else
                     {q: [] for q in self.quantities})
        self.meta: list = []
        self.d2h = 0                    # batched D2H copies performed
        self.denominator_failures = 0   # blocks whose per-cell counts disagree
        self._cur = None

    def put_meta(self, row: int, meta: dict) -> None:
        if row != len(self.meta):
            raise RuntimeError(
                f"BlockRecorder metadata order drift: row={row}, "
                f"next={len(self.meta)}")
        self.meta.append(meta)

    def put_batch_layer(self, lp: int, records: list[dict]) -> None:
        """Segment-reduce one packed forward directly into block rows.

        Each record supplies ``row``, ``n`` and token-level ``extra`` values.
        Contiguous prefix differences compute every block mean per quantity;
        the complete quantity packet crosses to CPU once for this layer/batch.
        """
        if self.n_blocks is None:
            raise RuntimeError("put_batch_layer requires n_blocks at construction")
        if not records:
            return
        lengths = torch.tensor([r["n"] for r in records], device=self.device,
                               dtype=torch.long)
        ends = lengths.cumsum(0)
        starts = ends - lengths
        means = []
        for q in self.quantities:
            x = torch.cat([r["extra"][q] for r in records], dim=0).float()
            prefix = torch.cat((torch.zeros_like(x[:1]), x.cumsum(0)), dim=0)
            means.append((prefix[ends] - prefix[starts])
                         / lengths.to(x.dtype).unsqueeze(-1))
        packet = torch.stack(means).cpu().numpy().astype(np.float32)
        self.d2h += 1
        rows = np.asarray([r["row"] for r in records], dtype=np.int64)
        for qi, q in enumerate(self.quantities):
            self.rows[q][rows, lp] = packet[qi]

    def begin(self, meta: dict) -> None:
        self._cur = {
            q: (torch.zeros(self.shape, dtype=torch.float64, device=self.device),
                torch.zeros(self.shape, dtype=torch.float64, device=self.device))
            for q in self.quantities
        }
        self._meta = meta

    def add(self, name: str, x: torch.Tensor, lp: int = None) -> None:
        """x: (L, H, T) full rows, or (H, T) for one layer's row when lp is given.

        With lp, ONLY that row's accumulator and count advance. Callers that
        zero-fill a full tensor and rely on a shared scalar count dilute every
        mean by the layer count; pass the real row instead.
        """
        if self._cur is None or name not in self._cur:
            return
        acc, cnt = self._cur[name]
        if lp is None:
            acc += x.to(torch.float64).sum(dim=-1)
            cnt += x.shape[-1]
        else:
            acc[lp] += x.to(torch.float64).sum(dim=-1)
            cnt[lp] += x.shape[-1]

    def end(self) -> None:
        """ONE batched D2H for the whole quantity set: sums and counts packed."""
        qs = self.quantities
        acc = torch.stack([self._cur[q][0] for q in qs])      # (n_q, L, H)
        cnt = torch.stack([self._cur[q][1] for q in qs])
        pack = torch.cat([acc.reshape(-1), cnt.reshape(-1)]).cpu().numpy()
        self.d2h += 1
        half = pack.size // 2
        sums = pack[:half].reshape(len(qs), *self.shape)
        cnts = pack[half:].reshape(len(qs), *self.shape)
        expected = cnts[0, 0, 0]
        if expected <= 0 or not np.all(cnts == expected):
            self.denominator_failures += 1
            raise RuntimeError(
                "BlockRecorder incomplete block: per-quantity/layer/head token "
                f"counts disagree (min={cnts.min()}, max={cnts.max()})")
        means = (sums / np.maximum(cnts, 1)).astype(np.float32)
        for i, q in enumerate(qs):
            self.rows[q].append(means[i])
        self.meta.append(self._meta)
        self._cur = None

    def finish(self) -> dict:
        if self.n_blocks is not None:
            return dict(self.rows)
        return {q: np.stack(v) if v else np.zeros((0, *self.shape), np.float32)
                for q, v in self.rows.items()}
