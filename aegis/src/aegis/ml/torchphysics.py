"""Differentiable re-implementations of the AEGIS physics in PyTorch.

Why this module exists
----------------------
The physics-informed conjunction GNN (contract section 15) is trained with a
loss that pushes its proposed burns through the *same* linearised dynamics,
B-plane geometry and Alfano probability integral that the certified
optimizer uses. That only works if two things hold at once: the torch
versions must agree with the numpy originals to well inside the tolerances
the optimizer itself works to, and every function must have a finite
gradient at every input the originals accept. Both are verified against the
originals rather than trusted, and the numbers are recorded in the
verification report that accompanies this module.

Each function here is a batched mirror of one specific numpy function, named
in its docstring. The originals remain the source of truth: this module
imports their private tolerances (``_N_FLOOR``, ``_ZERO``, the bisection
sentinel and iteration count) and their quadrature nodes rather than
restating them, so the two implementations cannot drift apart silently.

Conventions
-----------
* Every function accepts tensors or plain floats, broadcasts over any number
  of leading batch dimensions, and computes in the dtype of its tensor
  inputs (plain floats are promoted to ``float64``). ``torch.set_default_dtype``
  is never called: the caller owns the precision decision, and the
  verification that backs this module was run in ``float64``.
* Units are the codebase's: kilometres, km/s, seconds, radians per second.
* Degenerate branches (non-positive lead time, vanishing relative velocity,
  zero hard-body radius, clamped roots) are selected with ``torch.where`` on
  a *safe surrogate* -- the risky expression is evaluated on a harmless
  stand-in value and then discarded -- rather than by masking after the
  fact. Masking after a division by zero leaves ``0 * inf = NaN`` in the
  backward graph even when the forward value is correct; the surrogate
  pattern is what keeps the gradients finite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from ..constants import ALFANO_QUAD_ORDER, PC_TARGET_POST_MANEUVER
from ..fleetopt.bplane import _ZERO
from ..fleetopt.dynamics import _N_FLOOR
from ..fleetopt.errors import FleetOptError
from ..maneuver.miss import _SEARCH_ITERS, _SENTINEL_MISS_KM
from ..risk.alfano import _gauss_chebyshev_nodes

__all__ = [
    "cw_impulse_matrix_torch",
    "bplane_projector_torch",
    "collision_probability_torch",
    "log_collision_probability_torch",
    "max_collision_probability_torch",
    "dilution_sigma_torch",
    "required_miss_torch",
    "miss_vector_after_torch",
    "miss_after_torch",
    "ConstraintResiduals",
    "constraint_residuals_torch",
    "physics_loss_torch",
]

_SQRT_2 = math.sqrt(2.0)
_LOG_2 = math.log(2.0)

#: ``required_miss_distance_km`` refuses to search below this target when the
#: hard-body radius dominates the covariance; mirrored verbatim.
_HBR_DOMINATED_TARGET = 1e-16
_HBR_DOMINATED_RATIO = 3.0


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


def _coerce(*values: Tensor | float) -> tuple[Tensor, ...]:
    """Promote plain numbers to tensors, unify dtype and device, broadcast.

    Plain floats become ``float64`` because that is the precision the numpy
    originals run in; if the caller has already committed to ``float32``
    tensors the promotion rules pick ``float32`` for the arithmetic, which
    is the caller's choice to make, not this module's.
    """
    tensors = [v for v in values if isinstance(v, Tensor)]
    device = tensors[0].device if tensors else torch.device("cpu")
    dtype = torch.float64
    if tensors:
        dtype = tensors[0].dtype
        for t in tensors[1:]:
            dtype = torch.promote_types(dtype, t.dtype)
        if not dtype.is_floating_point:
            dtype = torch.float64
    converted = [
        v.to(dtype=dtype) if isinstance(v, Tensor) else torch.as_tensor(v, dtype=dtype, device=device)
        for v in values
    ]
    return tuple(torch.broadcast_tensors(*converted))


def _safe_norm(vector: Tensor, dim: int = -1) -> Tensor:
    """Euclidean norm whose gradient at the origin is zero rather than NaN.

    ``torch.linalg.norm`` differentiates ``sqrt(0)`` to ``inf`` and then
    multiplies by zero. Zero is a valid subgradient of the norm at the
    origin (the subdifferential is the unit ball), so the surrogate returns
    the right value and a usable gradient.
    """
    squared = torch.sum(vector * vector, dim=dim)
    positive = squared > 0
    safe = torch.where(positive, squared, torch.ones_like(squared))
    return torch.where(positive, torch.sqrt(safe), torch.zeros_like(squared))


# ---------------------------------------------------------------------------
# Clohessy-Wiltshire impulse block (mirrors fleetopt.dynamics.cw_impulse_matrix)
# ---------------------------------------------------------------------------


def cw_impulse_matrix_torch(n_rad_s: Tensor | float, lead_s: Tensor | float) -> Tensor:
    """Batched :func:`aegis.fleetopt.dynamics.cw_impulse_matrix`.

    Returns ``Phi(n, sigma)`` with shape ``(*batch, 3, 3)`` where ``batch``
    is the broadcast of the two inputs. Rows are RTN displacement, columns
    RTN impulse, exactly as in the original.

    Three regimes, selected with ``torch.where`` on safe surrogates so that
    the backward pass is finite in all of them:

    * ``lead_s <= 0`` returns zeros (an impulse cannot act before it is
      applied). The trigonometric block is still evaluated at ``sigma = 1``
      and discarded, so no ``0 * d(sin)/d(sigma)`` ambiguity reaches
      autograd. The derivative with respect to ``lead_s`` on that branch is
      zero, which is the correct one-sided derivative of a clamped function.
    * ``|n| < 1e-15`` degenerates to ``sigma * I3``; the division by ``n``
      happens on a stand-in ``n = 1``.
    * Otherwise the exact CW block, with the secular ``-3 n sigma`` term in
      the transverse-transverse entry.
    """
    n, sigma = _coerce(n_rad_s, lead_s)
    ones = torch.ones_like(sigma)
    zeros = torch.zeros_like(sigma)

    positive = sigma > 0
    sigma_safe = torch.where(positive, sigma, ones)
    rectilinear = torch.abs(n) < _N_FLOOR
    n_safe = torch.where(rectilinear, ones, n)

    nt = n_safe * sigma_safe
    s = torch.sin(nt)
    c = torch.cos(nt)
    block = torch.stack(
        [
            torch.stack([s / n_safe, 2.0 * (1.0 - c) / n_safe, zeros], dim=-1),
            torch.stack([2.0 * (c - 1.0) / n_safe, (4.0 * s - 3.0 * nt) / n_safe, zeros], dim=-1),
            torch.stack([zeros, zeros, s / n_safe], dim=-1),
        ],
        dim=-2,
    )

    eye = torch.eye(3, dtype=sigma.dtype, device=sigma.device)
    rect_block = sigma_safe[..., None, None] * eye
    phi = torch.where(rectilinear[..., None, None], rect_block, block)
    return torch.where(positive[..., None, None], phi, torch.zeros_like(phi))


# ---------------------------------------------------------------------------
# B-plane projector (mirrors fleetopt.bplane.bplane_projector)
# ---------------------------------------------------------------------------


def bplane_projector_torch(relative_velocity_km_s: Tensor) -> Tensor:
    """Batched :func:`aegis.fleetopt.bplane.bplane_projector`.

    ``P = I3 - w w^T / |w|^2`` for input shape ``(*batch, 3)``, output
    ``(*batch, 3, 3)``. The projector is written with ``|w|^2`` rather than
    the normalised ``w_hat`` so no square root sits on the gradient path;
    the numpy original's threshold ``|w| < 1e-12`` is applied as
    ``|w|^2 < 1e-24``, which is the same test without the root.

    A vanishing relative velocity returns the identity, for the reason the
    original gives: with no encounter plane every displacement direction
    counts, which is the conservative choice. The division is performed on
    a stand-in ``|w|^2 = 1`` on that branch so its gradient is finite (and
    zero, matching the constant identity output).
    """
    w = relative_velocity_km_s
    if not isinstance(w, Tensor):
        w = torch.as_tensor(w, dtype=torch.float64)
    if w.shape[-1] != 3:
        raise ValueError(f"relative velocity must have a trailing dimension of 3, got {tuple(w.shape)}")
    speed_sq = torch.sum(w * w, dim=-1)
    degenerate = speed_sq < _ZERO * _ZERO
    speed_sq_safe = torch.where(degenerate, torch.ones_like(speed_sq), speed_sq)
    outer = w[..., :, None] * w[..., None, :] / speed_sq_safe[..., None, None]
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(outer.shape)
    return torch.where(degenerate[..., None, None], eye, eye - outer)


# ---------------------------------------------------------------------------
# Alfano collision probability (mirrors risk.alfano.collision_probability)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _nodes(order: int, dtype: torch.dtype, device: str) -> tuple[Tensor, Tensor, Tensor]:
    """Gauss-Chebyshev nodes as tensors, taken from the numpy original.

    Reusing :func:`aegis.risk.alfano._gauss_chebyshev_nodes` rather than
    recomputing ``cos``/``sin`` in torch guarantees the two quadratures use
    bit-identical abscissae and weights, which is what makes the agreement
    below 1e-6 relative a property of ``erfc`` alone.
    """
    t, y, w = _gauss_chebyshev_nodes(order)
    to = dict(dtype=dtype, device=torch.device(device))
    return torch.as_tensor(t, **to), torch.as_tensor(y, **to), torch.as_tensor(w, **to)


def _alfano_terms(
    sigma_major: Tensor,
    sigma_minor: Tensor,
    miss_x: Tensor,
    miss_z: Tensor,
    hbr_safe: Tensor,
    order: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The per-node pieces of Alfano's integrand, shape ``(*batch, order)``."""
    t, y, w = _nodes(order, sigma_major.dtype, str(sigma_major.device))
    exponent = -(((miss_x[..., None] + hbr_safe[..., None] * t) / (sigma_major[..., None] * _SQRT_2)) ** 2)
    chord = hbr_safe[..., None] * y
    upper = (miss_z[..., None] + chord) / (sigma_minor[..., None] * _SQRT_2)
    lower = (miss_z[..., None] - chord) / (sigma_minor[..., None] * _SQRT_2)
    return w, exponent, upper, lower, chord


def _check_alfano_inputs(sigma_major: Tensor, sigma_minor: Tensor, hbr: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if bool(torch.any(sigma_major <= 0) | torch.any(sigma_minor <= 0)):
        raise ValueError("covariance standard deviations must be positive")
    zero_result = ~(hbr > 0)
    certain = torch.isinf(hbr) & (hbr > 0)
    hbr_safe = torch.where(zero_result | certain, torch.ones_like(hbr), hbr)
    return zero_result, certain, hbr_safe


def collision_probability_torch(
    sigma_major: Tensor | float,
    sigma_minor: Tensor | float,
    miss_x: Tensor | float,
    miss_z: Tensor | float,
    hard_body_radius: Tensor | float,
    *,
    order: int = ALFANO_QUAD_ORDER,
) -> Tensor:
    """Batched :func:`aegis.risk.alfano.collision_probability`.

    Same Gauss-Chebyshev quadrature of Alfano's one-dimensional integral,
    same nodes, same operation order, so the two agree to the precision of
    the ``erfc`` implementations (measured below 1e-12 relative over
    ``Pc`` in ``[1e-12, 1e-2]``; the contract asks for 1e-6).

    Tail behaviour, and why ``erfc`` rather than ``erf``
    ---------------------------------------------------
    The chord term is ``erf(upper) - erf(lower)``. Once the miss along the
    minor axis exceeds a few sigma both ``erf`` values round to exactly 1
    and the difference is exactly zero -- the result is destroyed, not
    merely imprecise. The original therefore evaluates the identical
    quantity as ``erfc(lower) - erfc(upper)``, which keeps its relative
    precision down to about ``1e-300``; ``torch.special.erfc`` does the
    same, and its backward pass is ``-2/sqrt(pi) * exp(-x^2)``, which
    underflows gracefully to zero rather than to NaN. That is why no
    gradient here can be non-finite: every intermediate is an ``exp`` of a
    finite negative number, an ``erfc`` of a finite argument, or a product
    of those, and the only divisions are by the (positive, checked) sigmas.

    The ``erfc`` identity is one-sided. For a *negative* ``miss_z`` large
    against ``sigma_minor`` both arguments are negative, both ``erfc``
    values saturate at 2, and tail terms below ``2e-16`` absolute are lost
    exactly as in the original. That is deliberately reproduced here rather
    than fixed, because (a) :mod:`aegis.risk.projection` takes the absolute
    value of both miss components before they reach Alfano, so the pipeline
    never exercises the negative tail, and (b) the contract measures this
    function against the original, and the lost terms are the smallest in
    the sum. :func:`log_collision_probability_torch` uses the evenness of
    the integral in both miss components instead and is symmetric.

    For the regime beyond ``Pc ~ 1e-300``, where even ``erfc`` underflows
    and the gradient of ``Pc`` is legitimately zero, use the log form,
    which stays informative there.

    Degenerate hard-body radii follow the original: ``hbr <= 0`` yields
    exactly 0 and ``hbr = +inf`` exactly 1, both selected by ``torch.where``
    on a surrogate ``hbr = 1`` so the quadrature never sees them.
    """
    sigma_major, sigma_minor, miss_x, miss_z, hbr = _coerce(
        sigma_major, sigma_minor, miss_x, miss_z, hard_body_radius
    )
    zero_result, certain, hbr_safe = _check_alfano_inputs(sigma_major, sigma_minor, hbr)

    w, exponent, upper, lower, _ = _alfano_terms(sigma_major, sigma_minor, miss_x, miss_z, hbr_safe, order)
    gaussian = torch.exp(exponent)
    chord_term = torch.special.erfc(lower) - torch.special.erfc(upper)
    total = torch.sum(w * gaussian * chord_term, dim=-1)
    probability = (hbr_safe / sigma_major) * total
    probability = torch.clamp(probability, 0.0, 1.0)

    probability = torch.where(certain, torch.ones_like(probability), probability)
    return torch.where(zero_result, torch.zeros_like(probability), probability)


def _log_erfc(x: Tensor) -> Tensor:
    """``log(erfc(x))`` without underflow, via ``log_ndtr``.

    ``erfc(x) = 2 * Phi(-x sqrt 2)`` where ``Phi`` is the standard normal
    CDF, and ``torch.special.log_ndtr`` evaluates ``log Phi`` with an
    asymptotic expansion in the tail. Measured against
    ``scipy.special.log_ndtr`` the absolute error is below 1e-12 out to
    ``x = 26``, i.e. the same relative precision on ``Pc`` as the direct
    path, but with no floor at 1e-300.
    """
    return _LOG_2 + torch.special.log_ndtr(-_SQRT_2 * x)


def log_collision_probability_torch(
    sigma_major: Tensor | float,
    sigma_minor: Tensor | float,
    miss_x: Tensor | float,
    miss_z: Tensor | float,
    hard_body_radius: Tensor | float,
    *,
    order: int = ALFANO_QUAD_ORDER,
) -> Tensor:
    """Natural log of :func:`collision_probability_torch`, computed in log space.

    The model consumes ``log10 Pc`` as a feature and target, and a loss on
    ``log Pc`` needs a gradient that does not vanish with ``Pc`` itself.
    Taking ``log`` of the direct result would return ``-inf`` with a NaN
    gradient once ``Pc`` underflows; instead each quadrature term is formed
    as ``log w + exponent + log(erfc(lower) - erfc(upper))`` and combined
    with ``logsumexp``. The chord log-difference is
    ``log_erfc(lower) + log(-expm1(log_erfc(upper) - log_erfc(lower)))``,
    valid because ``upper > lower`` strictly at every interior node
    (``chord > 0`` there), so the ``expm1`` argument is strictly negative.
    ``expm1`` rather than ``1 - exp`` keeps the difference accurate when
    the two ``erfc`` values are close, which is the small-chord regime.

    Both miss components enter as absolute values. Alfano's integrand is
    even in each of them (the disc and the Gaussian are both symmetric
    about the principal axes), so the value is unchanged; what changes is
    that ``upper`` is then always positive and the two ``erfc`` values can
    never both saturate at 2 -- the one-sided loss described on
    :func:`collision_probability_torch` would otherwise turn into an exact
    ``log(0)`` with a NaN gradient for a negative ``miss_z`` beyond about
    six ``sigma_minor``. The gradient with respect to a miss component
    picks up the sign through ``abs``; at exactly zero it is zero, which is
    the true derivative of an even function there.

    Known limit: when the chord is so small relative to the minor-axis miss
    that the two ``log_erfc`` values round to the same double
    (``hbr / sigma_minor`` below about 1e-15 at the node), the term
    collapses to ``-inf`` and its gradient to ``-inf``; the direct path has
    the same loss of information there. No such geometry occurs in LEO
    conjunction assessment (hard bodies are metres, sigmas are at most
    tens of kilometres) and the verification samples do not include it.

    ``hbr <= 0`` returns ``-inf`` (log of the original's exact zero) with a
    zero gradient; ``hbr = +inf`` returns 0. The clamp to ``[0, 1]`` in the
    original becomes a clamp of the log to ``(-inf, 0]``.
    """
    sigma_major, sigma_minor, miss_x, miss_z, hbr = _coerce(
        sigma_major, sigma_minor, miss_x, miss_z, hard_body_radius
    )
    zero_result, certain, hbr_safe = _check_alfano_inputs(sigma_major, sigma_minor, hbr)

    w, exponent, upper, lower, _ = _alfano_terms(
        sigma_major, sigma_minor, torch.abs(miss_x), torch.abs(miss_z), hbr_safe, order
    )
    log_lower = _log_erfc(lower)
    log_upper = _log_erfc(upper)
    log_chord = log_lower + torch.log(-torch.expm1(log_upper - log_lower))
    log_terms = torch.log(w) + exponent + log_chord
    log_total = torch.logsumexp(log_terms, dim=-1)
    log_probability = torch.log(hbr_safe / sigma_major) + log_total
    log_probability = torch.clamp(log_probability, max=0.0)

    log_probability = torch.where(certain, torch.zeros_like(log_probability), log_probability)
    return torch.where(
        zero_result, torch.full_like(log_probability, -math.inf), log_probability
    )


# ---------------------------------------------------------------------------
# Geometric ceiling and dilution (mirror risk.alfano)
# ---------------------------------------------------------------------------


def max_collision_probability_torch(
    hard_body_radius: Tensor | float, miss_distance: Tensor | float
) -> Tensor:
    """Batched :func:`aegis.risk.alfano.max_collision_probability`.

    ``Pc_max = R^2 / (e d^2)`` capped at 1, and exactly 1 for a
    non-positive miss (the original's guard). The division is done on a
    stand-in ``d = 1`` for that branch, so the gradient there is zero rather
    than ``inf``.
    """
    hbr, miss = _coerce(hard_body_radius, miss_distance)
    degenerate = ~(miss > 0)
    miss_safe = torch.where(degenerate, torch.ones_like(miss), miss)
    value = (hbr * hbr) / (math.e * miss_safe * miss_safe)
    value = torch.clamp(value, max=1.0)
    return torch.where(degenerate, torch.ones_like(value), value)


def dilution_sigma_torch(miss_distance: Tensor | float) -> Tensor:
    """Batched :func:`aegis.risk.alfano.dilution_sigma`: ``d / sqrt(2)``."""
    (miss,) = _coerce(miss_distance)
    return miss / _SQRT_2


# ---------------------------------------------------------------------------
# Required miss distance (mirrors maneuver.miss.required_miss_distance_km)
# ---------------------------------------------------------------------------


def _required_miss_bisection(
    sigma_major: Tensor, sigma_minor: Tensor, hbr: Tensor, target: Tensor, order: int
) -> tuple[Tensor, Tensor]:
    """The original's early exits and 80-step bisection, in tensor form.

    Returns ``(rho, interior)`` where ``interior`` marks the elements whose
    result came from the bisection proper -- those are the only ones with a
    non-zero derivative. The five early-exit tests are applied in the same
    precedence as the numpy code so the two agree element-for-element,
    including the sentinel cases.
    """
    with torch.no_grad():
        zero = torch.zeros_like(sigma_major)
        sentinel = torch.full_like(sigma_major, _SENTINEL_MISS_KM)

        m1 = ~(hbr > 0) | (target >= 1.0)
        if bool(torch.any((~m1) & ((sigma_major <= 0) | (sigma_minor <= 0)))):
            raise ValueError("covariance standard deviations must be positive")
        # Sigmas that only reach the quadrature on an early-exit branch are
        # replaced by 1 so the evaluation below cannot raise for them.
        sm = torch.where(m1, torch.ones_like(sigma_major), sigma_major)
        sn = torch.where(m1, torch.ones_like(sigma_minor), sigma_minor)

        def pc(miss_x: Tensor) -> Tensor:
            return collision_probability_torch(sm, sn, miss_x, zero, hbr, order=order)

        m2 = (~m1) & (target <= 0.0)
        m3 = (~m1) & (~m2) & (pc(zero) <= target)
        m4 = (~m1) & (~m2) & (~m3) & (pc(sentinel) > target)
        m5 = (
            (~m1) & (~m2) & (~m3) & (~m4)
            & (hbr >= _HBR_DOMINATED_RATIO * sm)
            & (target < _HBR_DOMINATED_TARGET)
        )
        interior = ~(m1 | m2 | m3 | m4 | m5)

        lo = zero.clone()
        hi = sentinel.clone()
        for _ in range(_SEARCH_ITERS):
            mid = 0.5 * (lo + hi)
            below = pc(mid) <= target
            hi = torch.where(below, mid, hi)
            lo = torch.where(below, lo, mid)

        rho = torch.where(m1, zero, torch.where(m2, sentinel, torch.where(m3, zero, torch.where(m4 | m5, sentinel, hi))))
    return rho, interior


class _RequiredMiss(torch.autograd.Function):
    """Implicit-function gradient for the bisection root.

    Forward: the non-differentiable bisection. Backward: differentiate the
    defining equation ``Pc(rho; theta) = target`` once,

        d rho / d theta  = -(dPc/dtheta) / (dPc/drho)
        d rho / d target =        1      / (dPc/drho)

    with both partials obtained by autograd through
    :func:`collision_probability_torch` at the converged root. Elements that
    took an early exit (clamped to 0 or to the sentinel) have zero gradient,
    which is the true one-sided derivative of a clamped function.

    If ``dPc/drho`` is exactly zero at an interior root the root is not
    locally unique and the derivative is genuinely undefined; that requires
    a target below the ``erfc`` underflow floor and cannot happen for the
    targets this codebase uses. Rather than emit ``inf``, such elements get
    a zero gradient and the condition is exposed to the caller through the
    ``strict`` flag of :func:`required_miss_torch`.
    """

    @staticmethod
    def forward(ctx, sigma_major: Tensor, sigma_minor: Tensor, hbr: Tensor, target: Tensor, order: int, strict: bool):  # type: ignore[override]
        rho, interior = _required_miss_bisection(sigma_major, sigma_minor, hbr, target, order)
        ctx.save_for_backward(sigma_major, sigma_minor, hbr, target, rho, interior)
        ctx.order = order
        ctx.strict = strict
        return rho

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        sigma_major, sigma_minor, hbr, target, rho, interior = ctx.saved_tensors
        with torch.enable_grad():
            sm = sigma_major.detach().requires_grad_(True)
            sn = sigma_minor.detach().requires_grad_(True)
            radius = hbr.detach().requires_grad_(True)
            miss = rho.detach().requires_grad_(True)
            # Early-exit elements may carry sigmas the quadrature rejects;
            # substitute 1 there, their gradient is discarded anyway.
            sm_eval = torch.where(interior, sm, torch.ones_like(sm))
            sn_eval = torch.where(interior, sn, torch.ones_like(sn))
            pc = collision_probability_torch(sm_eval, sn_eval, miss, torch.zeros_like(miss), radius, order=ctx.order)
            g_sm, g_sn, g_hbr, g_rho = torch.autograd.grad(pc.sum(), (sm, sn, radius, miss), allow_unused=False)

        usable = interior & (g_rho != 0)
        if ctx.strict and bool(torch.any(interior & ~usable)):
            raise FleetOptError(
                "required_miss_torch: dPc/drho vanished at an interior root; the implicit "
                "gradient is undefined there (target below the erfc underflow floor)"
            )
        g_rho_safe = torch.where(usable, g_rho, torch.ones_like(g_rho))
        scale = torch.where(usable, grad_output / g_rho_safe, torch.zeros_like(grad_output))
        return -scale * g_sm, -scale * g_sn, -scale * g_hbr, scale, None, None


def required_miss_torch(
    sigma_major: Tensor | float,
    sigma_minor: Tensor | float,
    hard_body_radius: Tensor | float,
    target_pc: Tensor | float = PC_TARGET_POST_MANEUVER,
    *,
    order: int = ALFANO_QUAD_ORDER,
    strict: bool = True,
) -> Tensor:
    """Differentiable :func:`aegis.maneuver.miss.required_miss_distance_km`.

    Note the argument order follows the contract (``sigma_major,
    sigma_minor, hard_body_radius, target_pc``), not the numpy original,
    which leads with the hard-body radius.

    Forward value
    -------------
    Exactly the original's algorithm: the same five early exits, then 80
    bisection steps on ``[0, 100] km`` with ``Pc(mid) <= target`` deciding
    the half, returning the upper end. Because both sides evaluate the same
    quadrature and ``erfc`` agrees to ~1e-13, a comparison can only flip
    when ``Pc(mid)`` is within that of the target -- which moves the answer
    by at most the bracket width at that step. Measured agreement with the
    original is at the 1e-13 km level over 300+ geometries (see the
    verification report), far below anything the optimizer resolves.

    Why an implicit-function gradient and not unrolled bisection
    ------------------------------------------------------------
    Unrolling 80 ``torch.where`` steps *is* differentiable, but the result
    is a convex combination of the two endpoints ``0`` and ``100 km``, both
    constants: autograd sees a piecewise-constant function of the inputs
    and returns a gradient that is identically zero. The information about
    how the root moves lives entirely in the comparisons, which have no
    gradient. The implicit function theorem recovers it exactly: at the
    root ``Pc(rho; theta) = target`` and so
    ``d rho / d theta = -(dPc/dtheta) / (dPc/drho)``. This is both the true
    derivative of the limit of the bisection and cheaper -- one extra
    quadrature evaluation with autograd, instead of 80 with a saved graph.
    Its accuracy is that of the quadrature's own gradient; ``gradcheck``
    against central differences of the bisection passes at torch's default
    ``float64`` tolerances.

    ``strict=True`` raises :class:`FleetOptError` if an interior root has a
    vanishing ``dPc/drho`` (see :class:`_RequiredMiss`); ``strict=False``
    returns a zero gradient for that element instead. Early-exit elements
    (result 0 or the 100 km sentinel) always have zero gradient.
    """
    sigma_major, sigma_minor, hbr, target = _coerce(sigma_major, sigma_minor, hard_body_radius, target_pc)
    return _RequiredMiss.apply(sigma_major, sigma_minor, hbr, target, int(order), bool(strict))


# ---------------------------------------------------------------------------
# Proposition 1 miss distance and the physics-loss residuals
# ---------------------------------------------------------------------------


def miss_vector_after_torch(miss_vector: Tensor, sensitivity: Tensor, x: Tensor) -> Tensor:
    """``d + B x`` for stacked constraints -- the vector inside Proposition 1.

    Shapes: ``miss_vector (*batch, J, 3)``, ``sensitivity (*batch, J, 3,
    n_vars)``, ``x (*batch, n_vars)``; result ``(*batch, J, 3)``. ``batch``
    may be empty on any argument and broadcasts. When the inputs are the
    projected quantities stored on :class:`aegis.fleetopt.bplane.MissSensitivity`
    (``P d`` and ``P B``) the result is ``P (d + B x)``; the projector is
    *not* applied here, because the latent ``grid`` constraints of section
    5 are deliberately unprojected.
    """
    displaced = torch.einsum("...jkn,...n->...jk", sensitivity, x)
    return miss_vector + displaced


def miss_after_torch(miss_vector: Tensor, sensitivity: Tensor, x: Tensor) -> Tensor:
    """Proposition 1: ``m_j(x) = ||d_j + B_j x||``, shape ``(*batch, J)``.

    Mirrors :meth:`aegis.fleetopt.bplane.MissSensitivity.miss_after_km`
    with the shape conventions of :func:`miss_vector_after_torch`. The norm
    uses the zero-subgradient surrogate at the origin so a plan that drives
    a miss vector exactly to zero (the worst case the loss is meant to push
    away from) still back-propagates a finite value.
    """
    return _safe_norm(miss_vector_after_torch(miss_vector, sensitivity, x), dim=-1)


@dataclass(frozen=True)
class ConstraintResiduals:
    """Signed margins of the linearised constraints at a candidate ``x``.

    ``resolve`` has shape ``(*batch, J)`` and holds
    ``u_j . (d_j + B_j x) - rho_j``; ``latent`` has shape ``(*batch, L)``
    and holds ``u_p . (d_p + B_p x) - floor_p``. Positive means satisfied.
    Because ``u`` is a unit vector, a non-negative resolve residual implies
    the true norm constraint by Proposition 2 -- that is what makes
    ``relu(-residual)`` the right penalty: it is a restriction, never a
    relaxation, of the safety requirement.

    ``resolve_mask`` and ``latent_mask`` are boolean tensors of the same
    shapes marking real (unpadded) rows. Padded rows have residual exactly
    zero, so any loss of the form ``f(residual)`` with ``f(0) = 0``
    contributes nothing and receives no gradient from them; the masks are
    kept for callers that count violations or average per row.
    """

    resolve: Tensor
    latent: Tensor
    resolve_mask: Tensor
    latent_mask: Tensor

    @property
    def resolve_violation(self) -> Tensor:
        """``relu(-resolve)`` -- how far each resolve row is from safe."""
        return torch.relu(-self.resolve)

    @property
    def latent_violation(self) -> Tensor:
        return torch.relu(-self.latent)


def _directional_residual(
    miss_vector: Tensor,
    sensitivity: Tensor,
    direction: Tensor,
    threshold: Tensor,
    mask: Tensor | None,
    x: Tensor,
    label: str,
) -> tuple[Tensor, Tensor]:
    """``u . (d + B x) - threshold`` with masking; shared by both row kinds."""
    after = miss_vector_after_torch(miss_vector, sensitivity, x)
    norm = _safe_norm(direction, dim=-1)
    residual_shape = torch.broadcast_shapes(after.shape[:-1], threshold.shape, norm.shape)
    if mask is None:
        mask = torch.ones(residual_shape, dtype=torch.bool, device=after.device)
    else:
        mask = mask.to(dtype=torch.bool).expand(residual_shape)
    degenerate = norm < _ZERO
    if bool(torch.any(degenerate & mask)):
        raise FleetOptError(f"{label} linearization direction must be non-zero on unmasked rows")
    # Mirrors ``linearize``: the direction is normalised here so callers may
    # pass raw vectors. Padded rows get a unit stand-in so the division is
    # finite; their residual is zeroed below.
    norm_safe = torch.where(degenerate, torch.ones_like(norm), norm)
    unit = direction / norm_safe[..., None]
    residual = torch.sum(unit * after, dim=-1) - threshold
    residual = residual.expand(residual_shape)
    return torch.where(mask, residual, torch.zeros_like(residual)), mask


def constraint_residuals_torch(
    x: Tensor,
    *,
    resolve_miss: Tensor,
    resolve_sensitivity: Tensor,
    resolve_direction: Tensor,
    required_miss: Tensor,
    resolve_mask: Tensor | None = None,
    latent_miss: Tensor | None = None,
    latent_sensitivity: Tensor | None = None,
    latent_direction: Tensor | None = None,
    latent_floor: Tensor | None = None,
    latent_mask: Tensor | None = None,
) -> ConstraintResiduals:
    """The two residual families of the physics loss (contract section 15.3).

    Inputs, with ``*batch`` broadcasting across all of them and ``J``/``L``
    padded to a common width within a batch:

    ``x``                   ``(*batch, n_vars)``  candidate decision vector, km/s
    ``resolve_miss``        ``(*batch, J, 3)``    ``P_j d_j`` from ``MissSensitivity.miss_vector_km``
    ``resolve_sensitivity`` ``(*batch, J, 3, n_vars)`` ``P_j B_j`` from ``MissSensitivity.sensitivity``
    ``resolve_direction``   ``(*batch, J, 3)``    linearisation direction ``u_j`` (normalised here)
    ``required_miss``       ``(*batch, J)``       ``rho_j``, km
    ``resolve_mask``        ``(*batch, J)`` bool  real rows; ``None`` means all real
    ``latent_miss``         ``(*batch, L, 3)``    ``d_p``; for ``kind="grid"`` rows this is
                                                  ``separation_km * direction`` (unprojected),
                                                  for ``kind="bplane_minimum"`` rows ``P d``
    ``latent_sensitivity``  ``(*batch, L, 3, n_vars)`` ``LatentConstraint.sensitivity``
    ``latent_direction``    ``(*batch, L, 3)``    ``u_p``
    ``latent_floor``        ``(*batch, L)``       ``floor_km`` (``s_min`` plus grid inflation)
    ``latent_mask``         ``(*batch, L)`` bool

    Returns :class:`ConstraintResiduals` with ``resolve (*batch, J)`` and
    ``latent (*batch, L)``. The latent family is optional: when all five
    latent arguments are ``None`` the latent residual is an empty tensor of
    shape ``(*batch, 0)`` (a ``fuel-only`` training target has no latent
    rows). Supplying only some of them is an error rather than a guess.

    This is the batched form of :func:`aegis.fleetopt.bplane.linearize`:
    the residual equals ``row @ x - rhs`` for the row and rhs that function
    returns, which the verification report checks directly. A zero
    direction on an unmasked row raises :class:`FleetOptError`, as
    ``linearize`` does; on a masked (padded) row it is permitted and the
    residual is exactly zero.
    """
    resolve, r_mask = _directional_residual(
        resolve_miss, resolve_sensitivity, resolve_direction, required_miss, resolve_mask, x, "resolve"
    )

    latent_args = (latent_miss, latent_sensitivity, latent_direction, latent_floor)
    if all(arg is None for arg in latent_args):
        if latent_mask is not None:
            raise FleetOptError("latent_mask supplied without latent constraint tensors")
        latent = resolve.new_zeros(resolve.shape[:-1] + (0,))
        l_mask = torch.zeros(latent.shape, dtype=torch.bool, device=latent.device)
    elif any(arg is None for arg in latent_args):
        raise FleetOptError(
            "latent_miss, latent_sensitivity, latent_direction and latent_floor must all be given or all be None"
        )
    else:
        latent, l_mask = _directional_residual(
            latent_miss, latent_sensitivity, latent_direction, latent_floor, latent_mask, x, "latent"  # type: ignore[arg-type]
        )
    return ConstraintResiduals(resolve=resolve, latent=latent, resolve_mask=r_mask, latent_mask=l_mask)


def physics_loss_torch(residuals: ConstraintResiduals, x: Tensor, dv_weight: float | Tensor) -> Tensor:
    """``L_phys`` of contract section 15.3, shape ``(*batch,)``.

    ``sum_j relu(-r_j)^2 + sum_p relu(-r_p)^2 + w_dv * ||x||_1``. The
    squared hinge is used rather than a plain hinge so the gradient goes to
    zero continuously as a constraint becomes satisfied; the L1 term is the
    same fuel proxy the LP minimises (with unit axis weights). Padded rows
    contribute exactly zero by construction of the residuals.
    """
    resolve_term = torch.sum(residuals.resolve_violation**2, dim=-1)
    latent_term = torch.sum(residuals.latent_violation**2, dim=-1)
    fuel_term = dv_weight * torch.sum(torch.abs(x), dim=-1)
    return resolve_term + latent_term + fuel_term
