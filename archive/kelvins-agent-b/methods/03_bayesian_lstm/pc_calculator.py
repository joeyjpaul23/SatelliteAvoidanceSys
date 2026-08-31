"""
2D probability-of-collision (Pc) calculator -- self-contained copy for
method 3 (each method is an independent pipeline per the shared contract,
so this is intentionally not imported from method 2's directory).

Same derivation as method 2's pc_calculator.py: JSpOC/Foster 2D Pc method
via the erf/square approximation, following this project's own
docs/probability-of-collision-deep-dive.md. The only difference from method
2's version is the input shape -- this one takes raw covariance matrix
entries directly (CR_R, CT_R, CT_T, CN_R, CN_T, CN_N), matching exactly what
kessler's CDM objects expose, instead of sigma+correlation pairs.
"""
import numpy as np
from scipy.special import erf

HBR_BY_TYPE = {
    "PAYLOAD": 5.0,
    "ROCKET BODY": 3.0,
    "UNKNOWN": 3.0,
    "DEBRIS": 1.0,
    "TBA": 3.0,
}
HBR_TARGET = 5.0


def cov_from_raw(cr_r, ct_r, ct_t, cn_r, cn_t, cn_n):
    """3x3 covariance in (R, T, N) order from raw upper-triangular terms,
    exactly as kessler's CDM stores them (CR_R = var(R), CT_R = cov(T,R),
    CT_T = var(T), CN_R = cov(N,R), CN_T = cov(N,T), CN_N = var(N))."""
    return np.array(
        [
            [cr_r, ct_r, cn_r],
            [ct_r, ct_t, cn_t],
            [cn_r, cn_t, cn_n],
        ]
    )


def encounter_plane_basis(v_rel):
    v_hat = v_rel / np.linalg.norm(v_rel)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, v_hat)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, v_hat) * v_hat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    return e1, e2


def pc_2d_square(mu2d, cov2d, d):
    eigvals, eigvecs = np.linalg.eigh(cov2d)
    eigvals = np.clip(eigvals, 1e-12, None)
    mu_prime = eigvecs.T @ mu2d
    sig = np.sqrt(eigvals)

    def axis_term(mu_i, sig_i):
        return erf((d - mu_i) / (np.sqrt(2) * sig_i)) - erf((-d - mu_i) / (np.sqrt(2) * sig_i))

    return 0.25 * axis_term(mu_prime[0], sig[0]) * axis_term(mu_prime[1], sig[1])


def compute_risk_from_raw(
    target_cov_terms, chaser_cov_terms, rel_pos_rtn, rel_vel_rtn, c_object_type,
    risk_floor=-30.0,
):
    """target_cov_terms / chaser_cov_terms: (CR_R, CT_R, CT_T, CN_R, CN_T, CN_N) tuples."""
    cov_t = cov_from_raw(*target_cov_terms)
    cov_c = cov_from_raw(*chaser_cov_terms)
    cov_combined = cov_t + cov_c

    v_rel = np.asarray(rel_vel_rtn, dtype=float)
    if np.linalg.norm(v_rel) < 1e-6 or not np.all(np.isfinite(v_rel)):
        return risk_floor

    e1, e2 = encounter_plane_basis(v_rel)
    proj = np.vstack([e1, e2])

    mu2d = proj @ np.asarray(rel_pos_rtn, dtype=float)
    cov2d = proj @ cov_combined @ proj.T

    if not np.all(np.isfinite(cov2d)) or not np.all(np.isfinite(mu2d)):
        return risk_floor

    hbr_chaser = HBR_BY_TYPE.get(c_object_type, 3.0)
    d = HBR_TARGET + hbr_chaser

    pc = pc_2d_square(mu2d, cov2d, d)
    pc = max(pc, 10**risk_floor)
    return max(np.log10(pc), risk_floor)
