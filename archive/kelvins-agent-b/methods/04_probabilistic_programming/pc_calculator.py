"""
2D probability-of-collision (Pc) calculator.

Implements the JSpOC/Foster 2D Pc method: reduce to the plane perpendicular
to relative velocity at TCA (rectilinear-motion assumption), combine both
objects' position covariances into one Gaussian on that plane, and integrate
its mass over the touching disk -- approximated, as JSpOC itself does, by the
smallest axis-aligned square (in the covariance's principal-axis frame) that
contains the disk, which reduces the 2D Gaussian integral to a product of two
1D erf terms.

This follows the derivation in this project's own
docs/probability-of-collision-deep-dive.md (itself based on Kopke, Snow &
Hejduk's JSpOC Pc memo), reused here for consistency rather than re-derived.

Default hard-body-radius (HBR) values (5 m payload/platform, 3 m rocket
body/unknown, 1 m debris) also follow that memo's cited defaults, since HBR
is not present in the Kelvins CDM feature set.
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
HBR_TARGET = 5.0  # target is always the ESA-operated payload


def build_covariance_rtn(sigma_r, sigma_t, sigma_n, corr_tr, corr_nr, corr_nt):
    """3x3 covariance in (R, T, N) order from stds + correlation coefficients."""
    cov = np.array(
        [
            [sigma_r**2, corr_tr * sigma_t * sigma_r, corr_nr * sigma_n * sigma_r],
            [corr_tr * sigma_t * sigma_r, sigma_t**2, corr_nt * sigma_n * sigma_t],
            [corr_nr * sigma_n * sigma_r, corr_nt * sigma_n * sigma_t, sigma_n**2],
        ]
    )
    return cov


def encounter_plane_basis(v_rel):
    """Two orthonormal vectors spanning the plane perpendicular to v_rel."""
    v_hat = v_rel / np.linalg.norm(v_rel)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, v_hat)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, v_hat) * v_hat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    return e1, e2


def pc_2d_square(mu2d, cov2d, d):
    """
    Pc via the erf/square approximation: diagonalize cov2d (principal axes),
    integrate the resulting axis-aligned Gaussian over the [-d, d]^2 square
    that circumscribes the collision disk of radius d.
    """
    eigvals, eigvecs = np.linalg.eigh(cov2d)
    eigvals = np.clip(eigvals, 1e-12, None)  # guard near-singular covariance
    mu_prime = eigvecs.T @ mu2d
    sig = np.sqrt(eigvals)

    def axis_term(mu_i, sig_i):
        return erf((d - mu_i) / (np.sqrt(2) * sig_i)) - erf((-d - mu_i) / (np.sqrt(2) * sig_i))

    px = axis_term(mu_prime[0], sig[0])
    py = axis_term(mu_prime[1], sig[1])
    return 0.25 * px * py


def compute_risk(
    t_sigma_r, t_sigma_t, t_sigma_n, t_ct_r, t_cn_r, t_cn_t,
    c_sigma_r, c_sigma_t, c_sigma_n, c_ct_r, c_cn_r, c_cn_t,
    rel_pos_rtn, rel_vel_rtn, c_object_type,
    risk_floor=-30.0,
):
    """Full pipeline: covariances -> encounter-plane projection -> Pc -> log10(Pc)."""
    cov_t = build_covariance_rtn(t_sigma_r, t_sigma_t, t_sigma_n, t_ct_r, t_cn_r, t_cn_t)
    cov_c = build_covariance_rtn(c_sigma_r, c_sigma_t, c_sigma_n, c_ct_r, c_cn_r, c_cn_t)
    cov_combined = cov_t + cov_c

    v_rel = np.asarray(rel_vel_rtn, dtype=float)
    if np.linalg.norm(v_rel) < 1e-6:
        return risk_floor  # degenerate encounter geometry, cannot define plane

    e1, e2 = encounter_plane_basis(v_rel)
    proj = np.vstack([e1, e2])  # 2x3

    mu2d = proj @ np.asarray(rel_pos_rtn, dtype=float)
    cov2d = proj @ cov_combined @ proj.T

    hbr_chaser = HBR_BY_TYPE.get(c_object_type, 3.0)
    d = HBR_TARGET + hbr_chaser

    pc = pc_2d_square(mu2d, cov2d, d)
    pc = max(pc, 10**risk_floor)
    return max(np.log10(pc), risk_floor)
