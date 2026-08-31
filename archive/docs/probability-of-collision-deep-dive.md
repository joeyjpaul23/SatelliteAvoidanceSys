# Deep Dive — "Probability of Collision in the Joint Space Operations Center" (Kopke, Snow & Hejduk)

*A section-by-section walkthrough of the reasoning and math behind the Space-Track/JSpOC Pc memo, written as a companion to `day3-study-guide.md`. The goal here isn't to restate what the doc says — it's to explain **why** each step is taken, what problem it's solving, and how it connects to the next section.*

---

## The framing you need before any of the sections make sense

The single most important sentence in the whole memo is buried in footnote 3: what JSpOC calls "probability of collision" is not actually the probability the two objects collide. It's **the probability that the two objects' centers of mass end up closer together than some threshold distance at TCA.** That threshold is chosen to be the largest possible distance at which the two objects, in the worst-case orientation, could still be touching.

This distinction matters because it explains almost every modeling choice downstream:

- Why object shape and orientation get thrown away in favor of a single number (a circumscribing sphere) — you don't know orientation at TCA, so you can't compute "do the actual shapes intersect." You can only bound it.
- Why every assumption in the doc is stated to bias the answer **upward** (conservative). This isn't incidental — it's a design principle. An operator who gets a slightly-too-high Pc might needlessly evaluate a maneuver; an operator who gets a too-low Pc might not evaluate a real collision risk at all. The entire method is built to fail safe.

Keep that "we are computing a worst-case-bound proximity probability, not literal collision probability" framing in your head — every later section is a consequence of it.

---

## Section: Object Size

**What it says:** circumscribe each object by a sphere, add the two radii together, and project the resulting "supervening sphere" into the conjunction plane as a circle of radius `d`.

**Why a sphere, specifically.** You don't know the orientation of either object at TCA (satellites tumble, have deployed panels, etc.), and you often don't know its exact shape either — debris in particular is frequently just a radar cross-section (RCS) number. A sphere is the only shape whose "does it touch another shape" test doesn't depend on orientation. Any other shape (box, cylinder) would require knowing attitude at TCA, which the JSpOC doesn't reliably have. So the sphere isn't chosen because it's realistic — it's chosen because it's the maximal, orientation-agnostic bound. This is the object-size instance of the "always be conservative" rule from the framing section.

**Why you add the two radii together.** Two spheres touch exactly when the distance between their centers is ≤ (r₁ + r₂). This is just geometry — but it's the key move that lets the whole problem be re-expressed as "is the *relative position* within a single circle of radius d = r₁ + r₂ centered at zero," rather than tracking two separate objects. This single-circle reformulation is what makes the later collision-plane integral a one-object problem (a combined Gaussian against a single disk) instead of a two-body problem.

**Why project onto a plane (going from a touching-condition in 3D to a circle in 2D) is deferred to later**, but it's worth flagging now: the projection is valid because of the rectilinear-motion assumption discussed under "Assumptions" — near TCA the relative trajectory is treated as a straight line, so whether the spheres ever touch reduces to whether that line ever passes within `d` of the origin, which is a purely 2D question in the plane perpendicular to that line.

**Why default sizes exist (5 m payload/platform, 3 m rocket body/unknown, 1 m debris).** These aren't arbitrary — footnote 8 says they came from an actual empirical study of the size distribution of cataloged objects by type. They exist because AREA_PC is frequently unreported (owner/operators don't always supply it, and debris obviously can't self-report), so the system needs a reasonable prior. Note the ordering of the defaults tracks known-size confidence: payloads are engineered objects with knowable envelopes, rocket bodies are big but variable, and debris is the least-known category — the value 1 m is a compromise, not a measurement.

**Why RCS is flagged as unreliable for size.** RCS is a radar-reflectivity measurement, not a physical dimension. A small object with a corner-reflector geometry can have a large RCS, and a large object with a radar-absorbing or edge-on orientation can have a small one. Using RCS as a stand-in for physical size embeds shape/orientation/material information you don't actually have, which is why the memo explicitly warns about it — it's a place where the "conservative by construction" property of the method can quietly break down if used carelessly.

---

## Section: Position and Velocity Vectors

**Why velocity is described in detail even though it "is not explicitly used in the computation of Pc."** This is worth sitting with because it looks contradictory at first. Velocity does three jobs upstream of the Pc formula itself:

1. **Finding TCA.** TCA is defined as the time when relative distance is minimized — you can't find that time without velocity (it's where the derivative of relative position, which depends on both objects' velocities, crosses zero / relative range rate flips sign).
2. **Defining the orientation of the collision plane.** The collision plane is defined as *perpendicular to the relative velocity vector at TCA* (this is stated explicitly in the Computation section). So even though velocity magnitude doesn't appear in the integral, velocity *direction* defines the coordinate system the entire integral is computed in.
3. **Coordinate transforms.** Converting from the object's native reference frame (ITRF) into the RTN/collision-plane frames used for the actual math requires the state vector, not just position.

So velocity is a "geometry-setup" input, not a "probability-math" input — it tells you *where to look*, not *how much probability mass is there*.

**Why ITRF, and why the RTN (Radial-Transverse-Normal) frame comes up separately.** ITRF is an Earth-fixed inertial-ish frame — a good universal frame for storing state vectors so that data from different sources (JSpOC-propagated, owner/operator-supplied) can be compared apples-to-apples. RTN, in contrast, is defined *relative to the orbit itself* (radial = away from Earth center, transverse ≈ along-track, normal = out of orbit plane) — it's the natural frame for describing *uncertainty*, because orbital error doesn't grow isotropically. Drag mismodeling, for instance, primarily corrupts your along-track (transverse) position prediction over time, so an error ellipsoid expressed in RTN naturally shows up elongated along the transverse axis (this is explicitly called out later, and is why RTN is the frame covariance is reported in — footnote 19 tells you the biggest axis is usually the drag-affected one).

**Why the state vector at TCA comes from ephemeris interpolation rather than a direct measurement.** JSpOC doesn't have a sensor observation sitting exactly at TCA — TCA is a predicted future (or very-near) time. So the process is: get an epoch state vector via orbit determination (differential correction against tracking observations), propagate it forward via special-perturbations (SP) numerical propagation to build an ephemeris (a table of position/velocity samples over time), then interpolate that table to the exact TCA time. This is the same conceptual chain as your Day 2 SGP4 propagation, just with a higher-fidelity numerical propagator (SP) instead of the analytic SGP4 model, because JSpOC's operational catalog uses SP for precision work.

---

## Section: Covariance and Error Ellipsoids

**What a covariance matrix is actually encoding.** The memo's own analogy is the right one: variance tells you how spread out a single number is around its mean; a covariance matrix does the same thing for a *vector* — it tells you how spread out (and correlated) each pair of components is. The diagonal terms (C_R_R, C_T_T, C_N_N) are ordinary variances of the radial, transverse, and normal position errors. The off-diagonal terms (C_T_R, etc.) tell you whether error in one direction tends to accompany error in another — e.g., if a long-arc orbit-fit error causes both an along-track and a slight radial offset together, that correlation shows up as a nonzero off-diagonal term.

**Why it's reported as a 6×6 matrix even though only the 3×3 position block is used.** The full 6×6 (position + velocity) covariance falls naturally out of the orbit-determination process — OD estimates the whole state vector jointly, so its uncertainty is naturally a joint position-velocity covariance. Pc, however, is a purely spatial "did they get close" question at one instant, so only the position-position 3×3 sub-block is relevant; the velocity uncertainty and the position-velocity cross terms just aren't needed for *this* calculation (they would matter if you were instead computing, say, uncertainty in the relative velocity or in a future TCA).

**Why each object's covariance is expressed in *its own* RTN frame rather than a shared one.** Because the physical error sources (drag, mismodeled gravity harmonics, tracking geometry) act relative to each object's own orbit, not relative to some external frame. Expressing each object's uncertainty in its own RTN frame is what makes the elongation pattern (large in transverse) physically interpretable. This is also *why*, in the Computation section, everything has to be rotated into a **common** frame before you can add the two covariances together — you can't sum two matrices that are each expressed in a different, object-specific coordinate system.

**Why 5-point Lagrange interpolation for covariance.** Same reasoning as position/velocity interpolation: the ephemeris only has covariance at discrete sample times, not exactly at TCA, so you need an interpolation scheme to get the value at the exact time you need. Lagrange interpolation (fitting a polynomial through nearby points) is a standard, well-behaved choice for smoothly-varying quantities like this over a short time window.

**Why diagonalization (eigendecomposition) gives you the "error ellipsoid."** A covariance matrix's diagonal form, obtained via eigendecomposition, separates the "how big" from the "which direction" information: the eigenvalues' square roots give you the lengths of the ellipsoid's three axes (this is literally footnote 18's construction), and the eigenvectors give you the orientation of those axes relative to RTN. This is the same math as principal component analysis — you're finding the directions along which the uncertainty is uncorrelated with itself, and how large it is along each of those directions. It's a visualization/intuition tool more than a computational necessity for Pc itself (the Pc integral works directly with the covariance matrix and its inverse, not with the eigendecomposition).

**Why epoch covariance and TCA covariance are called out as two different things (footnote 15).** The OD process gives you uncertainty *at the moment of the fit* (epoch), driven by how well your tracking observations constrained the state. But you need uncertainty *at TCA*, which may be hours or days later. Getting from one to the other means propagating the uncertainty forward through the same dynamics used to propagate the state itself — errors generally grow over the propagation interval (this is exactly why atmospheric drag mismodeling dominates the transverse-axis growth mentioned earlier). Conflating "uncertainty when we last observed it" with "uncertainty when it matters" would badly understate real risk for objects observed a while ago.

---

## Section: Assumptions

Each bullet here exists to make the integral in the Computation section tractable in closed form. Take them one at a time:

**"Object sizes are known or can otherwise be assigned an upper bound."** This just restates the Object Size section's role: the integral needs a concrete radius `d` to define its integration region. If you truly had zero information you couldn't compute a bounded integral at all — hence the fallback defaults.

**"The conjunction is hyperkinetic... allowing rectilinear relative motion."** This is the single most consequential assumption in the whole document, because it's what converts an inherently *dynamic* problem (two curved orbital trajectories, changing distance over time, uncertainty evolving over time) into a *static* one (a single snapshot at TCA). The argument: over the short window during which the two objects are actually close to each other (seconds, versus an orbital period of ~90 minutes), the curvature of each orbit due to gravity is negligible compared to the straight-line motion implied by their velocity at TCA. So instead of asking "integrate the probability of collision over the whole time window during which they're near each other, accounting for curving trajectories," you can ask the much simpler question "at the single instant of closest approach, treating both objects as moving in straight lines, what's the probability their positions land within `d` of each other." This is exactly the move that collapses the problem from 3D-and-time into a single 2D spatial integral — it's why the entire method is called "2D Pc."

**"Gaussian theory and statistics apply."** Orbit determination errors arise from many independent contributing error sources (tracking noise, force-model mismodeling, geometry dilution, etc.), and by a central-limit-theorem-style argument, sums of many small independent error contributions tend toward Gaussian. Practically, this assumption is also what buys you a *closed-form, fast-to-compute* answer — Gaussian integrals have well-studied closed forms (error functions), whereas a non-Gaussian uncertainty model would generally require numerical/Monte Carlo integration, which is far too slow to run for the tens of thousands of conjunctions JSpOC screens routinely.

**"Covariance for both objects is known and constant throughout the encounter."** Since the encounter window is (by the hyperkinetic assumption) very short, the change in uncertainty over that window is treated as negligible — you don't need to model covariance as time-varying *within* the encounter, only evaluate it once, at TCA.

**"Primary and secondary errors are independent... allowing combined covariance to be the simple sum."** This is a standard result from probability: if X and Y are independent random vectors, Var(X + Y) = Var(X) + Var(Y) — the covariance of a sum of independent random variables is the sum of their covariances. Physically, independence is justified because the two objects are tracked and orbit-determined completely separately (different tracking histories, different force-model errors, generally different sensors) — there's no shared error source linking the two objects' position uncertainties. This independence is also *why* the memo needs the "expressed in a common frame" step before summing — you can only add matrices representing the same coordinate basis.

**"Covariance is not too large" / "not too small."** These are opposite failure modes of the same underlying integral:
- If covariance is enormous (huge uncertainty), the Gaussian pdf becomes nearly flat over the small region of size `d` — Pc trends toward a small, uninformative number. This is a well-known pathology in conjunction assessment: an object whose position is essentially unknown can look *artificially safe*, because "we have no idea where it is" gets mathematically indistinguishable from "it's spread out and unlikely to be exactly here." That's obviously not what you want operationally, so SuperCOMBO refuses to compute (and instead flags) cases where covariance exceeds a threshold, rather than silently reporting a falsely-reassuring low Pc.
- If covariance is near zero, C becomes near-singular — Det(C) approaches zero, and the term `1/sqrt(|Det(C)|)` in front of the integral blows up. This is a numerical trap (footnote 20 calls it out as guarding against zero covariance specifically), not just a statistical judgment call — a genuinely zero covariance would make the formula literally undefined.

**Covariance realism (the closing paragraph).** This is a separate, more subtle concern from "too large/too small": even a *moderate*, well-conditioned covariance can still be *wrong* — specifically, too small relative to the true error, because OD covariance estimates are known to be statistically optimistic (they capture formal fit uncertainty but tend to under-represent real-world model errors like unmodeled atmospheric density variation). This is why operators apply empirical inflation/scaling factors to reported covariance — it's a bias-correction step layered on top of the raw statistical output, motivated by comparing predicted uncertainty against actual observed tracking residuals over many objects.

---

## Section: Computation — the actual math

This is where everything above gets assembled. Let's build the formula up piece by piece rather than just reading it.

**Step 1 — reduce to the collision plane.** Because of the rectilinear-motion assumption, near TCA the relative position vector between the two objects is (approximately) a straight line parameterized by time, moving in the direction of the relative velocity vector. All of the "risk" therefore lives in the two dimensions *perpendicular* to that line — the along-track dimension is just "where along the line are we right now," which for a straight-line approximation only matters insofar as it determines the point of closest approach (which is exactly TCA, by definition). So instead of integrating probability over a 3D volume across a time window, you fix time at TCA and integrate over the 2D **collision plane** (perpendicular to relative velocity) at that instant. This is the dynamic-to-static, 3D-to-2D reduction promised back in the Object Size section.

**Step 2 — combine the two objects into one Gaussian.** Both objects' position covariances get rotated into the common collision-plane frame and summed (justified by the independence assumption above) into a single combined 2×2 covariance matrix `C`. Physically: instead of tracking "primary's uncertainty blob" and "secondary's uncertainty blob" separately, you now have *one* combined uncertainty blob describing the uncertainty in their *relative* position, centered on the secondary's position relative to the primary, `r_S/P`. This is the "two little hills merge into one hill" intuition from footnote 25 — two independent 2D Gaussians "sitting over" their respective object positions combine into a single 2D Gaussian centered at the relative-position offset, with combined spread `C`.

**Step 3 — the exclusion region.** With the primary object placed at the origin of this 2D plane (by convention), the objects are considered "touching" whenever the *relative* position point `r = (x, y)` lands within distance `d` of the origin — recall `d` is the sum of the two circumscribing-sphere radii from the Object Size section. So the exclusion/collision region is simply the disk `x² + y² ≤ d²`.

**Step 4 — Pc is the probability mass of the combined Gaussian inside that disk.** This is exactly what the double integral computes:

```
Pc = (1 / (2π √|Det(C)|)) ∬_{x²+y²≤d²} exp[ -½ (r − r_S/P)ᵀ C⁻¹ (r − r_S/P) ] dx dy
```

Read the pieces:
- `(1/(2π√|Det(C)|)) · exp[-½ (r−r_S/P)ᵀC⁻¹(r−r_S/P)]` is nothing more than the standard 2D multivariate Gaussian probability density function, centered at `r_S/P`, with covariance `C`. If you've seen the general multivariate Gaussian pdf formula, this is a direct instance of it in 2 dimensions.
- The exponent `(r − r_S/P)ᵀ C⁻¹ (r − r_S/P)` is the squared **Mahalanobis distance** between the point `r` and the mean `r_S/P` — a distance measure that accounts for the shape and orientation of the uncertainty ellipse, not just raw Euclidean distance. This is the precise mathematical reason (foreshadowed in your Day 3 study guide) that "small miss distance" and "high risk" aren't the same thing: a point that's Euclidean-close but lies along the *long* axis of a stretched covariance ellipse has a *smaller* Mahalanobis distance penalty (i.e., is "less surprising," more probable) than the same Euclidean distance along the *short* axis. Pc is fundamentally a Mahalanobis-distance-based quantity, not a raw-distance-based one.
- Integrating this pdf over the disk of radius `d` is literally asking "what fraction of the total probability mass of the combined position-uncertainty distribution falls inside the region where the two objects would be touching." That's the whole calculation, conceptually: Pc = P(relative position lands inside the touching-disk).
- `C⁻¹` requires `C` to be invertible, i.e., `Det(C) ≠ 0` — which is exactly the numerical failure mode the "covariance not too small" assumption was guarding against.

**Why error functions (ERF), and why integrate over a circumscribing square instead of the exact circle.** The 2D Gaussian integral has a closed form in terms of the error function only when the integration region is axis-aligned and rectangular — a 1D Gaussian integral over an interval is expressible via erf, and a 2D integral over a rectangle (in the frame aligned with the covariance ellipse's principal axes) separates into a *product* of two such 1D erf integrals. A disk doesn't separate that way in Cartesian coordinates. So JSpOC's practical solution is to integrate over the smallest axis-aligned **square** that contains the disk of radius `d`, rather than the disk itself. Since the square strictly contains the circle, this integrates a bit of extra probability mass that lies in the square's corners but outside the actual touching-circle — which makes the reported Pc *slightly larger* than the geometrically exact value. Notice this is, again, the same conservative-by-construction philosophy from the very first section: given a choice between an exact-but-numerically-harder answer and an approximate-but-conservative one, the method consistently picks the one that can't underestimate risk.

---

## Section: Notes / Bibliography — why it closes the way it does

The closing material exists to answer a question a careful reader would naturally ask: "if I get a different Pc from a different agency for the same conjunction, is something wrong?" Not necessarily — different implementations can legitimately differ in exactly the details worked through above: square-vs-circle integration region, interpolation scheme for state/covariance at TCA, default size assumptions, and covariance-inflation policy. The memo's answer is pragmatic: small differences are expected and not a red flag; *large* differences warrant investigation. The provenance note (Foster's NASA work in the 1990s, matured by Chan into the standard reference monograph, validated numerically by AFSPC/A9 in 2000) is there to establish that this isn't an ad hoc formula — it's a specific, historically-traceable, independently-validated method, which matters if you're implementing your own Pc calculator on Day 7 and want a credible baseline to validate against (this is exactly why your build plan has you checking your implementation against Alfano's published test cases).

---

## How this maps onto your Day 7 build

When you implement the Pc calculator, the work breaks down along the same seams as this document:

1. **Geometry setup** (Object Size + Position/Velocity sections): get `d` (combined radius), find TCA, and construct the collision-plane rotation from the relative velocity vector.
2. **Statistics setup** (Covariance section + Assumptions): rotate both covariances into the common collision-plane frame, sum them into `C`, and sanity-check `Det(C)` isn't degenerate.
3. **The integral itself** (Computation section): evaluate the 2D Gaussian integral over the disk (or the circumscribing square, if you want to match JSpOC's exact convention) using `erf` — this is the only genuinely "new math" step; everything before it is bookkeeping to get `d`, `r_S/P`, and `C` into the right frame.

If your Day 7 numbers don't match Alfano's test cases, the two most likely culprits based on everything above are (a) a frame-rotation error when combining the two covariances, or (b) using the exact circle instead of JSpOC's circumscribing-square convention (or vice versa) — both are places where this document shows the "obvious" approach and the "operationally used" approach diverge slightly.
