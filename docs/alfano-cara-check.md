# Alfano / CARA independent Pc check

Independent reference-case check of the production Alfano
Gauss-Chebyshev collision probability against Chan's analytic method
(Chan 1997 / 2008, where the envelope allows), Alfano Simpson
quadrature, and the published geometric Pc_max formula.

NASA CARA MATLAB is not a dependency. These cases are the in-repo
cross-check against independent closed-form and quadrature referees.

| name | citation | Alfano (or pc_max) | referee | rel error | pass/fail |
| --- | --- | --- | --- | --- | --- |
| center_small | Chan isotropic analytic (same inputs); Alfano 2005 GC | 4.999875e-05 | 4.999875e-05 | 9.487006e-16 | pass |
| offset_isotropic | Chan isotropic analytic (same inputs); Alfano 2005 GC | 1.213001e-04 | 1.213001e-04 | 2.234546e-16 | pass |
| anisotropic_chan_ok | Chan 1997 / 2008 when the envelope allows; else Alfano Simpson | 2.423046e-05 | 2.423054e-05 | 3.320261e-06 | pass |
| far_tail | Far-tail; Alfano vs Chan or Simpson | 1.954035e-90 | 1.953940e-90 | 4.872874e-05 | pass |
| pc_max_unit | Alfano, Relating Position Uncertainty to Maximum Conjunction Probability, JAS 53(2), 2005 | 0.00367879 | 0.00367879 | 0 | pass |
| pc_max_zero_miss | Alfano, Relating Position Uncertainty to Maximum Conjunction Probability, JAS 53(2), 2005 | 1 | 1 | 0 | pass |

