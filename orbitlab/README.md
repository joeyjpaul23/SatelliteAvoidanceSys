# OrbitLab

A deterministic C++20 orbital-dynamics workbench with a dependency-free scientific web instrument.

The propagator integrates Cartesian ECI state with:

- EGM-96 Earth constants and point-mass gravity
- second zonal harmonic (`J2`) oblateness perturbation
- co-rotating exponential-atmosphere drag
- solar radiation pressure with cylindrical Earth eclipse gating
- fixed-step fourth-order Runge–Kutta integration
- live state-vector → osculating-element conversion
- specific-energy residual and discrete closest-approach screening

No remote data, API, runtime framework, package manager, or CDN is required.

## Build and generate

```bash
cmake -S orbitlab -B orbitlab/build -DCMAKE_BUILD_TYPE=Release
cmake --build orbitlab/build -j
cmake --build orbitlab/build --target generate
ctest --test-dir orbitlab/build --output-on-failure
```

The generator writes `web/telemetry.js`. A generated copy is committed so the viewer works without a C++ build; regenerating it changes the file. Serve the instrument from the repository root:

```bash
python3 -m http.server 8080 --directory orbitlab/web
```

Open `http://127.0.0.1:8080`.

## Data convention

All integrated positions are kilometers in GCRF/ECI-like J2000 axes, velocities are km/s,
times are SI seconds from the displayed epoch, and angles are degrees in the generated web
dataset. The included scenario is synthetic by design and exists for numerical research and
visualization—not operational collision avoidance.
