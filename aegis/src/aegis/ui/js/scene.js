import * as THREE from "three";

const R_EARTH_KM = 6378.137;
const CAM_NEAR = 7050;
const CAM_FAR = 48000;
const BAND_HEX = {
  CLEAR: 0x3dff8a,
  MONITOR: 0xe8c547,
  WATCH: 0xe07a3d,
  ACT: 0xe23b2f,
};

function temeToThree(x, y, z) {
  return new THREE.Vector3(x, z, -y);
}

function bandColor(band) {
  return BAND_HEX[band] || BAND_HEX.CLEAR;
}

function createEarthMaterial() {
  const uniforms = {
    landMap: { value: null },
    hasMap: { value: 0 },
    fallback: { value: new THREE.Color(0x14161b) },
    ambient: { value: 0.62 },
    lightDir: { value: new THREE.Vector3(-0.55, 0.42, 0.72).normalize() },
    lightColor: { value: new THREE.Color(0xf0eee8) },
  };
  return new THREE.ShaderMaterial({
    uniforms,
    vertexShader: `
      varying vec3 vWorldNormal;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D landMap;
      uniform float hasMap;
      uniform vec3 fallback;
      uniform float ambient;
      uniform vec3 lightDir;
      uniform vec3 lightColor;
      varying vec3 vWorldNormal;
      varying vec2 vUv;
      void main() {
        vec3 base = hasMap > 0.5 ? texture2D(landMap, vUv).rgb : fallback;
        float ndotl = max(dot(normalize(vWorldNormal), normalize(lightDir)), 0.0);
        vec3 lit = base * (ambient + (1.0 - ambient) * ndotl * lightColor);
        gl_FragColor = vec4(lit, 1.0);
      }
    `,
  });
}

function disposeObject(obj) {
  if (obj.geometry && !obj.geometry.userData.shared) obj.geometry.dispose();
  if (obj.material) {
    if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
    else obj.material.dispose();
  }
}

export function createGlobe(mount) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(mount.clientWidth, mount.clientHeight);
  renderer.setClearColor(0x07080a, 1);
  mount.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    42,
    mount.clientWidth / Math.max(mount.clientHeight, 1),
    1,
    80000,
  );
  camera.position.set(0, 6200, 22000);

  const earthMat = createEarthMaterial();
  const earth = new THREE.Mesh(new THREE.SphereGeometry(R_EARTH_KM, 96, 72), earthMat);
  scene.add(earth);

  new THREE.TextureLoader().load("/assets/earth-land.png?v=2", (map) => {
    map.colorSpace = THREE.SRGBColorSpace;
    map.anisotropy = renderer.capabilities.getMaxAnisotropy();
    earthMat.uniforms.landMap.value = map;
    earthMat.uniforms.hasMap.value = 1;
  });

  scene.add(
    new THREE.Mesh(
      new THREE.SphereGeometry(R_EARTH_KM * 1.018, 64, 48),
      new THREE.MeshBasicMaterial({
        color: 0x8b8e93,
        transparent: true,
        opacity: 0.045,
        side: THREE.BackSide,
      }),
    ),
  );

  scene.add(new THREE.AmbientLight(0xa8b8c8, 0.85));
  const sun = new THREE.DirectionalLight(0xfff4e6, 1.7);
  sun.position.set(-8000, 6000, 14000);
  scene.add(sun);

  const starPos = new Float32Array(1800 * 3);
  for (let i = 0; i < 1800; i += 1) {
    const u = Math.random() * 2 - 1;
    const v = Math.random() * 2 - 1;
    const w = Math.random() * 2 - 1;
    const n = Math.hypot(u, v, w) || 1;
    starPos.set([(u / n) * 48000, (v / n) * 48000, (w / n) * 48000], i * 3);
  }
  const stars = new THREE.BufferGeometry();
  stars.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
  scene.add(new THREE.Points(stars, new THREE.PointsMaterial({ color: 0x9aa0a8, size: 1.2 })));

  const tracksGroup = new THREE.Group();
  const nodesGroup = new THREE.Group();
  scene.add(tracksGroup, nodesGroup);

  function share(geo) {
    geo.userData.shared = true;
    return geo;
  }
  const busGeo = share(new THREE.BoxGeometry(1.35, 0.16, 0.72));
  const deckGeo = share(new THREE.BoxGeometry(1.18, 0.03, 0.62));
  const boomGeo = share(new THREE.BoxGeometry(0.1, 0.07, 0.62));
  const panelGeo = share(new THREE.BoxGeometry(1.45, 0.055, 1.55));
  const cellGeo = share(new THREE.BoxGeometry(1.25, 0.012, 0.28));
  const dishGeo = share(new THREE.CylinderGeometry(0.22, 0.22, 0.04, 16));
  const mastGeo = share(new THREE.CylinderGeometry(0.025, 0.018, 0.42, 8));
  const nozzleGeo = share(new THREE.ConeGeometry(0.09, 0.22, 10));
  const noseGeo = share(new THREE.BoxGeometry(0.18, 0.1, 0.28));
  const haloGeo = share(new THREE.RingGeometry(1.7, 2.05, 32));
  const pickGeo = share(new THREE.SphereGeometry(2.6, 12, 10));

  const chord = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0xe23b2f }),
  );
  chord.visible = false;
  scene.add(chord);

  const raycaster = new THREE.Raycaster();
  raycaster.params.Line = { threshold: 12 };
  const pointer = new THREE.Vector2();
  const ndc = new THREE.Vector3();
  const _along = new THREE.Vector3();
  const _radial = new THREE.Vector3();
  const _binormal = new THREE.Vector3();
  const _basis = new THREE.Matrix4();

  const state = {
    objects: [],
    conjunctions: [],
    nodes: new Map(),
    tracks: new Map(),
    timeIndex: 0,
    selectedObjectId: null,
    selectedEventId: null,
    hoverId: null,
    onSelect: null,
    onHover: null,
  };

  let dragging = false;
  let moved = 0;
  let lx = 0;
  let ly = 0;
  let yaw = 0.35;
  let pitch = 0.28;

  function viewSize() {
    const canvas = renderer.domElement;
    return {
      w: canvas.clientWidth || mount.clientWidth || 1,
      h: canvas.clientHeight || mount.clientHeight || 1,
    };
  }

  function placeCam() {
    const d = camera.position.length();
    camera.position.set(
      d * Math.cos(pitch) * Math.sin(yaw),
      d * Math.sin(pitch),
      d * Math.cos(pitch) * Math.cos(yaw),
    );
    camera.lookAt(0, 0, 0);
  }

  function zoomT() {
    const d = camera.position.length();
    return THREE.MathUtils.clamp((CAM_FAR - d) / (CAM_FAR - CAM_NEAR), 0, 1);
  }

  function craftScale() {
    const d = camera.position.length();
    const apparent = THREE.MathUtils.lerp(0.008, 0.018, zoomT());
    return THREE.MathUtils.clamp(d * apparent, 85, 220);
  }

  function poseCraft(group, pos, next) {
    group.position.copy(pos);
    _along.copy(next).sub(pos);
    if (_along.lengthSq() < 1e-10) _along.copy(pos).normalize();
    else _along.normalize();
    _radial.copy(pos).normalize();
    _binormal.crossVectors(_radial, _along);
    if (_binormal.lengthSq() < 1e-10) _binormal.set(0, 1, 0);
    else _binormal.normalize();
    _along.crossVectors(_binormal, _radial).normalize();
    _basis.makeBasis(_along, _radial, _binormal);
    group.quaternion.setFromRotationMatrix(_basis);
  }

  function projectLabel(id) {
    const entry = state.nodes.get(id);
    if (!entry) return null;
    const i = Math.max(0, Math.min(state.timeIndex, entry.track.length - 1));
    const p = entry.track[i];
    ndc.copy(p).project(camera);
    if (ndc.z < -1 || ndc.z > 1) return null;
    const { w, h } = viewSize();
    return {
      x: (ndc.x * 0.5 + 0.5) * w,
      y: (-ndc.y * 0.5 + 0.5) * h,
    };
  }

  function emitHover(id, clientX, clientY) {
    if (state.hoverId === id) {
      if (state.onHover) state.onHover({ objectId: id, x: clientX, y: clientY });
      return;
    }
    state.hoverId = id;
    applySelection();
    if (state.onHover) state.onHover({ objectId: id, x: clientX, y: clientY });
  }

  function pick(clientX, clientY) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(
      [...nodesGroup.children, ...tracksGroup.children],
      true,
    );
    for (const hit of hits) {
      let obj = hit.object;
      while (obj && !obj.userData.objectId) obj = obj.parent;
      if (obj?.userData.objectId) return obj.userData.objectId;
    }
    return null;
  }

  mount.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    dragging = true;
    moved = 0;
    lx = event.clientX;
    ly = event.clientY;
  });
  window.addEventListener("pointerup", () => {
    dragging = false;
  });
  window.addEventListener("pointermove", (event) => {
    if (dragging) {
      moved += Math.abs(event.clientX - lx) + Math.abs(event.clientY - ly);
      yaw += (event.clientX - lx) * 0.0055;
      pitch = Math.max(-1.15, Math.min(1.15, pitch + (event.clientY - ly) * 0.0045));
      lx = event.clientX;
      ly = event.clientY;
      return;
    }
    const over = event.target === renderer.domElement || mount.contains(event.target);
    if (!over) {
      emitHover(null, event.clientX, event.clientY);
      mount.style.cursor = "";
      return;
    }
    const id = pick(event.clientX, event.clientY);
    mount.style.cursor = id ? "pointer" : "";
    emitHover(id, event.clientX, event.clientY);
  });
  mount.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const d = camera.position.length();
      const factor = Math.exp(-event.deltaY * 0.0042);
      camera.position.setLength(THREE.MathUtils.clamp(d * factor, CAM_NEAR, CAM_FAR));
    },
    { passive: false },
  );

  mount.addEventListener("click", (event) => {
    if (moved > 6) return;
    const id = pick(event.clientX, event.clientY);
    if (id && state.onSelect) state.onSelect({ objectId: id, conjunctionId: null });
  });

  function onResize() {
    if (!mount.clientWidth || !mount.clientHeight) return;
    camera.aspect = mount.clientWidth / mount.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(mount.clientWidth, mount.clientHeight);
  }
  window.addEventListener("resize", onResize);

  function clearGroup(group) {
    while (group.children.length) {
      const child = group.children[0];
      group.remove(child);
      child.traverse(disposeObject);
    }
  }

  function makeTrack(points, color, opacity) {
    return new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: true,
      }),
    );
  }

  function makeCraft(color) {
    const group = new THREE.Group();
    const busMat = new THREE.MeshPhongMaterial({
      color: 0xb8bcc2,
      emissive: 0x14161a,
      shininess: 50,
    });
    const darkMat = new THREE.MeshPhongMaterial({
      color: 0x2a2d32,
      emissive: 0x0a0b0d,
      shininess: 18,
      side: THREE.DoubleSide,
    });
    const cellMat = new THREE.MeshPhongMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.28,
      shininess: 12,
      side: THREE.DoubleSide,
    });
    const metalMat = new THREE.MeshPhongMaterial({
      color: 0x8b8e93,
      emissive: 0x111214,
      shininess: 70,
    });

    const bus = new THREE.Mesh(busGeo, busMat);
    const deck = new THREE.Mesh(deckGeo, darkMat);
    deck.position.y = -0.1;
    const boomR = new THREE.Mesh(boomGeo, metalMat);
    const boomL = new THREE.Mesh(boomGeo, metalMat);
    boomR.position.set(0, 0.04, 1.05);
    boomL.position.set(0, 0.04, -1.05);
    const wingR = new THREE.Mesh(panelGeo, darkMat);
    const wingL = new THREE.Mesh(panelGeo, darkMat);
    wingR.position.set(0, 0.08, 2.15);
    wingL.position.set(0, 0.08, -2.15);
    wingR.rotation.x = 0.32;
    wingL.rotation.x = -0.32;
    const cells = [];
    for (let i = 0; i < 4; i += 1) {
      const z = 1.55 + i * 0.34;
      const right = new THREE.Mesh(cellGeo, cellMat);
      const left = new THREE.Mesh(cellGeo, cellMat);
      right.position.set(0, 0.12, z);
      left.position.set(0, 0.12, -z);
      right.rotation.x = 0.32;
      left.rotation.x = -0.32;
      cells.push(right, left);
    }
    const dish = new THREE.Mesh(dishGeo, metalMat);
    dish.position.y = -0.14;
    const mast = new THREE.Mesh(mastGeo, metalMat);
    mast.position.set(-0.35, 0.28, 0);
    const nozzle = new THREE.Mesh(nozzleGeo, darkMat);
    nozzle.rotation.z = Math.PI / 2;
    nozzle.position.set(-0.82, 0, 0);
    const nose = new THREE.Mesh(noseGeo, busMat);
    nose.position.set(0.78, 0, 0);

    const halo = new THREE.Mesh(
      haloGeo,
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    halo.rotation.x = Math.PI / 2;
    const pickSphere = new THREE.Mesh(
      pickGeo,
      new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0,
        depthWrite: false,
      }),
    );
    pickSphere.userData.skipFade = true;

    group.add(
      bus,
      deck,
      boomR,
      boomL,
      wingR,
      wingL,
      ...cells,
      dish,
      mast,
      nozzle,
      nose,
      halo,
      pickSphere,
    );
    group.userData.halo = halo;
    group.userData.panels = cells;
    return group;
  }

  function setSceneData(payload) {
    state.objects = payload.objects || [];
    state.conjunctions = payload.conjunctions || [];
    clearGroup(tracksGroup);
    clearGroup(nodesGroup);
    state.nodes.clear();
    state.tracks.clear();

    for (const obj of state.objects) {
      const track = obj.track || [];
      if (!track.length) continue;
      const pts = track.map((p) => temeToThree(p[0], p[1], p[2]));
      const color = bandColor(obj.color_band);
      const dim = obj.color_band === "CLEAR" ? 0.62 : 0.92;
      const line = makeTrack(pts, color, dim);
      line.userData.objectId = obj.id;
      tracksGroup.add(line);
      state.tracks.set(obj.id, line);

      // Uncertainty envelope stays in the inspector (σ RTN). Extra
      // offset traces read as fat / double-struck tracks on the globe.

      const craft = makeCraft(color);
      craft.userData.objectId = obj.id;
      nodesGroup.add(craft);
      state.nodes.set(obj.id, {
        node: craft,
        track: pts,
        color,
        band: obj.color_band,
      });
    }
    applySelection();
    setTimeIndex(state.timeIndex);
  }

  function setTimeIndex(index) {
    state.timeIndex = index;
    const scale = craftScale();
    for (const obj of state.objects) {
      const entry = state.nodes.get(obj.id);
      if (!entry || !entry.track.length) continue;
      const i = Math.max(0, Math.min(index, entry.track.length - 1));
      const next = entry.track[Math.min(i + 1, entry.track.length - 1)];
      poseCraft(entry.node, entry.track[i], next);
      entry.node.scale.setScalar(scale);
    }
  }

  function selectedIds() {
    const ids = new Set();
    if (state.selectedObjectId) ids.add(state.selectedObjectId);
    const event = state.conjunctions.find((row) => row.id === state.selectedEventId);
    if (event) {
      ids.add(event.primary_id);
      ids.add(event.secondary_id);
    }
    return ids;
  }

  function updateChord() {
    const event = state.conjunctions.find((row) => row.id === state.selectedEventId);
    if (event?.tca_primary_km && event.tca_secondary_km) {
      chord.geometry.dispose();
      chord.geometry = new THREE.BufferGeometry().setFromPoints([
        temeToThree(...event.tca_primary_km),
        temeToThree(...event.tca_secondary_km),
      ]);
      chord.material.color.setHex(bandColor(event.display_band || event.risk_level));
      chord.visible = true;
    } else {
      chord.visible = false;
    }
  }

  function applySelection() {
    const selectedIdsNow = selectedIds();

    for (const [id, line] of state.tracks) {
      const selected = selectedIdsNow.has(id);
      const hover = state.hoverId === id;
      const on = !selectedIdsNow.size || selected;
      const entry = state.nodes.get(id);
      line.material.opacity = on ? (selected ? 1 : hover ? 0.9 : 0.72) : 0.12;
      line.material.needsUpdate = true;
    }
    const scale = craftScale();
    for (const [id, entry] of state.nodes) {
      const selected = selectedIdsNow.has(id);
      const hover = state.hoverId === id;
      const on = !selectedIdsNow.size || selected;
      entry.node.visible = true;
      entry.node.scale.setScalar(scale * (selected ? 1.25 : hover ? 1.12 : 1));
      entry.node.traverse((child) => {
        if (!child.material || child === entry.node.userData.halo || child.userData.skipFade) {
          return;
        }
        child.material.transparent = true;
        child.material.opacity = on ? 1 : 0.18;
      });
      const halo = entry.node.userData.halo;
      if (halo) {
        halo.material.opacity = selected ? 0.85 : hover ? 0.45 : 0;
      }
    }
  }

  function setSelection({ objectId = null, conjunctionId = null } = {}) {
    state.selectedObjectId = objectId;
    state.selectedEventId = conjunctionId;
    updateChord();
    applySelection();
  }

  (function loop() {
    requestAnimationFrame(loop);
    placeCam();
    setTimeIndex(state.timeIndex);
    applySelection();
    renderer.render(scene, camera);
  })();

  return {
    setSceneData,
    setTimeIndex,
    setSelection,
    projectLabel,
    set onSelect(fn) {
      state.onSelect = fn;
    },
    set onHover(fn) {
      state.onHover = fn;
    },
    resize: onResize,
  };
}
