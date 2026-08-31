import { createGlobe } from "./scene.js?v=15";

const BAND_CLASS = {
  CLEAR: "c-ok",
  MONITOR: "c-mon",
  WATCH: "c-watch",
  ACT: "c-act",
};
const PILL_CLASS = {
  CLEAR: "clear",
  MONITOR: "mon",
  WATCH: "watch",
  ACT: "act",
};
const BAND_SHORT = {
  CLEAR: "CLR",
  MONITOR: "MON",
  WATCH: "WATCH",
  ACT: "ACT",
};

const globe = createGlobe(document.getElementById("globe"));

const els = {
  statusSource: document.getElementById("status-source"),
  statusFallback: document.getElementById("status-fallback"),
  statusCov: document.getElementById("status-cov"),
  statusN: document.getElementById("status-n"),
  srcPath: document.getElementById("src-path"),
  srcFetch: document.getElementById("src-fetch"),
  srcQuery: document.getElementById("src-query"),
  srcEpoch: document.getElementById("src-epoch"),
  srcCov: document.getElementById("src-cov"),
  honesty: document.getElementById("honesty"),
  objectSlider: document.getElementById("object-slider"),
  objectCountLabel: document.getElementById("object-count-label"),
  catWindow: document.getElementById("cat-window"),
  catStep: document.getElementById("cat-step"),
  catBox: document.getElementById("cat-box"),
  selectedTitle: document.getElementById("selected-title"),
  selectedPills: document.getElementById("selected-pills"),
  selectedKv: document.getElementById("selected-kv"),
  selectedFlags: document.getElementById("selected-flags"),
  eventsBody: document.getElementById("events-body"),
  objectsBody: document.getElementById("objects-body"),
  planKv: document.getElementById("plan-kv"),
  timeSlider: document.getElementById("time-slider"),
  scrubStart: document.getElementById("scrub-start"),
  scrubEnd: document.getElementById("scrub-end"),
  satHover: document.getElementById("sat-hover"),
};

let scene = null;
let selected = { objectId: null, conjunctionId: null };
let fetchTimer = 0;

function fmtIso(text) {
  if (!text) return "—";
  return String(text).replace(/\+00:00$/, "Z").replace(/\.\d+Z$/, "Z");
}

function fmtPc(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toExponential(1);
}

function fmtNum(value, digits) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function tPlus(seconds) {
  const sign = seconds < 0 ? "-" : "+";
  const abs = Math.abs(Math.round(seconds));
  const hh = String(Math.floor(abs / 3600)).padStart(2, "0");
  const mm = String(Math.floor((abs % 3600) / 60)).padStart(2, "0");
  return `T${sign}${hh}:${mm}`;
}

function tPlusFull(seconds) {
  const sign = seconds < 0 ? "-" : "+";
  const abs = Math.abs(Math.round(seconds));
  const hh = String(Math.floor(abs / 3600)).padStart(2, "0");
  const mm = String(Math.floor((abs % 3600) / 60)).padStart(2, "0");
  const ss = String(abs % 60).padStart(2, "0");
  return `T${sign}${hh}:${mm}:${ss}`;
}

function objectById(id) {
  return (scene?.objects || []).find((row) => row.id === id);
}

function eventById(id) {
  return (scene?.conjunctions || []).find((row) => row.id === id);
}

function bandOf(row) {
  return row.display_band || row.color_band || row.risk_level || "CLEAR";
}

function setKv(dl, rows) {
  dl.innerHTML = rows
    .map(([k, v, cls]) => `<dt>${k}</dt><dd class="${cls || ""}">${v}</dd>`)
    .join("");
}

function renderStatus() {
  if (!scene) return;
  els.statusSource.textContent = scene.source || "CELESTRAK";
  const fallback = scene.fallback;
  if (fallback === "slice") {
    els.statusFallback.textContent = "FALLBACK SLICE";
    els.statusFallback.classList.remove("hidden");
  } else {
    els.statusFallback.textContent = "";
    els.statusFallback.classList.add("hidden");
  }
  els.statusCov.textContent = scene.covariance_source || "SYNTHETIC_TLE";
  els.statusN.textContent = `N=${scene.objects?.length ?? 0}`;
}

function renderSource() {
  if (!scene) return;
  const group = /group=([^\s]+)/i.exec(scene.query || "")?.[1] || "STARLINK";
  els.srcPath.textContent = `${scene.source || "CELESTRAK"} / ${String(group).toUpperCase()}`;
  els.srcFetch.textContent = fmtIso(scene.fetched_at);
  els.srcQuery.textContent = scene.query || "—";
  els.srcEpoch.textContent = fmtIso(scene.epoch);
  els.srcCov.textContent = scene.covariance_source || "—";
  els.honesty.textContent = (scene.honesty || []).join("\n");
}

function renderCatalog() {
  if (!scene) return;
  const n = scene.max_objects || scene.objects?.length || 0;
  els.objectSlider.value = String(n);
  els.objectCountLabel.textContent = `${n} / 200`;
  els.catWindow.textContent = `${Math.round(scene.duration_s)} s`;
  els.catStep.textContent = `${Math.round(scene.step_s)} s`;
  const box = scene.box_km || [2, 44, 51];
  els.catBox.textContent = `${box[0]} × ${box[1]} × ${box[2]} km`;
}

function renderPlan() {
  if (!scene) return;
  const summary = scene.plan?.summary || {};
  const unresolved = scene.plan?.unresolved || [];
  const first = unresolved[0];
  const unresolvedText = first
    ? `${unresolved.length} · ${fmtNum(first.shortfall_km, 2)} km short`
    : "0";
  setKv(els.planKv, [
    ["BURNS", String(summary.total_burns ?? (scene.plan?.burns || []).length)],
    ["ΔV", `${fmtNum(summary.total_delta_v_mm_s, 1)} mm/s`],
    [
      "RESOLVED",
      `${summary.conjunctions_resolved ?? 0} / ${summary.conjunctions_addressed ?? 0}`,
    ],
    ["UNRESOLVED", unresolvedText],
    ["CONVERGED", summary.converged ? "TRUE" : "FALSE"],
  ]);
}

function altitudeKm(obj) {
  const track = obj?.track || [];
  if (!track.length) return null;
  const i = Math.max(0, Math.min(Number(els.timeSlider.value) || 0, track.length - 1));
  const p = track[i];
  return Math.hypot(p[0], p[1], p[2]) - 6378.137;
}

function renderObjects() {
  if (!els.objectsBody) return;
  els.objectsBody.innerHTML = "";
  for (const obj of scene?.objects || []) {
    const row = document.createElement("div");
    row.className = `object-row${obj.id === selected.objectId ? " sel" : ""}`;
    row.innerHTML = `<span class="name">${obj.name || obj.id}</span><span class="${BAND_CLASS[obj.color_band] || ""}">${BAND_SHORT[obj.color_band] || obj.color_band}</span>`;
    row.addEventListener("click", () => select({ objectId: obj.id, conjunctionId: null }));
    els.objectsBody.appendChild(row);
    if (obj.id === selected.objectId) row.scrollIntoView({ block: "nearest" });
  }
}

function renderEvents() {
  const rows = scene?.conjunctions || [];
  els.eventsBody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    const band = bandOf(row);
    if (row.id === selected.conjunctionId) tr.classList.add("sel");
    tr.innerHTML = `
      <td>${row.primary_id}/${row.secondary_id}</td>
      <td class="${BAND_CLASS[band] || ""}">${fmtPc(row.pc)}</td>
      <td>${fmtNum(row.miss_km, 2)}</td>
      <td class="${BAND_CLASS[band] || ""}">${BAND_SHORT[band] || band}</td>`;
    tr.addEventListener("click", () => select({ objectId: row.primary_id, conjunctionId: row.id }));
    els.eventsBody.appendChild(tr);
  }
}

function nearestTimeIndex(iso) {
  if (!scene?.times_s?.length || !scene.epoch || !iso) return 0;
  const dt = (Date.parse(iso) - Date.parse(scene.epoch)) / 1000;
  let best = 0;
  let bestDiff = Infinity;
  scene.times_s.forEach((t, i) => {
    const d = Math.abs(t - dt);
    if (d < bestDiff) {
      bestDiff = d;
      best = i;
    }
  });
  return best;
}

function renderSelected() {
  const event = eventById(selected.conjunctionId);
  const obj = objectById(selected.objectId);
  if (!event && !obj) {
    els.selectedTitle.textContent = "SELECTED";
    els.selectedPills.innerHTML = "";
    setKv(els.selectedKv, [["PC ALFANO", "—"]]);
    els.selectedFlags.innerHTML = `<p class="pick-hint">Click a satellite, a track, or a row</p>`;
    return;
  }
  if (event) {
    const band = bandOf(event);
    const inflated = event.display_band && event.display_band !== event.risk_level;
    els.selectedTitle.textContent = `SELECTED  ·  ${event.primary_id} / ${event.secondary_id}`;
    els.selectedPills.innerHTML =
      `<span class="pill ${PILL_CLASS[band] || ""}">${band}</span>` +
      (inflated
        ? `<span class="pill" style="margin-left:6px;color:var(--muted)">DISPLAY +1 TLE</span>`
        : "");
    const primary = objectById(event.primary_id);
    const sig = primary?.sigma_rtn_km || [0, 0, 0];
    const tcaOff = scene?.epoch ? (Date.parse(event.tca) - Date.parse(scene.epoch)) / 1000 : 0;
    const box = scene?.box_km || [2, 44, 51];
    setKv(els.selectedKv, [
      ["PC ALFANO", fmtPc(event.pc), BAND_CLASS[event.risk_level] || ""],
      ["PC CHAN", event.pc_chan === undefined ? "—" : fmtPc(event.pc_chan)],
      ["MISS", `${fmtNum(event.miss_km, 2)} km`],
      ["TCA", tPlusFull(tcaOff)],
      ["VREL", `${fmtNum(event.relative_speed_km_s, 2)} km/s`],
      ["MAHAL", fmtNum(event.mahalanobis, 1)],
      ["σ RTN", `${fmtNum(sig[0], 2)} / ${fmtNum(sig[1], 2)} / ${fmtNum(sig[2], 2)} km`],
      ["BOX RTN", `${box[0]} × ${box[1]} × ${box[2]} km`],
    ]);
    const flagNames = [
      ["DILUTION", event.dilution],
      ["REMEDIATED", event.remediated],
      ["SHORT OK", event.short_encounter_valid],
      ["LOW VREL", event.low_relative_velocity],
    ];
    els.selectedFlags.innerHTML = flagNames
      .map(([label, on]) => `<span class="flag${on ? " on" : ""}">${label}</span>`)
      .join("");
    return;
  }
  els.selectedTitle.textContent = `SELECTED  ·  ${obj.id}`;
  els.selectedPills.innerHTML = `<span class="pill ${PILL_CLASS[obj.color_band] || ""}">${obj.color_band}</span>`;
  const sig = obj.sigma_rtn_km || [0, 0, 0];
  const alt = altitudeKm(obj);
  const eventCount = (obj.conjunction_ids || []).length;
  setKv(els.selectedKv, [
    ["NAME", obj.name || obj.id],
    ["NORAD", obj.id],
    ["ALT", alt === null ? "—" : `${fmtNum(alt, 1)} km`],
    ["BAND", obj.color_band],
    ["EVENTS", String(eventCount)],
    ["σ RTN", `${fmtNum(sig[0], 2)} / ${fmtNum(sig[1], 2)} / ${fmtNum(sig[2], 2)} km`],
  ]);
  els.selectedFlags.innerHTML = eventCount
    ? `<p class="pick-hint">Open an event in the list to see Pc / miss</p>`
    : "";
}

function renderScrub() {
  if (!scene) return;
  const times = scene.times_s || [];
  const last = times.length ? times[times.length - 1] : scene.duration_s || 0;
  els.timeSlider.max = String(Math.max(0, times.length - 1));
  els.scrubStart.textContent = tPlus(times[0] || 0);
  els.scrubEnd.textContent = tPlus(last);
}

function select(next) {
  selected = next;
  globe.setSelection(next);
  if (next.conjunctionId) {
    els.timeSlider.value = String(nearestTimeIndex(eventById(next.conjunctionId)?.tca));
    globe.setTimeIndex(Number(els.timeSlider.value));
  }
  renderEvents();
  renderObjects();
  renderSelected();
}

globe.onSelect = (payload) => {
  select({ objectId: payload.objectId, conjunctionId: null });
};

globe.onHover = (payload) => {
  if (!els.satHover) return;
  if (!payload.objectId) {
    els.satHover.classList.add("hidden");
    return;
  }
  const obj = objectById(payload.objectId);
  const screen = globe.projectLabel(payload.objectId);
  els.satHover.textContent = obj ? `${obj.id}  ${obj.name || ""}`.trim() : payload.objectId;
  els.satHover.classList.remove("hidden");
  if (screen) {
    els.satHover.style.left = `${screen.x}px`;
    els.satHover.style.top = `${screen.y}px`;
  }
};

function applyScene(payload) {
  scene = payload;
  globe.setSceneData(payload);
  renderStatus();
  renderSource();
  renderCatalog();
  renderPlan();
  renderEvents();
  renderObjects();
  renderSelected();
  renderScrub();
  globe.setTimeIndex(Number(els.timeSlider.value) || 0);
}

async function loadScene(maxObjects) {
  const params = new URLSearchParams({
    max_objects: String(maxObjects),
    duration_s: "5400",
    step_s: "60",
    live: "true",
  });
  els.statusSource.textContent = "LOADING";
  const response = await fetch(`/api/scene?${params}`);
  if (!response.ok) {
    els.honesty.textContent = `scene request failed (${response.status})`;
    return;
  }
  applyScene(await response.json());
}

els.timeSlider.addEventListener("input", () => {
  globe.setTimeIndex(Number(els.timeSlider.value));
  if (selected.objectId && !selected.conjunctionId) renderSelected();
});

els.objectSlider.addEventListener("input", () => {
  els.objectCountLabel.textContent = `${els.objectSlider.value} / 200`;
  window.clearTimeout(fetchTimer);
  fetchTimer = window.setTimeout(() => {
    loadScene(Number(els.objectSlider.value));
  }, 280);
});

loadScene(Number(els.objectSlider.value) || 40);
