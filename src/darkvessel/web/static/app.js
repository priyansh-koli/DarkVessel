/* darkvessel viewer.
 *
 * Three sources, one payload shape:
 *   live      — `api/run` re-runs the pipeline per control change.
 *   static    — `darkvessel render` baked every control position; `data/manifest.json` maps a
 *               position to one of the distinct results in `data/runs/`. The structure
 *               register and the reception floor are applied here, which is exact: each only
 *               rewrites a dark row after matching (see `fusion.register`, `fusion.reception`),
 *               and every row already carries its own `reception_p`. They are re-applied in
 *               the pipeline's order — register first, then the floor — because a registered
 *               structure explains a detection better than a shadow does.
 *   snapshot  — an older bundle with only `data/run.json`; controls are read-only.
 *
 * The overlay lives inside a transformed container so pan and zoom stay on the compositor.
 * Marker geometry is in scene-pixel units, restyled on zoom so a marker keeps a constant
 * on-screen size without re-laying out on every pan frame.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";

  const STATUS_TEXT = {
    matched: "Matched",
    dark: "Dark",
    shadowed: "Shadowed",
    structure: "Structure",
    unsearched: "Unsearched",
  };

  /* The declaration side's verdicts. Only `undetected` is a finding; the others say why the
     radar's silence about a declared vessel means nothing. */
  const DECLARATION_TEXT = {
    explained: "Explained",
    undetected: "Undetected",
    below_detectable: "Below the detector's floor",
    outside_scene: "Outside the scene",
    masked: "On masked land",
  };

  /* Each slider: the request field it drives, its URL key, and how its value reads. In static
     mode the three baked sliders step through the manifest's grid by index instead. */
  const SLIDERS = {
    tolerance: { key: "tolerance_m", url: "tol", baked: true, fmt: (v) => `${v} m` },
    maxgap: { key: "max_gap_minutes", url: "gap", baked: true, fmt: (v) => `${v} min` },
    threshold: { key: "detector_threshold", url: "thr", baked: true, fmt: (v) => Number(v).toFixed(2) },
    "register-tolerance": { key: "register_tolerance_m", url: "regtol", baked: false, fmt: (v) => `${v} m` },
    // Not baked, for the same reason the register is not: it only ever rewrites a dark row
    // after matching, and `reception_p` is already on every row, so it re-applies exactly.
    "reception-floor": { key: "reception_floor", url: "rx", baked: false, fmt: (v) => Number(v).toFixed(2) },
  };

  const state = {
    source: "probing",   // probing | live | static | snapshot
    failed: false,       // the last request failed; the next change retries
    defaults: null,      // request values Reset returns to
    manifest: null,      // static only
    runs: new Map(),     // static only: run id -> Promise<payload>
    receptionRuns: new Map(),  // static only: reception id -> Promise<model>
    payload: null,
    selected: null,      // detection index
    filter: null,        // status string or null
    receptionOverlay: false,  // shade water the archive hears poorly
    register: [],        // [{x, y}] ground coordinates
    placing: false,
    dragged: false,      // the last pointer gesture panned, so its click is not a click
    zoom: 1,
    pan: { x: 0, y: 0 },
    base: { left: 0, top: 0, w: 0, h: 0 },
    pending: null,       // AbortController
    display: "radar",    // radar | sar
    grey: null,          // the scene's greyscale pixels, re-tinted per display
    sweepStart: performance.now(),
  };

  const els = {
    stage: $("viewer"),
    inner: $("stage-inner"),
    canvas: $("scene-canvas"),
    overlay: $("overlay"),
    stageState: $("stage-state"),
    counts: $("counts"),
    tbody: $("detections-body"),
    tableCount: $("table-count"),
    inspector: $("inspector"),
    sceneMeta: $("scene-meta"),
    aisSummary: $("ais-summary"),
    modeBadge: $("mode-badge"),
    registerList: $("register-list"),
    clearStructures: $("clear-structures"),
    addStructure: $("add-structure"),
    declarationSummary: $("declaration-summary"),
    zoomLevel: $("zoom-level"),
    scalebar: $("scalebar"),
    staticNote: $("static-note"),
    toast: $("toast"),
  };

  const azimuth = $("azimuth");
  const receptionOverlay = $("reception-overlay");

  /* ───────────────────────── formatting ───────────────────────── */

  const isAbsent = (v) => v === null || v === undefined || Number.isNaN(v);

  const fmtM = (v, digits = 0) => (isAbsent(v) ? "—" : `${Number(v).toFixed(digits)} m`);

  const fmtNum = (v, digits = 2) => (isAbsent(v) ? "—" : Number(v).toFixed(digits));

  const fmtSeconds = (s) => {
    if (isAbsent(s)) return "—";
    if (s < 90) return `${Math.round(s)} s`;
    return `${(s / 60).toFixed(1)} min`;
  };

  const fmtTime = (iso) => {
    if (!iso) return "—";
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toISOString().replace("T", " ").slice(0, 19) + "Z";
  };

  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const statusText = (status) => STATUS_TEXT[status] || status;

  const declarationText = (status) => DECLARATION_TEXT[status] || status;

  /** Reception as a share, or an em dash where nothing measured it. An unmeasured place and a
      badly heard one must never print the same, which is why this is not `fmtNum(p, 2)`. */
  const fmtReception = (d) =>
    isAbsent(d.reception_p) ? "—" : `${Math.round(d.reception_p * 100)}%`;

  function receptionTitle(d) {
    if (isAbsent(d.reception_p)) {
      return `No reception estimate here: ${d.reception_intervals ?? 0} report interval(s) ` +
        `nearby, too few to say. Reported dark on the search alone.`;
    }
    return `An AIS report would have placed a vessel here at the acquisition instant about ` +
      `${Math.round(d.reception_p * 100)}% of the time, from ${d.reception_intervals} ` +
      `nearby report interval(s).`;
  }

  /* ───────────────────────── controls: values in both modes ───────────────────────── */

  const gridFor = (id) =>
    state.source === "static" && SLIDERS[id].baked ? state.manifest.grid[SLIDERS[id].key] : null;

  function sliderValue(id) {
    const grid = gridFor(id);
    const raw = Number($(id).value);
    return grid ? grid[raw] : raw;
  }

  function setSliderValue(id, value) {
    const grid = gridFor(id);
    $(id).value = grid ? String(nearestIndex(grid, Number(value))) : String(value);
  }

  function nearestIndex(values, target) {
    let best = 0;
    for (let i = 1; i < values.length; i++) {
      if (Math.abs(values[i] - target) < Math.abs(values[best] - target)) best = i;
    }
    return best;
  }

  function syncOutputs() {
    for (const [id, spec] of Object.entries(SLIDERS)) {
      $(`${id}-out`).textContent = spec.fmt(sliderValue(id));
    }
  }

  function request() {
    return {
      tolerance_m: sliderValue("tolerance"),
      max_gap_minutes: sliderValue("maxgap"),
      detector_threshold: sliderValue("threshold"),
      apply_azimuth: azimuth.checked,
      register_tolerance_m: sliderValue("register-tolerance"),
      reception_floor: sliderValue("reception-floor"),
    };
  }

  function applyRequest(values) {
    for (const [id, spec] of Object.entries(SLIDERS)) {
      if (values[spec.key] !== undefined) setSliderValue(id, values[spec.key]);
    }
    if (values.apply_azimuth !== undefined) azimuth.checked = Boolean(values.apply_azimuth);
    syncOutputs();
  }

  function markupDefaults() {
    const read = (id) => Number($(id).getAttribute("value"));
    return {
      tolerance_m: read("tolerance"),
      max_gap_minutes: read("maxgap"),
      detector_threshold: read("threshold"),
      apply_azimuth: azimuth.hasAttribute("checked"),
      register_tolerance_m: read("register-tolerance"),
      reception_floor: read("reception-floor"),
    };
  }

  /* ───────────────────────── boot: find a source ───────────────────────── */

  async function boot() {
    bindControls();
    bindStage();
    bindDelegatedClicks();
    bindDisplay();
    setStageState("Loading scene…");

    // The content-type check matters: a static host with an SPA fallback answers every
    // unknown path with 200 and an HTML page, which would otherwise look like a live API.
    const health = await fetchJson("api/health");
    if (health) {
      state.source = "live";
      state.defaults = { ...markupDefaults(), ...(health.defaults || {}) };
    } else {
      const manifest = await fetchJson("data/manifest.json");
      if (manifest && manifest.version === 1) {
        state.source = "static";
        state.manifest = manifest;
        state.defaults = { ...markupDefaults(), ...manifest.defaults };
        configureBakedSliders();
      } else {
        state.source = "snapshot";
        state.defaults = markupDefaults();
      }
    }

    els.canvas.dataset.src = state.source === "live" ? "api/scene.png" : "assets/scene.png";
    applyRequest(state.defaults);
    readUrlState();
    setMode();
    await refresh();
  }

  async function fetchJson(url) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      const type = response.headers.get("content-type") || "";
      return response.ok && type.includes("application/json") ? await response.json() : null;
    } catch {
      return null;
    }
  }

  /** Static mode: a baked slider steps through the grid's values by index. */
  function configureBakedSliders() {
    for (const id of Object.keys(SLIDERS)) {
      const grid = gridFor(id);
      if (!grid) continue;
      const input = $(id);
      input.min = "0";
      input.max = String(grid.length - 1);
      input.step = "1";
    }
    els.staticNote.hidden = false;
    els.staticNote.innerHTML =
      `Static build: all ${state.manifest.runs.length.toLocaleString()} control positions were
       run ahead of time, so results are instant. Max AIS gap steps through
       ${state.manifest.grid.max_gap_minutes.length} preset values.`;
  }

  function setMode() {
    const badge = els.modeBadge;
    const label = badge.querySelector(".mode-label");
    const mode = state.failed ? "error" : state.source;
    badge.className = `mode is-${mode}`;
    label.textContent = {
      live: "Live — pipeline re-runs on change",
      static: "Static build — every control pre-computed",
      snapshot: "Snapshot — controls read-only",
      error: "Cannot reach the pipeline",
      probing: "connecting…",
    }[mode];

    if (state.source === "snapshot") {
      for (const id of Object.keys(SLIDERS)) $(id).disabled = true;
      azimuth.disabled = true;
      els.addStructure.disabled = true;
      $("reset").disabled = true;
    }
  }

  /* ───────────────────────── fetching ───────────────────────── */

  async function refresh() {
    if (state.pending) state.pending.abort();
    const controller = new AbortController();
    state.pending = controller;
    const busy = setTimeout(() => els.modeBadge.classList.add("is-busy"), 150);

    try {
      const payload = await load(controller.signal);
      if (controller.signal.aborted) return;
      state.payload = payload;
      state.failed = false;
      setMode();
      render();
      writeUrlState();
      closeShare();
    } catch (error) {
      if (error.name === "AbortError") return;
      state.failed = true;
      setMode();
      setStageState(
        "Could not load a run",
        state.payload
          ? "Showing the last successful result. Change a control to try again."
          : "Start the server with <code>darkvessel serve</code>, or build a static bundle with <code>darkvessel render</code>."
      );
    } finally {
      clearTimeout(busy);
      if (state.pending === controller) {
        state.pending = null;
        els.modeBadge.classList.remove("is-busy");
      }
    }
  }

  async function load(signal) {
    if (state.source === "live") return getJson(`api/run?${liveQuery()}`, signal);
    if (state.source === "snapshot") return getJson("data/run.json", signal);
    return bakedPayload();
  }

  async function getJson(url, signal) {
    const response = await fetch(url, { signal, cache: "no-store" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return response.json();
  }

  function liveQuery() {
    const r = request();
    const params = new URLSearchParams({
      tolerance_m: r.tolerance_m,
      max_gap_minutes: r.max_gap_minutes,
      detector_threshold: r.detector_threshold,
      apply_azimuth: r.apply_azimuth ? "true" : "false",
      register_tolerance_m: r.register_tolerance_m,
      reception_floor: r.reception_floor,
    });
    for (const p of state.register) params.append("register", `${p.x},${p.y}`);
    return params.toString();
  }

  /** The baked result for the current controls, with the register applied on top. */
  async function bakedPayload() {
    const { manifest } = state;
    const r = request();
    let flat = 0;
    for (const key of manifest.order) {
      const values = manifest.grid[key];
      flat = flat * values.length + values.indexOf(r[key]);
    }
    const id = manifest.runs[flat];
    // Both files are requested at once: the reception file does not depend on the run.
    const reception = bakedReception(r.max_gap_minutes);
    const base = await cachedJson(state.runs, id, `data/runs/${id}.json`);

    const payload = {
      ...base,
      reception: await reception,
      detections: base.detections.map((d) => ({ ...d })),
      register: state.register.map((p) => ({ ...p, ...groundToPixel(base.scene, p.x, p.y) })),
      config: {
        ...r,
        tile_px: manifest.tile_px,
        overlap_px: manifest.overlap_px,
        register_positions: state.register.length,
      },
    };

    // The run was baked at some floor, so first put every shadowed row back to dark. From
    // there the pipeline's own order applies: register (dark -> structure), then the floor
    // (whatever is still dark -> shadowed). Doing it the other way round would let a shadow
    // hide a registered structure, which the pipeline never does.
    for (const d of payload.detections) {
      if (d.status === "shadowed") d.status = "dark";
    }
    // Mirrors `fusion.register.Register.mark`: only a dark row changes, and only to structure.
    for (const d of payload.detections) {
      if (d.status !== "dark") continue;
      if (state.register.some((p) => Math.hypot(p.x - d.x, p.y - d.y) <= r.register_tolerance_m)) {
        d.status = "structure";
      }
    }
    // Mirrors `fusion.reception.Coverage.mark`: a measured estimate below the floor, never a
    // missing one.
    for (const d of payload.detections) {
      if (d.status === "dark" && !isAbsent(d.reception_p) && d.reception_p < r.reception_floor) {
        d.status = "shadowed";
      }
    }
    payload.counts = countStatuses(payload.detections);
    if (payload.reception) payload.reception = { ...payload.reception, floor: r.reception_floor };
    if (payload.agreement) {
      // `radar_only` is dark plus shadowed, so the floor cannot change it — but recount rather
      // than assert, because a baked payload is data this file did not produce.
      payload.agreement = {
        ...payload.agreement,
        radar_only: payload.counts.dark + payload.counts.shadowed,
      };
    }
    return payload;
  }

  /** The reception model for a max gap. `darkvessel render` writes one file per gap class
      rather than a copy inside every run that shares one, so this is a second lazy fetch. */
  function bakedReception(maxGapMinutes) {
    const index = state.manifest.reception;
    if (!index) return null;
    // Positional, by the gap's place in its own grid — the same lookup the runs use. Keying
    // by the gap's value would mean matching a JSON number against a string Python formatted.
    const id = index[state.manifest.grid.max_gap_minutes.indexOf(maxGapMinutes)];
    if (id === undefined || id === null) return null;
    return cachedJson(state.receptionRuns, id, `data/reception/${id}.json`);
  }

  /** One shared request per baked file. It takes no abort signal on purpose: neighbouring
      control positions share a file, so the refresh that superseded this one may be waiting
      on the same request, and aborting it would leave that refresh with nothing to render.
      `refresh` already ignores a result that arrives after it was superseded. */
  function cachedJson(cache, id, url) {
    if (!cache.has(id)) {
      const pending = getJson(url);
      cache.set(id, pending);
      pending.catch(() => cache.delete(id));
    }
    return cache.get(id);
  }

  function countStatuses(detections) {
    const counts = {
      total: detections.length, matched: 0, dark: 0, shadowed: 0, structure: 0, unsearched: 0,
    };
    for (const d of detections) counts[d.status] = (counts[d.status] || 0) + 1;
    return counts;
  }

  const scheduleRefresh = debounce(() => {
    if (state.source === "live") refresh();
  }, 180);

  /** Live mode waits for the slider to settle; a baked lookup is instant, so it just runs. */
  function controlsChanged() {
    if (state.source === "live") scheduleRefresh();
    else if (state.source === "static") refresh();
  }

  function debounce(fn, ms) {
    let handle;
    return (...args) => {
      clearTimeout(handle);
      handle = setTimeout(() => fn(...args), ms);
    };
  }

  /* ───────────────────────── shareable URL state ───────────────────────── */

  function readUrlState() {
    const params = new URLSearchParams(window.location.search);
    const values = {};
    for (const spec of Object.values(SLIDERS)) {
      const raw = params.get(spec.url);
      if (raw !== null && raw !== "" && Number.isFinite(Number(raw))) values[spec.key] = Number(raw);
    }
    if (params.has("az")) values.apply_azimuth = params.get("az") !== "0";
    if (state.source !== "snapshot") applyRequest(values);

    const reg = params.get("reg");
    if (reg && state.source !== "snapshot") {
      state.register = reg
        .split(";")
        .map((pair) => pair.split(",").map(Number))
        .filter((xy) => xy.length === 2 && xy.every(Number.isFinite))
        .map(([x, y]) => ({ x, y }));
    }

    // `?select=3` deep-links a detection, so a finding can be shared as a URL.
    const requested = Number(params.get("select"));
    if (Number.isInteger(requested) && requested > 0) state.selected = requested - 1;
  }

  /** Only what differs from the defaults goes in the URL, so a plain link stays plain. */
  function writeUrlState() {
    const url = new URL(window.location.href);
    const r = request();
    const d = state.defaults;
    for (const spec of Object.values(SLIDERS)) {
      if (r[spec.key] !== d[spec.key] && state.source !== "snapshot") {
        url.searchParams.set(spec.url, String(r[spec.key]));
      } else {
        url.searchParams.delete(spec.url);
      }
    }
    if (r.apply_azimuth !== d.apply_azimuth) url.searchParams.set("az", r.apply_azimuth ? "1" : "0");
    else url.searchParams.delete("az");

    if (state.register.length) {
      url.searchParams.set(
        "reg",
        state.register.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(";")
      );
    } else {
      url.searchParams.delete("reg");
    }

    if (state.selected === null) url.searchParams.delete("select");
    else url.searchParams.set("select", String(state.selected + 1));
    history.replaceState(null, "", url);
  }

  /* ───────────────────────── rendering ───────────────────────── */

  function render() {
    const data = state.payload;
    if (!data) return;

    if (state.selected !== null && !data.detections.some((d) => d.index === state.selected)) {
      state.selected = null;
    }
    // A filter on a status that no longer occurs would hide everything behind a disabled card.
    if (state.filter && !data.counts[state.filter]) state.filter = null;

    renderSceneMeta(data.scene);
    ensureSceneImage(data.scene);
    renderCounts(data.counts);
    renderTable(data.detections);
    renderOverlay(data);
    renderAis(data.ais, data.detections, data.reception);
    renderDeclarations(data);
    renderRegister();
    renderInspector();
    clearStageState();
  }

  function renderSceneMeta(scene) {
    els.sceneMeta.innerHTML = [
      ["Scene", scene.id],
      ["Acquired", fmtTime(scene.acquired_at)],
      ["CRS", scene.crs],
      ["Pixel", `${scene.pixel_size_m} m`],
      ["Geometry", `${scene.heading_deg}° hdg · ${scene.incidence_deg}° inc`],
    ]
      .map(
        ([term, value]) =>
          `<div><dt>${escapeHtml(term)}</dt><dd title="${escapeHtml(value)}">${escapeHtml(value)}</dd></div>`
      )
      .join("");
  }

  function ensureSceneImage(scene) {
    const canvas = els.canvas;
    if (canvas.dataset.loaded === scene.id || canvas.dataset.loading === scene.id) return;

    canvas.dataset.loading = scene.id;
    canvas.width = scene.width;
    canvas.height = scene.height;
    const image = new Image();
    image.onload = () => {
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      state.grey = context.getImageData(0, 0, canvas.width, canvas.height);
      canvas.dataset.loaded = scene.id;
      delete canvas.dataset.loading;
      paintScene();
      layoutStage();
    };
    image.onerror = () => {
      delete canvas.dataset.loading;
      setStageState("Scene image unavailable", "The overlay is still usable.");
    };
    image.src = canvas.dataset.src;
    layoutStage();
  }

  /** SAR mode shows the pixels as they are; radar mode maps them onto a phosphor ramp. */
  function paintScene() {
    if (!state.grey) return;
    const context = els.canvas.getContext("2d");
    if (state.display === "sar") {
      context.putImageData(state.grey, 0, 0);
      return;
    }
    const src = state.grey.data;
    const out = context.createImageData(state.grey.width, state.grey.height);
    for (let i = 0; i < src.length; i += 4) {
      // A gentle gamma lift so weak returns still read, as on a phosphor screen.
      const v = 255 * Math.pow(src[i] / 255, 0.8);
      out.data[i] = 0.22 * v + 2;
      out.data[i + 1] = v * 0.95 + 10;
      out.data[i + 2] = 0.55 * v + 6;
      out.data[i + 3] = 255;
    }
    context.putImageData(out, 0, 0);
  }

  function renderCounts(counts) {
    const cards = [
      ["total", "Detections", counts.total],
      ["matched", "Matched", counts.matched],
      ["dark", "Dark", counts.dark],
      ["shadowed", "Shadowed", counts.shadowed],
      ["structure", "Structure", counts.structure],
    ];
    if (counts.unsearched) cards.push(["unsearched", "Unsearched", counts.unsearched]);

    els.counts.innerHTML = cards
      .map(([key, label, value]) => {
        const filterable = key !== "total";
        const pressed = key === "total" ? state.filter === null : state.filter === key;
        return `<button type="button" class="stat stat-${key}" data-filter="${key}"
          aria-pressed="${pressed}" ${filterable && value === 0 ? "disabled" : ""}
          title="${filterable ? `Show only ${label.toLowerCase()} detections` : "Show all detections"}">
          <span class="stat-value">${value}</span>
          <span class="stat-label">${label}</span>
        </button>`;
      })
      .join("");
  }

  function renderTable(detections) {
    const visible = (d) => !state.filter || d.status === state.filter;
    els.tableCount.textContent = state.filter
      ? `${detections.filter(visible).length} of ${detections.length}`
      : `${detections.length}`;

    if (!detections.length) {
      els.tbody.innerHTML =
        `<tr class="empty-row"><td colspan="7">No detections at this threshold. Lower the detector threshold to find fainter targets.</td></tr>`;
      return;
    }

    els.tbody.innerHTML = detections
      .map((d) => {
        const dim = visible(d) ? "" : " is-dimmed";
        const selected = d.index === state.selected ? " is-selected" : "";
        return `<tr tabindex="0" data-index="${d.index}" class="${dim}${selected}"
            aria-label="Detection ${d.index + 1}, ${statusText(d.status)}">
          <td>${d.index + 1}</td>
          <td><span class="pill pill-${d.status}">${statusText(d.status)}</span></td>
          <td class="${d.mmsi ? "" : "muted"}">${d.mmsi ? escapeHtml(d.mmsi) : "—"}</td>
          <td class="right ${isAbsent(d.match_distance_m) ? "muted" : ""}">${fmtM(d.match_distance_m, 1)}</td>
          <td class="${d.position_basis ? "" : "muted"}">${d.position_basis ? escapeHtml(d.position_basis) : "—"}</td>
          <td class="right ${d.azimuth_shift_m ? "" : "muted"}">${fmtM(d.azimuth_shift_m, 0)}</td>
          <td class="right ${isAbsent(d.reception_p) ? "muted" : ""}"
              title="${escapeHtml(receptionTitle(d))}">${fmtReception(d)}</td>
        </tr>`;
      })
      .join("");
  }

  function renderOverlay(data) {
    const svg = els.overlay;
    const { scene } = data;
    svg.setAttribute("viewBox", `0 0 ${scene.width} ${scene.height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.textContent = "";

    const defs = el("defs");
    defs.append(el("clipPath", { id: "scene-clip" }));
    defs.firstChild.append(el("rect", { x: 0, y: 0, width: scene.width, height: scene.height }));
    svg.append(defs);
    const reception = group(svg, "reception");
    reception.setAttribute("clip-path", "url(#scene-clip)");
    reception.setAttribute("aria-hidden", "true");
    renderReceptionCells(reception, data);
    const grid = group(svg, "grid");
    grid.setAttribute("clip-path", "url(#scene-clip)");
    const trailLayer = group(svg, "trails");
    const lineLayer = group(svg, "lines");
    const declLayer = group(svg, "decls");
    const marks = group(svg, "marks");
    renderRadarGrid(grid, scene);

    for (const position of data.register) {
      lineLayer.append(
        el("circle", {
          class: "reg-ring",
          cx: position.px,
          cy: position.py,
          r: data.config.register_tolerance_m / scene.pixel_size_m,
        }),
        el("path", { class: "reg-mark", d: crossPath(position.px, position.py, 5) })
      );
    }

    const tolerancePx = data.config.tolerance_m / scene.pixel_size_m;
    const byTarget = new Map();
    for (const declaration of data.declarations) {
      // A declaration and the detection it explains move as one, at the declared velocity.
      const velocity = velocityOf(declaration.course_deg, declaration.speed_kn, scene);
      const lines = mover(lineLayer, velocity);
      const decls = mover(declLayer, velocity);
      trail(trailLayer, declaration.drawn.px, declaration.drawn.py, velocity,
        declaration.matched_index === null || declaration.matched_index === undefined
          ? "unsearched" : "matched");
      lines.append(
        el("circle", {
          class: `tol-ring${declaration.matched_index === null || declaration.matched_index === undefined ? "" : " is-used"}`,
          cx: declaration.drawn.px,
          cy: declaration.drawn.py,
          r: tolerancePx,
        })
      );

      if (declaration.azimuth_shift_m > 0.5) {
        lines.append(
          el("line", {
            class: "shift-line",
            x1: declaration.raw.px, y1: declaration.raw.py,
            x2: declaration.drawn.px, y2: declaration.drawn.py,
          })
        );
        decls.append(
          el("circle", { class: "decl-raw", cx: declaration.raw.px, cy: declaration.raw.py, r: 3 })
        );
      }

      if (declaration.matched_index !== null && declaration.matched_index !== undefined) {
        const target = data.detections[declaration.matched_index];
        if (target) {
          byTarget.set(target.index, { declaration, velocity });
          lines.append(
            el("line", {
              class: "match-line",
              x1: declaration.drawn.px, y1: declaration.drawn.py,
              x2: target.px, y2: target.py,
            })
          );
          // A speed vector from the target, as a radar plotting aid would draw it. Twenty
          // seconds rather than the usual six minutes: this scene is barely a kilometre across.
          if (declaration.course_deg !== null && declaration.speed_kn > 0.5) {
            const metres = (declaration.speed_kn * 1852 * VECTOR_SECONDS) / 3600;
            const length = metres / scene.pixel_size_m;
            const rad = (declaration.course_deg * Math.PI) / 180;
            const tipX = target.px + Math.sin(rad) * length;
            const tipY = target.py - Math.cos(rad) * length;
            lines.append(
              el("line", { class: "vector-line", x1: target.px, y1: target.py, x2: tipX, y2: tipY }),
              // Pinned in screen pixels like the markers, so the head does not grow on zoom.
              el("path", {
                class: "vector-head pinned",
                d: "M0 -5 L4 3 L-4 3 Z",
                "data-anchor": `${tipX},${tipY}`,
                "data-rotate": declaration.course_deg.toFixed(1),
              })
            );
          }
        }
      }

      const dot = el("circle", {
        class: `decl-dot${declaration.status === "undetected" ? " is-undetected" : ""}`,
        cx: declaration.drawn.px,
        cy: declaration.drawn.py,
        r: 3,
      });
      dot.append(title(declarationTitle(declaration)));
      decls.append(dot);

      // A declared vessel the radar drew nothing for gets a ring of its own, so the finding
      // is visible on the scene rather than only in a table.
      if (declaration.status === "undetected") {
        decls.append(
          el("path", {
            class: "decl-undetected pinned",
            d: "M0 -11 L11 0 L0 11 L-11 0 Z",
            "data-anchor": `${declaration.drawn.px},${declaration.drawn.py}`,
          })
        );
      }
    }

    for (const detection of data.detections) {
      const pairing = byTarget.get(detection.index);
      const declaration = pairing?.declaration;
      const velocity = pairing ? pairing.velocity : driftOf(detection, scene);
      if (!pairing) trail(trailLayer, detection.px, detection.py, velocity, detection.status);
      const dimmed = state.filter && detection.status !== state.filter ? " is-dimmed" : "";
      const selected = detection.index === state.selected ? " is-selected" : "";
      const mark = el("g", {
        class: `mark mark-${detection.status}${dimmed}${selected}`,
        tabindex: "0",
        role: "button",
        "aria-label": `Detection ${detection.index + 1}, ${statusText(detection.status)}`,
      });
      mark.dataset.index = String(detection.index);

      // Everything inside `.mark-body` is drawn in CSS pixels around (0, 0); restyleMarks
      // places and scales the group, so a marker keeps its on-screen size at every zoom.
      const body = el("g", { class: "mark-body" });
      body.append(
        el("rect", { class: "mark-halo", x: -17, y: -17, width: 34, height: 34 }),
        el("path", { class: "mark-brackets", d: bracketPath(16, 6) }),
        glyph(detection, declaration),
        markLabel(detection, declaration),
        title(markTitle(detection, declaration))
      );
      mark.append(body);
      mark.style.setProperty("--sweep-delay", `${sweepDelay(detection, scene)}s`);
      setVelocity(mark, velocity);
      marks.append(mark);
    }

    restyleMarks();
    applyMotion();
  }

  /** A status-specific glyph: a hull along its course for a vessel AIS explains, a diamond
      for a dark contact, a boxed cross for a structure, a plain blip where nothing is known. */
  function glyph(detection, declaration) {
    const g = el("g", { class: "glyph" });
    if (detection.status === "matched" && declaration && declaration.course_deg !== null) {
      g.setAttribute("transform", `rotate(${declaration.course_deg.toFixed(1)})`);
      g.append(el("path", { class: "glyph-shape", d: "M0 -12 L6 -3 L6 9 L-6 9 L-6 -3 Z" }));
    } else if (detection.status === "matched") {
      g.append(el("circle", { class: "glyph-shape", r: 6.5 }));
    } else if (detection.status === "dark") {
      g.append(
        el("circle", { class: "glyph-pulse", r: 7 }),
        el("path", { class: "glyph-shape", d: "M0 -8 L8 0 L0 8 L-8 0 Z" })
      );
    } else if (detection.status === "shadowed") {
      // The same diamond as a dark contact, hollow and without the pulse: the radar saw the
      // same thing, and only the strength of the claim about it is different.
      g.append(
        el("path", { class: "glyph-shape", d: "M0 -8 L8 0 L0 8 L-8 0 Z" }),
        el("path", { class: "glyph-detail", d: "M-4.5 0 L4.5 0" })
      );
    } else if (detection.status === "structure") {
      g.append(
        el("rect", { class: "glyph-shape", x: -6, y: -6, width: 12, height: 12 }),
        el("path", { class: "glyph-detail", d: "M-6 -6 L6 6 M6 -6 L-6 6" })
      );
    } else {
      g.append(el("circle", { class: "glyph-shape", r: 5 }));
    }
    return g;
  }

  /** The number, and the speed where known. On a narrow stage neighbouring labels would
      collide, so the speed is hidden there (it stays in the tooltip and the inspector). */
  function markLabel(detection, declaration) {
    const label = el("text", { class: "mark-label", x: 18, y: -10, "font-size": 10.5 });
    label.append(document.createTextNode(String(detection.index + 1)));
    if (declaration && declaration.speed_kn > 0.5) {
      const speed = el("tspan", { class: "mark-speed" });
      speed.textContent = ` · ${declaration.speed_kn.toFixed(1)} kn`;
      label.append(speed);
    }
    return label;
  }

  function markTitle(detection, declaration) {
    const parts = [`#${detection.index + 1} · ${statusText(detection.status)}`];
    if (declaration) {
      parts.push(`MMSI ${declaration.mmsi}`);
      if (declaration.length_m) parts.push(`${Math.round(declaration.length_m)} m long`);
      if (declaration.course_deg !== null) {
        parts.push(`course ${Math.round(declaration.course_deg)}° at ${declaration.speed_kn.toFixed(1)} kn`);
      }
    }
    return parts.join(" · ");
  }

  function bracketPath(r, arm) {
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    return corners
      .map(([sx, sy]) => `M${sx * r} ${sy * (r - arm)} V${sy * r} H${sx * (r - arm)}`)
      .join(" ");
  }

  /* ───────────────────────── radar display ───────────────────────── */

  const SWEEP_SECONDS = 6;
  const VECTOR_SECONDS = 20;

  /* ───────────────────────── motion ─────────────────────────
     The scene is one instant. To read as a live plot, every contact is dead-reckoned forward
     from that instant: a matched vessel at its declared course and speed, a dark or unsearched
     contact at an assumed slow drift (nothing declares its velocity), a structure not at all.
     Time runs MOTION_SPEEDUP times faster than real time and loops every MOTION_LOOP_SECONDS,
     fading out and back in at the join. Classification never moves: it belongs to the pass. */

  const MOTION_SPEEDUP = 3;
  const MOTION_LOOP_SECONDS = 30;
  const MOTION_FADE_SECONDS = 1.2;
  const KN_TO_MS = 1852 / 3600;

  const motion = { elapsed: 0, last: performance.now(), running: true, frame: 0 };

  function velocityOf(courseDeg, speedKn, scene) {
    if (isAbsent(courseDeg) || isAbsent(speedKn) || speedKn < 0.5) return null;
    const pxPerS = (speedKn * KN_TO_MS) / scene.pixel_size_m;
    const rad = (courseDeg * Math.PI) / 180;
    return { vx: Math.sin(rad) * pxPerS, vy: -Math.cos(rad) * pxPerS };
  }

  /** An assumed drift for a contact nothing declares: 2 to 5 knots on a course fixed by its
      position, so it stays the same across re-renders and control changes. */
  function driftOf(detection, scene) {
    if (detection.status === "structure" || detection.status === "matched") return null;
    const seed = Math.abs(Math.sin(detection.x * 12.9898 + detection.y * 78.233) * 43758.5453) % 1;
    return velocityOf((seed * 360 + 25) % 360, 2 + 3 * ((seed * 7) % 1), scene);
  }

  function setVelocity(node, velocity) {
    if (!velocity) return;
    node.dataset.vx = velocity.vx.toFixed(4);
    node.dataset.vy = velocity.vy.toFixed(4);
    node.classList.add("is-moving");
  }

  function mover(layer, velocity) {
    const g = el("g", { class: "mover" });
    setVelocity(g, velocity);
    layer.append(g);
    return g;
  }

  /** A dotted track from where the contact was at the pass to where it is now. */
  function trail(layer, x, y, velocity, status) {
    if (!velocity) return;
    const line = el("line", { class: `trail trail-${status}`, x1: x, y1: y, x2: x, y2: y });
    setVelocity(line, velocity);
    layer.append(line);
  }

  function simSeconds() {
    return (motion.elapsed % MOTION_LOOP_SECONDS) * MOTION_SPEEDUP;
  }

  function applyMotion() {
    const t = simSeconds();
    const intoLoop = motion.elapsed % MOTION_LOOP_SECONDS;
    const fade = Math.min(1, intoLoop / MOTION_FADE_SECONDS, (MOTION_LOOP_SECONDS - intoLoop) / MOTION_FADE_SECONDS);
    for (const node of els.overlay.querySelectorAll(".is-moving")) {
      const dx = Number(node.dataset.vx) * t;
      const dy = Number(node.dataset.vy) * t;
      if (node.tagName === "line") {
        node.setAttribute("x2", Number(node.getAttribute("x1")) + dx);
        node.setAttribute("y2", Number(node.getAttribute("y1")) + dy);
      } else {
        node.setAttribute("transform", `translate(${dx.toFixed(2)} ${dy.toFixed(2)})`);
      }
      node.style.setProperty("--fade", fade.toFixed(3));
    }
    const clock = $("motion-clock");
    if (clock) {
      const whole = Math.floor(t);
      clock.textContent = `T+${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")} after the pass`;
    }
  }

  function tickMotion(now) {
    if (motion.running) motion.elapsed += Math.min(0.1, (now - motion.last) / 1000);
    motion.last = now;
    applyMotion();
    motion.frame = motion.running ? requestAnimationFrame(tickMotion) : 0;
  }

  function setMotion(on) {
    motion.running = on;
    if (on && !motion.frame) {
      motion.last = performance.now();
      motion.frame = requestAnimationFrame(tickMotion);
    }
  }

  /** Range rings around the scene centre, bearing ticks, and a north mark. */
  function renderRadarGrid(layer, scene) {
    const cx = scene.width / 2;
    const cy = scene.height / 2;
    const reach = Math.hypot(cx, cy);
    const inner = Math.min(cx, cy);
    const ringM = niceStep((inner * scene.pixel_size_m) / 3);
    const ringPx = ringM / scene.pixel_size_m;

    for (let r = ringPx, n = 1; r <= reach; r += ringPx, n++) {
      layer.append(el("circle", { class: "radar-ring", cx, cy, r }));
      if (r < inner) {
        const at = Math.SQRT1_2 * r;
        const label = el("text", { class: "radar-label pinned", "data-anchor": `${cx + at},${cy + at}` });
        label.textContent = formatRange(ringM * n);
        layer.append(label);
      }
    }
    for (let bearing = 0; bearing < 360; bearing += 10) {
      const rad = (bearing * Math.PI) / 180;
      const major = bearing % 30 === 0;
      const r0 = inner * (major ? 0.93 : 0.965);
      layer.append(
        el("line", {
          class: major ? "radar-tick is-major" : "radar-tick",
          x1: cx + Math.sin(rad) * r0, y1: cy - Math.cos(rad) * r0,
          x2: cx + Math.sin(rad) * inner, y2: cy - Math.cos(rad) * inner,
        })
      );
      if (major) {
        const label = el("text", {
          class: "radar-label radar-bearing pinned",
          "data-anchor": `${cx + Math.sin(rad) * inner * 0.86},${cy - Math.cos(rad) * inner * 0.86}`,
        });
        label.textContent = bearing === 0 ? "N" : String(bearing).padStart(3, "0");
        layer.append(label);
      }
    }
    layer.append(
      el("line", { class: "radar-axis", x1: cx, y1: cy - inner, x2: cx, y2: cy + inner }),
      el("line", { class: "radar-axis", x1: cx - inner, y1: cy, x2: cx + inner, y2: cy }),
      el("circle", { class: "radar-centre", cx, cy, r: 1.5 })
    );
  }

  function niceStep(target) {
    const magnitude = 10 ** Math.floor(Math.log10(target));
    return magnitude * ([1, 2, 2.5, 5, 10].find((m) => magnitude * m >= target) ?? 10);
  }

  const formatRange = (m) => (m >= 1000 ? `${(m / 1000).toFixed(m % 1000 ? 1 : 0)} km` : `${m} m`);

  /** Grid bearing of a detection from the scene centre, clockwise from north. */
  function bearingOf(detection, scene) {
    const dx = detection.px - scene.width / 2;
    const dy = detection.py - scene.height / 2;
    return ((Math.atan2(dx, -dy) * 180) / Math.PI + 360) % 360;
  }

  /** A negative animation delay that brightens the blip exactly as the sweep crosses it.
      Both animations share the sweep's start time, so they stay in step across re-renders. */
  function sweepDelay(detection, scene) {
    const hitAt = (bearingOf(detection, scene) / 360) * SWEEP_SECONDS;
    const elapsed = ((performance.now() - state.sweepStart) / 1000) % SWEEP_SECONDS;
    return (((hitAt - elapsed) % SWEEP_SECONDS) + SWEEP_SECONDS) % SWEEP_SECONDS - SWEEP_SECONDS;
  }

  function setDisplay(display) {
    state.display = display;
    els.stage.classList.toggle("is-radar", display === "radar");
    for (const button of document.querySelectorAll("[data-display]")) {
      button.setAttribute("aria-pressed", String(button.dataset.display === display));
    }
    try {
      localStorage.setItem("darkvessel.display", display);
    } catch {
      // Storage refused (private window, embedded preview): the choice lasts for this visit.
    }
    paintScene();
  }

  function setSweep(on) {
    els.stage.classList.toggle("is-sweep-paused", !on);
    $("sweep-toggle").setAttribute("aria-pressed", String(on));
    $("sweep-toggle").textContent = on ? "Pause" : "Play";
    setMotion(on);
  }

  function bindDisplay() {
    let saved = null;
    try {
      saved = localStorage.getItem("darkvessel.display");
    } catch {
      saved = null;
    }
    setDisplay(saved === "sar" ? "sar" : "radar");
    for (const button of document.querySelectorAll("[data-display]")) {
      button.addEventListener("click", () => setDisplay(button.dataset.display));
    }
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setSweep(!still);
    $("sweep-toggle").addEventListener("click", () =>
      setSweep(els.stage.classList.contains("is-sweep-paused")));
    // The sweep element and every blip time themselves from this instant.
    $("sweep").style.animationDelay = "0s";
    state.sweepStart = performance.now();
  }

  function declarationTitle(declaration) {
    const parts = [`MMSI ${declaration.mmsi}`, declarationText(declaration.status)];
    parts.push(`position: ${declaration.position_basis}`);
    if (declaration.position_span_s > 0) {
      parts.push(`bracket: ${fmtSeconds(declaration.position_span_s)} wide`);
    }
    if (declaration.azimuth_shift_m > 0.5) {
      parts.push(`azimuth shift: ${fmtM(declaration.azimuth_shift_m)}`);
    }
    if (declaration.status === "undetected") {
      parts.push(`nearest detection: ${fmtM(declaration.nearest_detection_m, 0)}`);
    }
    return parts.join(" · ");
  }

  /** Place marker bodies and grid labels in CSS-pixel units, so they keep their on-screen
      size at every zoom while their positions stay in scene pixels. */
  function restyleMarks() {
    const data = state.payload;
    if (!data || !state.base.w) return;

    const unitsPerCssPx = data.scene.width / state.base.w / state.zoom;
    const u = unitsPerCssPx.toFixed(5);

    for (const mark of els.overlay.querySelectorAll(".mark")) {
      const detection = data.detections[Number(mark.dataset.index)];
      if (!detection) continue;
      mark.firstChild.setAttribute("transform", `translate(${detection.px} ${detection.py}) scale(${u})`);
    }
    for (const node of els.overlay.querySelectorAll(".pinned")) {
      const [x, y] = node.dataset.anchor.split(",");
      const turn = node.dataset.rotate ? ` rotate(${node.dataset.rotate})` : "";
      node.setAttribute("transform", `translate(${x} ${y}) scale(${u})${turn}`);
    }

    const dot = 4 * unitsPerCssPx;
    const rawDot = 3.5 * unitsPerCssPx;
    for (const circle of els.overlay.querySelectorAll(".decl-dot")) circle.setAttribute("r", dot);
    for (const circle of els.overlay.querySelectorAll(".decl-raw")) circle.setAttribute("r", rawDot);
  }

  /** Shade only the water that weakens a dark claim: cells the archive hears worse than the
      floor, and cells nothing measured. Cells above the floor are left clear — the overlay is
      there to show where the search fails, and painting the healthy majority would bury it. */
  function renderReceptionCells(layer, data) {
    if (!state.receptionOverlay || !data.reception) return;
    const floor = data.reception.floor;
    for (const cell of data.reception.cells) {
      const measured = !isAbsent(cell.reception_p);
      if (measured && cell.reception_p >= floor) continue;
      const shortfall = measured && floor > 0 ? (floor - cell.reception_p) / floor : 1;
      layer.append(
        el("rect", {
          class: `rx-cell ${measured ? "rx-low" : "rx-unknown"}`,
          x: cell.px,
          y: cell.py,
          width: cell.pw,
          height: cell.ph,
          "fill-opacity": (measured ? 0.1 + 0.32 * shortfall : 0.1).toFixed(3),
        })
      );
    }
  }

  /** The AIS side of the fusion: what the archive declared that the radar did not draw. */
  function renderDeclarations(data) {
    const counts = data.declaration_counts;
    const agreement = data.agreement;
    if (!counts || !counts.total) {
      els.declarationSummary.innerHTML =
        `<p class="hint" style="margin:0">No declarations were placed at this acquisition
         instant, so there is no other side to report.</p>`;
      return;
    }

    const undetected = data.declarations.filter((d) => d.status === "undetected");
    const recall = agreement && agreement.apparent_recall;

    els.declarationSummary.innerHTML = `
      <dl class="kv">
        <dt>Declarations placed</dt><dd>${counts.total}</dd>
        <dt>Explained by a detection</dt><dd>${counts.explained}</dd>
        <dt><span class="pill pill-undetected">Undetected</span></dt><dd>${counts.undetected}</dd>
        <dt>Below the detector's floor</dt><dd class="${counts.below_detectable ? "" : "zero"}">${counts.below_detectable}</dd>
        <dt>Outside the scene</dt><dd class="${counts.outside_scene ? "" : "zero"}">${counts.outside_scene}</dd>
        ${counts.masked ? `<dt>On masked land</dt><dd>${counts.masked}</dd>` : ""}
      </dl>
      ${undetected.length ? `<ul class="decl-list">${undetected
        .map((d) => `<li><strong>MMSI ${escapeHtml(d.mmsi)}</strong>
          <span>${fmtM(d.length_m)} long · nearest detection ${fmtM(d.nearest_detection_m, 0)}
          · position ${escapeHtml(d.position_basis)}</span></li>`)
        .join("")}</ul>` : ""}
      ${isAbsent(recall) ? "" : `
      <dl class="kv" style="margin-top:12px">
        <dt>Apparent recall</dt><dd>${(recall * 100).toFixed(0)}%</dd>
      </dl>
      <p class="rule-note">The share of declared, detectable, in-scene vessels the radar also
      drew — an estimate of this detector's recall that needs no labels. It is biased: vessels
      that declare themselves are larger and more cooperative than those that do not, and a
      spoofed declaration counts against the detector although nothing was there to find.</p>`}`;
  }

  function renderAis(ais, detections, reception) {
    if (!ais) {
      els.aisSummary.innerHTML =
        `<p class="hint" style="margin:0">No AIS archive is configured, so nothing was searched. Every detection is <em>unsearched</em> rather than dark.</p>`;
      return;
    }

    const declared = detections.find((d) => !isAbsent(d.declarations_searched));
    const searched = declared ? declared.declarations_searched : null;
    const rules = Object.entries(ais.removed);

    els.aisSummary.innerHTML = `
      <dl class="kv">
        <dt>Reports in archive</dt><dd>${ais.rows_in}</dd>
        <dt>Kept after cleaning</dt><dd>${ais.rows_kept}</dd>
        <dt>Distinct vessels</dt><dd>${ais.vessels}</dd>
        <dt>Declarations at acquisition</dt><dd>${searched ?? "—"}</dd>
      </dl>
      ${rules.length ? `<dl class="kv" style="margin-top:12px">${rules
        .map(([rule, count]) =>
          `<dt>${escapeHtml(rule.replace(/_/g, " "))}</dt><dd class="${count ? "" : "zero"}">−${count}</dd>`)
        .join("")}</dl>` : ""}
      <p class="rule-note">A dark claim rests on this: it says what the AIS search actually searched.</p>
      ${reception ? `
      <h3 class="sub-title">Reception</h3>
      <dl class="kv">
        <dt>Estimated on</dt><dd>${reception.report.used} report intervals</dd>
        <dt>Cell</dt><dd>${reception.cell_m} m, pooled over its 8 neighbours</dd>
        <dt>Evidence floor</dt><dd>${reception.min_intervals} intervals</dd>
        <dt>Claim floor</dt><dd>${Number(reception.floor).toFixed(2)}</dd>
      </dl>
      <p class="rule-note">How often the archive would have placed a transmitting vessel at
      this instant, measured from the vessels it does hold, at this max gap. Where it is low, a
      detection's silence is not evidence; where nothing measured it, the row says so and
      stays dark.</p>` : ""}`;
  }

  function renderRegister() {
    els.registerList.innerHTML = state.register
      .map(
        (p, i) =>
          `<li class="chip">${p.x.toFixed(0)}, ${p.y.toFixed(0)}
             <button type="button" data-remove="${i}" aria-label="Remove registered position ${i + 1}">×</button>
           </li>`
      )
      .join("");
    els.clearStructures.hidden = state.register.length === 0;
  }

  function renderInspector() {
    const data = state.payload;
    const detection = data?.detections.find((d) => d.index === state.selected);

    if (!detection) {
      els.inspector.innerHTML = `
        <h2 class="panel-title">Inspector</h2>
        <div class="inspector-empty">
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6v6H9z"/>
          </svg>
          <p style="margin:0">Select a detection on the scene or in the table to see why it was classified the way it was.</p>
          <p style="margin:0"><a href="guide.html">New here? Take the two-minute tour →</a></p>
        </div>`;
      return;
    }

    const declaration = data.declarations.find((d) => d.matched_index === detection.index);
    const position = data.detections.indexOf(detection);
    const previous = data.detections[position - 1];
    const next = data.detections[position + 1];
    els.inspector.innerHTML = `
      <div class="insp-top">
        <h2 class="panel-title">Inspector</h2>
        <div class="insp-nav">
          <button type="button" class="icon-btn" data-step="${previous ? previous.index : ""}"
            ${previous ? "" : "disabled"} aria-label="Previous detection">‹</button>
          <button type="button" class="icon-btn" data-step="${next ? next.index : ""}"
            ${next ? "" : "disabled"} aria-label="Next detection">›</button>
          <button type="button" class="icon-btn" data-close aria-label="Close inspector">×</button>
        </div>
      </div>
      <div class="insp-head">
        ${detection.crop
          ? `<img class="insp-crop" src="${detection.crop}" alt="SAR crop around detection ${detection.index + 1}">`
          : `<div class="insp-crop"></div>`}
        <div class="insp-title">
          <h3>Detection ${detection.index + 1}
            <span class="pill pill-${detection.status}">${statusText(detection.status)}</span>
          </h3>
          <p class="insp-sub">${fmtNum(detection.x, 1)}, ${fmtNum(detection.y, 1)}<br>
             row ${fmtNum(detection.row, 1)} · col ${fmtNum(detection.col, 1)}</p>
        </div>
      </div>

      <div class="verdict verdict-${detection.status}">${verdict(detection, declaration, data)}</div>

      ${declaration ? `
      <div class="insp-section">
        <h4>Declaration</h4>
        <dl class="kv">
          <dt>MMSI</dt><dd>${escapeHtml(declaration.mmsi)}</dd>
          <dt>Declared length</dt><dd>${fmtM(declaration.length_m)}</dd>
          <dt>Position basis</dt><dd>${escapeHtml(declaration.position_basis)}</dd>
          <dt>Report age</dt><dd>${fmtSeconds(declaration.position_age_s)}</dd>
          <dt>Azimuth shift</dt><dd>${fmtM(declaration.azimuth_shift_m, 1)}</dd>
          <dt>Match distance</dt><dd>${fmtM(detection.match_distance_m, 1)}</dd>
        </dl>
      </div>` : ""}

      <div class="insp-section">
        <h4>Evidence searched</h4>
        <dl class="kv">
          <dt>Tolerance</dt><dd>${fmtM(detection.tolerance_m)}</dd>
          <dt>Declarations searched</dt><dd>${detection.declarations_searched ?? "—"}</dd>
          <dt>Acquired at</dt><dd>${fmtTime(detection.acquired_at)}</dd>
          <dt>Scene</dt><dd>${escapeHtml(detection.scene ?? "—")}</dd>
        </dl>
      </div>

      <div class="insp-section">
        <h4>AIS reception here</h4>
        <dl class="kv">
          <dt>Reception</dt><dd class="${isAbsent(detection.reception_p) ? "zero" : ""}">${fmtReception(detection)}</dd>
          <dt>Basis</dt><dd class="${detection.reception_basis === "estimated" ? "" : "zero"}">${escapeHtml((detection.reception_basis ?? "—").replace(/_/g, " "))}</dd>
          <dt>Report intervals</dt><dd>${detection.reception_intervals ?? "—"}</dd>
          <dt>Claim floor</dt><dd>${Number(data.config.reception_floor ?? 0).toFixed(2)}</dd>
        </dl>
        <p class="rule-note">${escapeHtml(receptionTitle(detection))}</p>
      </div>

      <div class="insp-section">
        <h4>Contextual layers</h4>
        <dl class="kv">
          <dt>Distance to shore</dt><dd class="zero">${fmtM(detection.distance_to_shore_m)}</dd>
          <dt>Depth</dt><dd class="zero">${fmtM(detection.depth_m)}</dd>
          <dt>Fishing hours</dt><dd class="zero">${fmtNum(detection.fishing_hours, 1)}</dd>
          <dt>EEZ</dt><dd class="zero">${escapeHtml(detection.eez ?? "—")}</dd>
        </dl>
        <p class="rule-note">Declared but not yet sampled — the schema is fixed so a layer's
        presence never depends on whether the credentialed stage ran.</p>
      </div>`;
  }

  function verdict(detection, declaration, data) {
    if (detection.status === "unsearched") {
      return `<strong>Nothing was searched.</strong>No AIS archive was supplied, so this is not a
        dark claim — only a detection nobody has tried to explain yet.`;
    }

    if (detection.status === "structure") {
      return `<strong>Explained by recurrence, not by AIS.</strong>It stands within
        ${fmtM(data.config.register_tolerance_m)} of a registered fixed position. Structure
        exclusion runs after matching, so a vessel AIS explains is never reclassified this way.`;
    }

    if (detection.status === "shadowed") {
      return `<strong>Searched, but the search could not have reached here.</strong>
        The archive would have placed a transmitting vessel at this position and instant only
        about ${fmtReception(detection)} of the time — below the
        ${Number(data.config.reception_floor ?? 0).toFixed(2)} floor — measured from
        ${detection.reception_intervals} report intervals nearby. So its silence is not
        evidence of anything. Lower the floor to report it dark anyway; the estimate stays on
        the row either way.`;
    }

    if (detection.status === "dark") {
      const searched = detection.declarations_searched ?? 0;
      const reach = isAbsent(detection.reception_p)
        ? `Nothing nearby measures how well the archive hears this water
           (${detection.reception_intervals ?? 0} report interval${detection.reception_intervals === 1 ? "" : "s"}),
           so this rests on the search alone — a missing estimate is never treated as a poor one.`
        : `The archive would have placed a transmitting vessel here about
           ${fmtReception(detection)} of the time, so its silence carries weight.`;
      return `<strong>Searched, and nothing explains it.</strong>
        ${searched} declaration${searched === 1 ? " was" : "s were"} placed at the acquisition
        instant and none could be assigned to this detection within ${fmtM(detection.tolerance_m)}.
        A declaration may still lie within tolerance and belong to a nearer detection — the
        assignment is one-to-one. ${reach}`;
    }

    const bits = [];
    bits.push(`<strong>Explained by MMSI ${escapeHtml(detection.mmsi)}.</strong>`);
    bits.push(`It matched at ${fmtM(detection.match_distance_m, 1)}, inside the
      ${fmtM(detection.tolerance_m)} tolerance.`);

    if (detection.position_basis === "interpolated") {
      bits.push(`Its position was <em>interpolated</em> between two reports bracketing the
        acquisition — a single report taken as it stands could have been far enough away to
        report this vessel dark.`);
    } else if (detection.position_age_s > 0) {
      bits.push(`Its position came from the <em>nearest</em> report,
        ${fmtSeconds(detection.position_age_s)} from the acquisition; nothing is extrapolated
        past the end of a track.`);
    }

    if (declaration && declaration.azimuth_shift_m > 0.5) {
      bits.push(`Because it is moving, the radar draws it ${fmtM(declaration.azimuth_shift_m)}
        along-track from where it declared — the correction was applied before matching.`);
    }

    return bits.join(" ");
  }

  /* ───────────────────────── selection ───────────────────────── */

  function select(index, { toggle = true } = {}) {
    state.selected = toggle && state.selected === index ? null : index;
    writeUrlState();
    for (const row of els.tbody.querySelectorAll("tr[data-index]")) {
      row.classList.toggle("is-selected", Number(row.dataset.index) === state.selected);
    }
    for (const mark of els.overlay.querySelectorAll(".mark")) {
      mark.classList.toggle("is-selected", Number(mark.dataset.index) === state.selected);
    }
    renderInspector();
  }

  /* One listener per container, bound once: the contents are re-rendered on every run. */
  function bindDelegatedClicks() {
    els.counts.addEventListener("click", (event) => {
      const button = event.target.closest(".stat");
      if (!button || button.disabled) return;
      const key = button.dataset.filter;
      state.filter = key === "total" || state.filter === key ? null : key;
      render();
    });

    els.tbody.addEventListener("click", (event) => {
      const row = event.target.closest("tr[data-index]");
      if (row) select(Number(row.dataset.index));
    });
    els.tbody.addEventListener("keydown", (event) => {
      const row = event.target.closest("tr[data-index]");
      if (!row) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select(Number(row.dataset.index));
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const rows = [...els.tbody.querySelectorAll("tr[data-index]")];
        const next = rows[rows.indexOf(row) + (event.key === "ArrowDown" ? 1 : -1)];
        if (next) next.focus();
      }
    });

    // While placing, a click on a marker registers a position there — standing a register on
    // top of a dark detection is the whole point — so it falls through to the stage.
    els.overlay.addEventListener("click", (event) => {
      const mark = event.target.closest(".mark");
      if (!mark || state.placing) return;
      event.stopPropagation();
      if (!state.dragged) select(Number(mark.dataset.index));
    });
    els.overlay.addEventListener("keydown", (event) => {
      const mark = event.target.closest(".mark");
      if (mark && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        event.stopPropagation();
        select(Number(mark.dataset.index));
      }
    });

    els.inspector.addEventListener("click", (event) => {
      const step = event.target.closest("[data-step]");
      if (step && step.dataset.step !== "") {
        select(Number(step.dataset.step), { toggle: false });
        els.inspector.querySelector(`[data-step]:not([disabled])`)?.focus();
      } else if (event.target.closest("[data-close]")) {
        select(state.selected);
      }
    });

    els.registerList.addEventListener("click", (event) => {
      const button = event.target.closest("[data-remove]");
      if (!button) return;
      state.register.splice(Number(button.dataset.remove), 1);
      renderRegister();
      refresh();
    });
  }

  /* ───────────────────────── stage: layout, pan, zoom ───────────────────────── */

  function layoutStage() {
    const scene = state.payload?.scene;
    if (!scene) return;

    const rect = els.stage.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const sceneAspect = scene.width / scene.height;
    let w = rect.width;
    let h = rect.width / sceneAspect;
    if (h > rect.height) {
      h = rect.height;
      w = rect.height * sceneAspect;
    }

    state.base = { w, h, left: (rect.width - w) / 2, top: (rect.height - h) / 2 };
    els.stage.classList.toggle("is-compact", w < 520);
    Object.assign(els.inner.style, {
      left: `${state.base.left}px`,
      top: `${state.base.top}px`,
      width: `${w}px`,
      height: `${h}px`,
      right: "auto",
      bottom: "auto",
    });
    applyTransform();
  }

  function applyTransform() {
    els.inner.style.transform =
      `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`;
    els.zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
    $("zoom-out").disabled = state.zoom <= 1;
    $("zoom-in").disabled = state.zoom >= 12;
    restyleMarks();
    updateScalebar();
  }

  function updateScalebar() {
    const scene = state.payload?.scene;
    if (!scene || !state.base.w) return;

    const metresPerCssPx = (scene.pixel_size_m * scene.width) / (state.base.w * state.zoom);
    const target = 90 * metresPerCssPx;
    const magnitude = 10 ** Math.floor(Math.log10(target));
    const nice = [1, 2, 5, 10].find((m) => magnitude * m >= target) ?? 10;
    const metres = magnitude * nice;

    const bar = els.scalebar.querySelector("span");
    const label = els.scalebar.querySelector("em");
    bar.style.width = `${metres / metresPerCssPx}px`;
    label.textContent = metres >= 1000 ? `${(metres / 1000).toFixed(metres % 1000 ? 1 : 0)} km` : `${metres} m`;
  }

  function zoomTo(nextZoom, focus) {
    const clamped = Math.min(12, Math.max(1, nextZoom));
    const rect = els.stage.getBoundingClientRect();
    const cx = focus ? focus.x - rect.left - state.base.left : state.base.w / 2;
    const cy = focus ? focus.y - rect.top - state.base.top : state.base.h / 2;

    const ix = (cx - state.pan.x) / state.zoom;
    const iy = (cy - state.pan.y) / state.zoom;
    state.zoom = clamped;
    state.pan.x = cx - ix * clamped;
    state.pan.y = cy - iy * clamped;
    constrainPan();
    applyTransform();
  }

  function constrainPan() {
    const { w, h } = state.base;
    const overflowX = w * state.zoom - w;
    const overflowY = h * state.zoom - h;
    state.pan.x = Math.min(0, Math.max(-overflowX, state.pan.x));
    state.pan.y = Math.min(0, Math.max(-overflowY, state.pan.y));
  }

  function bindStage() {
    const DRAG_THRESHOLD_PX = 4;
    let drag = null;

    // Pointer capture starts only once the pointer has really moved. Capturing on
    // pointerdown retargets the click to the stage, and a click on a marker never arrives.
    els.stage.addEventListener("pointerdown", (event) => {
      state.dragged = false;
      if (state.placing || event.button !== 0) return;
      drag = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        pan: { ...state.pan },
        active: false,
      };
    });

    els.stage.addEventListener("pointermove", (event) => {
      if (!drag || event.pointerId !== drag.id) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (!drag.active) {
        if (Math.hypot(dx, dy) < DRAG_THRESHOLD_PX || state.zoom === 1) return;
        drag.active = true;
        state.dragged = true;
        els.stage.setPointerCapture(drag.id);
        els.stage.classList.add("is-dragging");
      }
      state.pan.x = drag.pan.x + dx;
      state.pan.y = drag.pan.y + dy;
      constrainPan();
      els.inner.style.transform =
        `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`;
    });

    const endDrag = (event) => {
      if (!drag || event.pointerId !== drag.id) return;
      if (drag.active && els.stage.hasPointerCapture?.(drag.id)) {
        els.stage.releasePointerCapture(drag.id);
      }
      els.stage.classList.remove("is-dragging");
      drag = null;
    };
    els.stage.addEventListener("pointerup", endDrag);
    els.stage.addEventListener("pointercancel", endDrag);

    els.stage.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const factor = Math.exp(-event.deltaY * 0.0015);
        zoomTo(state.zoom * factor, { x: event.clientX, y: event.clientY });
      },
      { passive: false }
    );

    els.stage.addEventListener("click", (event) => {
      if (!state.placing || state.dragged) return;
      const point = scenePointFromEvent(event);
      if (!point) return;
      state.register.push(point);
      setPlacing(false);
      renderRegister();
      refresh();
    });

    els.stage.addEventListener("keydown", (event) => {
      if (event.target !== els.stage) return;
      const step = 40;
      const keys = {
        ArrowLeft: () => (state.pan.x += step),
        ArrowRight: () => (state.pan.x -= step),
        ArrowUp: () => (state.pan.y += step),
        ArrowDown: () => (state.pan.y -= step),
      };
      if (keys[event.key]) {
        event.preventDefault();
        keys[event.key]();
        constrainPan();
        applyTransform();
      } else if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        zoomTo(state.zoom * 1.3);
      } else if (event.key === "-") {
        event.preventDefault();
        zoomTo(state.zoom / 1.3);
      } else if (event.key === "0") {
        event.preventDefault();
        resetView();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.placing) setPlacing(false);
    });

    $("zoom-in").addEventListener("click", () => zoomTo(state.zoom * 1.3));
    $("zoom-out").addEventListener("click", () => zoomTo(state.zoom / 1.3));
    $("zoom-reset").addEventListener("click", resetView);

    window.addEventListener("resize", debounce(layoutStage, 120));
  }

  function resetView() {
    state.zoom = 1;
    state.pan = { x: 0, y: 0 };
    applyTransform();
  }

  /** Ground coordinates of a click, via the scene transform the payload already carries. */
  function scenePointFromEvent(event) {
    const scene = state.payload?.scene;
    if (!scene || !state.base.w) return null;

    const rect = els.stage.getBoundingClientRect();
    const ix = (event.clientX - rect.left - state.base.left - state.pan.x) / state.zoom;
    const iy = (event.clientY - rect.top - state.base.top - state.pan.y) / state.zoom;
    if (ix < 0 || iy < 0 || ix > state.base.w || iy > state.base.h) return null;

    const col = (ix / state.base.w) * scene.width;
    const row = (iy / state.base.h) * scene.height;

    const [a, b, c, d, e, f] = scene.transform;
    return { x: a * col + b * row + c, y: d * col + e * row + f };
  }

  /** The inverse of the above, for positions that arrive as ground coordinates (a shared URL). */
  function groundToPixel(scene, x, y) {
    const [a, b, c, d, e, f] = scene.transform;
    const det = a * e - b * d;
    return { px: (e * (x - c) - b * (y - f)) / det, py: (a * (y - f) - d * (x - c)) / det };
  }

  function setPlacing(on) {
    state.placing = on;
    els.addStructure.setAttribute("aria-pressed", String(on));
    els.stage.classList.toggle("is-placing", on);
    els.addStructure.lastChild.textContent = on
      ? " Click the scene — Esc to cancel"
      : " Register by clicking the scene";
  }

  /* ───────────────────────── stage states + toast ───────────────────────── */

  function setStageState(title, detail = "") {
    els.stageState.hidden = false;
    els.stageState.innerHTML = `<strong>${escapeHtml(title)}</strong>${detail}`;
  }

  function clearStageState() {
    els.stageState.hidden = true;
  }

  let toastTimer;
  function toast(message) {
    els.toast.textContent = message;
    els.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (els.toast.hidden = true), 2600);
  }

  /* ───────────────────────── share + export ───────────────────────── */

  /* Share opens a panel rather than copying blindly: the clipboard API is refused in an
     embedded browser (an editor preview, an iframe) and `window.prompt` is often suppressed
     there too, which left the button doing nothing at all. The panel always shows the link. */
  const share = {
    button: $("copy-link"),
    pop: $("share-pop"),
    url: $("share-url"),
    status: $("share-status"),
    includes: $("share-includes"),
    native: $("share-native"),
  };

  function openShare() {
    writeUrlState();
    share.url.value = window.location.href;
    share.includes.innerHTML = shareSummary().map((line) => `<li>${escapeHtml(line)}</li>`).join("");
    share.status.textContent = "";
    share.native.hidden = typeof navigator.share !== "function";
    share.pop.hidden = false;
    share.button.setAttribute("aria-expanded", "true");
    share.url.focus();
    share.url.select();
    copyShareLink();
  }

  function closeShare({ restoreFocus = false } = {}) {
    if (share.pop.hidden) return;
    share.pop.hidden = true;
    share.button.setAttribute("aria-expanded", "false");
    if (restoreFocus) share.button.focus();
  }

  /** What the link will restore, so it is visible that it carries the view, not just the page. */
  function shareSummary() {
    const r = request();
    const d = state.defaults;
    const lines = [];
    if (r.tolerance_m !== d.tolerance_m) lines.push(`Match tolerance ${r.tolerance_m} m`);
    if (r.max_gap_minutes !== d.max_gap_minutes) lines.push(`Max AIS gap ${r.max_gap_minutes} min`);
    if (r.detector_threshold !== d.detector_threshold) {
      lines.push(`Brightness threshold ${Number(r.detector_threshold).toFixed(2)}`);
    }
    if (r.apply_azimuth !== d.apply_azimuth) {
      lines.push(`Azimuth correction ${r.apply_azimuth ? "on" : "off"}`);
    }
    if (state.register.length) {
      lines.push(`${state.register.length} registered position${state.register.length === 1 ? "" : "s"}`);
    }
    if (r.register_tolerance_m !== d.register_tolerance_m && state.register.length) {
      lines.push(`Same-position tolerance ${r.register_tolerance_m} m`);
    }
    if (r.reception_floor !== d.reception_floor) {
      lines.push(`AIS reception floor ${Number(r.reception_floor).toFixed(2)}`);
    }
    if (state.selected !== null) lines.push(`Detection ${state.selected + 1} selected`);
    return lines.length ? lines : ["Default settings — change a control to share a specific view"];
  }

  async function copyShareLink() {
    const link = share.url.value;
    let copied = false;
    try {
      await navigator.clipboard.writeText(link);
      copied = true;
    } catch {
      share.url.focus();
      share.url.select();
      try {
        copied = document.execCommand("copy");
      } catch {
        copied = false;
      }
    }
    if (copied) {
      share.status.textContent = "Copied to the clipboard.";
      share.status.className = "share-status is-ok";
    } else {
      share.url.select();
      share.status.textContent = `This browser blocked copying — the link is selected, press ${
        /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘C" : "Ctrl+C"}.`;
      share.status.className = "share-status";
    }
  }

  function bindShare() {
    share.button.addEventListener("click", () => (share.pop.hidden ? openShare() : closeShare()));
    $("share-copy").addEventListener("click", copyShareLink);
    share.native.addEventListener("click", async () => {
      try {
        await navigator.share({ title: document.title, url: share.url.value });
      } catch {
        // Dismissed by the person, or refused by the browser; the link is still in the field.
      }
    });
    share.pop.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        closeShare({ restoreFocus: true });
      }
    });
    document.addEventListener("pointerdown", (event) => {
      if (!event.target.closest(".share")) closeShare();
    });
  }

  const CSV_COLUMNS = [
    ["detection", (d) => d.index + 1],
    ["status", (d) => d.status],
    ["mmsi", (d) => d.mmsi],
    ["x", (d) => d.x],
    ["y", (d) => d.y],
    ["row", (d) => d.row],
    ["col", (d) => d.col],
    ["match_distance_m", (d) => d.match_distance_m],
    ["position_basis", (d) => d.position_basis],
    ["position_age_s", (d) => d.position_age_s],
    ["azimuth_shift_m", (d) => d.azimuth_shift_m],
    ["tolerance_m", (d) => d.tolerance_m],
    ["declarations_searched", (d) => d.declarations_searched],
    ["reception_p", (d) => d.reception_p],
    ["reception_basis", (d) => d.reception_basis],
    ["reception_intervals", (d) => d.reception_intervals],
    ["acquired_at", (d) => d.acquired_at],
    ["scene", (d) => d.scene],
  ];

  /* The other side of the fusion gets its own file: it has one row per declaration, not per
     detection, so folding it into the detections CSV would mean a table of two shapes. */
  const DECLARATION_CSV_COLUMNS = [
    ["mmsi", (d) => d.mmsi],
    ["status", (d) => d.status],
    ["length_m", (d) => d.length_m],
    ["detection", (d) => (isAbsent(d.matched_index) ? null : d.matched_index + 1)],
    ["nearest_detection_m", (d) => d.nearest_detection_m],
    ["position_basis", (d) => d.position_basis],
    ["position_age_s", (d) => d.position_age_s],
    ["position_span_s", (d) => d.position_span_s],
    ["azimuth_shift_m", (d) => d.azimuth_shift_m],
    ["declared_x", (d) => d.raw.x],
    ["declared_y", (d) => d.raw.y],
    ["drawn_x", (d) => d.drawn.x],
    ["drawn_y", (d) => d.drawn.y],
  ];

  function csvCell(v) {
    if (isAbsent(v)) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  function csvText(columns, rows) {
    const lines = [columns.map(([name]) => name).join(",")];
    for (const row of rows) lines.push(columns.map(([, get]) => csvCell(get(row))).join(","));
    return lines.join("\n") + "\n";
  }

  function exportCsv() {
    const data = state.payload;
    if (!data) return;
    download(
      `${data.scene.id}-detections.csv`,
      csvText(CSV_COLUMNS, data.detections),
      "text/csv"
    );
  }

  function exportDeclarations() {
    const data = state.payload;
    if (!data || !data.declarations.length) return;
    download(
      `${data.scene.id}-declarations.csv`,
      csvText(DECLARATION_CSV_COLUMNS, data.declarations),
      "text/csv"
    );
  }

  function exportJson() {
    const data = state.payload;
    if (!data) return;
    // Crops are base64 images; a run export is for analysis, so they are left out.
    const run = { ...data, detections: data.detections.map(({ crop, ...rest }) => rest) };
    download(`${data.scene.id}-run.json`, JSON.stringify(run, null, 2), "application/json");
  }

  function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ───────────────────────── svg helpers ───────────────────────── */

  function el(name, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function group(parent, name) {
    const g = el("g", { "data-layer": name });
    // Only the marks carry meaning: each is a focusable button naming its detection. The rest
    // of the overlay is decoration for the eye, and is hidden from assistive technology here
    // rather than on the whole overlay, which would have hidden the marks with it.
    if (name !== "marks") g.setAttribute("aria-hidden", "true");
    parent.append(g);
    return g;
  }

  function title(text) {
    return el("title", {}, text);
  }

  function crossPath(x, y, r) {
    return `M${x - r} ${y} H${x + r} M${x} ${y - r} V${y + r}`;
  }

  /* ───────────────────────── controls ───────────────────────── */

  function bindControls() {
    for (const [id, spec] of Object.entries(SLIDERS)) {
      const input = $(id);
      const output = $(`${id}-out`);
      input.addEventListener("input", () => {
        output.textContent = spec.fmt(sliderValue(id));
        // The register is applied in the browser in static mode, and re-run live otherwise.
        controlsChanged();
      });
    }
    syncOutputs();

    azimuth.addEventListener("change", controlsChanged);

    // Purely a view: nothing is re-run, the cells are already in the payload.
    receptionOverlay.addEventListener("change", () => {
      state.receptionOverlay = receptionOverlay.checked;
      render();
    });

    els.addStructure.addEventListener("click", () => setPlacing(!state.placing));
    els.clearStructures.addEventListener("click", () => {
      state.register = [];
      renderRegister();
      refresh();
    });

    $("reset").addEventListener("click", () => {
      applyRequest(state.defaults);
      state.register = [];
      state.filter = null;
      state.selected = null;
      setPlacing(false);
      resetView();
      renderRegister();
      refresh();
    });

    // An "i" inside the azimuth switch's label must open its tip, not flip the switch.
    for (const button of document.querySelectorAll(".info")) {
      button.addEventListener("click", (event) => event.preventDefault());
    }

    bindShare();
    $("export-csv").addEventListener("click", exportCsv);
    $("export-declarations").addEventListener("click", exportDeclarations);
    $("export-json").addEventListener("click", exportJson);
  }

  boot();
})();
