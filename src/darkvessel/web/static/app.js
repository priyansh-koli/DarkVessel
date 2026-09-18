/* darkvessel viewer.
 *
 * Two modes, one payload shape:
 *   live    — `api/run` re-runs the pipeline per control change.
 *   static  — a pre-rendered `data/run.json` from `darkvessel render`; controls are read-only.
 *
 * The overlay lives inside a transformed container so pan and zoom stay on the compositor.
 * Marker geometry is in scene-pixel units, restyled on zoom so a marker keeps a constant
 * on-screen size without re-laying out on every pan frame.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";

  const DEFAULTS = {
    tolerance: 200,
    maxgap: 10,
    threshold: 0.5,
    azimuth: true,
    registerTolerance: 100,
  };

  const STATUS_TEXT = {
    matched: "Matched",
    dark: "Dark",
    structure: "Structure",
    unsearched: "Unsearched",
  };

  const state = {
    mode: "probing",     // probing | live | static | error
    payload: null,
    selected: null,      // detection index
    filter: null,        // status string or null
    register: [],        // [{x, y}]
    placing: false,
    zoom: 1,
    pan: { x: 0, y: 0 },
    base: { left: 0, top: 0, w: 0, h: 0 },
    pending: null,       // AbortController
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
    zoomLevel: $("zoom-level"),
    scalebar: $("scalebar"),
  };

  const controls = {
    tolerance: $("tolerance"),
    maxgap: $("maxgap"),
    threshold: $("threshold"),
    azimuth: $("azimuth"),
    registerTolerance: $("register-tolerance"),
  };

  /* ───────────────────────── formatting ───────────────────────── */

  const fmtM = (v, digits = 0) =>
    v === null || v === undefined || Number.isNaN(v) ? "—" : `${Number(v).toFixed(digits)} m`;

  const fmtNum = (v, digits = 2) =>
    v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);

  const fmtSeconds = (s) => {
    if (s === null || s === undefined || Number.isNaN(s)) return "—";
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

  /* ───────────────────────── mode + fetching ───────────────────────── */

  async function boot() {
    bindControls();
    bindStage();
    setStageState("Loading scene…");

    // The content-type check matters: a static host with an SPA fallback answers every
    // unknown path with 200 and an HTML page, which would otherwise look like a live API.
    let live = false;
    try {
      const probe = await fetch("api/health", { cache: "no-store" });
      live = probe.ok && (probe.headers.get("content-type") || "").includes("application/json");
    } catch {
      live = false;
    }

    state.mode = live ? "live" : "static";
    setMode(state.mode);
    els.canvas.dataset.src = live ? "api/scene.png" : "assets/scene.png";

    // `?select=3` deep-links a detection, so a finding can be shared as a URL.
    const requested = Number(new URLSearchParams(window.location.search).get("select"));
    if (Number.isInteger(requested) && requested > 0) state.selected = requested - 1;

    await refresh();
  }

  function setMode(mode) {
    const badge = els.modeBadge;
    badge.className = `mode is-${mode}`;
    const label = badge.querySelector(".mode-label");
    if (mode === "live") {
      label.textContent = "Live — pipeline re-runs on change";
    } else if (mode === "static") {
      label.textContent = "Static build — controls read-only";
      for (const el of Object.values(controls)) el.disabled = true;
      els.addStructure.disabled = true;
      $("reset").disabled = true;
    } else {
      label.textContent = "Cannot reach the pipeline";
    }
  }

  function queryString() {
    const params = new URLSearchParams({
      tolerance_m: controls.tolerance.value,
      max_gap_minutes: controls.maxgap.value,
      detector_threshold: controls.threshold.value,
      apply_azimuth: controls.azimuth.checked ? "true" : "false",
      register_tolerance_m: controls.registerTolerance.value,
    });
    for (const p of state.register) params.append("register", `${p.x},${p.y}`);
    return params.toString();
  }

  async function refresh() {
    if (state.pending) state.pending.abort();
    const controller = new AbortController();
    state.pending = controller;

    try {
      const url = state.mode === "live" ? `api/run?${queryString()}` : "data/run.json";
      const response = await fetch(url, { signal: controller.signal, cache: "no-store" });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      state.payload = await response.json();
      state.pending = null;
      render();
    } catch (error) {
      if (error.name === "AbortError") return;
      state.pending = null;
      state.mode = "error";
      setMode("error");
      setStageState(
        "Could not load a run",
        state.payload
          ? "Showing the last successful result."
          : "Start the server with <code>darkvessel serve</code>, or build a static bundle with <code>darkvessel render</code>."
      );
    }
  }

  const scheduleRefresh = debounce(() => {
    if (state.mode === "live") refresh();
  }, 180);

  function debounce(fn, ms) {
    let handle;
    return (...args) => {
      clearTimeout(handle);
      handle = setTimeout(() => fn(...args), ms);
    };
  }

  /* ───────────────────────── rendering ───────────────────────── */

  function render() {
    const data = state.payload;
    if (!data) return;

    if (state.selected !== null && !data.detections.some((d) => d.index === state.selected)) {
      state.selected = null;
    }

    renderSceneMeta(data.scene);
    ensureSceneImage(data.scene);
    renderCounts(data.counts);
    renderTable(data.detections);
    renderOverlay(data);
    renderAis(data.ais, data.detections);
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
    if (canvas.dataset.loaded === scene.id) return;

    canvas.width = scene.width;
    canvas.height = scene.height;
    const image = new Image();
    image.onload = () => {
      canvas.getContext("2d").drawImage(image, 0, 0);
      canvas.dataset.loaded = scene.id;
      layoutStage();
    };
    image.onerror = () => setStageState("Scene image unavailable", "The overlay is still usable.");
    image.src = canvas.dataset.src;
    layoutStage();
  }

  function renderCounts(counts) {
    const cards = [
      ["total", "Detections", counts.total],
      ["matched", "Matched", counts.matched],
      ["dark", "Dark", counts.dark],
      ["structure", "Structure", counts.structure],
    ];
    if (counts.unsearched) cards.push(["unsearched", "Unsearched", counts.unsearched]);

    els.counts.innerHTML = cards
      .map(([key, label, value]) => {
        const filterable = key !== "total";
        const pressed = state.filter === key;
        return `<button type="button" class="stat stat-${key}" data-filter="${key}"
          aria-pressed="${pressed}" ${filterable && value === 0 ? "disabled" : ""}>
          <span class="stat-value">${value}</span>
          <span class="stat-label">${label}</span>
        </button>`;
      })
      .join("");

    for (const button of els.counts.querySelectorAll(".stat")) {
      button.addEventListener("click", () => {
        const key = button.dataset.filter;
        state.filter = key === "total" || state.filter === key ? null : key;
        render();
      });
    }
  }

  function renderTable(detections) {
    const visible = (d) => !state.filter || d.status === state.filter;
    els.tableCount.textContent = state.filter
      ? `${detections.filter(visible).length} of ${detections.length}`
      : `${detections.length}`;

    if (!detections.length) {
      els.tbody.innerHTML =
        `<tr class="empty-row"><td colspan="6">No detections at this threshold. Lower the detector threshold to find fainter targets.</td></tr>`;
      return;
    }

    els.tbody.innerHTML = detections
      .map((d) => {
        const dim = visible(d) ? "" : " is-dimmed";
        const selected = d.index === state.selected ? " is-selected" : "";
        return `<tr tabindex="0" data-index="${d.index}" class="${dim}${selected}"
            aria-label="Detection ${d.index + 1}, ${STATUS_TEXT[d.status] || d.status}">
          <td>${d.index + 1}</td>
          <td><span class="pill pill-${d.status}">${STATUS_TEXT[d.status] || d.status}</span></td>
          <td class="${d.mmsi ? "" : "muted"}">${d.mmsi ? escapeHtml(d.mmsi) : "—"}</td>
          <td class="right ${d.match_distance_m === null ? "muted" : ""}">${fmtM(d.match_distance_m, 1)}</td>
          <td class="${d.position_basis ? "" : "muted"}">${d.position_basis ? escapeHtml(d.position_basis) : "—"}</td>
          <td class="right ${d.azimuth_shift_m ? "" : "muted"}">${fmtM(d.azimuth_shift_m, 0)}</td>
        </tr>`;
      })
      .join("");

    for (const row of els.tbody.querySelectorAll("tr[data-index]")) {
      const index = Number(row.dataset.index);
      row.addEventListener("click", () => select(index));
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          select(index);
        } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          const rows = [...els.tbody.querySelectorAll("tr[data-index]")];
          const next = rows[rows.indexOf(row) + (event.key === "ArrowDown" ? 1 : -1)];
          if (next) next.focus();
        }
      });
    }
  }

  function renderOverlay(data) {
    const svg = els.overlay;
    svg.setAttribute("viewBox", `0 0 ${data.scene.width} ${data.scene.height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    svg.textContent = "";

    const lines = group(svg, "lines");
    const decls = group(svg, "decls");
    const marks = group(svg, "marks");

    for (const position of data.register) {
      const ring = el("circle", {
        class: "reg-ring",
        cx: position.px,
        cy: position.py,
        r: registerRadiusPx(data),
      });
      const cross = el("path", {
        class: "reg-mark",
        d: crossPath(position.px, position.py, 5),
      });
      lines.append(ring, cross);
    }

    for (const declaration of data.declarations) {
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
          lines.append(
            el("line", {
              class: "match-line",
              x1: declaration.drawn.px, y1: declaration.drawn.py,
              x2: target.px, y2: target.py,
            })
          );
        }
      }

      const dot = el("circle", {
        class: "decl-dot",
        cx: declaration.drawn.px,
        cy: declaration.drawn.py,
        r: 3,
      });
      dot.append(title(declarationTitle(declaration)));
      decls.append(dot);
    }

    for (const detection of data.detections) {
      const dimmed = state.filter && detection.status !== state.filter ? " is-dimmed" : "";
      const selected = detection.index === state.selected ? " is-selected" : "";
      const mark = el("g", {
        class: `mark mark-${detection.status}${dimmed}${selected}`,
        tabindex: "0",
        role: "button",
        "aria-label": `Detection ${detection.index + 1}, ${STATUS_TEXT[detection.status] || detection.status}`,
      });
      mark.dataset.index = String(detection.index);
      mark.append(
        el("rect", { class: "mark-halo", x: 0, y: 0, width: 1, height: 1 }),
        el("rect", { class: "mark-box", x: 0, y: 0, width: 1, height: 1 }),
        el("text", { class: "mark-label", x: 0, y: 0 }, String(detection.index + 1)),
        title(`#${detection.index + 1} · ${STATUS_TEXT[detection.status] || detection.status}`)
      );
      mark.addEventListener("click", (event) => {
        event.stopPropagation();
        select(detection.index);
      });
      mark.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          select(detection.index);
        }
      });
      marks.append(mark);
    }

    restyleMarks();
  }

  function declarationTitle(declaration) {
    const parts = [`MMSI ${declaration.mmsi}`, `position: ${declaration.position_basis}`];
    if (declaration.azimuth_shift_m > 0.5) {
      parts.push(`azimuth shift: ${fmtM(declaration.azimuth_shift_m)}`);
    }
    return parts.join(" · ");
  }

  function registerRadiusPx(data) {
    const registerTolerance = data.config.register_tolerance_m || 100;
    return registerTolerance / data.scene.pixel_size_m;
  }

  /** Size marks in scene units so their on-screen size stays constant across zoom levels. */
  function restyleMarks() {
    const data = state.payload;
    if (!data || !state.base.w) return;

    const unitsPerCssPx = data.scene.width / state.base.w / state.zoom;
    const box = 22 * unitsPerCssPx;
    const dot = 4 * unitsPerCssPx;
    const rawDot = 3.5 * unitsPerCssPx;
    const font = 10 * unitsPerCssPx;

    for (const mark of els.overlay.querySelectorAll(".mark")) {
      const detection = data.detections[Number(mark.dataset.index)];
      if (!detection) continue;
      for (const rect of mark.querySelectorAll("rect")) {
        const side = rect.classList.contains("mark-halo") ? box + 5 * unitsPerCssPx : box;
        rect.setAttribute("x", detection.px - side / 2);
        rect.setAttribute("y", detection.py - side / 2);
        rect.setAttribute("width", side);
        rect.setAttribute("height", side);
      }
      const label = mark.querySelector("text");
      label.setAttribute("x", detection.px + box / 2 + 3 * unitsPerCssPx);
      label.setAttribute("y", detection.py - box / 2 + font);
      label.setAttribute("font-size", font);
    }

    for (const circle of els.overlay.querySelectorAll(".decl-dot")) circle.setAttribute("r", dot);
    for (const circle of els.overlay.querySelectorAll(".decl-raw")) circle.setAttribute("r", rawDot);
  }

  function renderAis(ais, detections) {
    if (!ais) {
      els.aisSummary.innerHTML =
        `<p class="hint" style="margin:0">No AIS archive is configured, so nothing was searched. Every detection is <em>unsearched</em> rather than dark.</p>`;
      return;
    }

    const declared = detections.filter((d) => d.declarations_searched !== null);
    const searched = declared.length ? declared[0].declarations_searched : 0;
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
      <p class="rule-note">A dark claim rests on this: it says what the AIS search actually searched.</p>`;
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

    for (const button of els.registerList.querySelectorAll("[data-remove]")) {
      button.addEventListener("click", () => {
        state.register.splice(Number(button.dataset.remove), 1);
        renderRegister();
        refresh();
      });
    }
  }

  function renderInspector() {
    const data = state.payload;
    const detection = data?.detections.find((d) => d.index === state.selected);

    if (!detection) {
      els.inspector.innerHTML = `
        <h2 class="panel-title">Inspector</h2>
        <div class="inspector-empty">
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6v6H9z"/>
          </svg>
          <p style="margin:0">Select a detection on the scene or in the table to see why it was classified the way it was.</p>
        </div>`;
      return;
    }

    const declaration = data.declarations.find((d) => d.matched_index === detection.index);
    els.inspector.innerHTML = `
      <h2 class="panel-title">Inspector</h2>
      <div class="insp-head">
        ${detection.crop
          ? `<img class="insp-crop" src="${detection.crop}" alt="SAR crop around detection ${detection.index + 1}">`
          : `<div class="insp-crop"></div>`}
        <div class="insp-title">
          <h3>Detection ${detection.index + 1}
            <span class="pill pill-${detection.status}">${STATUS_TEXT[detection.status] || detection.status}</span>
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
        <h4>Contextual layers</h4>
        <dl class="kv">
          <dt>Distance to shore</dt><dd class="zero">${fmtM(detection.distance_to_shore_m)}</dd>
          <dt>Depth</dt><dd class="zero">${fmtM(detection.depth_m)}</dd>
          <dt>Fishing hours</dt><dd class="zero">${fmtNum(detection.fishing_hours, 1)}</dd>
          <dt>EEZ</dt><dd class="zero">${detection.eez ?? "—"}</dd>
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

    if (detection.status === "dark") {
      const searched = detection.declarations_searched ?? 0;
      return `<strong>Searched, and nothing explains it.</strong>
        ${searched} declaration${searched === 1 ? "" : "s"} were placed at the acquisition instant
        and none could be assigned to this detection within ${fmtM(detection.tolerance_m)}.
        A declaration may still lie within tolerance and belong to a nearer detection — the
        assignment is one-to-one.`;
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

  function select(index) {
    state.selected = state.selected === index ? null : index;
    const url = new URL(window.location.href);
    if (state.selected === null) url.searchParams.delete("select");
    else url.searchParams.set("select", String(state.selected + 1));
    history.replaceState(null, "", url);
    for (const row of els.tbody.querySelectorAll("tr[data-index]")) {
      row.classList.toggle("is-selected", Number(row.dataset.index) === state.selected);
    }
    for (const mark of els.overlay.querySelectorAll(".mark")) {
      mark.classList.toggle("is-selected", Number(mark.dataset.index) === state.selected);
    }
    renderInspector();
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
    let dragging = false;
    let moved = false;
    let origin = { x: 0, y: 0 };

    els.stage.addEventListener("pointerdown", (event) => {
      if (state.placing) return;
      dragging = true;
      moved = false;
      origin = { x: event.clientX - state.pan.x, y: event.clientY - state.pan.y };
      els.stage.setPointerCapture(event.pointerId);
    });

    els.stage.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      state.pan.x = event.clientX - origin.x;
      state.pan.y = event.clientY - origin.y;
      if (Math.abs(event.movementX) + Math.abs(event.movementY) > 0) moved = true;
      constrainPan();
      els.inner.style.transform =
        `translate(${state.pan.x}px, ${state.pan.y}px) scale(${state.zoom})`;
    });

    const endDrag = (event) => {
      if (dragging && els.stage.hasPointerCapture?.(event.pointerId)) {
        els.stage.releasePointerCapture(event.pointerId);
      }
      dragging = false;
      if (!moved) updateScalebar();
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
      if (!state.placing || moved) return;
      const point = scenePointFromEvent(event);
      if (!point) return;
      state.register.push(point);
      setPlacing(false);
      renderRegister();
      refresh();
    });

    els.stage.addEventListener("keydown", (event) => {
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
      } else if (event.key === "Escape" && state.placing) {
        setPlacing(false);
      }
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

  function setPlacing(on) {
    state.placing = on;
    els.addStructure.setAttribute("aria-pressed", String(on));
    els.stage.classList.toggle("is-placing", on);
    els.addStructure.lastChild.textContent = on
      ? " Click the scene to register a position"
      : " Register by clicking the scene";
  }

  /* ───────────────────────── stage states ───────────────────────── */

  function setStageState(title, detail = "") {
    els.stageState.hidden = false;
    els.stageState.innerHTML = `<strong>${escapeHtml(title)}</strong>${detail}`;
  }

  function clearStageState() {
    els.stageState.hidden = true;
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
    const outputs = {
      tolerance: (v) => `${v} m`,
      maxgap: (v) => `${v} min`,
      threshold: (v) => Number(v).toFixed(2),
      "register-tolerance": (v) => `${v} m`,
    };

    for (const [id, format] of Object.entries(outputs)) {
      const input = $(id);
      const output = $(`${id}-out`);
      const sync = () => (output.textContent = format(input.value));
      input.addEventListener("input", () => {
        sync();
        scheduleRefresh();
      });
      sync();
    }

    controls.azimuth.addEventListener("change", scheduleRefresh);

    els.addStructure.addEventListener("click", () => setPlacing(!state.placing));
    els.clearStructures.addEventListener("click", () => {
      state.register = [];
      renderRegister();
      refresh();
    });

    $("reset").addEventListener("click", () => {
      controls.tolerance.value = DEFAULTS.tolerance;
      controls.maxgap.value = DEFAULTS.maxgap;
      controls.threshold.value = DEFAULTS.threshold;
      controls.azimuth.checked = DEFAULTS.azimuth;
      controls.registerTolerance.value = DEFAULTS.registerTolerance;
      state.register = [];
      state.filter = null;
      state.selected = null;
      for (const id of ["tolerance", "maxgap", "threshold", "register-tolerance"]) {
        $(id).dispatchEvent(new Event("input"));
      }
      renderRegister();
      refresh();
    });
  }

  boot();
})();
