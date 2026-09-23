# darkvessel — final goal

## The goal

Build an open, reproducible pipeline that finds **dark vessels** — ships visible in Sentinel-1
radar but not broadcasting on AIS — over a real study area and time period, and that reports
where they concentrate, with every claim traceable back to the evidence behind it.

The finished project answers three questions:

1. **Which radar detections have no AIS declaration to explain them?** Per scene, per
   detection, with the reason stated (matched to what, at what distance, interpolated or
   nearest, how much the azimuth correction moved it).
2. **Where and when do dark vessels concentrate?** Across the whole archive, related to
   distance from shore, depth, fishing effort and EEZ boundaries.
3. **How far can those results be trusted?** Detector precision/recall, what the AIS search
   actually covered — including *where the archive could hear at all* — and which results are
   only candidates for inspection.

## Deliverables

- [x] **Tier 1 core** — tiling, pixel→ground, AIS interpolation, azimuth correction, optimal
      matching, structure register, recurrence clustering, CLI. Runs with no network.
- [x] **Interactive viewer** — live and static, showing why each detection got its status.
- [x] **Reception-weighted dark claims** — done 2026-09-23 (`fusion/reception.py`): every row
      carries how reliably the archive would have placed a transmitting vessel there at the
      acquisition instant, estimated from the archive's own reporting intervals at the same
      `max_gap` the matcher uses. Below a stated floor a `dark` detection is reported
      `shadowed`; a *missing* estimate never is. Validated only against the synthetic fixture.
- [x] **The declaration side** — done 2026-09-23 (`fusion/declarations.py`): declarations no
      detection explains, as `undetected` / `below_detectable` / `outside_scene`, written as a
      second GeoPackage layer, plus a label-free estimate of detector recall on the scene.
- [ ] **Real data run** — at least one real Sentinel-1 scene fused with real DMA AIS,
      producing a sensible result.
- [~] **Trained detector** — done 2026-09-18 (`models/README.md`): beats CFAR offshore
      (F1 0.87 vs 0.69) and on overall AP (0.63 vs 0.56), loses inshore (F1 0.19 vs 0.28)
      until land is masked. Original goal: a CNN trained on LS-SSDD, satisfying the `Detector` protocol,
      with reported precision/recall and compared against a simple CFAR baseline.
- [ ] **Tier 2 contextual layers** — distance to shore, depth, fishing effort, EEZ from
      Earth Engine, joined to every detection.
- [ ] **Contrastive embeddings** — detection crops embedded for structure/vessel separation
      and similarity search.
- [ ] **Archive-wide run** — every scene in the study period, processed from a single config.
- [ ] **Concentration analysis and static map** — where dark detections cluster, normalised
      for how often each area was imaged.
- [ ] **Write-up** — README/report covering method, the design decisions, results and their limits.
- [x] **Published** — code on GitHub with CI green; the viewer on GitHub Pages.

## Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | Core + viewer | ✅ 150 tests pass, viewer works in browser |
| 2 | Online | ✅ Repo pushed, CI green, Pages bundle live |
| 3 | First real scene | One Sentinel-1 scene + DMA AIS runs end to end; results inspected by eye in the viewer |
| 4 | Real detector | Trained model beats the CFAR baseline on LS-SSDD test split; plugged into the pipeline |
| 5 | Context | Each detection carries its contextual variables; missing values are absent, not zero |
| 6 | Archive | Full study period processed reproducibly |
| 7 | Analysis | Concentration map and statistics, corrected for revisit frequency **and for AIS reception** — a map of raw dark counts is partly a map of where the receivers are |
| 8 | Write-up | Report finished, results stated with their uncertainty |

## Definition of done

- Anyone can clone the repo, run the quick start with no credentials, and get the README's
  numbers.
- Anyone with the listed accounts can reproduce the real-data results from the configs.
- Every `dark` detection can be opened in the viewer and its status explained.
- Detector performance and the AIS coverage behind each `dark` claim are reported alongside
  the findings, including the estimated reception at that position and the floor it had to
  clear.
- The declarations the radar failed to explain are reported too, and not quietly dropped.

## Out of scope

- Identifying specific vessels or owners, or claiming wrongdoing. A dark detection is a claim
  about the evidence searched, not a verdict. An `undetected` declaration is likewise a claim
  about two sensors disagreeing, not an accusation of spoofing.
- Real-time or operational monitoring.
- SAR sensors other than Sentinel-1.

## Still to decide

- Where the reception floor should sit on real data. 0.5 suits the synthetic fixture and
  nothing more; on a real archive it should follow from what the estimate looks like over the
  study area, and the choice has to be stated rather than inherited.
- Whether a bracket of reports should be bounded by `max_gap` at all — see the progress log's
  open questions. `position_span_s` now records the width, but nothing acts on it.
- Exact study area (the default CRS, EPSG:25832, points at Danish waters, which matches the DMA
  AIS source) and time period.
- Whether the target is a paper, a portfolio project, or a tool for others — this changes how
  much the write-up and hosting matter.
